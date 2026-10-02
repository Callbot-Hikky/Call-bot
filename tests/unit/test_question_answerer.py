"""Réponse aux questions hors script, avec la base de connaissances.

Le défaut de départ : le modèle ne connaissait que les horaires et la
capacité. À « vous avez une terrasse ? », il inventait. Trois choses sont
vérifiées ici : il reçoit ce que le restaurateur a écrit, il ne reçoit
pas ce qui ne ressemble pas à la question, et quand rien ne répond il le
dit et remonte la question au lieu de broder.
"""

import asyncio
from datetime import time

from hikky.domain.conversation_brain import (
    REPLY_UNKNOWN,
    REPLY_UNKNOWN_REPORTED,
    QuestionAnswerer,
)
from hikky.domain.knowledge import KnowledgePassage
from hikky.domain.reservation_intent import ReservationIntent
from hikky.domain.restaurant_context import (
    OpeningHours,
    RestaurantContext,
    RestaurantRules,
)
from tests.fakes.fake_knowledge import FakeKnowledge
from tests.fakes.fake_language_model import FakeLanguageModel


def _contexte(**surcharges) -> RestaurantContext:
    base = dict(
        id="r1",
        name="Le Petit Sud",
        greeting="Bonjour.",
        opening_hours=[OpeningHours(weekday=4, opens=time(19, 0), closes=time(23, 0))],
        total_capacity=40,
        rules=RestaurantRules(max_group_size=8),
        fallback_message="Je vous rappelle.",
    )
    base.update(surcharges)
    return RestaurantContext(**base)


async def _repondre(answerer, question="Vous avez un menu enfant ?", contexte=None):
    reponse = await answerer.answer(
        user_text=question,
        intent=ReservationIntent(),
        context=contexte or _contexte(),
        history=[],
    )
    await answerer.wait_pending()
    return reponse


def _prompt(llm: FakeLanguageModel) -> str:
    return llm.calls[0][0]["content"]


async def test_le_modele_recoit_ce_que_le_restaurateur_a_ecrit():
    llm = FakeLanguageModel(["Oui, un menu enfant à neuf euros."])
    base = FakeKnowledge([KnowledgePassage("Menu enfant", "Un menu enfant à neuf euros.", 0.82)])

    reponse = await _repondre(QuestionAnswerer(llm, knowledge=base))

    assert reponse == "Oui, un menu enfant à neuf euros."
    assert base.searched == ["Vous avez un menu enfant ?"]
    assert "CE QUE LE RESTAURANT A ÉCRIT" in _prompt(llm)
    assert "Un menu enfant à neuf euros." in _prompt(llm)
    assert base.reported == []


async def test_un_passage_trop_eloigne_n_est_pas_montre_au_modele():
    llm = FakeLanguageModel(["INCONNU"])
    base = FakeKnowledge([KnowledgePassage("Parking", "Parking Jaurès à deux minutes.", 0.12)])

    await _repondre(QuestionAnswerer(llm, knowledge=base, min_score=0.35))

    assert "Parking Jaurès" not in _prompt(llm)
    assert "CE QUE LE RESTAURANT A ÉCRIT" not in _prompt(llm)


async def test_sans_reponse_le_bot_le_dit_et_remonte_la_question():
    llm = FakeLanguageModel(["INCONNU."])
    base = FakeKnowledge([])

    answerer = QuestionAnswerer(llm, knowledge=base)
    reponse = await _repondre(answerer, "On peut venir avec un chien ?")

    # La phrase est écrite par le code : le mot du modèle n'est jamais prononcé.
    assert reponse == REPLY_UNKNOWN_REPORTED
    assert base.reported == ["On peut venir avec un chien ?"]


async def test_sans_base_branchee_le_bot_ne_promet_pas_de_transmettre():
    llm = FakeLanguageModel(["inconnu"])

    reponse = await _repondre(QuestionAnswerer(llm))

    assert reponse == REPLY_UNKNOWN


async def test_les_attributs_du_restaurant_sont_dans_le_prompt():
    llm = FakeLanguageModel(["Oui, nous avons une terrasse."])
    contexte = _contexte(
        attributes={"terrasse": True, "parking": False, "cuisine": "méditerranéenne"}
    )

    await _repondre(QuestionAnswerer(llm), "Vous avez une terrasse ?", contexte)

    prompt = _prompt(llm)
    assert "INFORMATIONS PRATIQUES" in prompt
    assert "- terrasse : oui" in prompt
    assert "- parking : non" in prompt
    assert "- cuisine : méditerranéenne" in prompt


async def test_une_base_trop_lente_ne_fait_pas_attendre_le_client():
    class _Lente(FakeKnowledge):
        async def search(self, question):
            await asyncio.sleep(1)
            return [KnowledgePassage("Trop tard", "Arrive après le délai.", 0.9)]

    llm = FakeLanguageModel(["Je regarde."])
    answerer = QuestionAnswerer(llm, knowledge=_Lente(), knowledge_timeout_seconds=0.01)

    assert await _repondre(answerer) == "Je regarde."
    assert "Arrive après le délai." not in _prompt(llm)


async def test_le_seuil_par_defaut_separe_les_scores_mesures_sur_le_site():
    # Chèques vacances (0,34) ne doit pas passer ; « s'en griller une » (0,49) doit.
    llm = FakeLanguageModel(["INCONNU", "Oui, deux coins fumeurs."])
    hors_sujet = FakeKnowledge([KnowledgePassage("Coin fumeur", "Deux coins fumeurs.", 0.34)])
    a_propos = FakeKnowledge([KnowledgePassage("Coin fumeur", "Deux coins fumeurs.", 0.49)])

    await _repondre(QuestionAnswerer(llm, knowledge=hors_sujet), "Des chèques vacances ?")
    await _repondre(QuestionAnswerer(llm, knowledge=a_propos), "On peut s'en griller une ?")

    assert "Deux coins fumeurs." not in llm.calls[0][0]["content"]
    assert "Deux coins fumeurs." in llm.calls[1][0]["content"]


async def test_un_passage_long_est_tronque():
    llm = FakeLanguageModel(["D'accord."])
    base = FakeKnowledge([KnowledgePassage("Carte", "x" * 2000, 0.9)])

    await _repondre(QuestionAnswerer(llm, knowledge=base))

    assert "x" * 400 in _prompt(llm)
    assert "x" * 401 not in _prompt(llm)
