"""L'aiguilleur : le code decide, le modele reste pour l'intelligence.

Mesure qui motive ce module : Q4 et Q6 echouent IDENTIQUEMENT face a une
dizaine de contraintes de prompt — les deux reposent trois fois « a
quelle heure ? » apres avoir recu l'heure. Ce n'est pas la capacite du
modele qui est en cause mais la charge. Le code sait exactement ce qui
manque ; il n'a besoin de personne pour choisir la question suivante.

Le modele reste sollicite pour ce qu'il reussit : comprendre les phrases
(l'extracteur) et repondre aux questions hors script.
"""

from datetime import date, time

from hikky.domain.question_router import Action, route_turn
from hikky.domain.reservation_intent import ReservationIntent


def _complet():
    return (
        ReservationIntent()
        .with_date(date(2026, 7, 22))
        .with_time(time(12, 0))
        .with_party_size(2)
        .with_customer_name("Dupont")
    )


# ── Detection : question du client ou reponse ? ─────────────────────────


def test_client_question_is_routed_to_the_model():
    """« Vous etes ouverts dimanche ? » doit atteindre le LLM."""
    d = route_turn(ReservationIntent(), "Vous êtes ouverts le dimanche ?", False)
    assert d.action is Action.ANSWER_QUESTION


def test_asking_about_the_recorded_name_is_a_question():
    """Bug reel : « A quel nom ? » recevait « C'est a quel nom ? »."""
    d = route_turn(_complet(), "À quel nom ?", False)
    assert d.action is Action.ANSWER_QUESTION


def test_plain_answer_is_not_treated_as_a_question():
    d = route_turn(ReservationIntent(), "Deux personnes", False)
    assert d.action is not Action.ANSWER_QUESTION


def test_repeat_request_is_routed_to_the_model():
    d = route_turn(_complet(), "Tu peux répéter ?", False)
    assert d.action is Action.ANSWER_QUESTION


# ── Questions hors parcours, formulees a l'oral (sortie STT, sans « ? ») ──
#
# Bug client : « est-ce que le restaurant est halal ? », « vous avez une
# terrasse ? » restaient sans reponse. Transcriptions reelles relevees dans
# /workspace/stt.log : le « oui »/« non » de politesse en tete ou « est ce
# que » sans tiret suffisaient a rater la question.

import pytest

from hikky.domain.question_router import is_client_question


@pytest.mark.parametrize(
    "phrase",
    [
        # Transcriptions reelles (stt.log du pod).
        "Bonjour, je voulais vous demander est-ce que votre restaurant il est halage?",
        "Bonjour et je voulais vous demander est ce que votre restaurant Halal",
        "Oui, bonjour, je voulais savoir est ce que revient de Halal? ",
        "Non, non, je voulais vous poser la question, c'est quoi l'ambiance du restaurant s'il vous plaît? ",
        "Est-ce qu'il y a une terrasse? ",
        # Tournures orales sans point d'interrogation.
        "vous avez une terrasse",
        "c'est halal",
        "le restaurant est halal",
        "vous faites des plats végétariens",
        "il y a un parking",
        "vous acceptez les chiens",
        "on peut venir avec un chien",
        "c'est cher",
        "vous acceptez les tickets resto",
        "vous avez un menu enfant",
        "vous êtes ouverts le dimanche",
        "vous faites à emporter",
        "euh vous avez des options sans gluten",
        "oui et vous avez une terrasse",
        "c'est où exactement",
    ],
)
def test_oral_question_without_question_mark_is_detected(phrase):
    assert is_client_question(phrase), phrase
    assert route_turn(ReservationIntent(), phrase, False).action is Action.ANSWER_QUESTION


@pytest.mark.parametrize(
    "phrase",
    [
        "oui",
        "Oui, c'est cela.",
        "Ouais, à peu près",
        "non",
        "Non, pour Ryan.",
        "pour quatre personnes",
        "demain à 20 heures",
        "J'aimerais avoir une table pour ce soir à vingt heures.",
        "Je voudrais réserver pour demain.",
        "il y a quatre personnes",
        "on sera quatre avec deux enfants",
        "Ça fera au nom de Ryan.",
        "Merci, ma belle.",
        "soixante-dix personnes",
        "voilà",
        "Dupont",
    ],
)
def test_plain_answers_are_not_questions(phrase):
    assert not is_client_question(phrase), phrase


def test_agreement_with_pending_confirmation_still_books_despite_polite_prefix():
    assert route_turn(_complet(), "Oui oui, c'est bon pour moi", True).action is Action.BOOK


def test_refusal_with_pending_confirmation_still_corrects():
    assert route_turn(_complet(), "Non, pour Ryan", True).action is Action.CORRECT


def test_question_while_awaiting_confirmation_is_answered_not_booked():
    """« Oui, et vous avez une terrasse ? » : on repond, on ne reserve pas a l'aveugle."""
    d = route_turn(_complet(), "Oui, et vous avez une terrasse ?", True)
    assert d.action is Action.ANSWER_QUESTION


def test_slot_answer_with_polite_prefix_stays_on_track():
    intent = ReservationIntent().with_date(date(2026, 7, 22)).with_time(time(12))
    d = route_turn(intent, "Oui, bonjour, pour quatre personnes", False)
    assert d.action is Action.ASK_SLOT


# ── Collecte pilotee par le code ────────────────────────────────────────


def test_missing_date_is_asked_first():
    d = route_turn(ReservationIntent(), "bonjour", False)
    assert d.action is Action.ASK_SLOT
    assert d.slot == "date"


def test_time_is_asked_once_the_day_is_known():
    intent = ReservationIntent().with_date(date(2026, 7, 22))
    d = route_turn(intent, "demain", False)
    assert d.slot == "time"


def test_party_size_then_name():
    intent = ReservationIntent().with_date(date(2026, 7, 22)).with_time(time(12))
    assert route_turn(intent, "midi", False).slot == "party_size"
    intent = intent.with_party_size(2)
    assert route_turn(intent, "deux", False).slot == "customer_name"


def test_a_known_slot_is_never_asked_again():
    """Le defaut central : le modele redemandait l'heure en la connaissant."""
    intent = _complet()
    d = route_turn(intent, "voila", False)
    assert d.action is not Action.ASK_SLOT


# ── Confirmation et reservation ─────────────────────────────────────────


def test_complete_intent_asks_for_confirmation():
    d = route_turn(_complet(), "Dupont", False)
    assert d.action is Action.CONFIRM


def test_agreement_while_awaiting_confirmation_books():
    d = route_turn(_complet(), "oui c'est ça", True)
    assert d.action is Action.BOOK


def test_refusal_while_awaiting_confirmation_reopens():
    d = route_turn(_complet(), "non, pas du tout", True)
    assert d.action is Action.CORRECT


def test_agreement_without_pending_confirmation_does_not_book():
    d = route_turn(_complet(), "oui", False)
    assert d.action is not Action.BOOK


def test_incomplete_intent_never_books():
    intent = ReservationIntent().with_date(date(2026, 7, 22))
    d = route_turn(intent, "oui je confirme", True)
    assert d.action is not Action.BOOK


# ── Formulations naturelles et variees ──────────────────────────────────


def test_slot_questions_are_natural_french():
    from hikky.domain.question_router import phrase_for_slot

    for slot in ("date", "time", "party_size", "customer_name"):
        q = phrase_for_slot(slot, set())
        assert q.endswith("?"), q
        assert len(q.split()) <= 12, q


def test_phrasing_varies_between_turns():
    """Une formulation figee sonne robotique au bout de deux appels."""
    from hikky.domain.question_router import phrase_for_slot

    vues = {phrase_for_slot("time", set()) for _ in range(30)}
    assert len(vues) > 1, "formulation unique"


def test_phrasing_avoids_recently_used_wording():
    from hikky.domain.question_router import phrase_for_slot

    premiere = phrase_for_slot("time", set())
    seconde = phrase_for_slot("time", {premiere})
    assert seconde != premiere


# Appel du 2026-10-03 16:56 : deux questions sur le végétarien routées en « quel jour ? ».
QUESTIONS_INDIRECTES_OU_MAL_ENTENDUES = [
    # Question indirecte : « demander si » n'était pas une tournure reconnue.
    "Je voulais demander si le restaurant propose des plats végétariens ou végan.",
    "Je voudrais demander si vous avez une terrasse.",
    # « Est-ce que » transcrit « Parce que » par le STT : le sujet (le restaurant +
    # un thème) suffit — personne n'informe le bot sur son propre restaurant.
    "Parce que le restaurant propose des plats végétariens, ou végan.",
    "Le restaurant propose des plats halal.",
    "Le restaurant accepte les chiens ?",
]


@pytest.mark.parametrize("texte", QUESTIONS_INDIRECTES_OU_MAL_ENTENDUES)
def test_une_question_indirecte_ou_mal_entendue_reste_une_question(texte):
    assert is_client_question(texte), texte


@pytest.mark.parametrize("texte", [
    "Le restaurant, c'est pour demain soir.",
    "Parce que nous serons quatre.",
    "On sera six personnes.",
])
def test_une_reponse_sur_la_reservation_n_est_pas_une_question(texte):
    assert not is_client_question(texte), texte


# ── Phase 0 (2026-10-03) : le routeur reçoit le CONTEXTE ──────────────────────
# Un accord ou un refus n'a de sens qu'en réponse à une question fermée du bot.
# Appel réel : « Je vous ai demandé si le restaurant est halal » classé « accord »
# parce que « si » figurait dans la liste des accords — testé sur toute phrase.


def test_si_dans_une_question_indirecte_n_est_pas_un_accord():
    d = route_turn(ReservationIntent(), "Je vous ai demandé si le restaurant est halal.", False)
    assert d.action is Action.ANSWER_QUESTION


def test_si_seul_en_reponse_a_une_confirmation_est_un_accord():
    for texte in ("Si.", "si si", "Mais si !"):
        d = route_turn(_complet(), texte, True)
        assert d.action is Action.BOOK, texte


def test_oui_hors_confirmation_n_est_ni_question_ni_reservation():
    d = route_turn(ReservationIntent().with_date(date(2026, 10, 4)), "oui", False)
    assert d.action is Action.ASK_SLOT and d.slot == "time"


def test_une_reponse_incertaine_a_la_confirmation_refait_le_recapitulatif():
    for texte in ("euh, peut-être", "je ne sais pas", "je sais pas trop", "attendez"):
        d = route_turn(_complet(), texte, True)
        assert d.action is Action.RECONFIRM, texte


def test_une_demande_de_changement_pendant_la_confirmation_est_une_correction():
    for texte in ("non, plutôt 21 heures", "je voudrais changer l'heure", "modifier le nombre"):
        d = route_turn(_complet(), texte, True)
        assert d.action is Action.CORRECT, texte


def test_changer_hors_confirmation_n_est_pas_un_refus():
    # Sans question fermée en attente, « changer » n'a rien à refuser : on continue.
    d = route_turn(ReservationIntent().with_date(date(2026, 10, 4)), "je voudrais changer de jour", False)
    assert d.action is not Action.CORRECT


@pytest.mark.parametrize("texte", [
    "Je vous ai demandé si le restaurant est halal.",
    "Vous avez rien compris, je vous ai demandé si le restaurant est halal.",
    "J'ai demandé si vous aviez une terrasse.",
    "Je répète : est-ce que vous avez un parking ?",
])
def test_les_questions_repetees_restent_des_questions(texte):
    assert is_client_question(texte), texte
