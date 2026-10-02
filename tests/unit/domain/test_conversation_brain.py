"""L'answerer repond avec les faits du restaurant — et reste honnete sans eux.

Observe en production (orch_new.log, 2026-10-02 12:34) :

    CLIENT : Bonjour, je voulais vous demander est-ce que votre restaurant il est halage?
    route   : answer_question
    BOT    : Je suis désolé, mais je n'ai pas d'information concernant l'halal
             de notre restaurant. Je vous conseille de contacter notre gérant…

alors que le backend renvoyait `attributes.dietary.halal = true`. Le modele
n'avait simplement jamais recu l'information.
"""

from datetime import time

from hikky.domain.conversation_brain import (
    _FAUSSE_CONFIRMATION,
    _REPONSE_INCONNUE,
    _REPONSE_SURE,
    QuestionAnswerer,
    _format_facts,
)
from hikky.domain.reservation_intent import ReservationIntent
from hikky.domain.restaurant_context import (
    OpeningHours,
    RestaurantContext,
    RestaurantRules,
)

ATTRIBUTS_PETIT_SUD = {
    "details": {
        "ambiance": ["Chaleureuse", "Familiale"],
        "price_range": "€€",
        "cuisine_type": ["Française traditionnelle", "Pizzeria"],
    },
    "dietary": {"halal": True, "vegan": False, "vegetarian": True, "gluten_free": True},
    "equipments": {"terrace": True, "pets_allowed": False, "private_parking": False},
    "payments": {"meal_vouchers": True, "takeaway": True},
}


def _ctx(attributes=None, address="12 rue de la République, 69002 Lyon"):
    return RestaurantContext(
        id="r-1",
        name="Le Petit Sud",
        greeting="Bonjour",
        opening_hours=[OpeningHours(weekday=i, opens=time(12), closes=time(23)) for i in range(7)],
        total_capacity=40,
        rules=RestaurantRules(max_group_size=15),
        fallback_message="Au revoir",
        address=address,
        attributes=attributes if attributes is not None else ATTRIBUTS_PETIT_SUD,
    )


class FakeLLM:
    def __init__(self, reponse):
        self.reponse = reponse
        self.messages = None

    async def complete(self, messages):
        self.messages = messages
        return self.reponse


async def _answer(llm, question, ctx=None):
    return await QuestionAnswerer(llm).answer(
        user_text=question, intent=ReservationIntent(), context=ctx or _ctx(), history=[]
    )


# ── Les faits atteignent le modele ──────────────────────────────────────


async def test_the_prompt_carries_the_restaurant_facts():
    llm = FakeLLM("Oui, notre restaurant est halal.")
    await _answer(llm, "c'est halal ?")
    system = llm.messages[0]["content"]
    assert "- halal : oui" in system
    assert "- terrasse : oui" in system
    assert "- animaux acceptés : non" in system
    assert "- options végétariennes : oui" in system
    assert "- tickets restaurant acceptés : oui" in system
    assert "- gamme de prix : prix modérés" in system
    assert "- ambiance : Chaleureuse, Familiale" in system
    assert "ADRESSE : 12 rue de la République, 69002 Lyon" in system
    assert "quinze mots maximum" in system


async def test_a_factual_answer_is_spoken_as_is():
    assert await _answer(FakeLLM("Oui, nous avons une terrasse."), "vous avez une terrasse") == (
        "Oui, nous avons une terrasse."
    )


async def test_confirming_a_fact_is_not_mistaken_for_a_booking_confirmation():
    """« je vous confirme que nous avons une terrasse » est legitime."""
    reponse = "Oui, je vous confirme que nous avons une terrasse."
    assert await _answer(FakeLLM(reponse), "vous avez une terrasse") == reponse


async def test_a_fake_booking_confirmation_is_still_neutralised():
    for texte in (
        "Votre réservation est confirmée pour ce soir à vingt et une heures.",
        "C'est noté, à ce soir !",
        "Je vous confirme la réservation.",
        "Votre table est réservée.",
    ):
        assert await _answer(FakeLLM(texte), "c'est bon pour ce soir ?") == _REPONSE_SURE, texte


def test_guard_regex_does_not_match_legitimate_fact_answers():
    for texte in (
        "Oui, nous sommes halal.",
        "Oui, je vous confirme que nous acceptons les tickets restaurant.",
        "Non, les animaux ne sont pas acceptés.",
    ):
        assert not _FAUSSE_CONFIRMATION.search(texte), texte


# ── Sans information : court, honnete, et on avance ─────────────────────


async def test_unknown_answer_is_shortened_to_an_honest_sentence():
    long = (
        "Je suis désolé, mais je n'ai pas d'information concernant l'halal de "
        "notre restaurant. Je vous conseille de contacter notre gérant pour plus de détails."
    )
    reponse = await _answer(FakeLLM(long), "c'est halal ?", _ctx(attributes={}))
    assert reponse == _REPONSE_INCONNUE
    assert len(reponse.split()) <= 15


async def test_unknown_sentinel_from_prompt_is_recognised():
    reponse = await _answer(FakeLLM("Je n'ai pas cette information."), "vous avez un voiturier ?")
    assert reponse == _REPONSE_INCONNUE


def test_facts_without_attributes_say_so():
    assert "aucune information" in _format_facts({})
    assert "aucune information" in _format_facts(None)


def test_unknown_keys_are_still_exposed():
    faits = _format_facts({"custom": {"brunch_dimanche": True, "dress_code": "décontracté"}})
    assert "- brunch dimanche : oui" in faits
    assert "- dress code : décontracté" in faits
