from datetime import time

from hikky.domain.fallback_policy import FallbackDecision, FallbackPolicy
from hikky.domain.outcomes import CallOutcome
from hikky.domain.restaurant_context import (
    OpeningHours,
    RestaurantContext,
    RestaurantRules,
)


def _ctx(transfer_number: str | None = None, max_group: int = 8) -> RestaurantContext:
    return RestaurantContext(
        id="r-1",
        name="Chez Test",
        greeting="Bonjour.",
        opening_hours=[
            OpeningHours(weekday=d, opens=time(19, 0), closes=time(23, 0))
            for d in range(7)
        ],
        total_capacity=40,
        rules=RestaurantRules(max_group_size=max_group),
        transfer_number=transfer_number,
        fallback_message="…",
    )


def test_no_progress_three_turns_yields_callback():
    policy = FallbackPolicy()
    decision = policy.decide(
        _ctx(),
        consecutive_no_progress_turns=3,
        user_requested_human=False,
        group_size=None,
    )
    assert decision == FallbackDecision(
        outcome=CallOutcome.CALLBACK_REQUESTED, reason="no_progress"
    )


def test_user_request_human_with_transfer_configured_yields_transfer():
    policy = FallbackPolicy()
    decision = policy.decide(
        _ctx(transfer_number="+33100000099"),
        consecutive_no_progress_turns=0,
        user_requested_human=True,
        group_size=None,
    )
    assert decision.outcome == CallOutcome.TRANSFERRED
    assert decision.transfer_destination == "+33100000099"


def test_user_request_human_without_transfer_yields_callback():
    policy = FallbackPolicy()
    decision = policy.decide(
        _ctx(transfer_number=None),
        consecutive_no_progress_turns=0,
        user_requested_human=True,
        group_size=None,
    )
    assert decision.outcome == CallOutcome.CALLBACK_REQUESTED
    assert decision.reason == "human_requested_no_transfer_configured"


def test_oversized_group_yields_callback():
    policy = FallbackPolicy()
    decision = policy.decide(
        _ctx(max_group=8),
        consecutive_no_progress_turns=0,
        user_requested_human=False,
        group_size=20,
    )
    assert decision.outcome == CallOutcome.CALLBACK_REQUESTED
    assert decision.reason == "oversized_group"


def test_no_trigger_returns_none():
    policy = FallbackPolicy()
    assert (
        policy.decide(
            _ctx(),
            consecutive_no_progress_turns=1,
            user_requested_human=False,
            group_size=4,
        )
        is None
    )


def test_clarification_request_is_not_counted_as_stalling():
    """« Tu peux repeter ? » signale un bot defaillant, pas un client bloque.

    En appel reel, cette phrase a ete comptee comme un tour sans progres
    et a contribue a raccrocher au nez d'un client qui cooperait.
    """
    from hikky.domain.fallback_policy import is_clarification_request

    for phrase in [
        "Hein ? T'as dit quoi ? Tu peux répéter ?",
        "pardon ?",
        "je n'ai pas compris",
        "comment ?",
        "vous pouvez répéter s'il vous plaît",
    ]:
        assert is_clarification_request(phrase) is True, phrase


def test_normal_answers_are_not_clarification_requests():
    from hikky.domain.fallback_policy import is_clarification_request

    for phrase in ["demain matin", "pour quatre personnes", "au nom de Dupont", "oui"]:
        assert is_clarification_request(phrase) is False, phrase


def test_contesting_a_recorded_fact_is_not_stalling():
    """« Je t'ai jamais dit que je m'appelais X » signale une erreur du bot.

    Appel reel : le client a conteste trois fois un nom mal transcrit,
    et le repli l'a raccroche au nez pendant qu'il tentait de corriger.
    """
    from hikky.domain.fallback_policy import is_correction

    for phrase in [
        "Je t'ai jamais dit que je m'appelais Monsieur Medica",
        "Et qui est M. Medica ?",
        "ce n'est pas mon nom",
        "j'ai jamais dit ça",
        "non c'est faux",
    ]:
        assert is_correction(phrase) is True, phrase


def test_plain_answers_are_not_corrections():
    from hikky.domain.fallback_policy import is_correction

    for phrase in ["demain matin", "pour quatre personnes", "oui", "Dupont"]:
        assert is_correction(phrase) is False, phrase
