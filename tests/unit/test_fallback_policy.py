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
