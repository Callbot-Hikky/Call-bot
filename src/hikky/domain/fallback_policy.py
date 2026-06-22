from dataclasses import dataclass

from hikky.domain.outcomes import CallOutcome
from hikky.domain.restaurant_context import RestaurantContext

NO_PROGRESS_THRESHOLD = 3


@dataclass(frozen=True, slots=True)
class FallbackDecision:
    outcome: CallOutcome
    reason: str
    transfer_destination: str | None = None


class FallbackPolicy:
    def decide(
        self,
        ctx: RestaurantContext,
        *,
        consecutive_no_progress_turns: int,
        user_requested_human: bool,
        group_size: int | None,
    ) -> FallbackDecision | None:
        if user_requested_human:
            if ctx.transfer_number:
                return FallbackDecision(
                    outcome=CallOutcome.TRANSFERRED,
                    reason="human_requested",
                    transfer_destination=ctx.transfer_number,
                )
            return FallbackDecision(
                outcome=CallOutcome.CALLBACK_REQUESTED,
                reason="human_requested_no_transfer_configured",
            )

        if group_size is not None and group_size > ctx.rules.max_group_size:
            return FallbackDecision(
                outcome=CallOutcome.CALLBACK_REQUESTED, reason="oversized_group"
            )

        if consecutive_no_progress_turns >= NO_PROGRESS_THRESHOLD:
            return FallbackDecision(
                outcome=CallOutcome.CALLBACK_REQUESTED, reason="no_progress"
            )

        return None
