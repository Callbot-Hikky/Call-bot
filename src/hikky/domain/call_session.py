from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from hikky.domain.dialogue_engine import DialogueEngine, TurnResult
from hikky.domain.fallback_policy import FallbackDecision, FallbackPolicy
from hikky.domain.outcomes import CallOutcome
from hikky.domain.reservation_intent import ReservationIntent
from hikky.domain.restaurant_context import RestaurantContext
from hikky.ports.call_log import CallLogPort
from hikky.ports.notification import NotificationPort
from hikky.ports.reservation import ReservationPort


@dataclass(slots=True)
class _SessionState:
    intent: ReservationIntent = field(default_factory=ReservationIntent)
    last_bot_message: str = ""
    consecutive_no_progress_turns: int = 0
    started_at: datetime | None = None
    ended: bool = False


class CallSession:
    def __init__(
        self,
        *,
        call_id: str,
        context: RestaurantContext,
        dialogue_engine: DialogueEngine,
        fallback_policy: FallbackPolicy,
        reservation_port: ReservationPort,
        call_log: CallLogPort,
        notification: NotificationPort,
        clock: Callable[[], datetime],
    ) -> None:
        self.call_id = call_id
        self.context = context
        self._engine = dialogue_engine
        self._fallback = fallback_policy
        self._reservation = reservation_port
        self._log = call_log
        self._notif = notification
        self._clock = clock
        self._state = _SessionState()

    async def begin(self) -> None:
        now = self._clock()
        self._state.started_at = now
        await self._log.start(self.call_id, self.context.id, now)

    async def process_user_turn(
        self, user_text: str, slot_updates: dict[str, Any]
    ) -> TurnResult:
        before = self._state.intent
        result = await self._engine.handle_turn(
            user_text=user_text,
            current_intent=before,
            slot_updates=slot_updates,
        )
        if result.updated_intent == before:
            self._state.consecutive_no_progress_turns += 1
        else:
            self._state.consecutive_no_progress_turns = 0
        self._state.intent = result.updated_intent
        self._state.last_bot_message = result.bot_says
        return result

    def check_fallback(
        self, *, user_requested_human: bool, group_size: int | None
    ) -> FallbackDecision | None:
        return self._fallback.decide(
            self.context,
            consecutive_no_progress_turns=self._state.consecutive_no_progress_turns,
            user_requested_human=user_requested_human,
            group_size=group_size,
        )

    async def finalize_if_complete(self, customer_phone: str | None) -> CallOutcome | None:
        intent = self._state.intent
        if not intent.is_complete():
            return None
        assert intent.date_time is not None
        assert intent.party_size is not None
        assert intent.customer_name is not None
        reservation_id = await self._reservation.create(
            restaurant_id=self.context.id,
            date_time=intent.date_time,
            party_size=intent.party_size,
            customer_name=intent.customer_name,
            customer_phone=customer_phone,
        )
        await self._notif.send_confirmation(reservation_id)
        await self._end(CallOutcome.RESERVATION_CREATED)
        return CallOutcome.RESERVATION_CREATED

    async def end_with(self, outcome: CallOutcome) -> None:
        await self._end(outcome)

    async def _end(self, outcome: CallOutcome) -> None:
        if self._state.ended:
            return
        now = self._clock()
        duration = (now - (self._state.started_at or now)).total_seconds()
        await self._log.end(self.call_id, outcome, duration)
        self._state.ended = True
