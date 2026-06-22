from dataclasses import dataclass, field
from datetime import datetime, time
from typing import Any

from hikky.domain.call_session import CallSession
from hikky.domain.dialogue_engine import DialogueEngine
from hikky.domain.fallback_policy import FallbackPolicy
from hikky.domain.outcomes import CallOutcome
from hikky.domain.restaurant_context import (
    OpeningHours,
    RestaurantContext,
    RestaurantRules,
)
from tests.fakes.fake_call_log import FakeCallLog
from tests.fakes.fake_language_model import FakeLanguageModel
from tests.fakes.fake_notification import FakeNotification
from tests.fakes.fake_reservation import FakeReservation


@dataclass
class UserSays:
    text: str
    slot_updates: dict[str, Any] = field(default_factory=dict)
    user_requested_human: bool = False
    declared_group_size: int | None = None


@dataclass
class ExpectBotContains:
    fragment: str


@dataclass
class ExpectBackCalled:
    method: str  # "create" | "check_availability" | "create_callback_request"


@dataclass
class ExpectOutcome:
    outcome: CallOutcome


Step = UserSays | ExpectBotContains | ExpectBackCalled | ExpectOutcome


def _default_context(
    rest_id: str = "r-1", transfer_number: str | None = None
) -> RestaurantContext:
    return RestaurantContext(
        id=rest_id,
        name="Chez Test",
        greeting="Bonjour.",
        opening_hours=[
            OpeningHours(weekday=d, opens=time(0, 0), closes=time(23, 59))
            for d in range(7)
        ],
        total_capacity=40,
        rules=RestaurantRules(max_group_size=8),
        transfer_number=transfer_number,
        fallback_message="…",
    )


class ConversationReplay:
    def __init__(self) -> None:
        self._context: RestaurantContext | None = None
        self._availability: bool = True
        self._replies: list[str] = []
        self.last_outcome: CallOutcome | None = None
        self.reservation = FakeReservation()
        self.call_log = FakeCallLog()
        self.notification = FakeNotification()
        self._last_bot_message: str = ""

    def with_default_context(self, **kwargs: Any) -> "ConversationReplay":
        self._context = _default_context(**kwargs)
        return self

    def with_context(self, ctx: RestaurantContext) -> "ConversationReplay":
        self._context = ctx
        return self

    def with_availability(self, available: bool) -> "ConversationReplay":
        self._availability = available
        return self

    def with_llm_replies(self, replies: list[str]) -> "ConversationReplay":
        self._replies = list(replies)
        return self

    def _make_session(self, call_id: str) -> CallSession:
        assert self._context is not None, "Call with_context(...) or with_default_context() first"
        self.reservation.set_availability(self._context.id, self._availability)
        llm = FakeLanguageModel(replies=self._replies)
        engine = DialogueEngine(llm)
        return CallSession(
            call_id=call_id,
            context=self._context,
            dialogue_engine=engine,
            fallback_policy=FallbackPolicy(),
            reservation_port=self.reservation,
            call_log=self.call_log,
            notification=self.notification,
            clock=lambda: datetime(2026, 7, 1, 20, 0),
        )

    async def run(
        self,
        steps: list[Step],
        *,
        call_id: str = "c-1",
        customer_phone: str | None = None,
    ) -> None:
        session = self._make_session(call_id)
        await session.begin()

        for step in steps:
            if isinstance(step, UserSays):
                result = await session.process_user_turn(step.text, step.slot_updates)
                self._last_bot_message = result.bot_says

                fallback = session.check_fallback(
                    user_requested_human=step.user_requested_human,
                    group_size=step.declared_group_size,
                )
                if fallback is not None:
                    if fallback.outcome == CallOutcome.CALLBACK_REQUESTED:
                        await self.reservation.create_callback_request(
                            restaurant_id=session.context.id,
                            customer_phone=customer_phone or "",
                            preferred_slot=None,
                            note=fallback.reason,
                        )
                    await session.end_with(fallback.outcome)
                    self.last_outcome = fallback.outcome
                    continue

                outcome = await session.finalize_if_complete(customer_phone=customer_phone)
                if outcome is not None:
                    self.last_outcome = outcome

            elif isinstance(step, ExpectBotContains):
                assert step.fragment.lower() in self._last_bot_message.lower(), (
                    f"Expected bot to say something containing {step.fragment!r}, "
                    f"got {self._last_bot_message!r}"
                )

            elif isinstance(step, ExpectBackCalled):
                if step.method == "create":
                    assert self.reservation.reservations, "Reservation.create was not called"
                elif step.method == "create_callback_request":
                    assert self.reservation.callback_requests, "Callback request was not created"
                else:
                    raise AssertionError(f"Unknown back method: {step.method}")

            elif isinstance(step, ExpectOutcome):
                assert self.last_outcome == step.outcome, (
                    f"Expected outcome {step.outcome}, got {self.last_outcome}"
                )

            else:
                raise AssertionError(f"Unknown step type: {type(step)}")
