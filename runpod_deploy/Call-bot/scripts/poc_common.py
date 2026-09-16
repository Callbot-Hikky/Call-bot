from __future__ import annotations

import sys
from collections.abc import Callable
from datetime import datetime, time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from hikky.domain.call_session import CallSession  # noqa: E402
from hikky.domain.dialogue_engine import DialogueEngine  # noqa: E402
from hikky.domain.fallback_policy import FallbackPolicy  # noqa: E402
from hikky.domain.restaurant_context import (  # noqa: E402
    OpeningHours,
    RestaurantContext,
    RestaurantRules,
)


class StubRestaurantContextPort:
    def __init__(self, ctx: RestaurantContext | None = None) -> None:
        self._ctx = ctx or default_restaurant_context()

    async def load(self, called_number: str) -> RestaurantContext:
        return self._ctx


class NoopReservationPort:
    async def check_availability(self, *a, **k) -> bool:
        return True

    async def create(self, *a, **k) -> str:
        return "res-noop"

    async def create_callback_request(self, *a, **k) -> str:
        return "cb-noop"


class NoopCallLogPort:
    async def start(self, *a, **k) -> None:
        return None

    async def end(self, *a, **k) -> None:
        return None


class NoopNotificationPort:
    async def send_confirmation(self, *a, **k) -> None:
        return None


class NoopLanguageModel:
    async def complete(self, messages) -> str:
        return ""


def default_restaurant_context() -> RestaurantContext:
    return RestaurantContext(
        id="r-poc",
        name="Le Petit Sud",
        greeting="Bonjour, restaurant Le Petit Sud, que puis-je faire pour vous ?",
        opening_hours=[
            OpeningHours(weekday=i, opens=time(9, 0), closes=time(23, 0))
            for i in range(7)
        ],
        total_capacity=40,
        rules=RestaurantRules(),
        transfer_number=None,
        fallback_message="Je vous rappelle au plus vite, merci de votre appel.",
    )


def build_session_factory(
    llm, reservation_port=None
) -> Callable[[str, RestaurantContext], CallSession]:
    """`reservation_port` non fourni => port factice (rien n'est enregistré)."""

    def _factory(call_id: str, ctx: RestaurantContext) -> CallSession:
        return CallSession(
            call_id=call_id,
            context=ctx,
            dialogue_engine=DialogueEngine(llm),
            fallback_policy=FallbackPolicy(),
            reservation_port=reservation_port or NoopReservationPort(),
            call_log=NoopCallLogPort(),
            notification=NoopNotificationPort(),
            clock=datetime.now,
        )

    return _factory
