from datetime import datetime, time

import pytest

from hikky.domain.outcomes import CallOutcome
from hikky.domain.restaurant_context import (
    OpeningHours,
    RestaurantContext,
    RestaurantRules,
)
from hikky.exceptions import UnknownRestaurant
from tests.fakes.fake_call_log import FakeCallLog
from tests.fakes.fake_language_model import FakeLanguageModel
from tests.fakes.fake_notification import FakeNotification
from tests.fakes.fake_reservation import FakeReservation
from tests.fakes.fake_restaurant_context import FakeRestaurantContext
from tests.fakes.fake_telephony import FakeTelephony


def _ctx(rest_id="r-1") -> RestaurantContext:
    return RestaurantContext(
        id=rest_id,
        name="Chez Test",
        greeting="Bonjour.",
        opening_hours=[
            OpeningHours(weekday=d, opens=time(19, 0), closes=time(23, 0))
            for d in range(7)
        ],
        total_capacity=40,
        rules=RestaurantRules(),
        transfer_number=None,
        fallback_message="…",
    )


async def test_fake_restaurant_context_returns_registered():
    fake = FakeRestaurantContext()
    fake.register("+33100000001", _ctx("r-1"))
    ctx = await fake.load("+33100000001")
    assert ctx.id == "r-1"


async def test_fake_restaurant_context_raises_on_unknown():
    fake = FakeRestaurantContext()
    with pytest.raises(UnknownRestaurant):
        await fake.load("+33199999999")


async def test_fake_language_model_returns_scripted_in_order():
    fake = FakeLanguageModel(replies=["bonjour", "merci"])
    assert await fake.complete([]) == "bonjour"
    assert await fake.complete([]) == "merci"


async def test_fake_reservation_check_and_create_and_callback():
    fake = FakeReservation()
    fake.set_availability("r-1", available=True)
    assert await fake.check_availability("r-1", datetime(2026, 7, 1, 20), 4) is True
    rid = await fake.create("r-1", datetime(2026, 7, 1, 20), 4, "Dupont", "+33600000000")
    assert rid in fake.reservations
    cid = await fake.create_callback_request("r-1", "+33600000000", "ce soir", "trois échecs")
    assert cid in fake.callback_requests


async def test_fake_call_log_records_lifecycle():
    fake = FakeCallLog()
    await fake.start("c-1", "r-1", datetime(2026, 7, 1, 20))
    await fake.end("c-1", CallOutcome.RESERVATION_CREATED, 42.0)
    assert fake.entries[-1] == ("end", "c-1", CallOutcome.RESERVATION_CREATED, 42.0)


async def test_fake_telephony_records_actions():
    fake = FakeTelephony()
    await fake.send_audio("c-1", b"\x00\x01")
    await fake.transfer("c-1", "+33100000099")
    await fake.hang_up("c-1")
    kinds = [a[0] for a in fake.actions]
    assert kinds == ["send_audio", "transfer", "hang_up"]


async def test_fake_notification_records_sends():
    fake = FakeNotification()
    await fake.send_confirmation("res-123")
    assert fake.sent == ["res-123"]
