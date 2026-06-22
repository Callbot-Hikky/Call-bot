from datetime import datetime, time

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


def _ctx() -> RestaurantContext:
    return RestaurantContext(
        id="r-1",
        name="Chez Test",
        greeting="Bonjour, Chez Test.",
        opening_hours=[
            OpeningHours(weekday=d, opens=time(19, 0), closes=time(23, 0))
            for d in range(7)
        ],
        total_capacity=40,
        rules=RestaurantRules(max_group_size=8),
        transfer_number=None,
        fallback_message="…",
    )


async def _make_session(
    llm_replies: list[str],
) -> tuple[CallSession, FakeReservation, FakeCallLog, FakeNotification]:
    log = FakeCallLog()
    reservation = FakeReservation()
    notif = FakeNotification()
    session = CallSession(
        call_id="c-1",
        context=_ctx(),
        dialogue_engine=DialogueEngine(FakeLanguageModel(replies=llm_replies)),
        fallback_policy=FallbackPolicy(),
        reservation_port=reservation,
        call_log=log,
        notification=notif,
        clock=lambda: datetime(2026, 7, 1, 19, 30),
    )
    await session.begin()
    return session, reservation, log, notif


async def test_session_logs_start_on_begin():
    _, _, log, _ = await _make_session([])
    assert log.entries[0][0] == "start"


async def test_full_reservation_flow_ends_with_created_outcome():
    session, reservation, log, notif = await _make_session(
        llm_replies=[
            "À quelle heure ?",
            "Pour combien de personnes ?",
            "À quel nom ?",
            "Confirmé, à bientôt.",
        ]
    )
    reservation.set_availability("r-1", available=True)

    await session.process_user_turn("Je voudrais réserver", slot_updates={})
    await session.process_user_turn(
        "Demain 20h", slot_updates={"date_time": datetime(2026, 7, 2, 20)}
    )
    await session.process_user_turn("4 personnes", slot_updates={"party_size": 4})
    await session.process_user_turn("Dupont", slot_updates={"customer_name": "Dupont"})

    await session.finalize_if_complete(customer_phone="+33600000000")

    assert len(reservation.reservations) == 1
    assert notif.sent
    assert log.entries[-1][0] == "end"
    assert log.entries[-1][2] == CallOutcome.RESERVATION_CREATED


async def test_finalize_refuses_when_slot_unavailable():
    session, reservation, _, notif = await _make_session(
        llm_replies=["Quand ?", "Combien ?", "Nom ?", "..."]
    )
    reservation.set_availability("r-1", available=False)

    await session.process_user_turn(
        "Demain 20h", slot_updates={"date_time": datetime(2026, 7, 2, 20)}
    )
    await session.process_user_turn("4", slot_updates={"party_size": 4})
    await session.process_user_turn("Dupont", slot_updates={"customer_name": "Dupont"})

    outcome = await session.finalize_if_complete(customer_phone="+33600000000")
    assert outcome is None
    assert reservation.reservations == {}
    assert notif.sent == []
    assert session.last_availability_check is False


async def test_finalize_refuses_when_date_outside_opening_hours():
    session, reservation, _, notif = await _make_session(
        llm_replies=["Quand ?", "Combien ?", "Nom ?", "..."]
    )
    reservation.set_availability("r-1", available=True)

    # Restaurant ouvert 19h-23h ; on tente une réservation à 12h
    await session.process_user_turn(
        "Demain midi", slot_updates={"date_time": datetime(2026, 7, 2, 12)}
    )
    await session.process_user_turn("4", slot_updates={"party_size": 4})
    await session.process_user_turn("Dupont", slot_updates={"customer_name": "Dupont"})

    outcome = await session.finalize_if_complete(customer_phone="+33600000000")
    assert outcome is None
    assert reservation.reservations == {}
    assert notif.sent == []
    assert session.last_within_opening_hours is False
