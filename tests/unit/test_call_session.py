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


async def test_finalize_renvoie_none_sur_conflit_de_reservation():
    """Un 409 du Back (créneau qui vient d'être pris, résa déjà en attente…)
    ne doit PAS être présenté comme un succès : `finalize` renvoie None, rien
    n'est notifié, et le tour bascule sur « créneau indisponible »."""
    from hikky.exceptions import ReservationConflict

    session, reservation, _, notif = await _make_session(
        llm_replies=["Quand ?", "Combien ?", "Nom ?", "..."]
    )
    reservation.set_availability("r-1", available=True)
    reservation.create_error = ReservationConflict("POST /api/calls/ingest: 409")

    await session.process_user_turn(
        "Demain 20h", slot_updates={"date_time": datetime(2026, 7, 2, 20)}
    )
    await session.process_user_turn("4", slot_updates={"party_size": 4})
    await session.process_user_turn("Dupont", slot_updates={"customer_name": "Dupont"})

    outcome = await session.finalize_if_complete(customer_phone="+33600000000")
    assert outcome is None
    assert notif.sent == []


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


async def test_contested_name_is_cleared_from_the_intent():
    """Un slot doit pouvoir etre efface quand le client le conteste.

    Sans ca, une erreur de transcription devient definitive : le bot a
    appele le client « Monsieur Medica » jusqu'a le faire raccrocher.
    """
    from hikky.domain.reservation_intent import ReservationIntent

    intent = ReservationIntent().with_customer_name("Medica")
    cleared = intent.without("customer_name")
    assert cleared.customer_name is None
    assert "customer_name" in cleared.missing_slots()


async def test_clearing_an_unknown_slot_is_a_no_op():
    from hikky.domain.reservation_intent import ReservationIntent

    intent = ReservationIntent().with_party_size(4)
    assert intent.without("inexistant").party_size == 4


async def test_contesting_the_name_clears_it_and_does_not_count_as_stalling():
    """Scenario reel complet : nom mal transcrit, puis conteste trois fois."""
    session, _, _, _ = await _make_session(["ok", "ok", "ok"])

    await session.process_user_turn("Medica", {"customer_name": "Medica"})
    assert session.intent.customer_name == "Medica"

    await session.process_user_turn("Je t'ai jamais dit que je m'appelais Medica", {})
    assert session.intent.customer_name is None, "le slot conteste doit etre efface"
    assert session.check_fallback(user_requested_human=False, group_size=None) is None


async def test_apply_slots_fills_the_intent():
    from datetime import date as _d
    from datetime import time as _t

    session, _, _, _ = await _make_session([])
    session.apply_slots({"date": _d(2026, 7, 22), "time": _t(20, 0),
                         "party_size": 4, "customer_name": "Dupont"})
    assert session.intent.is_complete()


async def test_clear_slots_removes_contested_information():
    session, _, _, _ = await _make_session([])
    session.apply_slots({"customer_name": "Medica"})
    session.clear_slots(["customer_name"])
    assert session.intent.customer_name is None


async def test_check_availability_delegates_to_the_port():
    session, reservation, _, _ = await _make_session([])
    reservation.set_availability("r-1", available=False)
    assert await session.check_availability(datetime(2026, 7, 1, 20), 4) is False


async def test_book_creates_the_reservation_and_notifies():
    from datetime import date as _d
    from datetime import time as _t

    session, reservation, _, notif = await _make_session([])
    reservation.set_availability("r-1", available=True)
    session.apply_slots({"date": _d(2026, 7, 1), "time": _t(20, 0),
                         "party_size": 4, "customer_name": "Dupont"})
    outcome = await session.book("+33600000000")
    assert outcome == CallOutcome.RESERVATION_CREATED
    assert len(reservation.reservations) == 1
    assert notif.sent


async def test_book_refuses_an_incomplete_intent():
    session, reservation, _, _ = await _make_session([])
    assert await session.book(None) is None
    assert len(reservation.reservations) == 0
