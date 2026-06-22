from datetime import datetime, time

from hikky.adapters.back.contracts import (
    AvailabilityRequest,
    AvailabilityResponse,
    CallbackRequestPayload,
    CallbackRequestResponse,
    CallEndRequest,
    CallStartRequest,
    Endpoints,
    ReservationCreateRequest,
    ReservationCreateResponse,
    RestaurantContextResponse,
)
from hikky.domain.outcomes import CallOutcome
from hikky.domain.restaurant_context import (
    OpeningHours,
    RestaurantContext,
    RestaurantRules,
)


def test_endpoints_have_expected_paths():
    assert Endpoints.RESTAURANT_BY_PHONE == "/restaurants/by-phone/{phone_number}"
    assert Endpoints.AVAILABILITY == "/restaurants/{restaurant_id}/availability"
    assert Endpoints.RESERVATIONS == "/restaurants/{restaurant_id}/reservations"
    assert Endpoints.CALLBACK_REQUESTS == "/restaurants/{restaurant_id}/callback-requests"
    assert Endpoints.CALL_START == "/calls/{call_id}/start"
    assert Endpoints.CALL_END == "/calls/{call_id}/end"
    assert Endpoints.NOTIFICATION_CONFIRMATION == "/reservations/{reservation_id}/confirmation"


def test_restaurant_context_response_can_be_mapped_to_domain():
    payload = {
        "id": "r-1",
        "name": "Chez Test",
        "greeting": "Bonjour.",
        "opening_hours": [
            {"weekday": d, "opens": "19:00:00", "closes": "23:00:00"} for d in range(7)
        ],
        "total_capacity": 40,
        "rules": {"max_group_size": 8, "reservation_duration_minutes": 90},
        "transfer_number": None,
        "fallback_message": "…",
    }
    res = RestaurantContextResponse.model_validate(payload)
    ctx = RestaurantContext(
        id=res.id,
        name=res.name,
        greeting=res.greeting,
        opening_hours=res.opening_hours,
        total_capacity=res.total_capacity,
        rules=res.rules,
        transfer_number=res.transfer_number,
        fallback_message=res.fallback_message,
    )
    assert ctx.id == "r-1"
    assert ctx.opening_hours[0] == OpeningHours(weekday=0, opens=time(19, 0), closes=time(23, 0))
    assert ctx.rules == RestaurantRules(max_group_size=8, reservation_duration_minutes=90)


def test_availability_request_and_response_roundtrip():
    req = AvailabilityRequest(date_time=datetime(2026, 7, 1, 20), party_size=4)
    data = req.model_dump(mode="json")
    assert data["party_size"] == 4
    assert "2026-07-01" in data["date_time"]

    res = AvailabilityResponse.model_validate({"available": True})
    assert res.available is True


def test_reservation_create_request_rejects_zero_party_size():
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        ReservationCreateRequest(
            date_time=datetime(2026, 7, 1, 20),
            party_size=0,
            customer_name="X",
        )


def test_reservation_create_response_parses_id():
    res = ReservationCreateResponse.model_validate({"reservation_id": "res-abc"})
    assert res.reservation_id == "res-abc"


def test_callback_request_payload_and_response():
    payload = CallbackRequestPayload(
        customer_phone="+33600000000", preferred_slot="ce soir", note="trois échecs"
    )
    assert payload.customer_phone == "+33600000000"

    res = CallbackRequestResponse.model_validate({"callback_request_id": "cb-1"})
    assert res.callback_request_id == "cb-1"


def test_call_start_and_end_payloads():
    start = CallStartRequest(restaurant_id="r-1", started_at=datetime(2026, 7, 1, 20))
    assert start.restaurant_id == "r-1"

    end = CallEndRequest(outcome=CallOutcome.RESERVATION_CREATED, duration_seconds=42.5)
    data = end.model_dump(mode="json")
    assert data["outcome"] == "reservation_created"
    assert data["duration_seconds"] == 42.5
