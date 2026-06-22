from datetime import datetime

import pytest
from pytest_httpx import HTTPXMock

from hikky.adapters.back.http_client import BackHttpClient
from hikky.adapters.back.reservation_adapter import BackHttpReservationAdapter
from hikky.exceptions import BackUnavailable


async def test_check_availability_returns_true(
    httpx_mock: HTTPXMock, back_client: BackHttpClient, base_url: str
):
    httpx_mock.add_response(
        url=f"{base_url}/restaurants/r-1/availability",
        method="POST",
        json={"available": True},
    )
    adapter = BackHttpReservationAdapter(back_client)
    result = await adapter.check_availability("r-1", datetime(2026, 7, 1, 20), 4)
    assert result is True


async def test_check_availability_returns_false(
    httpx_mock: HTTPXMock, back_client: BackHttpClient, base_url: str
):
    httpx_mock.add_response(
        url=f"{base_url}/restaurants/r-1/availability",
        method="POST",
        json={"available": False},
    )
    adapter = BackHttpReservationAdapter(back_client)
    assert await adapter.check_availability("r-1", datetime(2026, 7, 1, 20), 4) is False


async def test_create_returns_reservation_id(
    httpx_mock: HTTPXMock, back_client: BackHttpClient, base_url: str
):
    httpx_mock.add_response(
        url=f"{base_url}/restaurants/r-1/reservations",
        method="POST",
        json={"reservation_id": "res-abc"},
    )
    adapter = BackHttpReservationAdapter(back_client)
    rid = await adapter.create(
        "r-1",
        datetime(2026, 7, 1, 20),
        4,
        "Dupont",
        "+33600000000",
    )
    assert rid == "res-abc"


async def test_create_sends_expected_body(
    httpx_mock: HTTPXMock, back_client: BackHttpClient, base_url: str
):
    httpx_mock.add_response(
        url=f"{base_url}/restaurants/r-1/reservations",
        method="POST",
        json={"reservation_id": "res-1"},
    )
    adapter = BackHttpReservationAdapter(back_client)
    await adapter.create("r-1", datetime(2026, 7, 1, 20), 4, "Dupont", "+33600000000")
    import json

    body = json.loads(httpx_mock.get_requests()[0].read())
    assert body["customer_name"] == "Dupont"
    assert body["party_size"] == 4
    assert "2026-07-01" in body["date_time"]


async def test_create_callback_request_returns_id(
    httpx_mock: HTTPXMock, back_client: BackHttpClient, base_url: str
):
    httpx_mock.add_response(
        url=f"{base_url}/restaurants/r-1/callback-requests",
        method="POST",
        json={"callback_request_id": "cb-1"},
    )
    adapter = BackHttpReservationAdapter(back_client)
    cid = await adapter.create_callback_request("r-1", "+33600000000", "ce soir", "trois échecs")
    assert cid == "cb-1"


async def test_back_unavailable_on_persistent_5xx(
    httpx_mock: HTTPXMock, back_client: BackHttpClient, base_url: str
):
    httpx_mock.add_response(
        url=f"{base_url}/restaurants/r-1/availability", method="POST", status_code=503
    )
    httpx_mock.add_response(
        url=f"{base_url}/restaurants/r-1/availability", method="POST", status_code=503
    )
    adapter = BackHttpReservationAdapter(back_client)
    with pytest.raises(BackUnavailable):
        await adapter.check_availability("r-1", datetime(2026, 7, 1, 20), 4)
