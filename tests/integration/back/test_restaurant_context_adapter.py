import pytest
from pytest_httpx import HTTPXMock

from hikky.adapters.back.http_client import BackHttpClient
from hikky.adapters.back.restaurant_context_adapter import (
    BackHttpRestaurantContextAdapter,
)
from hikky.exceptions import BackUnavailable, UnknownRestaurant


def _payload(rest_id: str = "r-1") -> dict:
    return {
        "id": rest_id,
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


async def test_load_returns_restaurant_context(
    httpx_mock: HTTPXMock, back_client: BackHttpClient, base_url: str
):
    httpx_mock.add_response(
        url=f"{base_url}/restaurants/by-phone/+33100000001", json=_payload("r-1")
    )
    adapter = BackHttpRestaurantContextAdapter(back_client)
    ctx = await adapter.load("+33100000001")
    assert ctx.id == "r-1"
    assert ctx.rules.max_group_size == 8


async def test_load_raises_unknown_restaurant_on_404(
    httpx_mock: HTTPXMock, back_client: BackHttpClient, base_url: str
):
    httpx_mock.add_response(
        url=f"{base_url}/restaurants/by-phone/+33199999999", status_code=404
    )
    adapter = BackHttpRestaurantContextAdapter(back_client)
    with pytest.raises(UnknownRestaurant):
        await adapter.load("+33199999999")


async def test_load_raises_back_unavailable_on_persistent_5xx(
    httpx_mock: HTTPXMock, back_client: BackHttpClient, base_url: str
):
    httpx_mock.add_response(url=f"{base_url}/restaurants/by-phone/+33100000001", status_code=503)
    httpx_mock.add_response(url=f"{base_url}/restaurants/by-phone/+33100000001", status_code=503)
    adapter = BackHttpRestaurantContextAdapter(back_client)
    with pytest.raises(BackUnavailable):
        await adapter.load("+33100000001")
