import pytest
from pytest_httpx import HTTPXMock

from hikky.adapters.back.http_client import BackHttpClient
from hikky.adapters.back.notification_adapter import BackHttpNotificationAdapter
from hikky.exceptions import BackUnavailable


async def test_send_confirmation_posts_to_expected_path(
    httpx_mock: HTTPXMock, back_client: BackHttpClient, base_url: str
):
    httpx_mock.add_response(
        url=f"{base_url}/reservations/res-1/confirmation", method="POST", status_code=204
    )
    adapter = BackHttpNotificationAdapter(back_client)
    await adapter.send_confirmation("res-1")
    assert len(httpx_mock.get_requests()) == 1


async def test_back_unavailable_on_persistent_5xx(
    httpx_mock: HTTPXMock, back_client: BackHttpClient, base_url: str
):
    httpx_mock.add_response(
        url=f"{base_url}/reservations/res-1/confirmation", method="POST", status_code=500
    )
    httpx_mock.add_response(
        url=f"{base_url}/reservations/res-1/confirmation", method="POST", status_code=500
    )
    adapter = BackHttpNotificationAdapter(back_client)
    with pytest.raises(BackUnavailable):
        await adapter.send_confirmation("res-1")
