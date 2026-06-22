import json
from datetime import datetime

import pytest
from pytest_httpx import HTTPXMock

from hikky.adapters.back.call_log_adapter import BackHttpCallLogAdapter
from hikky.adapters.back.http_client import BackHttpClient
from hikky.domain.outcomes import CallOutcome
from hikky.exceptions import BackUnavailable


async def test_start_posts_expected_payload(
    httpx_mock: HTTPXMock, back_client: BackHttpClient, base_url: str
):
    httpx_mock.add_response(url=f"{base_url}/calls/c-1/start", method="POST", status_code=204)
    adapter = BackHttpCallLogAdapter(back_client)
    await adapter.start("c-1", "r-1", datetime(2026, 7, 1, 20))
    body = json.loads(httpx_mock.get_requests()[0].read())
    assert body["restaurant_id"] == "r-1"
    assert "2026-07-01" in body["started_at"]


async def test_end_serializes_outcome_as_string(
    httpx_mock: HTTPXMock, back_client: BackHttpClient, base_url: str
):
    httpx_mock.add_response(url=f"{base_url}/calls/c-1/end", method="POST", status_code=204)
    adapter = BackHttpCallLogAdapter(back_client)
    await adapter.end("c-1", CallOutcome.RESERVATION_CREATED, 42.5)
    body = json.loads(httpx_mock.get_requests()[0].read())
    assert body["outcome"] == "reservation_created"
    assert body["duration_seconds"] == 42.5


async def test_back_unavailable_on_persistent_5xx(
    httpx_mock: HTTPXMock, back_client: BackHttpClient, base_url: str
):
    httpx_mock.add_response(url=f"{base_url}/calls/c-1/start", method="POST", status_code=503)
    httpx_mock.add_response(url=f"{base_url}/calls/c-1/start", method="POST", status_code=503)
    adapter = BackHttpCallLogAdapter(back_client)
    with pytest.raises(BackUnavailable):
        await adapter.start("c-1", "r-1", datetime(2026, 7, 1, 20))
