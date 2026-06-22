import pytest
from pytest_httpx import HTTPXMock

from hikky.adapters.back.http_client import BackHttpClient, NotFound
from hikky.exceptions import BackUnavailable


async def test_get_returns_parsed_json(
    httpx_mock: HTTPXMock, back_client: BackHttpClient, base_url: str
):
    httpx_mock.add_response(url=f"{base_url}/ping", json={"ok": True})
    result = await back_client.get("/ping")
    assert result == {"ok": True}


async def test_get_sends_bearer_auth_header(
    httpx_mock: HTTPXMock, back_client: BackHttpClient, base_url: str
):
    httpx_mock.add_response(url=f"{base_url}/ping", json={})
    await back_client.get("/ping")
    requests = httpx_mock.get_requests()
    assert requests[0].headers["Authorization"] == "Bearer test-key"


async def test_get_retries_once_on_5xx_then_succeeds(
    httpx_mock: HTTPXMock, back_client: BackHttpClient, base_url: str
):
    httpx_mock.add_response(url=f"{base_url}/ping", status_code=503)
    httpx_mock.add_response(url=f"{base_url}/ping", json={"ok": True})
    result = await back_client.get("/ping")
    assert result == {"ok": True}
    assert len(httpx_mock.get_requests()) == 2


async def test_get_raises_back_unavailable_on_persistent_5xx(
    httpx_mock: HTTPXMock, back_client: BackHttpClient, base_url: str
):
    httpx_mock.add_response(url=f"{base_url}/ping", status_code=503)
    httpx_mock.add_response(url=f"{base_url}/ping", status_code=503)
    with pytest.raises(BackUnavailable):
        await back_client.get("/ping")


async def test_get_raises_not_found_on_404(
    httpx_mock: HTTPXMock, back_client: BackHttpClient, base_url: str
):
    httpx_mock.add_response(url=f"{base_url}/missing", status_code=404)
    with pytest.raises(NotFound):
        await back_client.get("/missing")


async def test_post_sends_json_body(
    httpx_mock: HTTPXMock, back_client: BackHttpClient, base_url: str
):
    httpx_mock.add_response(url=f"{base_url}/echo", json={"received": True})
    result = await back_client.post("/echo", json={"a": 1})
    assert result == {"received": True}
    request = httpx_mock.get_requests()[0]
    import json

    assert json.loads(request.read()) == {"a": 1}


async def test_post_returns_empty_dict_on_204(
    httpx_mock: HTTPXMock, back_client: BackHttpClient, base_url: str
):
    httpx_mock.add_response(url=f"{base_url}/noop", status_code=204)
    result = await back_client.post("/noop")
    assert result == {}
