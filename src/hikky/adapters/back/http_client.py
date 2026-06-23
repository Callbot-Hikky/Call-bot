"""Client HTTP partagé par tous les adapters Back.

Encapsule httpx async, gère le retry (1 tentative avec backoff ≤ 300 ms),
et traduit les erreurs réseau / 5xx en `BackUnavailable`.

Les 404 sont propagés tels quels — les adapters décident ce qu'ils en font
(ex. l'adapter Restaurant les traduit en `UnknownRestaurant`).
"""

import asyncio
from typing import Any

import httpx

from hikky.exceptions import BackUnavailable

RETRY_BACKOFF_SECONDS = 0.3
TIMEOUT_SECONDS = 5.0


class NotFound(Exception):
    """404 levée par le client, à interpréter par l'adapter."""


class BackHttpClient:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        *,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._headers = {"Authorization": f"Bearer {api_key}"}
        self._client = client or httpx.AsyncClient(timeout=TIMEOUT_SECONDS)

    async def get(self, path: str) -> dict[str, Any]:
        return await self._request("GET", path)

    async def post(
        self,
        path: str,
        json: dict[str, Any] | None = None,
        *,
        headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        return await self._request("POST", path, json=json, extra_headers=headers)

    async def _request(
        self,
        method: str,
        path: str,
        *,
        json: dict[str, Any] | None = None,
        extra_headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        url = f"{self._base_url}{path}"
        merged_headers = (
            self._headers if not extra_headers else {**self._headers, **extra_headers}
        )
        last_error: Exception | None = None
        for attempt in range(2):  # 1 try + 1 retry
            try:
                response = await self._client.request(
                    method, url, json=json, headers=merged_headers
                )
            except httpx.HTTPError as exc:
                last_error = exc
                if attempt == 0:
                    await asyncio.sleep(RETRY_BACKOFF_SECONDS)
                    continue
                raise BackUnavailable(f"{method} {path}: {exc}") from exc

            if response.status_code == 404:
                raise NotFound(f"{method} {path}: 404")

            if 500 <= response.status_code < 600:
                last_error = BackUnavailable(
                    f"{method} {path}: HTTP {response.status_code}"
                )
                if attempt == 0:
                    await asyncio.sleep(RETRY_BACKOFF_SECONDS)
                    continue
                raise last_error

            if response.status_code >= 400:
                raise BackUnavailable(f"{method} {path}: HTTP {response.status_code}")

            if response.status_code == 204 or not response.content:
                return {}
            return response.json()

        # Should be unreachable, but keep the type checker happy.
        raise BackUnavailable(f"{method} {path}: {last_error}")

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> "BackHttpClient":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.aclose()
