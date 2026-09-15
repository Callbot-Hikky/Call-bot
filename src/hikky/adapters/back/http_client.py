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


class Conflict(Exception):
    """409 levée par le client, à interpréter par l'adapter.

    `code` porte le motif métier renvoyé par le backend (champ `error` du
    corps JSON), ex. `table_overlap` : la table proposée vient d'être prise.
    L'adapter s'en sert pour décider s'il réessaie (autre table) ou renonce.
    """

    def __init__(self, message: str, *, code: str | None = None) -> None:
        super().__init__(message)
        self.code = code


class BackHttpClient:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        *,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        # Le backend Spring authentifie les appels machine par `X-Api-Key`
        # (cf. ServiceApiKeyFilter). On conserve `Authorization` pour les
        # déploiements qui attendent encore un jeton porteur.
        self._headers = {
            "Authorization": f"Bearer {api_key}",
            "X-Api-Key": api_key,
        }
        self._client = client or httpx.AsyncClient(timeout=TIMEOUT_SECONDS)

    async def get(
        self, path: str, *, params: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        return await self._request("GET", path, params=params)

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
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        url = f"{self._base_url}{path}"
        merged_headers = (
            self._headers if not extra_headers else {**self._headers, **extra_headers}
        )
        last_error: Exception | None = None
        for attempt in range(2):  # 1 try + 1 retry
            try:
                response = await self._client.request(
                    method, url, json=json, headers=merged_headers, params=params
                )
            except httpx.HTTPError as exc:
                last_error = exc
                if attempt == 0:
                    await asyncio.sleep(RETRY_BACKOFF_SECONDS)
                    continue
                raise BackUnavailable(f"{method} {path}: {exc}") from exc

            if response.status_code == 404:
                raise NotFound(f"{method} {path}: 404")

            if response.status_code == 409:
                code = None
                try:
                    payload = response.json()
                    if isinstance(payload, dict):
                        code = payload.get("error")
                except (ValueError, TypeError):
                    pass
                raise Conflict(f"{method} {path}: 409", code=code)

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
