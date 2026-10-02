"""Base de connaissances du restaurant, lue sur les routes du callbot.

    GET  /api/calls/knowledge   passages proches d'une question, avec score
    POST /api/calls/unanswered  question à laquelle on n'a pas su répondre

Le backend calcule les embeddings : ici on n'échange que du texte. Le
restaurant est désigné par son numéro d'appel, comme pour le contexte.

Deux règles, toutes deux pour protéger l'appel en cours :

- **aucune exception ne remonte**. Ces routes servent une réponse annexe ;
  si elles tombent, le bot dit qu'il ne sait pas, il ne raccroche pas.
- **le signalement ne remplace pas la réponse**. On remonte la question au
  restaurateur, qui y répondra une fois ; ce que dit un client n'entre
  jamais tel quel dans la base.
"""

from __future__ import annotations

import logging
from typing import Any

from hikky.adapters.back.http_client import NotFound
from hikky.domain.knowledge import KnowledgePassage
from hikky.exceptions import BackUnavailable
from hikky.ports.knowledge import KnowledgePort

logger = logging.getLogger("hikky.back.knowledge")

DEFAULT_LIMIT = 3


class BackendKnowledgeAdapter(KnowledgePort):
    def __init__(
        self, client: Any, *, restaurant_phone: str, limit: int = DEFAULT_LIMIT
    ) -> None:
        self._client = client
        self._phone = restaurant_phone
        self._limit = limit

    async def search(self, question: str) -> list[KnowledgePassage]:
        try:
            payload = await self._client.get(
                "/api/calls/knowledge",
                params={
                    "restaurantPhone": self._phone,
                    "question": question,
                    "limit": self._limit,
                },
            )
        except (BackUnavailable, NotFound):
            logger.warning("base de connaissances injoignable", exc_info=True)
            return []

        passages: list[KnowledgePassage] = []
        matches = payload.get("matches") if isinstance(payload, dict) else None
        for match in matches or []:
            if not isinstance(match, dict):
                continue
            content = match.get("content")
            if not isinstance(content, str) or not content.strip():
                continue
            passages.append(
                KnowledgePassage(
                    title=str(match.get("title") or ""),
                    content=content.strip(),
                    score=_score(match.get("score")),
                )
            )
        return passages

    async def report_unanswered(self, question: str) -> None:
        try:
            await self._client.post(
                "/api/calls/unanswered",
                json={"restaurantPhone": self._phone, "question": question},
            )
        except (BackUnavailable, NotFound):
            logger.warning("question sans réponse non remontée", exc_info=True)


def _score(valeur: Any) -> float:
    try:
        return float(valeur)
    except (TypeError, ValueError):
        return 0.0
