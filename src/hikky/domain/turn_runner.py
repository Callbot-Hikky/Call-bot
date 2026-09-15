"""Interface d'extraction de slots, dans le domaine.

Historiquement ce module portait aussi un orchestrateur de tour (`run_turn`),
remplacé par `routed_turn`. Il ne reste que le port `SlotExtractor` : il vit
ici (et non dans `pipeline/`) pour que le domaine et le chemin AudioSocket
puissent l'utiliser sans dépendance transport.
"""

from __future__ import annotations

from typing import Any


class SlotExtractor:
    """Interface async : extrait des slots structurés d'un tour de parole."""

    async def extract(self, user_text: str) -> dict[str, Any]:  # pragma: no cover
        raise NotImplementedError
