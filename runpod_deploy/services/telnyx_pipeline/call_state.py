"""L'état d'un appel, en un seul endroit.

Tout ce qui changeait au fil de l'appel vivait dans des variables partagées par
huit fonctions imbriquées ; le pire bug de la semaine (le bot muet après une
interruption) est né là. Ici chaque champ a un nom, un commentaire, et les règles
qui le touchent sont des méthodes testées.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

PENDING_MAX_AGE_S = 6.0  # un énoncé gardé plus vieux est périmé : le client a déjà avancé


@dataclass
class CallState:
    # ── identité de l'appel ──
    stream_id: str | None = None
    alaw: bool = True  # codec entrant, lu dans l'événement start
    call_tag: str = field(default_factory=lambda: time.strftime("%H%M%S"))
    session: Any = None  # la session de réservation (domaine)

    # ── dialogue ──
    history: list[dict[str, str]] = field(default_factory=list)
    recent_phrasings: set[str] = field(default_factory=set)
    awaiting_confirmation: bool = False

    # ── parole ──
    speaking: bool = False  # le bot est en train d'envoyer de l'audio
    stop_requested: bool = False  # barge-in : arrêter le flux en cours
    stop_event: asyncio.Event = field(default_factory=asyncio.Event)  # réveille l'attente d'envoi
    last_send_ts: float = 0.0  # pour espacer les messages media (Telnyx)

    # ── compréhension ──
    utterances: int = 0  # numéro d'énoncé (nom des WAV de débogage)
    misunderstood: int = 0  # énoncés vides consécutifs
    last_user_text_key: str = ""  # pour reconnaître une répétition
    pending: tuple[float, bytes] | None = None  # (horodatage, audio) dit pendant que le bot parlait

    # ── énoncé gardé pendant que le bot parle ──
    def keep_pending(self, pcm: bytes, *, now: float | None = None) -> None:
        """Seul le plus récent compte : si le client a parlé deux fois, la première
        est déjà dépassée."""
        self.pending = (time.time() if now is None else now, pcm)

    def pop_pending(self, *, now: float | None = None) -> bytes | None:
        """L'énoncé gardé s'il est encore d'actualité, sinon None (et on l'oublie).

        Appel réel (Paul) : la répétition d'une question, gardée puis rejouée après la
        réponse, faisait répondre deux fois à la même question — un tour de retard.
        """
        if self.pending is None:
            return None
        kept_at, pcm = self.pending
        self.pending = None
        age = (time.time() if now is None else now) - kept_at
        return pcm if age <= PENDING_MAX_AGE_S else None

    # ── répétitions ──
    def is_repeat(self, text: str) -> bool:
        """Vrai si `text` redit la dernière phrase traitée (ponctuation et casse à part).
        Mémorise `text` comme dernière phrase dans tous les cas."""
        key = "".join(c for c in text.lower() if c.isalnum())
        repeat = bool(key) and key == self.last_user_text_key
        self.last_user_text_key = key
        return repeat
