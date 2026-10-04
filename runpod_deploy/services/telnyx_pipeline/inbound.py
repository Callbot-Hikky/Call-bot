"""Quand le téléphone sonne : la consigne rendue à Telnyx, et un seul flux par appel.

Telnyx appelle `POST /telnyx/inbound` ; on lui rend un TeXML « connecte cet appel
à mon WebSocket ». Observé en appel réel : DEUX requêtes pour UN appel, à 7 s
d'écart (rappel de statut ou nouvelle tentative). Chacune recevait la consigne ->
deux bots parlaient sur la même ligne. D'où le registre : un `Stream` par CallSid.
"""

from __future__ import annotations

import time
from urllib.parse import parse_qs

EMPTY_TEXML = '<?xml version="1.0" encoding="UTF-8"?><Response/>'


def texml_connect(public_host: str) -> str:
    """La consigne : ouvrir un flux bidirectionnel vers notre WebSocket, en A-law (PCMA),
    le codec que la ligne européenne nous envoie déjà : un seul codec dans les deux sens."""
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        f'<Response><Connect><Stream url="wss://{public_host}/telnyx/stream" '
        'bidirectionalMode="rtp" bidirectionalCodec="PCMA"/></Connect></Response>'
    )


def parse_form(body: bytes) -> dict[str, str]:
    """Le corps `application/x-www-form-urlencoded` envoyé par Telnyx, à plat."""
    return {k: v[0] for k, v in parse_qs(body.decode(errors="replace")).items()}


class StreamRegistry:
    """Mémorise les appels déjà connectés, pendant `ttl_s` secondes."""

    def __init__(self, ttl_s: float = 3600.0) -> None:
        self._ttl = ttl_s
        self._seen: dict[str, float] = {}

    def accept(self, call_sid: str, *, now: float | None = None) -> bool:
        """Vrai si c'est la première fois qu'on voit cet appel (-> on ouvre le flux)."""
        now = time.time() if now is None else now
        for sid in [s for s, ts in self._seen.items() if now - ts > self._ttl]:
            self._seen.pop(sid, None)
        if not call_sid:
            return True  # sans identifiant on ne peut pas dédoublonner : on ouvre
        if call_sid in self._seen:
            return False
        self._seen[call_sid] = now
        return True
