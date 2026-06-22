"""Adapter Twilio Media Streams pour le port `TelephonyPort`.

L'adapter ne crée pas la WebSocket lui-même — c'est l'application FastAPI
qui l'instancie pour chaque appel entrant et appelle `bind_call(...)`. À la
fin de la conversation, l'application appelle `unbind_call(...)`.

`send_audio` encode l'audio en média Twilio et le pousse sur la WS.
`hang_up` envoie un event `clear` puis ferme la WS.
"""

import json
from typing import Protocol

from hikky.adapters.telephony.twilio_protocol import (
    encode_clear,
    encode_outbound_media,
)
from hikky.exceptions import TelephonyError
from hikky.ports.telephony import TelephonyPort


class _SupportsWebSocketSend(Protocol):
    """Forme minimale d'une WebSocket Starlette/FastAPI pour cet adapter."""

    async def send_text(self, data: str) -> None: ...

    async def close(self, code: int = 1000) -> None: ...


class TwilioMediaStreamsAdapter(TelephonyPort):
    def __init__(self) -> None:
        self._connections: dict[str, _SupportsWebSocketSend] = {}
        self._stream_sids: dict[str, str] = {}

    def bind_call(
        self, call_id: str, websocket: _SupportsWebSocketSend, stream_sid: str
    ) -> None:
        self._connections[call_id] = websocket
        self._stream_sids[call_id] = stream_sid

    def unbind_call(self, call_id: str) -> None:
        self._connections.pop(call_id, None)
        self._stream_sids.pop(call_id, None)

    async def send_audio(self, call_id: str, audio_chunk: bytes) -> None:
        ws = self._connections.get(call_id)
        stream_sid = self._stream_sids.get(call_id)
        if ws is None or stream_sid is None:
            raise TelephonyError(f"No active stream for call_id={call_id}")
        payload = encode_outbound_media(stream_sid, audio_chunk)
        await ws.send_text(json.dumps(payload))

    async def transfer(self, call_id: str, destination_number: str) -> None:
        # Le transfert d'un appel actif Twilio se fait via l'API REST Twilio
        # (REST `Calls.update(twiml=...)`), pas via la WebSocket Media Streams.
        # On laisse cet appel non implémenté ici ; il sera branché en plan E
        # quand on intégrera le client twilio-python.
        raise NotImplementedError(
            "Transfer is not implemented in the walking skeleton (Plan E)."
        )

    async def hang_up(self, call_id: str) -> None:
        ws = self._connections.get(call_id)
        stream_sid = self._stream_sids.get(call_id)
        if ws is None:
            return
        if stream_sid is not None:
            try:
                await ws.send_text(json.dumps(encode_clear(stream_sid)))
            except Exception:  # noqa: BLE001 — best-effort cleanup
                pass
        await ws.close()
        self.unbind_call(call_id)
