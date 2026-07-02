"""Adapter Asterisk AudioSocket pour le port `TelephonyPort`.

L'adapter ne crée pas le socket TCP lui-même — c'est le serveur asyncio
(`asterisk_audiosocket_server.py`) qui accepte les connexions Asterisk et
appelle `bind_call(...)`. À la fin de l'appel, le serveur appelle
`unbind_call(...)`.

Cet adapter est le miroir de `TwilioMediaStreamsAdapter` : mêmes hooks
(`bind_call`, `unbind_call`), mêmes méthodes du port (`send_audio`,
`hang_up`, `transfer`). Différence : le transport sous-jacent est un
`asyncio.StreamWriter` (TCP binaire) au lieu d'une WebSocket JSON.
"""

from __future__ import annotations

import asyncio
from typing import Protocol

from hikky.adapters.telephony.audiosocket_protocol import (
    encode_audio,
    encode_hangup,
)
from hikky.exceptions import TelephonyError
from hikky.ports.telephony import TelephonyPort


class _SupportsBinaryWrite(Protocol):
    """Forme minimale d'un `asyncio.StreamWriter` pour cet adapter."""

    def write(self, data: bytes) -> None: ...

    async def drain(self) -> None: ...

    def close(self) -> None: ...

    async def wait_closed(self) -> None: ...


class AsteriskAudioSocketAdapter(TelephonyPort):
    """Implémentation du `TelephonyPort` pour Asterisk AudioSocket.

    Multi-appels : un même adapter gère N appels simultanés, chacun
    identifié par un `call_id` (typiquement le UUID envoyé par Asterisk
    au début de la connexion, sérialisé en string).
    """

    def __init__(self) -> None:
        self._connections: dict[str, _SupportsBinaryWrite] = {}
        # Verrou par appel : `write()` + `drain()` doivent être atomiques
        # pour éviter d'entrelacer des paquets audio depuis 2 coroutines.
        self._locks: dict[str, asyncio.Lock] = {}

    def bind_call(self, call_id: str, writer: _SupportsBinaryWrite) -> None:
        self._connections[call_id] = writer
        self._locks[call_id] = asyncio.Lock()

    def unbind_call(self, call_id: str) -> None:
        self._connections.pop(call_id, None)
        self._locks.pop(call_id, None)

    async def send_audio(self, call_id: str, audio_chunk: bytes) -> None:
        writer = self._connections.get(call_id)
        lock = self._locks.get(call_id)
        if writer is None or lock is None:
            raise TelephonyError(f"No active AudioSocket stream for call_id={call_id}")
        packet = encode_audio(audio_chunk)
        async with lock:
            writer.write(packet)
            await writer.drain()

    async def transfer(self, call_id: str, destination_number: str) -> None:
        # Le protocole AudioSocket ne gère PAS le transfert d'appel — il
        # n'y a pas d'action de contrôle "transfer" dans le protocole. Un
        # transfert vers un humain devrait passer par une action ARI
        # `Channel.redirect()` en parallèle. Pas dans le POC.
        raise NotImplementedError(
            "AudioSocket does not support call transfer; use ARI Channel.redirect() instead."
        )

    async def hang_up(self, call_id: str) -> None:
        writer = self._connections.get(call_id)
        if writer is None:
            return
        try:
            writer.write(encode_hangup())
            await writer.drain()
        except Exception:  # noqa: BLE001 — best-effort cleanup
            pass
        writer.close()
        try:
            await writer.wait_closed()
        except Exception:  # noqa: BLE001
            pass
        self.unbind_call(call_id)
