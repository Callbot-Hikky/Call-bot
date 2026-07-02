"""Serveur TCP asyncio qui accepte les connexions Asterisk AudioSocket.

Contrairement au chemin Twilio, on n'utilise **pas Pipecat** ici. Pour un POC
AudioSocket, une boucle asyncio simple qui appelle directement les ports
STT/TTS + `CallSession` est suffisante — Pipecat apporte de la valeur pour
la gestion fine des interruptions/VAD, ce qu'on ajoutera plus tard.

Chaque connexion Asterisk suit ce cycle :

    1. Asterisk envoie un paquet UUID (16 octets) — c'est le `call_id`.
    2. Le serveur charge le `RestaurantContext` (via le port, résolvant
       le numéro appelé injecté au dialplan ou par défaut).
    3. Le serveur instancie une `CallSession` et démarre la boucle de
       dialogue :
         - accumule les paquets audio inbound
         - à un seuil (silence détecté ou N ms), demande transcription
         - transmet le texte à la `CallSession`
         - stream la réponse TTS en paquets AudioSocket sortants
    4. Sur `HangupFrame` ou fermeture du socket, la session est clôturée
       proprement (`session.end_with(...)` + `unbind_call`).

Pour un premier "hello world" du transport, on peut faire tourner le
serveur avec un `stt_adapter=None` et `tts_adapter=None` : dans ce cas
le serveur se contente de logguer les paquets reçus et de raccrocher
au bout de 10 secondes — c'est le mode "smoke test".
"""

from __future__ import annotations

import asyncio
import logging
from asyncio import IncompleteReadError
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from hikky.adapters.telephony.asterisk_audiosocket_adapter import (
    AsteriskAudioSocketAdapter,
)
from hikky.adapters.telephony.audiosocket_protocol import (
    AUDIOSOCKET_CHUNK_20MS_BYTES,
    AudioFrame,
    DTMFFrame,
    ErrorFrame,
    HangupFrame,
    UuidFrame,
    read_packet,
)
from hikky.exceptions import UnknownRestaurant

if TYPE_CHECKING:
    from hikky.domain.call_session import CallSession
    from hikky.ports.restaurant_context import RestaurantContextPort


logger = logging.getLogger("hikky.asterisk_audiosocket")


@dataclass(slots=True)
class AudioSocketServerConfig:
    """Config du serveur TCP AudioSocket.

    - `host` / `port` : bind du serveur (0.0.0.0 pour accepter Asterisk
      depuis n'importe quel réseau, y compris la VM FreePBX en bridged).
    - `default_called_number` : numéro utilisé pour charger le
      `RestaurantContext` si Asterisk n'a pas transmis d'info métier
      (ex. via un canal ARI ou un dialplan qui set une variable).
    - `smoke_test`: si True, le serveur log les paquets et raccroche
      après `smoke_test_duration_s`. Sert à valider le transport avant
      d'avoir tous les adapters IA.
    """

    host: str = "0.0.0.0"
    port: int = 6666
    default_called_number: str = "+33000000000"
    smoke_test: bool = False
    smoke_test_duration_s: float = 10.0


@dataclass(slots=True)
class AudioSocketServerDeps:
    """Dépendances injectables — mirror de `AppDependencies` mais orienté
    AudioSocket. On les passe explicitement pour pouvoir tester
    unitairement avec des fakes.
    """

    session_factory: Callable[[str, object], "CallSession"]
    restaurant_context_port: "RestaurantContextPort"
    # STT / TTS optionnels — si None, on tourne en `smoke_test`.
    stt_adapter: object | None = None
    tts_adapter: object | None = None


async def handle_connection(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    *,
    adapter: AsteriskAudioSocketAdapter,
    deps: AudioSocketServerDeps,
    config: AudioSocketServerConfig,
) -> None:
    """Gère UNE connexion Asterisk du début à la fin.

    Publique pour permettre les tests d'intégration (créer un couple
    reader/writer via `asyncio.open_connection` sur le loopback).
    """
    peer = writer.get_extra_info("peername")
    logger.info("audiosocket connection opened", extra={"peer": peer})

    call_id: str | None = None
    session: CallSession | None = None

    try:
        # 1. Attendre le paquet UUID (obligatoire, envoyé en premier par Asterisk)
        first = await read_packet(reader)
        if not isinstance(first, UuidFrame):
            logger.warning(
                "expected UUID as first packet, got %s — closing", type(first).__name__
            )
            return
        call_id = str(first.call_uuid)
        adapter.bind_call(call_id, writer)
        logger.info("call bound", extra={"call_id": call_id})

        # 2. Charger le contexte restaurant
        # (Pour le POC on utilise le default_called_number ; à raffiner
        # quand Asterisk enverra le numéro appelé via une variable de canal.)
        try:
            ctx = await deps.restaurant_context_port.load(config.default_called_number)
        except UnknownRestaurant:
            logger.warning(
                "unknown restaurant for %s", config.default_called_number
            )
            return

        session = deps.session_factory(call_id, ctx)
        await session.begin()

        # 3. Boucle d'appel
        if config.smoke_test:
            await _run_smoke_test(reader, config.smoke_test_duration_s)
        else:
            await _run_ai_loop(reader, adapter, call_id, session, deps)

    except IncompleteReadError:
        logger.info("peer closed socket", extra={"call_id": call_id})
    except Exception:  # noqa: BLE001 — boundary handler
        logger.exception("audiosocket handler crashed", extra={"call_id": call_id})
    finally:
        # Filet de sécurité — CallSession.end_with est idempotent
        if session is not None:
            from hikky.domain.outcomes import CallOutcome

            try:
                await session.end_with(CallOutcome.TECHNICAL_ERROR)
            except Exception:  # noqa: BLE001
                logger.exception("failed to close session")
        if call_id is not None:
            adapter.unbind_call(call_id)
        try:
            writer.close()
            await writer.wait_closed()
        except Exception:  # noqa: BLE001
            pass
        logger.info("audiosocket connection closed", extra={"call_id": call_id})


async def _run_smoke_test(reader: asyncio.StreamReader, duration_s: float) -> None:
    """Boucle de test : log les paquets reçus, raccroche après `duration_s`."""

    async def _read_loop():
        received_audio_bytes = 0
        received_packets = 0
        while True:
            try:
                frame = await read_packet(reader)
            except IncompleteReadError:
                logger.info(
                    "smoke test: peer closed",
                    extra={
                        "packets": received_packets,
                        "audio_bytes": received_audio_bytes,
                    },
                )
                return
            received_packets += 1
            if isinstance(frame, AudioFrame):
                received_audio_bytes += len(frame.audio)
            elif isinstance(frame, HangupFrame):
                logger.info("smoke test: hangup packet from Asterisk")
                return
            elif isinstance(frame, DTMFFrame):
                logger.info("smoke test: DTMF digit", extra={"digit": frame.digit})
            elif isinstance(frame, ErrorFrame):
                logger.warning("smoke test: error packet", extra={"code": frame.code})

    try:
        await asyncio.wait_for(_read_loop(), timeout=duration_s)
    except TimeoutError:
        logger.info("smoke test: timeout reached, hanging up")


async def _run_ai_loop(
    reader: asyncio.StreamReader,
    adapter: AsteriskAudioSocketAdapter,
    call_id: str,
    session: "CallSession",
    deps: AudioSocketServerDeps,
) -> None:
    """Boucle IA minimale.

    Cette version est volontairement simple : elle accumule des paquets
    audio jusqu'à un `silence_ms` (détection sommaire par seuil de
    "packets audio sans changement significatif"), envoie le buffer au
    STT, passe le texte à la `CallSession`, envoie la réponse au TTS,
    stream l'audio en retour.

    Pour un vrai "prod-ready" il faudrait un VAD (webrtcvad, silero).
    Pour un POC, un timeout de silence "grossier" suffit.
    """
    if deps.stt_adapter is None or deps.tts_adapter is None:
        logger.error(
            "AI loop requires stt_adapter and tts_adapter — set them or use smoke_test",
        )
        return

    # NOTE: implémentation détaillée à venir dans une itération suivante.
    # Pour l'instant on log et raccroche — c'est mieux qu'un crash silencieux.
    logger.warning(
        "_run_ai_loop is a stub. Use smoke_test=True until the STT/TTS wiring is complete.",
        extra={"call_id": call_id, "chunk_size_hint": AUDIOSOCKET_CHUNK_20MS_BYTES},
    )


async def start_server(
    deps: AudioSocketServerDeps,
    config: AudioSocketServerConfig | None = None,
) -> tuple[asyncio.Server, AsteriskAudioSocketAdapter]:
    """Démarre le serveur TCP AudioSocket.

    Renvoie `(server, adapter)`. Le serveur peut ensuite être laissé
    tourner en arrière-plan avec `server.serve_forever()` — c'est le
    rôle du lifespan FastAPI dans `main.py`.
    """
    config = config or AudioSocketServerConfig()
    adapter = AsteriskAudioSocketAdapter()

    async def _handler(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        await handle_connection(
            reader, writer, adapter=adapter, deps=deps, config=config
        )

    server = await asyncio.start_server(_handler, host=config.host, port=config.port)
    sockets = server.sockets or ()
    addresses = [s.getsockname() for s in sockets]
    logger.info(
        "AudioSocket server listening",
        extra={"addresses": addresses, "smoke_test": config.smoke_test},
    )
    return server, adapter
