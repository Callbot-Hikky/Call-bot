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
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from hikky.observability.logging import clear_call_context, set_call_context

from hikky.adapters.telephony.asterisk_audiosocket_adapter import (
    AsteriskAudioSocketAdapter,
)
from hikky.adapters.telephony.audio_resample import (
    StreamingResampler,
    frame_rms,
    resample_pcm16,
)
from hikky.adapters.telephony.audiosocket_protocol import (
    AUDIOSOCKET_CHUNK_20MS_BYTES,
    AUDIOSOCKET_SAMPLE_RATE_HZ,
    AudioFrame,
    DTMFFrame,
    ErrorFrame,
    HangupFrame,
    UuidFrame,
    read_packet,
)
from hikky.adapters.telephony.barge_in import (
    SEUIL_INTERRUPTION_PAR_DEFAUT,
    TRAMES_CONSECUTIVES_PAR_DEFAUT,
    TRAMES_DE_GARDE_PAR_DEFAUT,
    DetecteurInterruption,
)
from hikky.adapters.telephony.vad_endpointer import VadEndpointer
from hikky.domain.routed_turn import run_routed_turn
from hikky.exceptions import TelephonyError, UnknownRestaurant

WHISPER_SAMPLE_RATE_HZ = 16000
_STT_CHUNK_BYTES = 3200

if TYPE_CHECKING:
    from hikky.domain.call_session import CallSession
    from hikky.ports.restaurant_context import RestaurantContextPort


logger = logging.getLogger("hikky.asterisk_audiosocket")

# Journal dédié à la transcription des échanges : sans lui, les logs
# montrent qu'un tour a eu lieu mais jamais ce qui s'est dit, ce qui
# rend tout diagnostic conversationnel impossible.
conversation = logging.getLogger("hikky.conversation")


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
    sample_rate: int = AUDIOSOCKET_SAMPLE_RATE_HZ
    tts_sample_rate: int = 22050
    silence_rms_threshold: float = 500.0
    silence_frames_to_end: int = 25
    min_speech_frames: int = 10
    max_utterance_frames: int = 750
    realtime_playback: bool = True
    # VAD d'endpointing : agressivité webrtcvad (0 = laisse tout passer,
    # 3 = ne garde que la parole franche) et préroll conservé avant l'attaque
    # pour ne pas rogner le premier phonème.
    vad_aggressiveness: int = 2
    vad_padding_frames: int = 5
    # Relance sur inactivité : si aucun tour de parole valable n'est produit
    # pendant `inactivite_relance_s` après le dernier échange (réponse non
    # captée, écho nettoyé par le VAD…), le bot relance (« vous êtes toujours
    # là ? ») au lieu de rester muet. Il relance jusqu'à `inactivite_max_relances`
    # fois — on NE raccroche PAS vite sur un client présent dont la voix passe
    # mal — puis clôt seulement après la dernière relance restée sans réponse.
    # `lecture_timeout_s` évite de bloquer sur une connexion à moitié ouverte.
    inactivite_relance_s: float = 10.0
    inactivite_max_relances: int = 3
    lecture_timeout_s: float = 2.0
    # Nombre de trames de 20 ms écrites entre deux `drain`. Les paquets
    # AudioSocket restent à 20 ms — Asterisk n'en accepte pas d'autres,
    # un paquet plus gros casse la lecture. Ce qui est groupé, c'est
    # l'attente réseau : un drain toutes les 20 ms à travers un tunnel
    # distant coûte ~100 ms par trame, soit 28 s pour 4 s de parole.
    #
    # Compromis : trop peu de trames par lot et le réseau étrangle la
    # diffusion ; trop et Asterisk reçoit par rafales, épuise son tampon
    # entre deux envois et grésille. 5 trames = 100 ms, cinq fois moins
    # d'allers-retours qu'en trame par trame, sans à-coups audibles.
    frames_per_batch: int = 5

    # Barge-in : écouter pendant que le bot parle. Le seuil est très
    # au-dessus de celui de fin de tour (500) car la ligne renvoie un
    # écho de la voix du bot ; le confondre avec le client ferait couper
    # le bot par lui-même à chaque phrase.
    barge_in_enabled: bool = True
    barge_in_rms_threshold: float = SEUIL_INTERRUPTION_PAR_DEFAUT
    barge_in_frames: int = TRAMES_CONSECUTIVES_PAR_DEFAUT
    barge_in_guard_frames: int = TRAMES_DE_GARDE_PAR_DEFAUT


@dataclass(slots=True)
class AudioSocketServerDeps:
    """Dépendances injectables — mirror de `AppDependencies` mais orienté
    AudioSocket. On les passe explicitement pour pouvoir tester
    unitairement avec des fakes.
    """

    session_factory: Callable[[str, object], CallSession]
    restaurant_context_port: RestaurantContextPort
    # STT / TTS optionnels — si None, on tourne en `smoke_test`.
    stt_adapter: object | None = None
    tts_adapter: object | None = None
    slot_extractor: object | None = None
    customer_phone: str | None = None
    # Répondeur hors script. Le parcours est piloté par le code
    # (`question_router`/`routed_turn`) et le modèle n'intervient que sur les
    # vraies questions du client.
    answerer: object | None = None
    # Met en mots la question choisie par le code. Absent, le bot retombe
    # sur les formulations figées — correctes, mais robotiques.
    phraseur: object | None = None


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
        # Clé d'idempotence UNIQUE par connexion pour l'ingestion. On ne se
        # fie PAS au call_id d'Asterisk : selon le dialplan, l'UUID AudioSocket
        # peut être FIGÉ (le même à chaque appel), et toutes les réservations
        # retombaient alors sur la même clé (twilioCallSid) — le backend
        # rejetait la 2ᵉ en 409 et elle n'était jamais créée. Un `ingest_ref`
        # minté ici garantit l'unicité quel que soit le comportement d'Asterisk,
        # tout en dédupliquant un même appel rejoué dans la même connexion.
        set_call_context(call_id=call_id, ingest_ref=uuid4().hex)
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
            await run_ai_loop(reader, adapter, call_id, session, deps, config)

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
        clear_call_context()


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


async def run_ai_loop(
    reader: asyncio.StreamReader,
    adapter: AsteriskAudioSocketAdapter,
    call_id: str,
    session: CallSession,
    deps: AudioSocketServerDeps,
    config: AudioSocketServerConfig,
) -> None:
    if deps.stt_adapter is None or deps.tts_adapter is None:
        logger.error(
            "AI loop requires stt_adapter and tts_adapter — set them or use smoke_test",
            extra={"call_id": call_id},
        )
        return

    await _speak(adapter, call_id, session.context.greeting, deps, config, reader=reader)

    endpointer = _build_endpointer(config)
    loop = asyncio.get_running_loop()
    derniere_activite = loop.time()
    relances_faites = 0

    while True:
        try:
            frame = await asyncio.wait_for(
                read_packet(reader), timeout=config.lecture_timeout_s
            )
        except IncompleteReadError:
            logger.info("peer closed during AI loop", extra={"call_id": call_id})
            return
        except TimeoutError:
            # Aucun paquet reçu (connexion à moitié ouverte) : on ne bloque pas,
            # on retombe sur la vérification d'inactivité ci-dessous.
            frame = None

        if isinstance(frame, HangupFrame):
            logger.info("hangup from Asterisk", extra={"call_id": call_id})
            return
        if isinstance(frame, ErrorFrame):
            logger.warning(
                "error frame from Asterisk",
                extra={"call_id": call_id, "code": frame.code},
            )
            continue

        pcm = None
        if isinstance(frame, AudioFrame):
            # Le VAD décide seul du point de coupure : il renvoie l'énoncé
            # complet quand la parole s'arrête (hangover) ou à la longueur max,
            # et None tant qu'on accumule.
            pcm = endpointer.feed(frame.audio)

        if pcm is None:
            # Pas de tour prêt : surveille l'inactivité pour ne pas rester muet.
            inactif = loop.time() - derniere_activite
            action = _action_inactivite(
                inactif, relances_faites=relances_faites, config=config
            )
            if action == "relance":
                await _speak(
                    adapter, call_id,
                    "Êtes-vous toujours là ? Je reste à votre écoute.",
                    deps, config, reader=reader,
                )
                relances_faites += 1
            elif action == "fin":
                await _speak(
                    adapter, call_id,
                    "Je n'ai pas de réponse, je vous laisse rappeler. Bonne journée !",
                    deps, config, reader=reader,
                )
                logger.info("clôture sur inactivité", extra={"call_id": call_id})
                return
            continue

        fini, audio_repris = await _process_turn(
            pcm, adapter, call_id, session, deps, config, reader=reader
        )
        # Un vrai tour a eu lieu : on repart à zéro sur l'inactivité.
        derniere_activite = loop.time()
        relances_faites = 0
        if fini:
            return
        if audio_repris:
            # Le client a coupé le bot : sa phrase a déjà commencé, on réinjecte
            # ce qu'on a entendu pour reprendre la collecte sans la perdre.
            endpointer.seed(audio_repris)


def _action_inactivite(
    inactif_s: float, *, relances_faites: int, config: AudioSocketServerConfig
) -> str:
    """Décide quoi faire après `inactif_s` sans tour de parole valable.

    Le bot relance à chaque palier de `inactivite_relance_s`, jusqu'à
    `inactivite_max_relances` fois — on ne raccroche pas vite sur un client
    présent dont la voix passe mal (écho). On ne clôt (`"fin"`) qu'après le
    dernier palier resté sans réponse. Renvoie "fin", "relance" ou "rien".
    Pur et sans effet de bord pour être testable sans socket.
    """
    seuil = config.inactivite_relance_s
    if (
        relances_faites < config.inactivite_max_relances
        and inactif_s >= seuil * (relances_faites + 1)
    ):
        return "relance"
    if inactif_s >= seuil * (config.inactivite_max_relances + 1):
        return "fin"
    return "rien"


def _build_endpointer(config: AudioSocketServerConfig) -> VadEndpointer:
    """Construit l'endpointer VAD, avec repli sur un VAD énergie (RMS) si
    `webrtcvad` n'est pas installé — le bot reste fonctionnel dans tous les cas.
    """
    common = dict(
        frame_bytes=AUDIOSOCKET_CHUNK_20MS_BYTES,
        sample_rate=config.sample_rate,
        min_speech_frames=config.min_speech_frames,
        hangover_frames=config.silence_frames_to_end,
        start_padding_frames=config.vad_padding_frames,
        max_utterance_frames=config.max_utterance_frames,
    )
    try:
        import webrtcvad  # noqa: F401 — sonde la présence de la lib

        return VadEndpointer(aggressiveness=config.vad_aggressiveness, **common)
    except ImportError:
        seuil = config.silence_rms_threshold
        logger.warning(
            "webrtcvad absent — repli sur un VAD d'énergie (RMS, seuil %.0f)",
            seuil,
        )
        return VadEndpointer(vad=lambda f, sr: frame_rms(f) >= seuil, **common)


async def _process_turn(
    pcm: bytes,
    adapter: AsteriskAudioSocketAdapter,
    call_id: str,
    session: CallSession,
    deps: AudioSocketServerDeps,
    config: AudioSocketServerConfig,
    reader: Any = None,
) -> tuple[bool, bytes]:
    # Ce que le client dit pendant la réponse du bot : il a coupé, et sa
    # phrase commence là. La perdre l'obligerait à tout répéter.
    interruption: dict[str, bytes] = {"audio": b""}

    text = await _transcribe(deps.stt_adapter, pcm, config)
    if not text.strip():
        logger.info("empty transcription, ignoring turn", extra={"call_id": call_id})
        return False, interruption["audio"]

    conversation.info("CLIENT : %s", text)

    async def _say(message: str) -> None:
        resultat = await _speak(adapter, call_id, message, deps, config, reader=reader)
        if resultat.interrompu and resultat.audio_client:
            interruption["audio"] = resultat.audio_client

    # Le parcours est piloté par le code : `routed_turn` collecte les slots,
    # ne sollicite le modèle que sur les vraies questions (`answerer`), et
    # garde la main sur la vérification de dispo et la réservation.
    state = _routing_state(session)
    outcome = await run_routed_turn(
        session=session,
        user_text=text,
        customer_phone=deps.customer_phone,
        history=_history_for(session),
        extractor=deps.slot_extractor,
        answerer=deps.answerer,
        awaiting_confirmation=state["awaiting"],
        speak=_say,
        recent_phrasings=state["phrasings"],
        phraseur=deps.phraseur,
    )
    state["awaiting"] = outcome.awaiting_confirmation
    return outcome.should_end, interruption["audio"]


def _routing_state(session: object) -> dict:
    """État de routage porté par la session (attente de confirmation,
    formulations déjà employées)."""
    state = getattr(session, "_hikky_routing", None)
    if state is None:
        state = {"awaiting": False, "phrasings": set()}
        try:
            session._hikky_routing = state
        except AttributeError:
            logger.warning("session sans attribut libre — état de routage volatil")
    return state


def _history_for(session: object) -> list:
    """Historique porté par la session, créé à la volée au premier tour."""
    history = getattr(session, "_hikky_history", None)
    if history is None:
        history = []
        try:
            session._hikky_history = history
        except AttributeError:
            logger.warning("session sans attribut libre — historique non conservé")
    return history


async def _transcribe(
    stt: object, pcm: bytes, config: AudioSocketServerConfig
) -> str:
    audio = resample_pcm16(pcm, config.sample_rate, WHISPER_SAMPLE_RATE_HZ)

    async def _chunks():
        for offset in range(0, len(audio), _STT_CHUNK_BYTES):
            yield audio[offset : offset + _STT_CHUNK_BYTES]

    stream = await stt.transcribe(_chunks())
    parts = [part.strip() async for part in stream]
    return " ".join(part for part in parts if part)


@dataclass(slots=True)
class ResultatParole:
    """Ce qu'il s'est passé pendant que le bot parlait.

    `audio_client` porte le début de la phrase du client, entendu avant
    que la diffusion ne s'arrête. Le jeter ferait commencer la
    transcription au milieu d'un mot.
    """

    interrompu: bool = False
    audio_client: bytes = b""
    raccroche: bool = False


async def _speak(
    adapter: AsteriskAudioSocketAdapter,
    call_id: str,
    text: str,
    deps: AudioSocketServerDeps,
    config: AudioSocketServerConfig,
    reader: Any = None,
) -> ResultatParole:
    if not text:
        return ResultatParole()

    conversation.info("BOT    : %s", text)

    # Le taux réel de la voix prime sur la valeur de config : les voix Piper
    # vont de 16 kHz (gilles-low) à 44,1 kHz (tom-medium). Rééchantillonner
    # depuis un taux supposé faux jouerait l'audio à la mauvaise vitesse.
    source_rate = getattr(deps.tts_adapter, "sample_rate", None) or config.tts_sample_rate

    frame_bytes = AUDIOSOCKET_CHUNK_20MS_BYTES
    frame_seconds = (frame_bytes / 2) / config.sample_rate
    per_batch = max(1, config.frames_per_batch)
    loop = asyncio.get_running_loop()
    deadline = loop.time()

    resampler = StreamingResampler(source_rate, config.sample_rate)
    pending = bytearray()
    worst_lag_ms = 0.0
    stopped = False

    resultat = ResultatParole()
    ecoute: asyncio.Task | None = None
    interruption = asyncio.Event()

    if reader is not None and config.barge_in_enabled:
        detecteur = DetecteurInterruption(
            seuil_rms=config.barge_in_rms_threshold,
            trames_consecutives=config.barge_in_frames,
            trames_de_garde=config.barge_in_guard_frames,
        )

        async def _ecouter() -> None:
            """Lit le flux entrant pendant que le bot parle."""
            while not interruption.is_set():
                try:
                    frame = await read_packet(reader)
                except (IncompleteReadError, ConnectionError, OSError):
                    resultat.raccroche = True
                    interruption.set()
                    return
                if isinstance(frame, HangupFrame):
                    resultat.raccroche = True
                    interruption.set()
                    return
                if not isinstance(frame, AudioFrame):
                    continue
                if detecteur.observer(frame.audio):
                    resultat.interrompu = True
                    resultat.audio_client = detecteur.audio_capte()
                    interruption.set()
                    return

        ecoute = asyncio.create_task(_ecouter())

    async def _drain(force: bool) -> bool:
        """Découpe `pending` en trames et les envoie. Rend False si coupé."""
        nonlocal worst_lag_ms, deadline
        while len(pending) >= frame_bytes * (1 if force else per_batch):
            if interruption.is_set():
                # Le client a repris la parole : ce qui reste ne doit pas
                # être joué, sinon l'interruption ne s'entend pas.
                return False
            frames = []
            while pending and len(frames) < per_batch:
                chunk = bytes(pending[:frame_bytes])
                del pending[:frame_bytes]
                if len(chunk) < frame_bytes:
                    chunk += b"\x00" * (frame_bytes - len(chunk))
                frames.append(chunk)
            if not frames:
                break
            try:
                await adapter.send_audio_frames(call_id, frames)
            except (TelephonyError, ConnectionError, OSError):
                logger.info("flux coupé pendant la parole", extra={"call_id": call_id})
                return False
            if config.realtime_playback:
                deadline += frame_seconds * len(frames)
                delay = deadline - loop.time()
                if delay > 0:
                    await asyncio.sleep(delay)
                elif delay < -frame_seconds:
                    # On n'alimente plus Asterisk en temps réel : son tampon
                    # se vide entre deux envois, ce qui s'entend comme un
                    # grésillement.
                    worst_lag_ms = max(worst_lag_ms, -delay * 1000)
        return True

    # On émet au fil de la synthèse : XTTS livre son premier morceau en
    # ~360 ms mais met plus de 4 s pour une phrase entière. Attendre la
    # totalité ajoutait ces secondes au silence perçu par l'appelant.
    stream = await deps.tts_adapter.synthesize(text)
    async for chunk in stream:
        pending.extend(resampler.push(chunk))
        if not await _drain(force=False):
            stopped = True
            break

    if not stopped:
        pending.extend(resampler.flush())
        await _drain(force=True)

    if worst_lag_ms > 0:
        logger.warning(
            "diffusion en retard de %.0f ms — Asterisk risque de grésiller",
            worst_lag_ms,
            extra={"call_id": call_id},
        )

    if ecoute is not None:
        interruption.set()
        ecoute.cancel()
        try:
            await ecoute
        except (asyncio.CancelledError, Exception):  # noqa: BLE001
            pass

    if resultat.interrompu:
        conversation.info("  (interrompu par le client)")
    return resultat


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
