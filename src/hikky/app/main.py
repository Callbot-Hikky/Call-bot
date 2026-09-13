"""Application FastAPI — point d'entrée HTTP/WS du callbot Hikky.

Lancement local (machine GPU avec extras `[voice]` et `[pipeline]` installés) :

    uvicorn hikky.app.main:app --host 0.0.0.0 --port 8000

Exposition à Twilio en développement :

    ngrok http 8000

Côté Twilio : configurer le numéro pour qu'au décrochage il
<Connect><Stream> vers `wss://<host-ngrok>/twilio/{{CallSid}}` avec
`<Parameter name="To" value="{{To}}" />` dans le `<Stream>` pour qu'on
puisse identifier le restaurant côté Hikky.

Pipeline appliquée à chaque appel (Pipecat) :

    Twilio WS  ─►  FastAPIWebsocketTransport (in, TwilioFrameSerializer)
              ─►  HikkySTTService (faster-whisper)
              ─►  DialogueProcessor (CallSession + FallbackPolicy)
              ─►  HikkyTTSService (Piper)
              ─►  FastAPIWebsocketTransport (out)  ─►  Twilio WS

Note de testabilité : le handler WS complet exige une vraie WebSocket
+ machinerie audio Pipecat — il n'est pas exécutable en test unitaire.
Le smoke test (TestClient) vérifie seulement que `/health` répond et
que `/twilio/{call_sid}` accepte la connexion. Le runtime réel est
validé sur la machine GPU avec un vrai appel Twilio.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from hikky.adapters.telephony.asterisk_audiosocket_server import (
    AudioSocketServerConfig,
    AudioSocketServerDeps,
)
from hikky.adapters.telephony.asterisk_audiosocket_server import (
    start_server as start_audiosocket_server,
)
from hikky.domain.outcomes import CallOutcome
from hikky.exceptions import UnknownRestaurant
from hikky.observability.logging import (
    clear_call_context,
    configure_json_logging,
    set_call_context,
)

if TYPE_CHECKING:
    from hikky.domain.call_session import CallSession
    from hikky.ports.restaurant_context import RestaurantContextPort
    from hikky.ports.speech_recognition import SpeechRecognitionPort
    from hikky.ports.speech_synthesis import SpeechSynthesisPort

logger = logging.getLogger("hikky.app")


@dataclass(slots=True)
class AppDependencies:
    """Briques injectées dans l'app — mockables en test."""

    stt_adapter: SpeechRecognitionPort
    tts_adapter: SpeechSynthesisPort
    restaurant_context_port: RestaurantContextPort
    session_factory: Callable[[str, object], CallSession]
    # session_factory(call_sid, restaurant_context) → CallSession câblée

    slot_extractor: object | None = None  # SlotExtractor concret (LLM-driven en prod)
    tts_sample_rate: int = 22050


def _audiosocket_config_from_env() -> AudioSocketServerConfig | None:
    """Charge la config du serveur AudioSocket depuis les variables d'env.

    Retourne `None` si `HIKKY_AUDIOSOCKET_ENABLED` n'est pas à `1` — dans
    ce cas le serveur AudioSocket n'est pas démarré (le transport Twilio
    continue de fonctionner normalement).

    Variables :
    - `HIKKY_AUDIOSOCKET_ENABLED` : "1" pour activer (défaut : "0")
    - `HIKKY_AUDIOSOCKET_HOST` : bind host (défaut : "0.0.0.0")
    - `HIKKY_AUDIOSOCKET_PORT` : bind port (défaut : 6666)
    - `HIKKY_AUDIOSOCKET_DEFAULT_CALLED_NUMBER` : fallback quand Asterisk
      n'envoie pas de numéro (défaut : "+33000000000")
    - `HIKKY_AUDIOSOCKET_SMOKE_TEST` : "1" pour le mode smoke test
      (défaut : "0") — utile pour valider le transport avant d'avoir
      les vrais adapters IA en place.
    """
    if os.environ.get("HIKKY_AUDIOSOCKET_ENABLED", "0") != "1":
        return None
    return AudioSocketServerConfig(
        host=os.environ.get("HIKKY_AUDIOSOCKET_HOST", "0.0.0.0"),
        port=int(os.environ.get("HIKKY_AUDIOSOCKET_PORT", "6666")),
        default_called_number=os.environ.get(
            "HIKKY_AUDIOSOCKET_DEFAULT_CALLED_NUMBER", "+33000000000"
        ),
        smoke_test=os.environ.get("HIKKY_AUDIOSOCKET_SMOKE_TEST", "0") == "1",
    )


def create_app(
    deps: AppDependencies | None = None,
    audiosocket_config: AudioSocketServerConfig | None = None,
) -> FastAPI:
    configure_json_logging()

    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        """Démarre le serveur AudioSocket en parallèle de FastAPI si activé.

        Si `audiosocket_config is None` OU si `deps is None` (pas d'IA
        configurée), on skip — l'app démarre normalement pour servir
        Twilio et `/health`.
        """
        server_task: asyncio.Task | None = None
        server = None
        if audiosocket_config is not None and deps is not None:
            audiosocket_deps = AudioSocketServerDeps(
                session_factory=deps.session_factory,
                restaurant_context_port=deps.restaurant_context_port,
                stt_adapter=deps.stt_adapter,
                tts_adapter=deps.tts_adapter,
                slot_extractor=deps.slot_extractor,
            )
            audiosocket_config.tts_sample_rate = deps.tts_sample_rate
            server, _adapter = await start_audiosocket_server(
                audiosocket_deps, audiosocket_config
            )
            app.state.audiosocket_server = server
            server_task = asyncio.create_task(server.serve_forever())
            logger.info(
                "AudioSocket server started",
                extra={
                    "host": audiosocket_config.host,
                    "port": audiosocket_config.port,
                    "smoke_test": audiosocket_config.smoke_test,
                },
            )
        try:
            yield
        finally:
            if server is not None:
                server.close()
                await server.wait_closed()
            if server_task is not None:
                server_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await server_task
                logger.info("AudioSocket server stopped")

    app = FastAPI(title="Hikky IA", lifespan=lifespan)
    app.state.deps = deps

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.websocket("/twilio/{call_sid}")
    async def twilio_stream(websocket: WebSocket, call_sid: str) -> None:
        await websocket.accept()
        set_call_context(call_id=call_sid)
        try:
            if deps is None:
                logger.warning(
                    "WS opened without configured AppDependencies — closing",
                    extra={"call_sid": call_sid},
                )
                await websocket.close(code=1011)
                return

            await _run_call(websocket, call_sid, deps)

        except WebSocketDisconnect:
            logger.info("WS disconnected")
        except Exception:  # noqa: BLE001 — boundary handler
            logger.exception("WS handler crashed")
            try:
                await websocket.close(code=1011)
            except Exception:  # noqa: BLE001
                pass
        finally:
            clear_call_context()

    return app


async def _run_call(
    websocket: WebSocket, call_sid: str, deps: AppDependencies
) -> None:
    """Boucle d'appel : lit le `start` de Twilio, charge le restaurant, build
    la pipeline Pipecat et la lance.

    Cette fonction nécessite le runtime Pipecat complet ; elle n'est pas
    exécutée dans la suite de tests. Le déploiement GPU la valide.
    """
    # 1. Lire le premier event Twilio pour récupérer streamSid + called number
    raw = await websocket.receive_text()
    payload = json.loads(raw)
    if payload.get("event") != "start":
        logger.warning("Expected Twilio 'start' event, got %s", payload.get("event"))
        await websocket.close(code=1011)
        return

    start = payload.get("start", {})
    stream_sid = payload.get("streamSid") or start.get("streamSid", "")
    custom = start.get("customParameters", {}) or {}
    called_number = custom.get("To") or custom.get("to") or ""

    # 2. Charger le contexte restaurant
    try:
        ctx = await deps.restaurant_context_port.load(called_number)
    except UnknownRestaurant:
        logger.warning("Unknown restaurant for number %s", called_number)
        await websocket.close(code=1011)
        return

    set_call_context(call_id=call_sid, restaurant_id=ctx.id)
    logger.info("starting call", extra={"called_number": called_number})

    # 3. Build + run la Pipeline Pipecat
    from pipecat.pipeline.runner import PipelineRunner
    from pipecat.serializers.twilio import TwilioFrameSerializer
    from pipecat.transports.websocket.fastapi import (
        FastAPIWebsocketParams,
        FastAPIWebsocketTransport,
    )

    from hikky.pipeline.builder import build_pipeline_task

    transport = FastAPIWebsocketTransport(
        websocket=websocket,
        params=FastAPIWebsocketParams(
            serializer=TwilioFrameSerializer(stream_sid=stream_sid, call_sid=call_sid),
            audio_in_enabled=True,
            audio_out_enabled=True,
            add_wav_header=False,
        ),
    )

    session = deps.session_factory(call_sid, ctx)
    try:
        build = build_pipeline_task(
            transport_input=transport.input(),
            transport_output=transport.output(),
            stt_adapter=deps.stt_adapter,
            tts_adapter=deps.tts_adapter,
            session=session,
            slot_extractor=deps.slot_extractor,  # type: ignore[arg-type]
            tts_sample_rate=deps.tts_sample_rate,
        )

        runner = PipelineRunner()
        await runner.run(build.task)
    finally:
        # Filet de sécurité : si la Pipeline a planté avant que le
        # DialogueProcessor n'ait pu fermer la session, on s'assure que
        # l'appel est journalisé comme `technical_error`. `end_with` est
        # idempotent côté CallSession — no-op si déjà terminé.
        try:
            await session.end_with(CallOutcome.TECHNICAL_ERROR)
        except Exception:  # noqa: BLE001 — meilleur effort, on quitte
            logger.exception("Failed to end session in cleanup")


def _build_default_dependencies() -> AppDependencies | None:
    """Tente de construire les dépendances réelles depuis l'environnement.
    Si une variable est absente, log un avertissement et renvoie None —
    l'app démarre quand même (pour `/health`) mais refuse les appels."""
    try:
        from hikky.app.config import MissingConfig, load_from_env
        from hikky.app.dependencies import build_app_dependencies
    except ImportError:
        logger.warning("Dependency assembly modules unavailable")
        return None

    try:
        config = load_from_env()
    except MissingConfig as exc:
        logger.warning(
            "Missing env var %s — app will refuse Twilio connections "
            "(set %s and restart to enable)",
            exc,
            exc,
        )
        return None

    return build_app_dependencies(config)


# Instance par défaut, utilisable par `uvicorn hikky.app.main:app`. Si les
# variables d'environnement requises sont présentes, les vraies dépendances
# sont câblées ; sinon l'app accepte `/health` mais refuse `/twilio/...`.
# Le serveur AudioSocket est démarré si `HIKKY_AUDIOSOCKET_ENABLED=1`.
app = create_app(
    deps=_build_default_dependencies(),
    audiosocket_config=_audiosocket_config_from_env(),
)
