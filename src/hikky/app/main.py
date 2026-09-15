"""Application FastAPI — point d'entrée HTTP du callbot Hikky.

Lancement local (machine GPU avec extras `[voice]` installés) :

    uvicorn hikky.app.main:app --host 0.0.0.0 --port 8000

Le transport voix passe par le **serveur AudioSocket** (Asterisk), démarré
dans le `lifespan` quand `HIKKY_AUDIOSOCKET_ENABLED=1`. FastAPI n'expose
plus que `/health` ; l'audio ne transite pas par HTTP.

Note de testabilité : la boucle d'appel réelle vit dans
`asterisk_audiosocket_server.py` et est validée sur la machine GPU avec un
vrai appel. Le smoke test (TestClient) vérifie seulement que `/health`
répond.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from fastapi import FastAPI

from hikky.adapters.telephony.asterisk_audiosocket_server import (
    AudioSocketServerConfig,
    AudioSocketServerDeps,
)
from hikky.adapters.telephony.asterisk_audiosocket_server import (
    start_server as start_audiosocket_server,
)
from hikky.observability.logging import configure_json_logging

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
    ce cas le serveur AudioSocket n'est pas démarré (l'app sert `/health`).

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
        `/health`.
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

    return app


def _build_default_dependencies() -> AppDependencies | None:
    """Tente de construire les dépendances réelles depuis l'environnement.
    Si une variable est absente, log un avertissement et renvoie None —
    l'app démarre quand même (pour `/health`) mais aucun appel n'est pris."""
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
            "Missing env var %s — app will refuse calls "
            "(set %s and restart to enable)",
            exc,
            exc,
        )
        return None

    return build_app_dependencies(config)


# Instance par défaut, utilisable par `uvicorn hikky.app.main:app`. Si les
# variables d'environnement requises sont présentes, les vraies dépendances
# sont câblées ; sinon l'app accepte `/health` mais ne prend aucun appel.
# Le serveur AudioSocket est démarré si `HIKKY_AUDIOSOCKET_ENABLED=1`.
app = create_app(
    deps=_build_default_dependencies(),
    audiosocket_config=_audiosocket_config_from_env(),
)
