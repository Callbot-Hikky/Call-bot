"""Lance le serveur AudioSocket seul, en mode smoke test.

Utile pour valider que le protocole AudioSocket fonctionne bout-en-bout
entre Asterisk et le bot Hikky, AVANT même d'avoir configuré Whisper /
LlamaCpp / Piper.

Usage :

    cd Call-bot
    python -m scripts.run_audiosocket_smoke

Ou directement :

    PYTHONPATH=src python scripts/run_audiosocket_smoke.py

Puis, depuis un softphone enregistré sur l'extension 1000 (cf. FreePBX),
compose 2000. Tu dois voir des logs "audiosocket connection opened" et
"call bound" côté serveur — c'est la preuve que le transport marche.
"""

from __future__ import annotations

import asyncio
import logging
import sys
from datetime import datetime, time
from pathlib import Path

# Permet de lancer le script sans installation en dev
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from hikky.adapters.telephony.asterisk_audiosocket_server import (  # noqa: E402
    AudioSocketServerConfig,
    AudioSocketServerDeps,
    start_server,
)
from hikky.domain.call_session import CallSession  # noqa: E402
from hikky.domain.dialogue_engine import DialogueEngine  # noqa: E402
from hikky.domain.fallback_policy import FallbackPolicy  # noqa: E402
from hikky.domain.restaurant_context import (  # noqa: E402
    OpeningHours,
    RestaurantContext,
    RestaurantRules,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("smoke")


# ── Stubs (aucun modèle IA n'est chargé en smoke test) ──────────────────────


class _FakeRestaurantCtxPort:
    async def load(self, called_number):
        return RestaurantContext(
            id="r-poc",
            name="Le Petit Sud (POC)",
            greeting="Bonjour, POC AudioSocket.",
            opening_hours=[
                OpeningHours(weekday=i, opens=time(9, 0), closes=time(23, 0))
                for i in range(7)
            ],
            total_capacity=40,
            rules=RestaurantRules(),
            transfer_number=None,
            fallback_message="Merci de rappeler plus tard.",
        )


class _NoopReservationPort:
    async def check_availability(self, *a, **k):
        return True

    async def create(self, *a, **k):
        return "res-noop"

    async def create_callback_request(self, *a, **k):
        return "cb-noop"


class _NoopCallLogPort:
    async def start(self, *a, **k):
        pass

    async def end(self, *a, **k):
        pass


class _NoopNotifPort:
    async def send_confirmation(self, *a, **k):
        pass


class _NoopLLM:
    async def complete(self, messages):
        return ""


def _session_factory(call_id: str, ctx: RestaurantContext) -> CallSession:
    return CallSession(
        call_id=call_id,
        context=ctx,
        dialogue_engine=DialogueEngine(_NoopLLM()),  # type: ignore[arg-type]
        fallback_policy=FallbackPolicy(),
        reservation_port=_NoopReservationPort(),  # type: ignore[arg-type]
        call_log=_NoopCallLogPort(),  # type: ignore[arg-type]
        notification=_NoopNotifPort(),  # type: ignore[arg-type]
        clock=datetime.now,
    )


async def main():
    deps = AudioSocketServerDeps(
        session_factory=_session_factory,
        restaurant_context_port=_FakeRestaurantCtxPort(),  # type: ignore[arg-type]
    )
    config = AudioSocketServerConfig(
        host="0.0.0.0",
        port=6666,
        smoke_test=True,
        smoke_test_duration_s=15.0,
    )
    server, _adapter = await start_server(deps, config)

    logger.info("AudioSocket smoke test server on 0.0.0.0:6666 — Ctrl+C to stop.")
    logger.info(
        "Dialplan Asterisk attendu : "
        "AudioSocket(${CHANNEL(uuid)},<IP-de-cette-machine>:6666)"
    )
    try:
        await server.serve_forever()
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        server.close()
        await server.wait_closed()
        logger.info("server stopped")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
