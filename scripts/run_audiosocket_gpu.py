"""Serveur AudioSocket avec les vrais modèles IA et un Back stubbé.

Destiné à la machine GPU (RunPod). Variables d'environnement :

    HIKKY_LLAMA_MODEL_PATH   (requis)  GGUF pour llama.cpp
    HIKKY_PIPER_MODEL_PATH   (requis)  voix Piper .onnx
    HIKKY_WHISPER_MODEL      large-v3
    HIKKY_WHISPER_DEVICE     cuda
    HIKKY_WHISPER_COMPUTE    float16
    HIKKY_LLAMA_N_CTX        4096
    HIKKY_LLAMA_N_GPU_LAYERS -1
    HIKKY_TTS_SAMPLE_RATE    22050
    HIKKY_AUDIOSOCKET_PORT   6666
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys

from poc_common import (
    StubRestaurantContextPort,
    build_session_factory,
    default_restaurant_context,
)

from hikky.adapters.back.backend_restaurant_context import (
    BackendRestaurantContextAdapter,
)
from hikky.adapters.back.call_ingest_adapter import CallIngestAdapter
from hikky.adapters.back.http_client import BackHttpClient
from hikky.adapters.telephony.asterisk_audiosocket_server import (
    AudioSocketServerConfig,
    AudioSocketServerDeps,
    start_server,
)
from hikky.adapters.telephony.audio_resample import resample_pcm16
from hikky.adapters.voice.faster_whisper_stt import FasterWhisperSTTAdapter
from hikky.adapters.voice.llama_cpp_llm import LlamaCppLLMAdapter
from hikky.adapters.voice.piper_tts import PiperTTSAdapter
from hikky.adapters.voice.xtts_tts import XttsTTSAdapter
from hikky.domain.conversation_brain import ConversationBrain, QuestionAnswerer
from hikky.domain.phraseur import Phraseur
from hikky.pipeline.llm_slot_extractor import LLMSlotExtractor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("gpu")


def _require(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        logger.error("missing required env var %s", name)
        sys.exit(1)
    return value


async def _warmup(stt, tts, llm) -> None:
    """Charge les trois modèles avant d'accepter le premier appel.

    Sans ça, l'appelant paie le chargement GPU au premier tour de parole
    (~2,4 s pour Whisper large-v3, plus le GGUF de llama.cpp) — soit
    précisément le moment où l'on veut faire bonne impression.
    """
    import time

    t0 = time.time()
    try:
        buf = bytearray()
        async for chunk in await tts.synthesize("Bonjour."):
            buf.extend(chunk)
        logger.info("TTS chaud (%d octets @ %s Hz)", len(buf), tts.sample_rate)

        pcm16 = resample_pcm16(
            resample_pcm16(bytes(buf), tts.sample_rate or 22050, 8000), 8000, 16000
        )

        async def _chunks():
            for offset in range(0, len(pcm16), 3200):
                yield pcm16[offset : offset + 3200]

        text = " ".join([t async for t in await stt.transcribe(_chunks())]).strip()
        logger.info("STT chaud (%r)", text[:60])

        await llm.complete([{"role": "user", "content": "Bonjour"}])
        logger.info("LLM chaud")
    except Exception:
        logger.exception("prechauffage incomplet — le premier appel sera plus lent")
    logger.info("prechauffage termine en %.1fs", time.time() - t0)


async def main() -> None:
    llama_path = _require("HIKKY_LLAMA_MODEL_PATH")
    piper_path = os.environ.get("HIKKY_PIPER_MODEL_PATH", "")

    logger.info("loading models (first load can take a minute)")

    llm = LlamaCppLLMAdapter(
        model_path=llama_path,
        n_ctx=int(os.environ.get("HIKKY_LLAMA_N_CTX", "4096")),
        n_gpu_layers=int(os.environ.get("HIKKY_LLAMA_N_GPU_LAYERS", "-1")),
    )
    stt = FasterWhisperSTTAdapter(
        model_name=os.environ.get("HIKKY_WHISPER_MODEL", "large-v3"),
        device=os.environ.get("HIKKY_WHISPER_DEVICE", "cuda"),
        compute_type=os.environ.get("HIKKY_WHISPER_COMPUTE", "float16"),
    )
    engine = os.environ.get("HIKKY_TTS_ENGINE", "xtts").lower()
    if engine == "xtts":
        tts = XttsTTSAdapter(speaker=os.environ.get("HIKKY_XTTS_SPEAKER", "Lilya Stainthorpe"))
        logger.info("TTS = XTTS-v2 (%s)", os.environ.get("HIKKY_XTTS_SPEAKER", "Lilya Stainthorpe"))
    else:
        tts = PiperTTSAdapter(model_path=piper_path)
        logger.info("TTS = Piper (%s)", piper_path)

    # Backend réel si configuré, sinon ports factices. Sans cette bascule,
    # le bot annonce des réservations qui ne sont enregistrées nulle part.
    back_url = os.environ.get("HIKKY_BACK_BASE_URL")
    reservation_port = None
    context_port = None
    if back_url:
        client = BackHttpClient(
            base_url=back_url,
            api_key=os.environ.get("HIKKY_BACK_API_KEY", ""),
        )
        reservation_port = CallIngestAdapter(
            client,
            restaurant_phone=os.environ.get(
                "HIKKY_RESTAURANT_PHONE", "+33000000000"
            ),
        )
        # Le restaurant est résolu par son numéro d'appel : un identifiant
        # inventé côté bot ferait échouer toutes les requêtes (Spring
        # renvoie 401 sur un UUID invalide).
        context_port = BackendRestaurantContextAdapter(client)
        logger.info("Back HTTP réel : %s", back_url)
    else:
        logger.warning("HIKKY_BACK_BASE_URL absent — réservations NON enregistrées")

    deps = AudioSocketServerDeps(
        session_factory=build_session_factory(llm, reservation_port=reservation_port),
        restaurant_context_port=context_port
        or StubRestaurantContextPort(default_restaurant_context()),
        stt_adapter=stt,
        tts_adapter=tts,
        slot_extractor=LLMSlotExtractor(llm),
        brain=ConversationBrain(llm),
        answerer=QuestionAnswerer(llm),
        phraseur=Phraseur(llm),
    )
    config = AudioSocketServerConfig(
        host=os.environ.get("HIKKY_AUDIOSOCKET_HOST", "0.0.0.0"),
        port=int(os.environ.get("HIKKY_AUDIOSOCKET_PORT", "6666")),
        smoke_test=False,
        # Asterisk ne transmet pas encore le numéro appelé : on utilise
        # celui du restaurant configuré pour résoudre le contexte.
        default_called_number=os.environ.get(
            "HIKKY_RESTAURANT_PHONE", "+33000000000"
        ),
        tts_sample_rate=int(os.environ.get("HIKKY_TTS_SAMPLE_RATE", "22050")),
    )

    await _warmup(stt, tts, llm)

    server, _adapter = await start_server(deps, config)
    logger.info("AudioSocket ready on %s:%s", config.host, config.port)
    try:
        await server.serve_forever()
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        server.close()
        await server.wait_closed()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
