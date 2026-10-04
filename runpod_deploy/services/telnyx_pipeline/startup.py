"""Démarrage et arrêt de l'orchestrateur : charger les modèles, brancher les voisins.

Tout ce qui coûte cher (LLM 32B, client HTTP, salutation synthétisée) est construit UNE
fois ici, au démarrage du serveur, et rangé dans le dictionnaire `state` partagé par tous
les appels. À l'arrêt, le client HTTP est fermé proprement.

`make_lifespan` remplace `@app.on_event("startup")`, déprécié par FastAPI : un seul
endroit décrit le démarrage (avant le `yield`) ET l'arrêt (après).
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import AsyncGenerator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager

import httpx
from fastapi import FastAPI

from . import audio
from .call_state import CallState
from .config import Settings
from .speaker import TelnyxSpeaker

log = logging.getLogger("telnyx")


async def build_services(settings: Settings, state: dict) -> None:
    """Construit les services partagés et les range dans `state`."""
    from poc_common import build_session_factory

    from hikky.adapters.back.backend_restaurant_context import BackendRestaurantContextAdapter
    from hikky.adapters.back.call_ingest_adapter import CallIngestAdapter
    from hikky.adapters.back.http_client import BackHttpClient
    from hikky.adapters.back.knowledge_adapter import BackendKnowledgeAdapter
    from hikky.adapters.voice.llama_cpp_llm import LlamaCppLLMAdapter
    from hikky.domain.conversation_brain import QuestionAnswerer
    from hikky.domain.phraseur import Phraseur
    from hikky.pipeline.llm_slot_extractor import LLMSlotExtractor

    t = time.time()
    log.info("chargement LLM Qwen 32B...")
    llm = LlamaCppLLMAdapter(model_path=settings.llm_gguf, n_ctx=4096, n_gpu_layers=-1)
    log.info("LLM chargé en %.1fs", time.time() - t)
    client = BackHttpClient(base_url=settings.back_base_url, api_key=settings.back_api_key)
    state["context_port"] = BackendRestaurantContextAdapter(client)
    state["session_factory"] = build_session_factory(
        llm, reservation_port=CallIngestAdapter(client, restaurant_phone=settings.restaurant_phone)
    )
    # Base de connaissances du restaurant : le port ne lève jamais, backend injoignable
    # = base vide, l'appel continue.
    knowledge = BackendKnowledgeAdapter(client, restaurant_phone=settings.restaurant_phone)
    state["answerer"] = QuestionAnswerer(llm, knowledge=knowledge)
    state["phraseur"] = Phraseur(llm)
    state["extractor"] = LLMSlotExtractor(llm)
    state["http"] = httpx.AsyncClient(timeout=60.0)
    log.info(
        "heure locale du bot : %s (%s)", time.strftime("%A %d %B %Y %H:%M"), os.environ.get("TZ")
    )

    # Salutation : texte fixe, synthétisée une fois, trames prêtes.
    try:
        probe = TelnyxSpeaker(
            ws=None,
            http=state["http"],
            state=CallState(),
            tts_url=settings.tts_url,
            tts_stream_url=settings.tts_stream_url,
            streaming=False,
        )
        pcm, rate = await probe.synthesize(settings.greeting)
        state["greeting_frames"] = audio.pcm_to_telephony_frames(pcm, rate)
        log.info("salutation cachée: %d frames", len(state["greeting_frames"]))
    except Exception as e:  # noqa: BLE001
        state["greeting_frames"] = None
        log.warning("cache salutation échoué: %s", e)
    try:
        t = time.time()
        await llm.complete([{"role": "user", "content": "Bonjour"}])
        log.info("LLM préchauffé en %.1fs", time.time() - t)
    except Exception as e:  # noqa: BLE001
        log.warning("préchauffage LLM échoué: %s", e)
    log.info("orchestrateur prêt")


async def close_services(state: dict) -> None:
    """Ferme ce qui doit l'être à l'arrêt ; ne plante jamais."""
    client = state.pop("http", None)
    if client is not None:
        try:
            await client.aclose()
        except Exception as e:  # noqa: BLE001 — l'arrêt ne doit jamais planter
            log.warning("fermeture du client HTTP échouée: %s", e)


def make_lifespan(
    settings: Settings, state: dict
) -> Callable[[FastAPI], AbstractAsyncContextManager[None]]:
    """Fabrique le gestionnaire de cycle de vie à donner à `FastAPI(lifespan=...)`."""

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncGenerator[None]:
        await build_services(settings, state)  # démarrage
        yield  # le serveur tourne
        await close_services(state)  # arrêt

    return lifespan
