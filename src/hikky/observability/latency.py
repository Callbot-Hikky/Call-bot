"""Mesure de latence par étape (STT, LLM, TTS…).

Usage :

    async with measure_latency("stt"):
        text = await stt.transcribe(...)

Émet un log JSON avec `latency_step` et `latency_ms` à la fin de chaque
bloc. Permet d'agréger les durées et de surveiller le respect du budget
de latence §5 du spec.
"""

import logging
import time
from contextlib import asynccontextmanager
from typing import Any

logger = logging.getLogger("hikky.latency")


@asynccontextmanager
async def measure_latency(step: str, **extra: Any):
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        logger.info(
            "latency",
            extra={"latency_step": step, "latency_ms": round(elapsed_ms, 2), **extra},
        )
