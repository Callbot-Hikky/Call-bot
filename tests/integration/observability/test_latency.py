import io
import json
import logging

from hikky.observability.latency import measure_latency
from hikky.observability.logging import JsonFormatter


def _attach_capture() -> io.StringIO:
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(JsonFormatter())
    logger = logging.getLogger("hikky.latency")
    logger.handlers.clear()
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return stream


async def test_measure_latency_emits_log_with_step_and_ms():
    stream = _attach_capture()
    async with measure_latency("stt"):
        pass
    payload = json.loads(stream.getvalue().strip())
    assert payload["latency_step"] == "stt"
    assert "latency_ms" in payload
    assert payload["latency_ms"] >= 0.0


async def test_measure_latency_records_elapsed_time():
    import asyncio

    stream = _attach_capture()
    async with measure_latency("slow"):
        await asyncio.sleep(0.02)
    payload = json.loads(stream.getvalue().strip())
    assert payload["latency_ms"] >= 15.0


async def test_measure_latency_passes_extra_fields():
    stream = _attach_capture()
    async with measure_latency("llm", model="mistral-7b"):
        pass
    payload = json.loads(stream.getvalue().strip())
    assert payload["model"] == "mistral-7b"


async def test_measure_latency_emits_even_on_exception():
    stream = _attach_capture()
    with __import__("contextlib").suppress(ValueError):
        async with measure_latency("crashing"):
            raise ValueError("boom")
    payload = json.loads(stream.getvalue().strip())
    assert payload["latency_step"] == "crashing"
