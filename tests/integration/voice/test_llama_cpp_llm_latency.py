"""Vérifie que LlamaCppLLMAdapter émet une métrique de latence labelée
`llm` à chaque appel `complete(...)`. C'est ce qui rend la latence LLM
isolable dans les logs structurés (cf. §5 du spec : décomposition du
budget de latence par étape)."""

import io
import json
import logging
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from hikky.adapters.voice.llama_cpp_llm import LlamaCppLLMAdapter
from hikky.observability.logging import JsonFormatter


@pytest.fixture
def fake_llama_module(monkeypatch):
    instance = MagicMock(name="LlamaInstance")
    instance.create_chat_completion.return_value = {
        "choices": [{"message": {"content": "réponse"}}]
    }
    llama_cls = MagicMock(name="Llama", return_value=instance)
    fake_module = SimpleNamespace(Llama=llama_cls)
    monkeypatch.setitem(sys.modules, "llama_cpp", fake_module)
    return llama_cls, instance


@pytest.fixture
def latency_capture():
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(JsonFormatter())
    logger = logging.getLogger("hikky.latency")
    logger.handlers.clear()
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return stream


async def test_complete_emits_latency_log_with_llm_step(
    fake_llama_module, latency_capture
):
    adapter = LlamaCppLLMAdapter(model_path="/m.gguf")
    await adapter.complete([{"role": "user", "content": "salut"}])
    line = latency_capture.getvalue().strip()
    payload = json.loads(line)
    assert payload["latency_step"] == "llm"
    assert "latency_ms" in payload
    assert payload["message_count"] == 1
