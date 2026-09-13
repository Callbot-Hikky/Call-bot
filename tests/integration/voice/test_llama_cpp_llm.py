import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from hikky.adapters.voice.llama_cpp_llm import LlamaCppLLMAdapter
from hikky.exceptions import LLMOverloaded


@pytest.fixture
def fake_llama_module(monkeypatch):
    instance = MagicMock(name="LlamaInstance")
    instance.create_chat_completion.return_value = {
        "choices": [{"message": {"content": "Bonjour, je peux vous aider à réserver."}}]
    }
    llama_cls = MagicMock(name="Llama", return_value=instance)
    fake_module = SimpleNamespace(Llama=llama_cls)
    monkeypatch.setitem(sys.modules, "llama_cpp", fake_module)
    return llama_cls, instance


async def test_complete_loads_model_with_constructor_args(fake_llama_module):
    llama_cls, _ = fake_llama_module
    adapter = LlamaCppLLMAdapter(model_path="/models/mistral.gguf", n_ctx=8192, n_gpu_layers=99)
    await adapter.complete([{"role": "user", "content": "salut"}])
    llama_cls.assert_called_once_with(
        model_path="/models/mistral.gguf",
        n_ctx=8192,
        n_gpu_layers=99,
        flash_attn=True,
        verbose=False,
    )


async def test_complete_passes_messages_and_returns_content(fake_llama_module):
    _, instance = fake_llama_module
    adapter = LlamaCppLLMAdapter(model_path="/m.gguf")
    messages = [
        {"role": "system", "content": "Tu es l'assistant d'un restaurant."},
        {"role": "user", "content": "Je veux réserver"},
    ]
    reply = await adapter.complete(messages)
    assert reply == "Bonjour, je peux vous aider à réserver."
    call_kwargs = instance.create_chat_completion.call_args.kwargs
    assert call_kwargs["messages"] == messages


async def test_complete_raises_llm_overloaded_on_library_exception(fake_llama_module):
    _, instance = fake_llama_module
    instance.create_chat_completion.side_effect = RuntimeError("boom")
    adapter = LlamaCppLLMAdapter(model_path="/m.gguf")
    with pytest.raises(LLMOverloaded):
        await adapter.complete([{"role": "user", "content": "x"}])


async def test_complete_lets_keyboard_interrupt_propagate(fake_llama_module):
    """L'adapter capture les erreurs de la librairie mais doit laisser
    passer les exceptions système (KeyboardInterrupt, SystemExit)."""
    _, instance = fake_llama_module
    instance.create_chat_completion.side_effect = KeyboardInterrupt()
    adapter = LlamaCppLLMAdapter(model_path="/m.gguf")
    with pytest.raises(KeyboardInterrupt):
        await adapter.complete([{"role": "user", "content": "x"}])


async def test_complete_does_not_block_event_loop(fake_llama_module):
    import asyncio
    import time

    _, instance = fake_llama_module

    def _slow(*_a, **_kw):
        time.sleep(0.1)
        return {"choices": [{"message": {"content": "ok"}}]}

    instance.create_chat_completion.side_effect = _slow

    adapter = LlamaCppLLMAdapter(model_path="/m.gguf")
    progress = []

    async def tick():
        for _ in range(5):
            await asyncio.sleep(0.02)
            progress.append("tick")

    async def complete_once():
        await adapter.complete([{"role": "user", "content": "x"}])

    await asyncio.gather(tick(), complete_once())
    assert len(progress) >= 3


async def test_complete_raises_llm_overloaded_on_malformed_response(fake_llama_module):
    _, instance = fake_llama_module
    instance.create_chat_completion.return_value = {"unexpected": "shape"}
    adapter = LlamaCppLLMAdapter(model_path="/m.gguf")
    with pytest.raises(LLMOverloaded):
        await adapter.complete([{"role": "user", "content": "x"}])


async def test_model_loaded_once_across_calls(fake_llama_module):
    llama_cls, _ = fake_llama_module
    adapter = LlamaCppLLMAdapter(model_path="/m.gguf")
    await adapter.complete([{"role": "user", "content": "a"}])
    await adapter.complete([{"role": "user", "content": "b"}])
    assert llama_cls.call_count == 1


# ── Acces serialise au modele ───────────────────────────────────────────
#
# Crash en production : llama.cpp n'est pas thread-safe. L'extracteur et
# le conversationnel partagent la meme instance ; deux generations qui se
# chevauchent corrompent l'etat interne et tuent le processus :
#   IndexError: index 917 is out of bounds for axis 0 with size 5

async def test_concurrent_completions_are_serialised(fake_llama_module):
    import asyncio

    _, model = fake_llama_module
    en_cours = []
    chevauchements = []

    def _lent(*args, **kwargs):
        import time

        en_cours.append(1)
        if len(en_cours) > 1:
            chevauchements.append(1)
        time.sleep(0.05)
        en_cours.pop()
        return {"choices": [{"message": {"content": "ok"}}]}

    model.create_chat_completion.side_effect = _lent
    adapter = LlamaCppLLMAdapter(model_path="/m.gguf")

    await asyncio.gather(
        *(adapter.complete([{"role": "user", "content": f"m{i}"}]) for i in range(4))
    )
    assert not chevauchements, "generations concurrentes : llama.cpp va crasher"
