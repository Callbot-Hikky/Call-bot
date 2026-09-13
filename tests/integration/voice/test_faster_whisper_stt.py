"""Tests de FasterWhisperSTTAdapter sans la librairie faster-whisper installée.

On mocke l'import via `sys.modules['faster_whisper']` ; ça permet de valider
le câblage de l'adapter sans dépendance lourde.
"""

import asyncio
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from hikky.adapters.voice.faster_whisper_stt import FasterWhisperSTTAdapter
from hikky.exceptions import STTTimeout


@pytest.fixture
def fake_whisper_module(monkeypatch):
    """Injecte un faux module `faster_whisper` avec un `WhisperModel` mock."""
    model_instance = MagicMock(name="WhisperModelInstance")
    # Par défaut : 2 segments
    model_instance.transcribe.return_value = (
        [SimpleNamespace(text="bonjour "), SimpleNamespace(text="je voudrais réserver")],
        SimpleNamespace(language="fr"),
    )
    whisper_model_cls = MagicMock(name="WhisperModel", return_value=model_instance)
    fake_module = SimpleNamespace(WhisperModel=whisper_model_cls)
    monkeypatch.setitem(sys.modules, "faster_whisper", fake_module)
    return whisper_model_cls, model_instance


@pytest.fixture
def passthrough_audio_conversion(monkeypatch):
    """Évite la dépendance à numpy dans les tests : la conversion devient
    une identité (les bytes sont passés directement à `model.transcribe`)."""
    monkeypatch.setattr(
        FasterWhisperSTTAdapter, "_to_whisper_audio", lambda self, b: b
    )


async def _audio(chunks: list[bytes]):
    for c in chunks:
        yield c


async def test_transcribe_loads_model_with_constructor_args(
    fake_whisper_module, passthrough_audio_conversion
):
    whisper_model_cls, model_instance = fake_whisper_module
    adapter = FasterWhisperSTTAdapter(
        model_name="distil-large-v3", device="cuda", compute_type="int8"
    )
    stream = await adapter.transcribe(_audio([b"\x00\x01"]))
    # Itérer pour déclencher la transcription
    [x async for x in stream]
    whisper_model_cls.assert_called_once_with(
        "distil-large-v3", device="cuda", compute_type="int8"
    )


async def test_transcribe_passes_audio_buffer_and_french_language(
    fake_whisper_module, passthrough_audio_conversion
):
    _, model_instance = fake_whisper_module
    adapter = FasterWhisperSTTAdapter()
    stream = await adapter.transcribe(_audio([b"\x10\x20", b"\x30\x40"]))
    [x async for x in stream]
    call = model_instance.transcribe.call_args
    assert call.args[0] == b"\x10\x20\x30\x40"
    assert call.kwargs["language"] == "fr"


async def test_transcribe_yields_segment_texts_in_order(
    fake_whisper_module, passthrough_audio_conversion
):
    adapter = FasterWhisperSTTAdapter()
    stream = await adapter.transcribe(_audio([b"\x00"]))
    texts = [text async for text in stream]
    assert texts == ["bonjour ", "je voudrais réserver"]


async def test_model_is_loaded_once_across_calls(
    fake_whisper_module, passthrough_audio_conversion
):
    whisper_model_cls, _ = fake_whisper_module
    adapter = FasterWhisperSTTAdapter()
    stream1 = await adapter.transcribe(_audio([b"\x00"]))
    [x async for x in stream1]
    stream2 = await adapter.transcribe(_audio([b"\x00"]))
    [x async for x in stream2]
    assert whisper_model_cls.call_count == 1


async def test_transcribe_raises_stt_timeout_when_model_too_slow(
    fake_whisper_module, passthrough_audio_conversion
):
    import time

    _, model_instance = fake_whisper_module

    def _slow_transcribe(*_args, **_kwargs):
        time.sleep(0.5)
        return ([], SimpleNamespace(language="fr"))

    model_instance.transcribe.side_effect = _slow_transcribe

    adapter = FasterWhisperSTTAdapter(timeout_seconds=0.1)
    with pytest.raises(STTTimeout):
        stream = await adapter.transcribe(_audio([b"\x00"]))
        [text async for text in stream]


async def test_transcribe_does_not_block_event_loop(
    fake_whisper_module, passthrough_audio_conversion
):
    """Le modèle Whisper est synchrone — on doit l'invoquer dans un thread
    pour ne pas bloquer l'event loop. Concrètement : pendant qu'on
    transcribe, une autre coroutine doit pouvoir avancer."""
    import time

    _, model_instance = fake_whisper_module

    def _blocking_transcribe(*_args, **_kwargs):
        time.sleep(0.1)
        return ([SimpleNamespace(text="ok")], SimpleNamespace(language="fr"))

    model_instance.transcribe.side_effect = _blocking_transcribe

    adapter = FasterWhisperSTTAdapter()
    progress = []

    async def tick():
        for _ in range(5):
            await asyncio.sleep(0.02)
            progress.append("tick")

    async def transcribe_one():
        stream = await adapter.transcribe(_audio([b"\x00"]))
        [x async for x in stream]

    await asyncio.gather(tick(), transcribe_one())
    # Si transcribe bloquait l'event loop, `tick` ne serait jamais appelé
    # plus de 1 fois. Avec to_thread, il doit avancer en parallèle.
    assert len(progress) >= 3


# ── Contexte lexical ────────────────────────────────────────────────────
#
# Appel simule : « Au nom de Rian » transcrit « Au nom de rien ». Le
# 8 kHz telephonique abime les fins de mots, et Whisper choisit le mot
# courant plutot que le prenom. Un prompt de contexte oriente le decodage
# vers le vocabulaire du domaine.

async def test_domain_prompt_is_passed_to_whisper(fake_whisper_module):
    _, model = fake_whisper_module
    adapter = FasterWhisperSTTAdapter()

    async def _chunks():
        yield b"\x00" * 320

    stream = await adapter.transcribe(_chunks())
    [t async for t in stream]

    kwargs = model.transcribe.call_args.kwargs
    prompt = kwargs.get("initial_prompt") or ""
    assert "réservation" in prompt.lower()
    assert "table" in prompt.lower()


async def test_domain_prompt_can_be_overridden(fake_whisper_module):
    _, model = fake_whisper_module
    adapter = FasterWhisperSTTAdapter(initial_prompt="vocabulaire maison")

    async def _chunks():
        yield b"\x00" * 320

    stream = await adapter.transcribe(_chunks())
    [t async for t in stream]

    assert model.transcribe.call_args.kwargs["initial_prompt"] == "vocabulaire maison"
