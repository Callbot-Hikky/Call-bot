"""Tests de FasterWhisperSTTAdapter sans la librairie faster-whisper installée.

On mocke l'import via `sys.modules['faster_whisper']` ; ça permet de valider
le câblage de l'adapter sans dépendance lourde.
"""

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from hikky.adapters.voice.faster_whisper_stt import FasterWhisperSTTAdapter


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


async def _audio(chunks: list[bytes]):
    for c in chunks:
        yield c


async def test_transcribe_loads_model_with_constructor_args(fake_whisper_module):
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


async def test_transcribe_passes_audio_buffer_and_french_language(fake_whisper_module):
    _, model_instance = fake_whisper_module
    adapter = FasterWhisperSTTAdapter()
    stream = await adapter.transcribe(_audio([b"\x10\x20", b"\x30\x40"]))
    [x async for x in stream]
    call = model_instance.transcribe.call_args
    assert call.args[0] == b"\x10\x20\x30\x40"
    assert call.kwargs["language"] == "fr"


async def test_transcribe_yields_segment_texts_in_order(fake_whisper_module):
    adapter = FasterWhisperSTTAdapter()
    stream = await adapter.transcribe(_audio([b"\x00"]))
    texts = [text async for text in stream]
    assert texts == ["bonjour ", "je voudrais réserver"]


async def test_model_is_loaded_once_across_calls(fake_whisper_module):
    whisper_model_cls, _ = fake_whisper_module
    adapter = FasterWhisperSTTAdapter()
    stream1 = await adapter.transcribe(_audio([b"\x00"]))
    [x async for x in stream1]
    stream2 = await adapter.transcribe(_audio([b"\x00"]))
    [x async for x in stream2]
    assert whisper_model_cls.call_count == 1
