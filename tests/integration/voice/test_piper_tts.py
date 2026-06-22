import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from hikky.adapters.voice.piper_tts import PiperTTSAdapter


@pytest.fixture
def fake_piper_module(monkeypatch):
    voice_instance = MagicMock(name="PiperVoiceInstance")
    voice_instance.synthesize_stream_raw.return_value = iter(
        [b"\x00\x01", b"\x02\x03", b"\x04\x05"]
    )
    piper_voice_cls = MagicMock(name="PiperVoice")
    piper_voice_cls.load = MagicMock(return_value=voice_instance)
    fake_module = SimpleNamespace(PiperVoice=piper_voice_cls)
    monkeypatch.setitem(sys.modules, "piper", fake_module)
    return piper_voice_cls, voice_instance


async def test_synthesize_loads_voice_from_model_path(fake_piper_module):
    piper_voice_cls, _ = fake_piper_module
    adapter = PiperTTSAdapter(model_path="/voices/fr_FR-siwis-medium.onnx")
    stream = await adapter.synthesize("bonjour")
    [c async for c in stream]
    piper_voice_cls.load.assert_called_once_with("/voices/fr_FR-siwis-medium.onnx")


async def test_synthesize_passes_text_to_piper(fake_piper_module):
    _, voice = fake_piper_module
    adapter = PiperTTSAdapter(model_path="/v.onnx")
    stream = await adapter.synthesize("merci pour votre réservation")
    [c async for c in stream]
    voice.synthesize_stream_raw.assert_called_once_with("merci pour votre réservation")


async def test_synthesize_yields_chunks_in_order(fake_piper_module):
    adapter = PiperTTSAdapter(model_path="/v.onnx")
    stream = await adapter.synthesize("salut")
    chunks = [c async for c in stream]
    assert chunks == [b"\x00\x01", b"\x02\x03", b"\x04\x05"]


async def test_voice_loaded_once_across_calls(fake_piper_module):
    piper_voice_cls, voice = fake_piper_module
    # Renouveler le générateur entre appels
    voice.synthesize_stream_raw.side_effect = lambda _t: iter([b"\xaa"])
    adapter = PiperTTSAdapter(model_path="/v.onnx")
    await adapter.synthesize("un")
    await adapter.synthesize("deux")
    assert piper_voice_cls.load.call_count == 1


async def test_synthesize_yields_nothing_when_piper_returns_empty(fake_piper_module):
    _, voice = fake_piper_module
    voice.synthesize_stream_raw.return_value = iter([])
    adapter = PiperTTSAdapter(model_path="/v.onnx")
    stream = await adapter.synthesize("")
    chunks = [c async for c in stream]
    assert chunks == []
