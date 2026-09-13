import sys
from types import SimpleNamespace

import pytest

from hikky.adapters.voice.piper_tts import PiperTTSAdapter


class FakeAudioChunk:
    """Reproduit la forme réelle de `piper.voice.AudioChunk`.

    Classe explicite et non MagicMock : un mock auto-génère n'importe quel
    attribut, ce qui laissait passer des tests verts alors que l'API réelle
    de Piper avait changé.
    """

    def __init__(self, audio: bytes, sample_rate: int = 22050) -> None:
        self.audio_int16_bytes = audio
        self.sample_rate = sample_rate
        self.sample_width = 2
        self.sample_channels = 1


class FakeVoice:
    def __init__(self, chunks: list[bytes] | None = None, sample_rate: int = 22050) -> None:
        self._chunks = [b"\x00\x01", b"\x02\x03", b"\x04\x05"] if chunks is None else chunks
        self._sample_rate = sample_rate
        self.calls: list[str] = []
        self.on_synthesize = None

    def synthesize(self, text: str):
        self.calls.append(text)
        if self.on_synthesize is not None:
            self.on_synthesize(text)
        return iter(
            FakeAudioChunk(c, sample_rate=self._sample_rate) for c in self._chunks
        )


class FakeVoiceClass:
    def __init__(self, voice: FakeVoice) -> None:
        self._voice = voice
        self.load_calls: list[str] = []

    def load(self, model_path: str) -> FakeVoice:
        self.load_calls.append(model_path)
        return self._voice


@pytest.fixture
def fake_piper(monkeypatch):
    voice = FakeVoice()
    cls = FakeVoiceClass(voice)
    monkeypatch.setitem(sys.modules, "piper", SimpleNamespace(PiperVoice=cls))
    return cls, voice


async def test_synthesize_loads_voice_from_model_path(fake_piper):
    cls, _ = fake_piper
    adapter = PiperTTSAdapter(model_path="/voices/fr_FR-siwis-medium.onnx")
    stream = await adapter.synthesize("bonjour")
    [c async for c in stream]
    assert cls.load_calls == ["/voices/fr_FR-siwis-medium.onnx"]


async def test_synthesize_passes_text_to_piper(fake_piper):
    _, voice = fake_piper
    adapter = PiperTTSAdapter(model_path="/v.onnx")
    stream = await adapter.synthesize("merci pour votre réservation")
    [c async for c in stream]
    assert voice.calls == ["merci pour votre réservation"]


async def test_synthesize_yields_raw_pcm_bytes_in_order(fake_piper):
    adapter = PiperTTSAdapter(model_path="/v.onnx")
    stream = await adapter.synthesize("salut")
    chunks = [c async for c in stream]
    assert chunks == [b"\x00\x01", b"\x02\x03", b"\x04\x05"]


async def test_sample_rate_is_reported_from_piper(fake_piper):
    _, voice = fake_piper
    voice._sample_rate = 16000
    adapter = PiperTTSAdapter(model_path="/v.onnx")
    stream = await adapter.synthesize("salut")
    [c async for c in stream]
    assert adapter.sample_rate == 16000


async def test_sample_rate_is_none_before_first_synthesis(fake_piper):
    adapter = PiperTTSAdapter(model_path="/v.onnx")
    assert adapter.sample_rate is None


async def test_voice_loaded_once_across_calls(fake_piper):
    cls, _ = fake_piper
    adapter = PiperTTSAdapter(model_path="/v.onnx")
    await adapter.synthesize("un")
    await adapter.synthesize("deux")
    assert len(cls.load_calls) == 1


async def test_synthesize_does_not_block_event_loop(fake_piper):
    import asyncio
    import time

    _, voice = fake_piper
    voice.on_synthesize = lambda _t: time.sleep(0.1)

    adapter = PiperTTSAdapter(model_path="/v.onnx")
    progress = []

    async def tick():
        for _ in range(5):
            await asyncio.sleep(0.02)
            progress.append("tick")

    async def synth_once():
        stream = await adapter.synthesize("hello")
        [c async for c in stream]

    await asyncio.gather(tick(), synth_once())
    assert len(progress) >= 3


async def test_synthesize_yields_nothing_when_piper_returns_empty(fake_piper):
    _, voice = fake_piper
    voice._chunks = []
    adapter = PiperTTSAdapter(model_path="/v.onnx")
    stream = await adapter.synthesize("")
    assert [c async for c in stream] == []
