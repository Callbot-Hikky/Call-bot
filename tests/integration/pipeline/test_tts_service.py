from collections.abc import AsyncIterator

from pipecat.frames.frames import TTSAudioRawFrame

from hikky.pipeline.tts_service import HikkyTTSService
from hikky.ports.speech_synthesis import SpeechSynthesisPort


class _FakeTTSAdapter(SpeechSynthesisPort):
    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = chunks
        self.texts: list[str] = []

    async def synthesize(self, text: str) -> AsyncIterator[bytes]:
        self.texts.append(text)
        my_chunks = self._chunks

        async def _stream() -> AsyncIterator[bytes]:
            for c in my_chunks:
                yield c

        return _stream()


async def test_run_tts_yields_audio_frames_with_pcm_chunks():
    adapter = _FakeTTSAdapter([b"\x00\x01", b"\x02\x03"])
    service = HikkyTTSService(adapter, sample_rate=22050)
    frames = [f async for f in service.run_tts("bonjour", context_id="ctx-1")]
    assert len(frames) == 2
    for f in frames:
        assert isinstance(f, TTSAudioRawFrame)
        assert f.sample_rate == 22050
        assert f.num_channels == 1
        assert f.context_id == "ctx-1"
    assert frames[0].audio == b"\x00\x01"
    assert frames[1].audio == b"\x02\x03"


async def test_run_tts_passes_text_to_adapter():
    adapter = _FakeTTSAdapter([b"\x00"])
    service = HikkyTTSService(adapter)
    [_ async for _ in service.run_tts("merci pour votre réservation", context_id="x")]
    assert adapter.texts == ["merci pour votre réservation"]


async def test_run_tts_yields_nothing_when_adapter_emits_no_chunks():
    adapter = _FakeTTSAdapter([])
    service = HikkyTTSService(adapter)
    frames = [f async for f in service.run_tts("rien", context_id="x")]
    assert frames == []
