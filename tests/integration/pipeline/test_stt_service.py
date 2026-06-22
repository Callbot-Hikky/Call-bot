from collections.abc import AsyncIterator

from pipecat.frames.frames import TranscriptionFrame

from hikky.pipeline.stt_service import HikkySTTService
from hikky.ports.speech_recognition import SpeechRecognitionPort


class _FakeSTTAdapter(SpeechRecognitionPort):
    def __init__(self, segments: list[str]) -> None:
        self._segments = segments
        self.calls: list[bytes] = []

    async def transcribe(self, audio_chunks: AsyncIterator[bytes]) -> AsyncIterator[str]:
        buf = bytearray()
        async for c in audio_chunks:
            buf.extend(c)
        self.calls.append(bytes(buf))
        segments = self._segments

        async def _stream() -> AsyncIterator[str]:
            for s in segments:
                yield s

        return _stream()


async def test_run_stt_yields_transcription_frame_with_concatenated_text():
    adapter = _FakeSTTAdapter(["bonjour ", "je voudrais réserver"])
    service = HikkySTTService(adapter)
    frames = [f async for f in service.run_stt(b"\x00\x01\x02")]
    assert len(frames) == 1
    frame = frames[0]
    assert isinstance(frame, TranscriptionFrame)
    assert frame.text == "bonjour je voudrais réserver"
    assert frame.finalized is True


async def test_run_stt_yields_none_when_no_segments():
    adapter = _FakeSTTAdapter([])
    service = HikkySTTService(adapter)
    frames = [f async for f in service.run_stt(b"\x00")]
    assert frames == [None]


async def test_run_stt_yields_none_when_only_whitespace():
    adapter = _FakeSTTAdapter(["   ", "\n"])
    service = HikkySTTService(adapter)
    frames = [f async for f in service.run_stt(b"\x00")]
    assert frames == [None]


async def test_run_stt_passes_audio_to_adapter():
    adapter = _FakeSTTAdapter(["x"])
    service = HikkySTTService(adapter)
    [_ async for _ in service.run_stt(b"\xaa\xbb")]
    assert adapter.calls == [b"\xaa\xbb"]
