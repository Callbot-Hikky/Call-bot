from collections.abc import AsyncIterator

from hikky.ports.speech_synthesis import SpeechSynthesisPort


class FakeSpeechSynthesis(SpeechSynthesisPort):
    def __init__(self) -> None:
        self.spoken: list[str] = []

    async def synthesize(self, text: str) -> AsyncIterator[bytes]:
        self.spoken.append(text)

        async def _stream() -> AsyncIterator[bytes]:
            yield b"\x00" * 16

        return _stream()
