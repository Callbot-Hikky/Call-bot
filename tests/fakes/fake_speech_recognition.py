from collections.abc import AsyncIterator

from hikky.ports.speech_recognition import SpeechRecognitionPort


class FakeSpeechRecognition(SpeechRecognitionPort):
    """Pour les tests cœur, on injecte du texte directement, pas via la transcription.

    Le fake est fourni pour complétude et tests d'isolation.
    """

    def __init__(self, scripted_chunks: list[str] | None = None) -> None:
        self._scripted = list(scripted_chunks or [])

    async def transcribe(self, audio_chunks: AsyncIterator[bytes]) -> AsyncIterator[str]:
        async def _stream() -> AsyncIterator[str]:
            for chunk in self._scripted:
                yield chunk

        return _stream()
