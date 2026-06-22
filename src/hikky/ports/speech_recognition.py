from abc import ABC, abstractmethod
from collections.abc import AsyncIterator


class SpeechRecognitionPort(ABC):
    @abstractmethod
    async def transcribe(self, audio_chunks: AsyncIterator[bytes]) -> AsyncIterator[str]:
        """Reçoit des chunks audio PCM, livre des morceaux de texte transcrit."""
