from abc import ABC, abstractmethod
from collections.abc import AsyncIterator


class SpeechSynthesisPort(ABC):
    @abstractmethod
    async def synthesize(self, text: str) -> AsyncIterator[bytes]:
        """Reçoit du texte, livre des chunks audio PCM en streaming."""
