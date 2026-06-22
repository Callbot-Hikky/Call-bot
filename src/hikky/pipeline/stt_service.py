"""Wrapper Pipecat STTService → SpeechRecognitionPort.

Plug n'importe quel adapter qui implémente `SpeechRecognitionPort` dans
une `Pipeline` Pipecat. On délègue tout le travail réel à l'adapter sous-
jacent ; ce wrapper convertit juste les types Pipecat ↔ port.

Émet une `TranscriptionFrame` finalisée par segment transcrit.
"""

from collections.abc import AsyncGenerator
from datetime import UTC, datetime

from pipecat.frames.frames import Frame, TranscriptionFrame
from pipecat.services.stt_service import STTService

from hikky.observability.latency import measure_latency
from hikky.ports.speech_recognition import SpeechRecognitionPort


class HikkySTTService(STTService):
    def __init__(self, adapter: SpeechRecognitionPort, *, user_id: str = "caller") -> None:
        super().__init__()
        self._adapter = adapter
        self._user_id = user_id

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame | None, None]:
        async def _one_chunk():
            yield audio

        async with measure_latency("stt"):
            stream = await self._adapter.transcribe(_one_chunk())
            texts: list[str] = [t async for t in stream]

        if not texts:
            yield None
            return

        full = "".join(texts).strip()
        if not full:
            yield None
            return

        yield TranscriptionFrame(
            text=full,
            user_id=self._user_id,
            timestamp=datetime.now(UTC).isoformat(),
            finalized=True,
        )
