"""Wrapper Pipecat TTSService → SpeechSynthesisPort.

Plug n'importe quel adapter qui implémente `SpeechSynthesisPort` dans
une `Pipeline` Pipecat. On délègue tout le travail réel à l'adapter ;
ce wrapper convertit juste les types Pipecat ↔ port.

Émet une `TTSAudioRawFrame` par chunk audio produit par l'adapter.
"""

from collections.abc import AsyncGenerator

from pipecat.frames.frames import Frame, TTSAudioRawFrame
from pipecat.services.tts_service import TTSService

from hikky.observability.latency import measure_latency
from hikky.ports.speech_synthesis import SpeechSynthesisPort

DEFAULT_SAMPLE_RATE = 22050  # Piper voix françaises medium = 22050 Hz


class HikkyTTSService(TTSService):
    def __init__(
        self,
        adapter: SpeechSynthesisPort,
        *,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        num_channels: int = 1,
    ) -> None:
        super().__init__()
        self._adapter = adapter
        self._sample_rate = sample_rate
        self._num_channels = num_channels

    async def run_tts(
        self, text: str, context_id: str
    ) -> AsyncGenerator[Frame | None, None]:
        async with measure_latency("tts", text_length=len(text)):
            stream = await self._adapter.synthesize(text)
            async for chunk in stream:
                yield TTSAudioRawFrame(
                    audio=chunk,
                    sample_rate=self._sample_rate,
                    num_channels=self._num_channels,
                    context_id=context_id,
                )
