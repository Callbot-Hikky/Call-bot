"""Adapter TTS basé sur Piper.

Installation : `pip install -e ".[voice]"` puis télécharger une voix Piper
française (par exemple `fr_FR-siwis-medium.onnx` depuis le hub Piper).

API Piper utilisée : `PiperVoice.synthesize(text)` renvoie un itérable de
`AudioChunk`, dont `audio_int16_bytes` porte le PCM 16-bit signé et
`sample_rate` le taux réel de la voix (22050 Hz pour siwis-medium).

Le taux n'est pas supposé : il est relevé sur les chunks et exposé via
`sample_rate`, car le rééchantillonnage vers le 8 kHz de la téléphonie en
dépend directement.

La synthèse tourne dans un thread (`asyncio.to_thread`) pour ne pas
bloquer l'event loop.

Lazy import — la suite de tests tourne sans la lib installée.
"""

import asyncio
from collections.abc import AsyncIterator
from typing import Any

from hikky.ports.speech_synthesis import SpeechSynthesisPort


class PiperTTSAdapter(SpeechSynthesisPort):
    def __init__(self, *, model_path: str) -> None:
        self._model_path = model_path
        self._voice: Any | None = None
        self._sample_rate: int | None = None

    @property
    def sample_rate(self) -> int | None:
        """Taux d'échantillonnage relevé lors de la dernière synthèse."""
        return self._sample_rate

    def _load_voice(self) -> Any:
        if self._voice is None:
            from piper import PiperVoice  # lazy

            self._voice = PiperVoice.load(self._model_path)
        return self._voice

    async def synthesize(self, text: str) -> AsyncIterator[bytes]:
        voice = self._load_voice()

        def _collect() -> tuple[list[bytes], int | None]:
            audio: list[bytes] = []
            rate: int | None = None
            for chunk in voice.synthesize(text):
                audio.append(chunk.audio_int16_bytes)
                rate = chunk.sample_rate
            return audio, rate

        chunks, rate = await asyncio.to_thread(_collect)
        if rate is not None:
            self._sample_rate = rate

        async def _stream() -> AsyncIterator[bytes]:
            for chunk in chunks:
                yield chunk

        return _stream()
