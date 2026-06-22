"""Adapter STT basé sur faster-whisper.

Installation : `pip install -e ".[voice]"` (puis voir la doc faster-whisper
pour le téléchargement du modèle). Nécessite aussi `numpy` (tiré comme
dépendance transitive de faster-whisper).

Modèle recommandé en français : `distil-large-v3` (rapide) ou `medium`
(meilleure qualité). Voir `model_name` ci-dessous.

Format audio attendu : **PCM 16-bit signé little-endian, 16 kHz, mono**.
Le transcodage depuis le format Twilio (μ-law 8 kHz) doit être fait en
amont — Pipecat le fera nativement quand on l'intégrera en plan E.

Imports `faster_whisper` et `numpy` paresseux — la suite de tests tourne
sans ces librairies installées.
"""

import asyncio
from collections.abc import AsyncIterator
from typing import Any

from hikky.exceptions import STTTimeout
from hikky.ports.speech_recognition import SpeechRecognitionPort

DEFAULT_TIMEOUT_SECONDS = 3.0


class FasterWhisperSTTAdapter(SpeechRecognitionPort):
    def __init__(
        self,
        *,
        model_name: str = "distil-large-v3",
        device: str = "cuda",
        compute_type: str = "int8",
        language: str = "fr",
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        self._model_name = model_name
        self._device = device
        self._compute_type = compute_type
        self._language = language
        self._timeout_seconds = timeout_seconds
        self._model: Any | None = None

    def _load_model(self) -> Any:
        if self._model is None:
            from faster_whisper import WhisperModel  # lazy

            self._model = WhisperModel(
                self._model_name,
                device=self._device,
                compute_type=self._compute_type,
            )
        return self._model

    def _to_whisper_audio(self, pcm16_bytes: bytes) -> Any:
        """Convertit du PCM16 16 kHz en numpy float32 [-1, 1] (format Whisper)."""
        import numpy as np  # lazy

        return np.frombuffer(pcm16_bytes, dtype=np.int16).astype(np.float32) / 32768.0

    async def transcribe(self, audio_chunks: AsyncIterator[bytes]) -> AsyncIterator[str]:
        model = self._load_model()
        buffer = bytearray()
        async for chunk in audio_chunks:
            buffer.extend(chunk)
        audio = self._to_whisper_audio(bytes(buffer))

        try:
            segments, _info = await asyncio.wait_for(
                asyncio.to_thread(
                    model.transcribe,
                    audio,
                    language=self._language,
                    beam_size=1,
                ),
                timeout=self._timeout_seconds,
            )
        except TimeoutError as exc:
            raise STTTimeout(
                f"faster-whisper exceeded {self._timeout_seconds}s"
            ) from exc

        async def _stream() -> AsyncIterator[str]:
            for segment in segments:
                yield segment.text

        return _stream()
