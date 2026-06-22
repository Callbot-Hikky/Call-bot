"""Adapter STT basé sur faster-whisper.

Installation : `pip install -e ".[voice]"` (puis voir la doc faster-whisper
pour le téléchargement du modèle).

Modèle recommandé en français : `distil-large-v3` (rapide) ou `medium`
(meilleure qualité). Voir `model_name` ci-dessous.

L'import de `faster_whisper` est paresseux — fait au premier `transcribe(...)`.
Ça permet à la suite de tests de tourner sans avoir la librairie installée.
"""

from collections.abc import AsyncIterator
from typing import Any

from hikky.ports.speech_recognition import SpeechRecognitionPort


class FasterWhisperSTTAdapter(SpeechRecognitionPort):
    def __init__(
        self,
        *,
        model_name: str = "distil-large-v3",
        device: str = "cuda",
        compute_type: str = "int8",
        language: str = "fr",
    ) -> None:
        self._model_name = model_name
        self._device = device
        self._compute_type = compute_type
        self._language = language
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

    async def transcribe(self, audio_chunks: AsyncIterator[bytes]) -> AsyncIterator[str]:
        model = self._load_model()
        buffer = bytearray()
        async for chunk in audio_chunks:
            buffer.extend(chunk)

        segments, _info = model.transcribe(
            bytes(buffer),
            language=self._language,
            beam_size=1,
        )

        async def _stream() -> AsyncIterator[str]:
            for segment in segments:
                yield segment.text

        return _stream()
