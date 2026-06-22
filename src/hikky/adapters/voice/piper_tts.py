"""Adapter TTS basé sur Piper.

Installation : `pip install -e ".[voice]"` puis télécharger une voix Piper
française (par exemple `fr_FR-siwis-medium.onnx` depuis le hub Piper).

Piper synthétise en PCM 16-bit signé à 22050 Hz par défaut (selon la voix).
On livre les chunks au fur et à mesure qu'ils sortent du modèle, pour le
streaming.

Lazy import — la suite de tests tourne sans la lib installée.
"""

from collections.abc import AsyncIterator
from typing import Any

from hikky.ports.speech_synthesis import SpeechSynthesisPort


class PiperTTSAdapter(SpeechSynthesisPort):
    def __init__(self, *, model_path: str) -> None:
        self._model_path = model_path
        self._voice: Any | None = None

    def _load_voice(self) -> Any:
        if self._voice is None:
            from piper import PiperVoice  # lazy

            self._voice = PiperVoice.load(self._model_path)
        return self._voice

    async def synthesize(self, text: str) -> AsyncIterator[bytes]:
        voice = self._load_voice()
        # Piper expose `synthesize_stream_raw(text)` qui yield des bytes PCM.
        chunks = list(voice.synthesize_stream_raw(text))

        async def _stream() -> AsyncIterator[bytes]:
            for chunk in chunks:
                yield chunk

        return _stream()
