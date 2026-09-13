"""Adapter STT basé sur faster-whisper.

Installation : `pip install -e ".[voice]"` (puis voir la doc faster-whisper
pour le téléchargement du modèle). Nécessite aussi `numpy` (tiré comme
dépendance transitive de faster-whisper).

Modèle en français : `large-v3`. Les variantes distil-whisper
(`distil-large-v3`) sont **anglaises uniquement** — leur passer
`language="fr"` produit une traduction, pas une transcription.

Mesuré sur RTX 3090 (float16), phrase de 5,5 s passée par le 8 kHz
téléphonique, modèle déjà chargé :

    large-v3   0,29 s  — "au nom de Duchesne"  (exact)
    medium     0,25 s  — "au nom de Duchesse"  (nom faux)

L'écart de latence est négligeable, celui de précision ne l'est pas :
`medium` écorche les noms propres, ce qui est rédhibitoire pour une
réservation. D'où `large-v3` par défaut.

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

# À chaud la transcription prend ~0,3 s ; le premier appel inclut le
# chargement du modèle (~2,4 s). 3 s était trop juste et faisait lever
# STTTimeout sur le premier tour de parole. On garde une marge large :
# ce timeout sert à détecter un blocage, pas à cadencer le nominal.
DEFAULT_TIMEOUT_SECONDS = 15.0

# Contexte lexical soufflé à Whisper. Le 8 kHz téléphonique abîme les fins
# de mots et le décodeur choisit alors le terme le plus courant : « Au nom
# de Rian » est devenu « Au nom de rien », et le prénom a été perdu.
# Orienter le décodage vers le vocabulaire du domaine réduit nettement ces
# confusions, surtout sur les noms propres et les horaires.
DOMAIN_PROMPT = (
    "Réservation de table au restaurant. Le client indique le jour, "
    "l'heure, le nombre de personnes et son nom de famille. Vocabulaire : "
    "réserver, réservation, table, couverts, personnes, midi, soir, "
    "demain, ce soir, heures, et demie, au nom de, monsieur, madame."
)


# La precision suit le materiel. `int8` etait servi a un adapter tournant
# sur CUDA : c'est le reglage prevu pour le CPU. Sur un GPU Ampere, les
# Tensor Cores sont concus pour le float16, qui y est plus rapide ET plus
# fidele — double gain sur un callbot, ou chaque mot mal transcrit
# devient une information mal comprise.
PRECISION_GPU = "float16"
PRECISION_CPU = "int8"

# Repli si la VRAM manque : les couches non quantisees restent en FP16.
PRECISION_GPU_ECONOME = "int8_float16"


class FasterWhisperSTTAdapter(SpeechRecognitionPort):
    def __init__(
        self,
        *,
        model_name: str = "large-v3",
        device: str = "cuda",
        compute_type: str | None = None,
        language: str = "fr",
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        initial_prompt: str | None = None,
    ) -> None:
        self._initial_prompt = initial_prompt or DOMAIN_PROMPT
        self._model_name = model_name
        self._device = device
        self._compute_type = compute_type or (
            PRECISION_GPU if device.startswith("cuda") else PRECISION_CPU
        )
        self._language = language
        self._timeout_seconds = timeout_seconds
        self._model: Any | None = None

    @property
    def compute_type(self) -> str:
        return self._compute_type

    @property
    def model_name(self) -> str:
        return self._model_name

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
                    initial_prompt=self._initial_prompt,
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
