"""Adapter TTS basé sur XTTS-v2 (Coqui).

Retenu pour la qualité de prosodie, très supérieure à Piper sur du
français. En contrepartie il est ~5x plus lent : on utilise donc
`inference_stream`, qui livre l'audio au fil de la génération au lieu
d'attendre la phrase entière. L'appelant entend le début de la réponse
après quelques centaines de millisecondes plutôt qu'après 1,5 s.

⚠️ Licence : XTTS-v2 est publié sous Coqui Public Model License, qui
**interdit l'usage commercial**. Utilisable pour un projet académique ;
à remplacer avant toute exploitation commerciale de Hikky.

⚠️ Dépendances : `coqui-tts` exige `transformers>=4.57,<5` et
`torch>=2.5`. Coqui n'ayant plus de mainteneur, ces bornes sont figées
dans `pyproject.toml` — ne pas les élargir sans revérifier.

Le modèle sort du 24 kHz : `sample_rate` l'expose pour que le
rééchantillonnage vers le 8 kHz téléphonique parte du bon taux.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
from collections.abc import AsyncIterator
from typing import Any

from hikky.ports.speech_synthesis import SpeechSynthesisPort

logger = logging.getLogger("hikky.xtts")

XTTS_SAMPLE_RATE_HZ = 24000
XTTS_MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"

_INT16_MAX = 32767


def _load_model(model_name: str = XTTS_MODEL_NAME) -> Any:
    """Charge XTTS sur GPU. Isolé pour être substituable en test."""
    os.environ.setdefault("COQUI_TOS_AGREED", "1")

    import torch

    # Sur certains hôtes GPU, le cuDNN 9.x embarqué par torch échoue à
    # s'initialiser (CUDNN_STATUS_NOT_INITIALIZED dès la première conv1d du
    # vocodeur HiFi-GAN), alors que le GPU est libre. On désactive cuDNN : la
    # conv passe par le noyau CUDA natif, produit exactement le même audio, et
    # le coût est négligeable sur les petites convolutions du décodeur.
    torch.backends.cudnn.enabled = False

    from TTS.tts.configs.xtts_config import XttsConfig
    from TTS.tts.models.xtts import Xtts
    from TTS.utils.manage import ModelManager

    path, _, _ = ModelManager().download_model(model_name)
    config = XttsConfig()
    config.load_json(f"{path}/config.json")
    model = Xtts.init_from_config(config)
    model.load_checkpoint(config, checkpoint_dir=path, eval=True)
    return model.cuda()


class XttsTTSAdapter(SpeechSynthesisPort):
    def __init__(
        self,
        *,
        speaker: str = "Lilya Stainthorpe",
        language: str = "fr",
        model_name: str = XTTS_MODEL_NAME,
    ) -> None:
        self._speaker = speaker
        self._language = language
        self._model_name = model_name
        self._model: Any | None = None

    @property
    def sample_rate(self) -> int:
        return XTTS_SAMPLE_RATE_HZ

    def _get_model(self) -> Any:
        if self._model is None:
            self._model = _load_model(self._model_name)
        return self._model

    def _latents(self, model: Any) -> tuple[Any, Any]:
        speakers = model.speaker_manager.speakers
        entry = speakers.get(self._speaker)
        if entry is None:
            raise ValueError(
                f"locuteur inconnu: {self._speaker!r}. "
                f"Disponibles: {', '.join(sorted(speakers))}"
            )
        return entry["gpt_cond_latent"], entry["speaker_embedding"]

    async def synthesize(self, text: str) -> AsyncIterator[bytes]:
        model = self._get_model()
        gpt_cond_latent, speaker_embedding = self._latents(model)

        # La génération est synchrone et bloquante : elle tourne dans un
        # thread qui pousse chaque morceau dans une file au fur et à
        # mesure. Accumuler puis livrer ferait perdre tout l'intérêt du
        # flux — 1,6 s avant le premier son au lieu de 0,33 s.
        queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_running_loop()
        done = object()

        def _produce() -> None:
            try:
                for chunk in model.inference_stream(
                    text, self._language, gpt_cond_latent, speaker_embedding
                ):
                    loop.call_soon_threadsafe(queue.put_nowait, _to_pcm16(chunk))
            except Exception as exc:  # noqa: BLE001 — relayé au consommateur
                loop.call_soon_threadsafe(queue.put_nowait, exc)
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, done)

        producer = asyncio.create_task(asyncio.to_thread(_produce))

        async def _stream() -> AsyncIterator[bytes]:
            try:
                while True:
                    item = await queue.get()
                    if item is done:
                        return
                    if isinstance(item, Exception):
                        raise item
                    yield item
            finally:
                producer.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await producer

        return _stream()


def _to_pcm16(chunk: Any) -> bytes:
    import struct

    values = chunk.tolist() if hasattr(chunk, "tolist") else list(chunk)
    out = []
    for v in values:
        scaled = int(v * _INT16_MAX)
        out.append(max(-_INT16_MAX, min(_INT16_MAX, scaled)))
    return struct.pack(f"<{len(out)}h", *out)
