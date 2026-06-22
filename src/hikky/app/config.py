"""Configuration de l'app via variables d'environnement.

À setter dans le `.env` de prod / sur la VM GPU. Tout est optionnel ;
si une valeur manque, `AppDependencies` n'est pas construite et l'app
refuse poliment les connexions WS (avec un log d'avertissement).
"""

from __future__ import annotations

import os
from dataclasses import dataclass


class MissingConfig(Exception):
    """Une variable d'environnement requise manque."""


@dataclass(frozen=True, slots=True)
class AppConfig:
    back_base_url: str
    back_api_key: str
    whisper_model: str
    whisper_device: str
    whisper_compute_type: str
    llama_model_path: str
    llama_n_ctx: int
    llama_n_gpu_layers: int
    piper_model_path: str
    tts_sample_rate: int


def _req(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise MissingConfig(name)
    return value


def _opt(name: str, default: str) -> str:
    return os.environ.get(name) or default


def load_from_env() -> AppConfig:
    """Charge la config depuis les variables d'env. Lève `MissingConfig`
    pour la première variable requise absente."""

    return AppConfig(
        back_base_url=_req("HIKKY_BACK_BASE_URL"),
        back_api_key=_req("HIKKY_BACK_API_KEY"),
        whisper_model=_opt("HIKKY_WHISPER_MODEL", "distil-large-v3"),
        whisper_device=_opt("HIKKY_WHISPER_DEVICE", "cuda"),
        whisper_compute_type=_opt("HIKKY_WHISPER_COMPUTE_TYPE", "int8"),
        llama_model_path=_req("HIKKY_LLAMA_MODEL_PATH"),
        llama_n_ctx=int(_opt("HIKKY_LLAMA_N_CTX", "4096")),
        llama_n_gpu_layers=int(_opt("HIKKY_LLAMA_N_GPU_LAYERS", "-1")),
        piper_model_path=_req("HIKKY_PIPER_MODEL_PATH"),
        tts_sample_rate=int(_opt("HIKKY_TTS_SAMPLE_RATE", "22050")),
    )
