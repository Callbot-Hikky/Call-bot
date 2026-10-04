"""La configuration du pipeline, lue une fois depuis l'environnement.

Tout ce qui dépend de la machine ou du déploiement est ici, et nulle part ailleurs :
adresses des services voisins, chemins, numéro du restaurant, salutation. Le code
métier reçoit un objet `Settings` et ne lit jamais `os.environ` lui-même.

Variables (toutes préfixées `HIKKY_`, sauf les deux héritées `PUBLIC_HOST` et `TTS_STREAM`) :

    PUBLIC_HOST               hôte public du pod (proxy RunPod)            OBLIGATOIRE
    TTS_STREAM                "1" = voix en flux (défaut : synthèse complète)
    HIKKY_STT_URL             http://127.0.0.1:8801/transcribe
    HIKKY_TTS_URL             http://127.0.0.1:8802/synthesize
    HIKKY_TTS_STREAM_URL      http://127.0.0.1:8802/synthesize_stream
    HIKKY_BACK_BASE_URL       URL du backend                               OBLIGATOIRE
    HIKKY_BACK_API_KEY        clé API du backend
    HIKKY_RESTAURANT_PHONE    +33472100100
    HIKKY_LLM_GGUF            /workspace/models/Qwen2.5-32B-Instruct-Q5_K_M.gguf
    HIKKY_GREETING            « Bonjour, vous êtes au restaurant Le Petit Sud, je vous écoute. »
    HIKKY_DEBUG_DIR           /workspace/debug (WAV de chaque énoncé ; vide = pas d'enregistrement)
    HIKKY_SRC_DIRS            /workspace/Call-bot/src:/workspace/Call-bot/scripts (séparés par :)
    HIKKY_ENV_FILE            /workspace/bot_back.env (fichier KEY=VALUE chargé au démarrage)
    HIKKY_TIMEZONE            Europe/Paris

Une variable obligatoire absente arrête le processus au démarrage, avec son nom :
une erreur de déploiement se voit tout de suite, jamais en plein appel.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def load_env_file(path: str) -> None:
    """Charge un fichier `KEY=VALUE` dans l'environnement, sans écraser l'existant.

    Lignes vides et commentaires ignorés ; coupe au PREMIER `=` (une valeur peut en
    contenir un). Fichier absent = rien à charger.
    """
    file = Path(path)
    if not path or not file.exists():
        return
    for raw in file.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


def _required(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise SystemExit(
            f"{name} manquant : à définir dans l'environnement (voir telnyx_pipeline/config.py)"
        )
    return value


@dataclass(frozen=True)
class Settings:
    public_host: str
    streaming: bool
    stt_url: str
    tts_url: str
    tts_stream_url: str
    back_base_url: str
    back_api_key: str
    restaurant_phone: str
    llm_gguf: str
    greeting: str
    debug_dir: str
    src_dirs: tuple[str, ...]
    timezone: str

    @classmethod
    def from_env(cls) -> Settings:
        env = os.environ.get
        return cls(
            public_host=_required("PUBLIC_HOST"),
            streaming=bool(env("TTS_STREAM")),
            stt_url=env("HIKKY_STT_URL", "http://127.0.0.1:8801/transcribe"),
            tts_url=env("HIKKY_TTS_URL", "http://127.0.0.1:8802/synthesize"),
            tts_stream_url=env("HIKKY_TTS_STREAM_URL", "http://127.0.0.1:8802/synthesize_stream"),
            back_base_url=_required("HIKKY_BACK_BASE_URL"),
            back_api_key=env("HIKKY_BACK_API_KEY", ""),
            restaurant_phone=env("HIKKY_RESTAURANT_PHONE", "+33472100100"),
            llm_gguf=env("HIKKY_LLM_GGUF", "/workspace/models/Qwen2.5-32B-Instruct-Q5_K_M.gguf"),
            greeting=env(
                "HIKKY_GREETING",
                "Bonjour, vous êtes au restaurant Le Petit Sud, je vous écoute.",
            ),
            debug_dir=env("HIKKY_DEBUG_DIR", "/workspace/debug"),
            src_dirs=tuple(
                d
                for d in env(
                    "HIKKY_SRC_DIRS", "/workspace/Call-bot/src:/workspace/Call-bot/scripts"
                ).split(":")
                if d
            ),
            timezone=env("HIKKY_TIMEZONE", "Europe/Paris"),
        )
