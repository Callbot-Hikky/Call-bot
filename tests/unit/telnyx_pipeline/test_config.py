"""La configuration vient de l'environnement ; une variable obligatoire absente arrête tout."""

import pytest
from telnyx_pipeline.config import Settings, load_env_file

OBLIGATOIRES = {
    "PUBLIC_HOST": "pod-19123.proxy.runpod.net",
    "HIKKY_BACK_BASE_URL": "https://back.test",
}


def _env(monkeypatch, **extra):
    for k in (
        "PUBLIC_HOST",
        "HIKKY_BACK_BASE_URL",
        "TTS_STREAM",
        "HIKKY_STT_URL",
        "HIKKY_GREETING",
    ):
        monkeypatch.delenv(k, raising=False)
    for k, v in {**OBLIGATOIRES, **extra}.items():
        monkeypatch.setenv(k, v)


def test_les_valeurs_par_defaut_pointent_sur_les_services_locaux(monkeypatch):
    _env(monkeypatch)
    s = Settings.from_env()
    assert s.stt_url == "http://127.0.0.1:8801/transcribe"
    assert s.tts_stream_url == "http://127.0.0.1:8802/synthesize_stream"
    assert s.streaming is False
    assert s.restaurant_phone == "+33472100100"
    assert s.src_dirs == ("/workspace/Call-bot/src", "/workspace/Call-bot/scripts")


def test_l_environnement_l_emporte_sur_les_defauts(monkeypatch):
    _env(
        monkeypatch,
        TTS_STREAM="1",
        HIKKY_STT_URL="http://stt:9000/transcribe",
        HIKKY_GREETING="Allô ?",
    )
    s = Settings.from_env()
    assert s.streaming is True
    assert s.stt_url == "http://stt:9000/transcribe"
    assert s.greeting == "Allô ?"


def test_sans_public_host_le_processus_refuse_de_demarrer(monkeypatch):
    """Avant, une valeur par défaut pointait sur un pod mort : Telnyx se connectait
    dans le vide sans aucune erreur visible. Une dépendance absente doit se voir."""
    _env(monkeypatch)
    monkeypatch.delenv("PUBLIC_HOST")
    with pytest.raises(SystemExit, match="PUBLIC_HOST"):
        Settings.from_env()


def test_le_fichier_env_ne_remplace_pas_une_variable_deja_definie(monkeypatch, tmp_path):
    f = tmp_path / "bot_back.env"
    f.write_text(
        "# commentaire\nHIKKY_BACK_API_KEY=cle=avec=egal\nHIKKY_RESTAURANT_PHONE=+33100000000\n\n"
    )
    monkeypatch.setenv("HIKKY_RESTAURANT_PHONE", "+33999999999")
    monkeypatch.delenv("HIKKY_BACK_API_KEY", raising=False)
    load_env_file(str(f))
    import os

    assert os.environ["HIKKY_BACK_API_KEY"] == "cle=avec=egal"  # coupé au premier =
    assert os.environ["HIKKY_RESTAURANT_PHONE"] == "+33999999999"  # l'existant gagne


def test_un_fichier_env_absent_ne_casse_rien(tmp_path):
    load_env_file(str(tmp_path / "inexistant.env"))
