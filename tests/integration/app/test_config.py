import pytest

from hikky.app.config import MissingConfig, load_from_env


def test_load_from_env_returns_config_when_all_required_set(monkeypatch):
    monkeypatch.setenv("HIKKY_BACK_BASE_URL", "https://back.example")
    monkeypatch.setenv("HIKKY_BACK_API_KEY", "secret")
    monkeypatch.setenv("HIKKY_LLAMA_MODEL_PATH", "/models/m.gguf")
    monkeypatch.setenv("HIKKY_PIPER_MODEL_PATH", "/voices/v.onnx")
    config = load_from_env()
    assert config.back_base_url == "https://back.example"
    assert config.back_api_key == "secret"
    assert config.llama_model_path == "/models/m.gguf"
    assert config.piper_model_path == "/voices/v.onnx"
    # defaults
    assert config.whisper_model == "distil-large-v3"
    assert config.whisper_device == "cuda"
    assert config.llama_n_ctx == 4096
    assert config.tts_sample_rate == 22050


def test_load_from_env_raises_missing_config_on_missing_required(monkeypatch):
    monkeypatch.delenv("HIKKY_BACK_BASE_URL", raising=False)
    monkeypatch.setenv("HIKKY_BACK_API_KEY", "k")
    monkeypatch.setenv("HIKKY_LLAMA_MODEL_PATH", "/m")
    monkeypatch.setenv("HIKKY_PIPER_MODEL_PATH", "/v")
    with pytest.raises(MissingConfig) as exc:
        load_from_env()
    assert "HIKKY_BACK_BASE_URL" in str(exc.value)


def test_load_from_env_overrides_defaults(monkeypatch):
    monkeypatch.setenv("HIKKY_BACK_BASE_URL", "https://back.example")
    monkeypatch.setenv("HIKKY_BACK_API_KEY", "k")
    monkeypatch.setenv("HIKKY_LLAMA_MODEL_PATH", "/m")
    monkeypatch.setenv("HIKKY_PIPER_MODEL_PATH", "/v")
    monkeypatch.setenv("HIKKY_WHISPER_MODEL", "medium")
    monkeypatch.setenv("HIKKY_WHISPER_DEVICE", "cpu")
    monkeypatch.setenv("HIKKY_LLAMA_N_CTX", "8192")
    monkeypatch.setenv("HIKKY_LLAMA_N_GPU_LAYERS", "20")
    monkeypatch.setenv("HIKKY_TTS_SAMPLE_RATE", "16000")
    config = load_from_env()
    assert config.whisper_model == "medium"
    assert config.whisper_device == "cpu"
    assert config.llama_n_ctx == 8192
    assert config.llama_n_gpu_layers == 20
    assert config.tts_sample_rate == 16000
