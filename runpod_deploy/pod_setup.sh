#!/bin/bash
# Reconstruction du pod hikky (recette RESUME.md "Rebuild FROM SCRATCH").
# Lancé détaché ; journal dans /workspace/setup.log ; fin marquée par SETUP_DONE / SETUP_FAIL.
set -uo pipefail
export HF_HOME=/workspace/hf
export PIP_DISABLE_PIP_VERSION_CHECK=1
mkdir -p /workspace/models /workspace/services /workspace/hf
cd /workspace
step() { echo "[setup $(date -u +%H:%M:%S)] $*"; }

step "outil hf"
pip install -q -U "huggingface_hub[cli]" hf_transfer 2>&1 | tail -1
export HF_HUB_ENABLE_HF_TRANSFER=1

step "téléchargements modèles (en parallèle)"
( hf download Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice >/workspace/dl_tts.log 2>&1 && echo "DL_TTS_OK" || echo "DL_TTS_FAIL" ) &
( hf download nvidia/nemotron-3.5-asr-streaming-0.6b >/workspace/dl_stt.log 2>&1 && echo "DL_STT_OK" || echo "DL_STT_FAIL" ) &
( hf download bartowski/Qwen2.5-32B-Instruct-GGUF Qwen2.5-32B-Instruct-Q5_K_M.gguf --local-dir /workspace/models >/workspace/dl_gguf.log 2>&1 && echo "DL_GGUF_OK" || echo "DL_GGUF_FAIL" ) &

step "venv-bot (llama-cpp wheel cu124)"
python3 -m venv /workspace/venv-bot
/workspace/venv-bot/bin/pip install -q llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124 2>&1 | tail -1
/workspace/venv-bot/bin/pip install -q fastapi uvicorn httpx websockets numpy pydantic soxr 2>&1 | tail -1
/workspace/venv-bot/bin/python -c "from llama_cpp import llama_supports_gpu_offload as f; print('llama gpu offload:', f())"

step "venv-telnyx (TTS, hérite torch système)"
python3 -m venv --system-site-packages /workspace/venv-telnyx
/workspace/venv-telnyx/bin/pip install -q qwen-tts fastapi uvicorn 2>&1 | tail -1
/workspace/venv-telnyx/bin/python -c "import torch, qwen_tts; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

step "venv-stt (faster-whisper large-v3 ; Nemotron conservé en secours : transformers/accelerate)"
python3 -m venv /workspace/venv-stt
/workspace/venv-stt/bin/pip install -q torch --index-url https://download.pytorch.org/whl/cu124 2>&1 | tail -1
# faster-whisper = STT en production (stt_server.py). transformers/accelerate ne servent
# qu'au serveur de secours stt_server_nemotron.py. httpx : scripts de mesure (labs/).
/workspace/venv-stt/bin/pip install -q faster-whisper httpx "transformers>=5.13" accelerate numpy soundfile librosa uvicorn fastapi "nvidia-cudnn-cu12>=9" soxr 2>&1 | tail -1
/workspace/venv-stt/bin/python -c "import faster_whisper, transformers; print('faster-whisper', faster_whisper.__version__, '| transformers', transformers.__version__)"

step "attente fin des téléchargements"
wait
ls -la /workspace/models/
du -sh /workspace/hf 2>/dev/null

if [ -f /workspace/models/Qwen2.5-32B-Instruct-Q5_K_M.gguf ] && [ -x /workspace/venv-stt/bin/uvicorn ] && [ -x /workspace/venv-telnyx/bin/uvicorn ] && [ -x /workspace/venv-bot/bin/uvicorn ]; then
  step "SETUP_DONE"
else
  step "SETUP_FAIL"
fi
