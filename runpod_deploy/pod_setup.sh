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

step "venv-bot313 (Python 3.13 : audioop-lts + llama-cpp compilé CUDA)"
# Pourquoi 3.13 : le codec G.711 du pipeline passe par `audioop`, retiré de la stdlib en 3.13 ;
# son backport officiel `audioop-lts` n'existe QUE pour >= 3.13. Les roues CUDA précompilées de
# llama-cpp-python s'arrêtent à cp312 -> on compile (~15 min sur 96 coeurs, A40 = sm_86).
# Python est installé sur le volume (/workspace/uv-python) : il survit à un redémarrage du pod.
pip install -q uv 2>&1 | tail -1
export UV_PYTHON_INSTALL_DIR=/workspace/uv-python
uv python install 3.13 2>&1 | tail -1
PY313=$(uv python find 3.13)
uv venv /workspace/venv-bot313 --python "$PY313" 2>&1 | tail -1
uv pip install --python /workspace/venv-bot313/bin/python -q pip audioop-lts fastapi uvicorn httpx websockets numpy pydantic soxr 2>&1 | tail -1
PATH=/usr/local/cuda/bin:$PATH CUDACXX=/usr/local/cuda/bin/nvcc \
  CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=86" FORCE_CMAKE=1 CMAKE_BUILD_PARALLEL_LEVEL=32 \
  /workspace/venv-bot313/bin/python -m pip install -q --no-cache-dir "llama-cpp-python==0.3.36" 2>&1 | tail -1
/workspace/venv-bot313/bin/python -c "import audioop, soxr; from llama_cpp import llama_supports_gpu_offload as f; print('python 3.13 | audioop:', audioop.__file__.split('/')[-3], '| llama gpu offload:', f())"

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

if [ -f /workspace/models/Qwen2.5-32B-Instruct-Q5_K_M.gguf ] && [ -x /workspace/venv-stt/bin/uvicorn ] && [ -x /workspace/venv-telnyx/bin/uvicorn ] && [ -x /workspace/venv-bot313/bin/uvicorn ]; then
  step "SETUP_DONE"
else
  step "SETUP_FAIL"
fi
