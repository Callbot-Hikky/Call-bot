#!/bin/bash
# Lance les 3 services detaches (STT :8801, TTS :8802, orchestrateur :19123).
# Recette, pieges et historique : runpod_deploy/RESUME.md. PUBLIC_HOST = proxy du pod courant.
export HF_HOME=/workspace/hf
export TZ=Europe/Paris   # le pod est en UTC ; les dates du dialogue sont celles du restaurant
CUDNN=/workspace/venv-stt/lib/python3.11/site-packages/nvidia/cudnn/lib
cd /workspace/services
# STT (venv-stt) avec cuDNN9
setsid bash -c "export HF_HOME=/workspace/hf; export LD_LIBRARY_PATH=$CUDNN:\$LD_LIBRARY_PATH; exec /workspace/venv-stt/bin/uvicorn stt_server:app --host 127.0.0.1 --port 8801" </dev/null >>/workspace/stt.log 2>&1 & disown
# TTS (venv-telnyx)
setsid bash -c "export HF_HOME=/workspace/hf; export TTS_FAST=1; exec /workspace/venv-telnyx/bin/uvicorn tts_server:app --host 127.0.0.1 --port 8802" </dev/null >>/workspace/tts.log 2>&1 & disown
# L'orchestrateur met la salutation en cache au démarrage en appelant le TTS : lancés ensemble,
# il arrivait avant lui (« cache salutation échoué ») et la première phrase de chaque appel
# partait avec ~2 s de retard. On attend donc que le TTS réponde (modèle chargé : ~60-90 s).
for _ in $(seq 1 120); do
  curl -sf localhost:8802/health >/dev/null 2>&1 && break
  sleep 5
done
# Orchestrateur (venv-bot313 : Python 3.13 + audioop-lts, llama-cpp compilé CUDA).
setsid bash -c "export HF_HOME=/workspace/hf; export TZ=Europe/Paris; export TTS_STREAM=1; export PUBLIC_HOST=apv5oebaknccuj-19123.proxy.runpod.net; exec /workspace/venv-bot313/bin/uvicorn ${TELNYX_APP:-telnyx_pipeline.server:app} --host 0.0.0.0 --port 19123" </dev/null >>/workspace/orch_new.log 2>&1 & disown
echo "3 services lances"
