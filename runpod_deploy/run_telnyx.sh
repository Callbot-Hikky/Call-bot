#!/bin/bash
# Lance les 3 services detaches. cf HANDOFF.
export HF_HOME=/workspace/hf
export TZ=Europe/Paris   # le pod est en UTC ; les dates du dialogue sont celles du restaurant
CUDNN=/workspace/venv-stt/lib/python3.11/site-packages/nvidia/cudnn/lib
cd /workspace/services
# STT (venv-stt) avec cuDNN9
setsid bash -c "export HF_HOME=/workspace/hf; export LD_LIBRARY_PATH=$CUDNN:\$LD_LIBRARY_PATH; exec /workspace/venv-stt/bin/uvicorn stt_server:app --host 127.0.0.1 --port 8801" </dev/null >>/workspace/stt.log 2>&1 & disown
# TTS (venv-telnyx)
setsid bash -c "export HF_HOME=/workspace/hf; export TTS_FAST=1; exec /workspace/venv-telnyx/bin/uvicorn tts_server:app --host 127.0.0.1 --port 8802" </dev/null >>/workspace/tts.log 2>&1 & disown
# Orchestrateur (venv-bot)
setsid bash -c "export HF_HOME=/workspace/hf; export TTS_STREAM=1; export PUBLIC_HOST=e1yc63e2u7dqxw-19123.proxy.runpod.net; exec /workspace/venv-bot/bin/uvicorn telnyx_bot:app --host 0.0.0.0 --port 19123" </dev/null >>/workspace/orch_new.log 2>&1 & disown
echo "3 services lances"
