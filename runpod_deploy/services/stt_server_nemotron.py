"""Service STT Nemotron 3.5 ASR (francais). venv-stt (torch>=2.5, cuDNN9). Port 127.0.0.1:8801.
POST /transcribe : corps = PCM16 mono 16 kHz brut -> {"text": ...}."""
import os, time, numpy as np, torch
from fastapi import FastAPI, Request

os.environ.setdefault("HF_HOME", "/workspace/hf")
MID = "nvidia/nemotron-3.5-asr-streaming-0.6b"
app = FastAPI()
_proc = None
_model = None


def load():
    global _proc, _model
    if _model is None:
        from transformers import AutoModelForRNNT, AutoProcessor
        t = time.time()
        _proc = AutoProcessor.from_pretrained(MID)
        _model = AutoModelForRNNT.from_pretrained(MID, device_map="auto", dtype=torch.float16)
        print(f"[stt] Nemotron charge en {time.time()-t:.1f}s", flush=True)
    return _proc, _model


@app.on_event("startup")
def _warm():
    proc, model = load()
    # Première inférence à froid mesurée à 5 s en appel réel (noyaux CUDA, allocateur) :
    # la 1re phrase du client était découpée et perdue. On paie ce coût au boot, pas
    # sur le client. Deux longueurs pour couvrir les tailles de graphe courantes.
    t = time.time()
    for secs in (2.0, 4.0):
        audio = (np.random.default_rng(0).standard_normal(int(16000 * secs)) * 0.01).astype(np.float32)
        inputs = proc(audio, sampling_rate=16000, language="fr-FR").to(model.device, dtype=model.dtype)
        with torch.no_grad():
            model.generate(**inputs, return_dict_in_generate=True)
    print(f"[stt] prechauffage en {time.time()-t:.1f}s", flush=True)


@app.post("/transcribe")
async def transcribe(request: Request):
    proc, model = load()
    raw = await request.body()
    if len(raw) < 320:
        return {"text": ""}
    audio = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    t = time.time()
    inputs = proc(audio, sampling_rate=16000, language="fr-FR")
    inputs = inputs.to(model.device, dtype=model.dtype)
    with torch.no_grad():
        out = model.generate(**inputs, return_dict_in_generate=True)
    txt = proc.decode(out.sequences, skip_special_tokens=True)
    if isinstance(txt, list):
        txt = txt[0] if txt else ""
    print(f"[stt] {len(audio)/16000:.2f}s -> {txt!r} in {time.time()-t:.2f}s", flush=True)
    return {"text": (txt or "").strip()}


@app.get("/health")
def health():
    return {"ok": _model is not None}
