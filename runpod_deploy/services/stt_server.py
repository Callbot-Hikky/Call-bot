"""Service STT faster-whisper large-v3 (français). venv-stt. Port 127.0.0.1:8801.
POST /transcribe : corps = PCM16 mono 16 kHz brut -> {"text": ...}.

Plan B appliqué le 2026-10-03 après A/B sur 10 énoncés RÉELS d'un appel (stt_ab.py,
/workspace/debug/ab.json) : Nemotron rendait VIDE 3 énoncés parfaitement audibles
(« Au vegan », « Et en parking ? », « Où est-ce qu'on mange halal chez vous ? ») et
déformait les autres (« Stamps » / « statement » pour « restaurant », « J'ai sûr que »
pour « Est-ce que ») ; Whisper large-v3 les a tous compris, à vitesse égale sur A40
(0,2-0,7 s). L'ancien serveur est conservé : stt_server_nemotron.py."""
import os, time, numpy as np
from fastapi import FastAPI, Request

os.environ.setdefault("HF_HOME", "/workspace/hf")
MODEL = os.environ.get("WHISPER_MODEL", "large-v3")
# Vocabulaire du domaine : guide le décodage des mots rares au téléphone (noms propres,
# « halal » entendu « à l'al »). Pas de phrase complète : Whisper la recopierait.
PROMPT = ("Appel au restaurant Le Petit Sud. Réservation, table, couverts, ce soir, demain, "
          "midi, vingt heures, halal, casher, végétarien, végan, terrasse, parking, fumoir, "
          "climatisation, chaises hautes, accès fauteuil roulant, carte bancaire, tickets restaurant.")
app = FastAPI()
_model = None


def load():
    global _model
    if _model is None:
        from faster_whisper import WhisperModel
        t = time.time()
        _model = WhisperModel(MODEL, device="cuda", compute_type="float16")
        print(f"[stt] whisper {MODEL} charge en {time.time()-t:.1f}s", flush=True)
    return _model


def _run(audio: np.ndarray) -> str:
    segs, info = load().transcribe(
        audio, language="fr", beam_size=5, best_of=5, temperature=0.0,
        initial_prompt=PROMPT, condition_on_previous_text=False,
        vad_filter=False,            # l'orchestrateur a déjà segmenté (VAD énergie + pré-roll)
        no_speech_threshold=0.6, log_prob_threshold=-1.0,
    )
    parts = []
    for s in segs:
        # Whisper hallucine sur le bruit (« Sous-titres réalisés par… ») : un segment jugé
        # « sans parole » ou très improbable est ignoré plutôt que prononcé au client.
        if s.no_speech_prob > 0.6 and s.avg_logprob < -1.0:
            continue
        parts.append(s.text.strip())
    return " ".join(p for p in parts if p).strip()


@app.on_event("startup")
def _warm():
    load()
    # Première inférence à froid coûteuse (noyaux CUDA) : payée au boot, pas sur le client.
    t = time.time()
    for secs in (2.0, 4.0):
        _run((np.random.default_rng(0).standard_normal(int(16000 * secs)) * 0.01).astype(np.float32))
    print(f"[stt] prechauffage en {time.time()-t:.1f}s", flush=True)


@app.post("/transcribe")
async def transcribe(request: Request):
    raw = await request.body()
    if len(raw) < 320:
        return {"text": ""}
    audio = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    t = time.time()
    txt = _run(audio)
    print(f"[stt] {len(audio)/16000:.2f}s -> {txt!r} in {time.time()-t:.2f}s", flush=True)
    return {"text": txt}


@app.get("/health")
def health():
    return {"ok": _model is not None, "engine": f"faster-whisper {MODEL}"}
