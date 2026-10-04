"""Service TTS Qwen3-TTS (francais). venv-telnyx. Port 127.0.0.1:8802.
POST /synthesize {"text": "..."} -> PCM16 mono bytes ; en-tete X-Sample-Rate.
POST /synthesize_stream : idem en streaming (via streaming_engine, CUDA graphs)."""
import struct
import io, os, time, struct, numpy as np, torch
from fastapi import FastAPI, Response
from pydantic import BaseModel

os.environ.setdefault("HF_HOME", "/workspace/hf")
MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
SPEAKER = "ono_anna"
# Instruction en ANGLAIS, sobre. Mesuré le 2026-10-03 (debug/runaway_*.json, 60 synthèses
# de phrases courtes par variante) : avec l'instruction française, 6 emballements dont
# 12,8 s ; en anglais sobre, 1 emballement ; combinée à l'arrêt sur boucle (fast_tts),
# pire cas 4,5 s. Le labo émotions avait déjà vu que l'instruction française sur-joue.
INSTRUCT = os.environ.get("TTS_INSTRUCT", "Speak in a warm, calm and professional tone, like a restaurant host.")

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

app = FastAPI()
from fastapi.responses import StreamingResponse, JSONResponse
import threading, asyncio, faulthandler, signal
import queue as _queue
faulthandler.register(signal.SIGUSR1, all_threads=True)  # kill -USR1 <pid> -> piles de tous les threads
import streaming_engine as _se   # fichier du dépôt : absent = copie ratée, on échoue au démarrage

_model = None
_fast = None


def get_model():
    global _model
    if _model is None:
        from qwen_tts import Qwen3TTSModel
        t = time.time()
        _model = Qwen3TTSModel.from_pretrained(MODEL_ID, device_map="cuda:0", dtype=torch.bfloat16, attn_implementation="sdpa")
        print(f"[tts] Qwen3-TTS charge en {time.time()-t:.1f}s", flush=True)
    return _model


class Req(BaseModel):
    text: str


@app.on_event("startup")
def _warm():
    m = get_model()
    global _fast
    if os.environ.get("TTS_FAST"):
        try:
            import time as _t, traceback
            import fast_tts
            _s = _t.time()
            _fast = fast_tts.FastTTS(m)
            print("[tts] FAST engine pret (CUDA graphs) en %.1fs" % (_t.time() - _s), flush=True)
        except Exception as _e:
            traceback.print_exc()
            _fast = None
            print("[tts] FAST engine indisponible: %s" % _e, flush=True)


# --- Verrou du moteur CUDA-graph : jamais une attente silencieuse de 60 s.
# Observe en prod : un stream abandonne par le client a garde le verrou pour
# toujours -> TTFA 67 s, 87 s, 100 s, 889 s en file derriere lui.
LOCK_ACQUIRE_TIMEOUT = 10.0   # au-dela : 503 immediat, l'orchestrateur reagit
LOCK_STUCK_AFTER = 60.0       # un detenteur plus vieux = anomalie (health ok:false)
_lock_held_since = None

def _acquire(lock):
    global _lock_held_since
    got = lock.acquire(timeout=LOCK_ACQUIRE_TIMEOUT)
    if got:
        _lock_held_since = time.time()
    else:
        age = time.time() - (_lock_held_since or time.time())
        print(f"[tts] VERROU OCCUPE depuis {age:.0f}s - requete refusee (503)", flush=True)
    return got

def _release(lock):
    global _lock_held_since
    _lock_held_since = None
    lock.release()

def _lock_stuck():
    return _lock_held_since is not None and time.time() - _lock_held_since > LOCK_STUCK_AFTER

def _busy_503():
    return JSONResponse({"error": "tts busy"}, status_code=503)


@app.post("/synthesize")
def synthesize(req: Req):
    m = get_model()
    t = time.time()
    if _fast is not None:
        import fast_tts
        if not _acquire(fast_tts.LOCK):
            return _busy_503()
        try:
            _w, sr, _nf = _fast.generate(req.text, SPEAKER, INSTRUCT)
        finally:
            _release(fast_tts.LOCK)
        wav = np.asarray(_w, dtype=np.float32)
    else:
        wavs, sr = m.generate_custom_voice(text=req.text, language="French", speaker=SPEAKER, instruct=INSTRUCT)
        wav = np.asarray(wavs[0], dtype=np.float32)
    pcm16 = np.clip(wav, -1.0, 1.0)
    pcm16 = (pcm16 * 32767.0).astype("<i2").tobytes()
    print(f"[tts] synth {len(req.text)}c -> {len(wav)/sr:.2f}s audio in {time.time()-t:.2f}s (sr={sr})", flush=True)
    return Response(content=pcm16, media_type="application/octet-stream",
                    headers={"X-Sample-Rate": str(int(sr))})


@app.post("/synthesize_stream")
async def synthesize_stream(req: Req):
    m = get_model()
    import fast_tts
    loop = asyncio.get_running_loop()
    if _fast is not None:
        # acquisition dans un thread : ne bloque jamais la boucle d'evenements
        if not await loop.run_in_executor(None, _acquire, fast_tts.LOCK):
            return _busy_503()
    q = _queue.Queue()            # non borne : <= ~13 chunks de ~60 Ko, jamais bloquant
    stop = threading.Event()      # leve quand le client coupe (raccrochage, barge-in)
    _END = object()

    def worker():
        t0 = time.time(); first = True
        it = None
        try:
            it = (_fast.generate_stream(req.text, SPEAKER, INSTRUCT) if _fast is not None
                  else _se.stream_custom_voice(m, text=req.text, speaker=SPEAKER,
                                               language="French", instruct=INSTRUCT))
            for pcm_f32, sr in it:
                if stop.is_set():
                    break                      # client parti : on arrete de generer
                if first:
                    print("[tts] FAST TTFA=%.3fs" % (time.time() - t0), flush=True); first = False
                b = (np.clip(pcm_f32, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
                q.put(struct.pack(">I", len(b)) + b)
        except Exception as e:  # noqa: BLE001
            print("[tts] stream erreur: %r" % (e,), flush=True)
        finally:
            close = getattr(it, "close", None)
            if close is not None:
                try: close()                   # finalise le generateur
                except Exception: pass         # noqa: BLE001
            if _fast is not None:
                _release(fast_tts.LOCK)        # TOUJOURS rendu, meme si le client a raccroche
            q.put(_END)

    threading.Thread(target=worker, daemon=True).start()

    async def agen():
        try:
            while True:
                item = await loop.run_in_executor(None, q.get)
                if item is _END:
                    break
                yield item
        finally:
            stop.set()   # Starlette ferme l'async-gen a la deconnexion -> on arrete le worker

    return StreamingResponse(agen(), media_type="application/octet-stream",
                             headers={"X-Sample-Rate": "24000"})


@app.get("/health")
def health():
    stuck = _lock_stuck()
    held = None if _lock_held_since is None else round(time.time() - _lock_held_since, 1)
    return {"ok": _model is not None and not stuck, "lock_held_s": held, "lock_stuck": stuck}
