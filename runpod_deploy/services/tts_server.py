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
INSTRUCT = "Ton chaleureux et professionnel d une hotesse de restaurant francais"

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

app = FastAPI()
from fastapi.responses import StreamingResponse
try:
    import streaming_engine as _se
    _STREAM_OK = True
except Exception as _e:
    _se = None
    _STREAM_OK = False
    print("[tts] streaming_engine indisponible: %s" % _e, flush=True)

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


@app.post("/synthesize")
def synthesize(req: Req):
    m = get_model()
    t = time.time()
    if _fast is not None:
        import fast_tts
        with fast_tts.LOCK:
            _w, sr, _nf = _fast.generate(req.text, SPEAKER, INSTRUCT)
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
def synthesize_stream(req: Req):
    m = get_model()
    def gen():
        t0 = time.time(); first = True
        if _fast is not None:
            import fast_tts
            with fast_tts.LOCK:
                for pcm_f32, sr in _fast.generate_stream(req.text, SPEAKER, INSTRUCT):
                    if first:
                        print("[tts] FAST TTFA=%.3fs" % (time.time() - t0), flush=True); first = False
                    b = (np.clip(pcm_f32, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
                    yield struct.pack(">I", len(b)) + b
        else:
            for pcm_f32, sr in _se.stream_custom_voice(m, text=req.text, speaker=SPEAKER,
                                                       language="French", instruct=INSTRUCT):
                if first:
                    print("[tts] TTFA=%.3fs" % (time.time() - t0), flush=True); first = False
                b = (np.clip(pcm_f32, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
                yield struct.pack(">I", len(b)) + b
    return StreamingResponse(gen(), media_type="application/octet-stream",
                             headers={"X-Sample-Rate": "24000"})


@app.get("/health")
def health():
    return {"ok": _model is not None}
