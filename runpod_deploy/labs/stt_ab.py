"""A/B STT sur les énoncés réels de l'appel : Nemotron (serveur prod :8801, même chemin
que le bot : 8k->16k soxr + 1 s de silence en queue) vs faster-whisper large-v3 (fr).
Sortie : une ligne par WAV, les deux transcriptions."""
import glob, json, sys, time, wave
import numpy as np, soxr, httpx

files = sorted(glob.glob("/workspace/debug/*.wav"))
print(f"{len(files)} énoncés", flush=True)

from faster_whisper import WhisperModel
t = time.time()
wm = WhisperModel("large-v3", device="cuda", compute_type="float16")
print(f"whisper large-v3 chargé en {time.time()-t:.1f}s", flush=True)

rows = []
for f in files:
    with wave.open(f) as w:
        pcm8 = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
    x16 = soxr.resample(pcm8, 8000, 16000, quality="HQ").astype("<i2")
    buf = x16.tobytes() + b"\x00" * 32000
    t = time.time()
    nem = httpx.post("http://127.0.0.1:8801/transcribe", content=buf, timeout=30).json()["text"]
    t_nem = time.time() - t
    t = time.time()
    segs, _ = wm.transcribe(x16.astype(np.float32) / 32768.0, language="fr", beam_size=5,
                            vad_filter=False, condition_on_previous_text=False)
    wh = " ".join(s.text.strip() for s in segs)
    t_wh = time.time() - t
    name = f.rsplit("/", 1)[1]
    rows.append({"wav": name, "dur_s": round(len(pcm8) / 8000, 2),
                 "nemotron": nem, "t_nem": round(t_nem, 2), "whisper": wh, "t_wh": round(t_wh, 2)})
    print(f"{name} {len(pcm8)/8000:.2f}s | NEM({t_nem:.2f}s): {nem!r} | WHI({t_wh:.2f}s): {wh!r}", flush=True)

json.dump(rows, open("/workspace/debug/ab.json", "w"), ensure_ascii=False, indent=1)
print("AB_DONE", flush=True)
