"""Mesure l'emballement TTS : N synthèses de phrases courtes, compte celles dont la durée
dépasse 2x la durée attendue (~1,1 frame/caractère) ou atteint le plafond.
Usage : python bench_runaway.py <label> [N]  -> écrit /workspace/debug/runaway_<label>.json"""
import json, sys, time, os
os.environ.setdefault("HF_HOME", "/workspace/hf")
sys.path.insert(0, "/workspace/services")
import torch
from qwen_tts import Qwen3TTSModel
from fast_tts import FastTTS

label = sys.argv[1]; N = int(sys.argv[2]) if len(sys.argv) > 2 else 60
PHRASES = ["Puis-je avoir votre nom ?", "Pour quel jour souhaitez-vous réserver ?",
           "À quelle heure souhaitez-vous venir ?", "Oui, nous avons une terrasse.",
           "C'est bien cela ?"]
SPEAKER = os.environ.get("SPEAKER", "ono_anna")
INSTRUCT = os.environ.get("INSTRUCT", "Ton chaleureux et professionnel d une hotesse de restaurant francais")

m = Qwen3TTSModel.from_pretrained("Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice", device_map="cuda", dtype=torch.bfloat16)
eng = FastTTS(m)
rows = []; t0 = time.time()
for i in range(N):
    text = PHRASES[i % len(PHRASES)]
    attendu = 1.1 * len(text)
    plafond = eng.frames_plafond(text) if hasattr(eng, "frames_plafond") else 160
    _w, fs, nf = eng.generate(text, SPEAKER, INSTRUCT)
    rows.append({"text": text, "frames": int(nf), "attendu": round(attendu, 1), "plafond": plafond,
                 "emballement": bool(nf >= 2 * attendu), "plafond_atteint": bool(nf >= plafond)})
emb = sum(r["emballement"] for r in rows); cap = sum(r["plafond_atteint"] for r in rows)
res = {"label": label, "n": N, "emballements": emb, "plafond_atteints": cap,
       "frames_moy": round(sum(r["frames"] for r in rows) / N, 1),
       "frames_max": max(r["frames"] for r in rows), "duree_s": round(time.time() - t0, 1), "rows": rows}
json.dump(res, open(f"/workspace/debug/runaway_{label}.json", "w"), ensure_ascii=False, indent=1)
print(f"RESULT {label}: n={N} emballements={emb} plafond_atteints={cap} frames_moy={res['frames_moy']} max={res['frames_max']} ({res['duree_s']}s)", flush=True)
