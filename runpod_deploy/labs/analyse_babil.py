"""Que contient la queue d'un emballement ? Pour chaque synthèse qui dépasse 2x la durée
attendue : codes du codebook 0 (répétition ?) et énergie RMS par frame de 80 ms du wav
décodé (silence ou babil ?). But : trouver un critère d'arrêt précoce fiable.
Sortie : /workspace/debug/babil.json + résumé sur stdout."""
import json, sys, os, time
os.environ.setdefault("HF_HOME", "/workspace/hf")
sys.path.insert(0, "/workspace/services")
import numpy as np, torch
from qwen_tts import Qwen3TTSModel
from fast_tts import FastTTS

N = int(sys.argv[1]) if len(sys.argv) > 1 else 24
TEXT = "Puis-je avoir votre nom ?"
SPEAKER = "ono_anna"; INSTRUCT = "Ton chaleureux et professionnel d une hotesse de restaurant francais"
m = Qwen3TTSModel.from_pretrained("Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice", device_map="cuda", dtype=torch.bfloat16)
eng = FastTTS(m)

# On patche generate pour récupérer les codes : on relit la fonction, mais le plus simple
# est de reproduire la boucle via generate_stream qui conserve `codes` ... generate renvoie
# seulement le nombre de frames. On capture donc les tokens via le sampler.
tokens_log = []
_orig = eng._sample
def _spy(logits, temperature, top_k, top_p, do_sample):
    t = _orig(logits, temperature, top_k, top_p, do_sample)
    tokens_log.append(int(t[0]))
    return t
eng._sample = _spy

out = []
for i in range(N):
    tokens_log.clear()
    w, fs, nf = eng.generate(TEXT, SPEAKER, INSTRUCT, max_frames=160)  # sans plafond serré : voir la vraie queue
    w = np.asarray(w, dtype=np.float32)
    spf = int(fs * 0.08)
    rms = [float(np.sqrt(np.mean(w[k:k + spf] ** 2)) + 1e-9) for k in range(0, len(w) - spf + 1, spf)]
    db = [round(20 * np.log10(r), 1) for r in rms]
    cb0 = list(tokens_log)
    emb = nf >= 2 * 1.1 * len(TEXT)
    # diversité du codebook 0 sur fenêtres glissantes de 10 frames
    div = [len(set(cb0[k:k + 10])) for k in range(0, max(1, len(cb0) - 9))]
    out.append({"i": i, "frames": int(nf), "emballement": bool(emb), "db_par_frame": db,
                "cb0": cb0, "diversite_10": div})
    print(f"#{i:02d} frames={nf:3d} {'EMBALLEMENT' if emb else 'ok':12} "
          f"dB[0:10]={db[:10]} dB[fin]={db[-8:]} div10[fin]={div[-8:]}", flush=True)
json.dump(out, open("/workspace/debug/babil.json", "w"))
print("ANALYSE_DONE", sum(o["emballement"] for o in out), "emballements /", N, flush=True)
