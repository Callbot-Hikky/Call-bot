import json
d = json.load(open("/workspace/debug/babil.json"))
for o in d:
    if not o["emballement"]:
        continue
    cb = o["cb0"]; n = o["frames"]; div = o["diversite_10"]; db = o["db_par_frame"]
    rep = sum(1 for k in range(26, len(cb)) if cb[k] == cb[k - 1])
    queue = db[25:] or [0]
    i = o["i"]
    print(f"#{i:02d} frames={n:3d} repet_consec_apres25={rep}/{max(1, len(cb) - 26)} "
          f"div10 min(apres20)={min(div[20:]) if len(div) > 20 else None} "
          f"queue_dB_moy={sum(queue) / len(queue):.1f}")
ok = [o for o in d if not o["emballement"]]
print("OK : div10 min", min(min(o["diversite_10"]) for o in ok), "frames", sorted(o["frames"] for o in ok))
