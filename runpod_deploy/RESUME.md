# Hikky callbot — état & reprise (sauvegarde 2026-09-17)

Sauvegarde de tout le code déployé sur le pod RunPod. **Rien d'irremplaçable n'est
sur le pod** : ce dossier contient les fichiers exacts qui tournent. Les modèles
et venvs sont volumineux mais reproductibles (voir "Rebuild from scratch").

## Accès au pod
- Pod ACTUEL : `8b10kksz3ra5iw` (A40 48 Go, image `runpod/pytorch:2.4.0-py3.11-cuda12.4.1`), reconstruit 2026-09-18.
- SSH direct : `ssh -i ~/.ssh/id_ed25519 root@194.68.245.239 -p 22145`
  (via rtk : `rtk proxy ssh -i ~/.ssh/id_ed25519 root@194.68.245.239 -p 22145 '<cmd>'`)
  Note : sshd peut mettre ~4-5 min à répondre après le démarrage d'un pod neuf.
- URL publique orchestrateur : `https://8b10kksz3ra5iw-19123.proxy.runpod.net`
- `PUBLIC_HOST` (env orchestrateur) = `8b10kksz3ra5iw-19123.proxy.runpod.net` (déjà dans run_telnyx.sh).
- Ancien pod `uv9jklqxm6vwnj` (69.30.85.117:22062) : ARRÊTÉ, jamais pu redémarrer (pas de GPU libre) → d'où ce rebuild. À supprimer quand plus utile.
- Port SSH direct peut changer : `runpodctl pod list` + API GraphQL pour les ports
  (`curl "https://api.runpod.io/graphql?api_key=$KEY" -d '{"query":"query{pod(input:{podId:\"uv9jklqxm6vwnj\"}){runtime{ports{ip privatePort publicPort type}}}}"}'`)

## Architecture (3 services sur le pod, /workspace)
| Service | Port | venv | Rôle |
|---|---|---|---|
| STT | 8801 | venv-stt | Nemotron `nvidia/nemotron-3.5-asr-streaming-0.6b` (fr) |
| TTS | 8802 | venv-telnyx | Qwen3-TTS-12Hz-1.7B-CustomVoice, voix `ono_anna`, **CUDA graphs** |
| Orchestrateur | 19123 | venv-bot | Qwen 32B GGUF (llama-cpp) + domaine Call-bot + WS Telnyx |

Lancement : `bash /workspace/run_telnyx.sh` (contient `TTS_FAST=1` et `TTS_STREAM=1`).
LLM 32B : préchauffage ~70-80 s au boot avant "orchestrateur pret".

## Ce qui a été gagné (à NE PAS reperdre)
1. **Codec A-law** : la ligne européenne envoie du PCMA (A-law). Entrée décodée en
   `audioop.alaw2lin` (dynamique via `media_format.encoding`), sortie PCMU. → STT enfin compréhensible.
2. **Voix `ono_anna`** (accent français) au lieu de Serena.
3. **TTS accéléré CUDA graphs** (`fast_tts.py`) : RTF **1,86 → 0,41** (5×). Remplace la
   double boucle autoregressive HF (talker + sous-talker) par 2 graphes CUDA rejoués
   (sous-talker 15 pas en 1 replay, greedy ; codebook0 garde le sampling complet).
   Activé par `TTS_FAST=1`. Fallback = `generate_custom_voice`.
4. **Streaming TTS** (`fast_tts.generate_stream` + `/synthesize_stream` framing
   longueur-préfixée + `telnyx_bot.speak_stream`) : **TTFA ~0,55 s** (le bot parle
   pendant qu'il génère). Chunks de 16 frames (~1,3 s) > écart Telnyx 1,05 s (anti-trou).
   Rééchantillonnage à état continu (pas de clic). Activé par `TTS_STREAM=1`. Fallback auto.
5. **Robustesse** : STT vide → le bot redemande ("pouvez-vous répéter ?") au lieu de
   rester muet ; `speaking` réinitialisé dans un `finally` (pas de tour gelé).
6. **Brièveté** : récap sans préambule + sans jour de semaine, clôture courte
   (`routed_turn.py`), `SIL_FRAMES=25` (fin de parole 0,5 s).
7. Latence tour de parole : **~15-20 s → ~2-5 s**. Réservation complète OK end-to-end.

## PIÈGES CRITIQUES (m'ont coûté des heures)
- **NE JAMAIS `pkill -f "tts_server"`** : la commande de lancement contient
  `uvicorn tts_server:app`, donc le pkill matche et tue le shell qui lance → morts
  intermittentes. **Toujours tuer par PID** : `ss -tlnp | grep :PORT` → `kill -9 <pid>`.
- **Zombies de port** : un ancien process peut garder le port et empêcher le nouveau de
  bind (le nouveau meurt). Vérifier `ss -tlnp | grep :PORT` et tuer par PID avant relance.
- **venv-stt** a besoin de `accelerate` (Nemotron `device_map`) — sinon crash au boot.
- **venv-bot** : `llama-cpp-python` via wheel CUDA pré-compilé
  (`pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124`),
  PAS de compilation (nvcc pas sur PATH). Vérifier `llama_supports_gpu_offload()==True`.
- **rtk avale la sortie** quand un job est backgroundé dans la commande SSH → lancer
  détaché puis vérifier dans des commandes séparées (lire les logs).
- Salutation cachée au boot de l'orchestrateur (via TTS) → TTS doit être prêt avant.

## Fichiers de ce dossier (= ce qui tourne sur le pod)
- `services/telnyx_bot.py` — orchestrateur (WS Telnyx, A-law, streaming speak, robustesse)
- `services/tts_server.py` — serveur TTS (/synthesize + /synthesize_stream, flags TTS_FAST)
- `services/fast_tts.py` — moteur CUDA graphs (generate + generate_stream)
- `services/optimized_talker.py` — TalkerGraph + PredictorGraph (bug lazy_init corrigé au runtime dans fast_tts)
- `services/streaming_engine.py` — helpers (_build_talker_inputs, _sample_next_token)
- `services/stt_server.py` — serveur Nemotron
- `services/poc_fast.py`, `bench_graphs.py`, `instrument.py` — bench/PoC (référence)
- `run_telnyx.sh` — lance les 3 services (TTS_FAST=1, TTS_STREAM=1)
- `bot_back.env` — URL/clé backend, téléphone resto
- `Call-bot/src/hikky/domain/{routed_turn,phraseur,conversation_brain}.py` — patches domaine
- `Call-bot/scripts/poc_common.py` — build_session_factory

## Telnyx
- TeXML app `3049671651756082810`. `voice_url` doit pointer sur
  `https://uv9jklqxm6vwnj-19123.proxy.runpod.net/telnyx/inbound`.
  PATCH : `curl -X PATCH https://api.telnyx.com/v2/texml_applications/3049671651756082810
  -H "Authorization: Bearer <TELNYX_KEY>" -H "Content-Type: application/json"
  -d '{"voice_url":"https://uv9jklqxm6vwnj-19123.proxy.runpod.net/telnyx/inbound"}'`
- Numéro de test : **+33974067183**.

## Reprise DEMAIN (pod juste arrêté/stoppé, /workspace intact)
1. Démarrer le pod (console RunPod ou `runpodctl`).
2. Récupérer le port SSH (peut avoir changé).
3. `bash /workspace/run_telnyx.sh`
4. Attendre ~80 s (LLM), vérifier `curl localhost:{8801,8802,19123}/health` = `{"ok":true}`.
5. Vérifier/mettre à jour le `voice_url` Telnyx (l'URL proxy est stable tant que le pod ID
   ne change pas). Tester un appel.

## Rebuild FROM SCRATCH (si pod terminé / crédit épuisé)
1. Créer un pod A40 avec la même image + `PUBLIC_KEY` (clé SSH) dans l'env.
2. Modèles (dans /workspace/models et le cache HF) :
   - `hf download Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice`
   - `hf download nvidia/nemotron-3.5-asr-streaming-0.6b`
   - GGUF 32B : `Qwen2.5-32B-Instruct-Q5_K_M.gguf` → /workspace/models/
3. venvs :
   - venv-bot : `pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124 fastapi uvicorn httpx websockets numpy pydantic`
   - venv-telnyx (`--system-site-packages`, hérite torch 2.4.1) : `pip install qwen-tts fastapi uvicorn`
   - venv-stt (frais) : `pip install torch --index-url .../cu124` puis `pip install "transformers>=5.13" accelerate numpy soundfile librosa uvicorn fastapi "nvidia-cudnn-cu12>=9"`
4. Copier ce dossier vers /workspace (services/, run_telnyx.sh, bot_back.env, Call-bot/).
   Recopier tout le src Call-bot depuis le repo git (les 3 fichiers domaine ici sont les
   patchés ; le reste vient du repo).
5. `bash /workspace/run_telnyx.sh`, vérifier health, PATCH Telnyx, tester.

## À FAIRE (pour ne plus jamais perdre)
- **Network volume RunPod** : monter un volume réseau sur /workspace → survit à la
  terminaison du pod et permet de rattacher à un nouveau pod (résout le "pas de GPU libre").
- Durcir le streaming (concurrence : lock déjà présent côté TTS).
- Écho STT possible après réponse du bot (à surveiller) — mitigé par le re-prompt.
