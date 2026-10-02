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
8. **Dialogue (2026-10-02)** : le code ne contredit plus le modèle, et le modèle ne peut
   plus affirmer un état. `_EXPLICIT_HOUR` reconnaît « vingt et une heures » (avant :
   l'heure correctement extraite par Qwen était JETÉE par `_drop_invented_time`).
   Phraseur : une demande de slot doit être une question sans vocabulaire d'affirmation
   (sinon repli figé) — il avait sorti « Je confirme 21h ? ». Answerer : ne peut jamais
   annoncer une réservation faite (garde `_FAUSSE_CONFIRMATION`) — il avait inventé
   « confirmée sous le nom Général ».
9. **Front-end audio STT, plan A (2026-10-02)** : la recherche (*Beyond Fresh Starts*,
   Nemotron en agent VAD) montre un WER du 1er mot de 51 % quand chaque clip démarre à
   froid et que le VAD énergie rogne l'attaque. Correctifs : pré-roll **400 ms d'audio
   réel** avant le seuil, silence en tête supprimé, silence en queue **1,0 s** (vidage du
   lookahead Nemotron), rééchantillonnage 8→16 kHz **soxr HQ** (repli ratecv),
   `MIN_SPEECH` 240 ms (pas 500 : ça jetterait « oui »). Diagnostic STT : 4/11
   transcriptions fausses quand le texte est juste, Qwen comprend du 1er coup.
   Si ça ne suffit pas → **plan B** : A/B `faster_whisper` large-v3 (adaptateur dans le
   repo, fr 5,5 % WER vs 9,0 % Nemotron, robuste 8 kHz).
10. **Verrou TTS (2026-10-02)** : un stream abandonné (raccrochage en pleine phrase)
    laissait le verrou CUDA-graph tenu pour toujours → TTS muet (TTFA 67 s… 889 s en
    file). `/synthesize_stream` = générateur async + thread + drapeau d'arrêt, verrou rendu
    en `finally` ; acquisition avec timeout 10 s → 503 (plus 60 s de silence) ;
    `/health` expose `lock_held_s` / `lock_stuck`. Prouvé sur vrai uvicorn + socket coupé.
11. **Questions hors parcours (2026-10-02)** : « est-ce que c'est halal ? », « vous avez une
    terrasse ? » restaient sans réponse. Cause racine PROUVÉE (log + `curl /api/calls/context`) :
    le backend renvoie déjà `attributes` (dietary.halal, equipments.terrace, vegetarian,
    gluten_free, pets_allowed, private_parking, meal_vouchers, price_range, ambiance,
    cuisine_type…) et `backend_restaurant_context._to_context` les JETAIT. 2e cause :
    `is_client_question` court-circuité par « oui, bonjour… » en tête et aveugle à « est ce
    que » sans tiret / tournures orales sans « ? ». Correctif : attributs transmis dans
    `RestaurantContext.attributes` et lus par l'answerer (section « CE QUE PROPOSE LE
    RESTAURANT »), détecteur réécrit (normalisation STT, interjections ignorées, tournures
    fortes + faibles conditionnées à un thème restaurant, 0 appel LLM), réponse inconnue
    courte + log `hikky.questions_sans_reponse`. 279 tests verts. Limites : le menu
    (`restaurant_menus`) n'est pas exposé par `/api/calls/context` ; pas de rappel client tant
    que le backend n'a pas de route de callback.
12. **Émotions TTS, labo mesuré (2026-10-02, 150 synthèses, `/workspace/emotion_lab/`)** :
    Qwen3-TTS suit bien une instruction `instruct` par phrase. Verdicts : (a) `fastS` =
    sous-talker ÉCHANTILLONNÉ dans le graphe (Gumbel-max, `torch.multinomial` non capturable)
    ramène l'expressivité au niveau de la référence pour +2 % de RTF (0,38 vs 1,62) et est
    plus stable que le greedy → classe `SampledPredictorGraph` dans `emotion_lab/lab.py`
    (25 lignes) ; (b) le gain est PAR INTENTION : confirmation enthousiaste (+32 Hz, F0 std
    2,8→4,0 st) et patience, pas sur la demande ; (c) l'instruction FRANÇAISE « enjouée »
    SUR-JOUE et déstabilise l'accueil (50 % de runs 1,5-2,2× trop longs) → pour l'accueil,
    instruction en ANGLAIS (0 run anormal) ; (d) l'excuse doit être PLUS basse/douce (F0 std
    qui baisse = effet voulu) ; (e) aucune limite de prefill : 134 tokens OK (vraie limite
    prefill+160 ≤ 1024) ; (f) l'émotion survit au téléphone (±0,1 st).
    ⚠ `ono_anna` est une voix JAPONAISE selon la fiche officielle (plafond de qualité fr
    indépendant de l'émotion) ; alternative à tester : `vivian`. Plan d'intégration minimal :
    `Req.emotion/instruct` + dict EMOTIONS avec repli = comportement actuel ; flag
    `TTS_SUBTALKER_SAMPLE=1` pour fastS ; orchestrateur : salutation cachée en « accueil »
    + garde durée (>6,5 s → regénérer), re-prompts → patience/excuse, classement par texte
    des réponses fixes (« C'est réservé »→confirmation, « Pardon »→excuse, « pas
    disponible »→indispo, « C'est bien cela ? »→recap). Échantillons : `emotion_lab/best/`.

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
  `run_telnyx.sh` lance les 3 en même temps → la salutation échoue (« cache salutation
  echoue ») : relancer l'orchestrateur seul une fois le TTS prêt.
- **TTS muet / salutation en ReadTimeout alors que `/health` est OK** = verrou bloqué.
  Vérifier `curl localhost:8802/health` → `lock_held_s` qui grimpe / `lock_stuck:true`.
  Diagnostic sans deviner : `kill -USR1 <pid tts>` → piles de tous les threads dans
  `tts.log` (faulthandler). Remède : relancer le TTS par PID, puis l'orchestrateur.
- `runpod_deploy/` est **superposé** au repo lors d'un rebuild (3 fichiers domaine +
  services) : tout correctif dans `src/hikky/domain/{phraseur,conversation_brain,
  routed_turn}.py` doit aussi être recopié dans `runpod_deploy/`, sinon le prochain
  rebuild l'écrase. (Le `routed_turn.py` du repo n'a PAS les raccourcissements récap/
  clôture qui tournent sur le pod — seule la copie `runpod_deploy/` les a.)
- `run_telnyx.sh` de la sauvegarde n'a pas de `PUBLIC_HOST` : l'ajouter à la ligne
  orchestrateur à chaque nouveau pod (`export PUBLIC_HOST=<podid>-19123.proxy.runpod.net`),
  sinon le TeXML pointe sur l'ancien pod.
- Fichiers domaine SANS copie `runpod_deploy/` (viennent du repo au rebuild) :
  `question_router.py`, `restaurant_context.py`, `adapters/back/backend_restaurant_context.py`.
  Ceux AVEC copie (superposée) : `phraseur.py`, `conversation_brain.py`, `routed_turn.py`.

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
