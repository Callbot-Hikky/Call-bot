# Hikky callbot — état & reprise (sauvegarde 2026-09-17)

Sauvegarde de tout le code déployé sur le pod RunPod. **Rien d'irremplaçable n'est
sur le pod** : ce dossier contient les fichiers exacts qui tournent. Les modèles
et venvs sont volumineux mais reproductibles (voir "Rebuild from scratch").

## Accès au pod
- **AUCUN POD (2026-10-02)** : `hikky-bot-4` (`4syj7wqd2o39x8`) et `hikky-bot-5` (`8b10kksz3ra5iw`)
  SUPPRIMÉS pour couper la facturation (un pod arrêté facture encore ~24 $/mois de volume).
  Prochaine session = « Rebuild from scratch » ci-dessous, puis mettre à jour ce bloc
  (id, IP/port SSH, `PUBLIC_HOST` dans `run_telnyx.sh` — il vaut encore
  `8b10kksz3ra5iw-19123.proxy.runpod.net`, OBSOLÈTE).
- Perdu avec le pod : `/workspace/emotion_lab/` et `/workspace/voice_lab/` (scripts et WAV).
  Les conclusions sont dans les items 12 et 14 ; à regénérer (~10 min) si l'écoute aveugle
  des voix (vivian / ryan / ono_anna) est encore voulue. Leçon : versionner les labos
  dans `runpod_deploy/labs/` dès qu'ils produisent un résultat.
- Dernier pod (`8b10kksz3ra5iw`) pour mémoire : A40 48 Go, image `runpod/pytorch:2.4.0-py3.11-cuda12.4.1`,
  SSH `ssh -i ~/.ssh/id_ed25519 root@194.68.245.239 -p 22145`, proxy `https://8b10kksz3ra5iw-19123.proxy.runpod.net`.
  Note : sshd peut mettre ~4-5 min à répondre après le démarrage d'un pod neuf.
- Un pod arrêté ne redémarre souvent PAS (« not enough free GPUs on the host ») → l'interface
  web permet un démarrage à 0 GPU pour récupérer le volume avant suppression.
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
13. **Plus de relance commerciale systématique (2026-10-02)** : « à chaque fin de réponse l'IA
    me casse la tête pour réserver ». `_ensure_progress` (routed_turn.py) collait d'office la
    question du créneau manquant après CHAQUE réponse hors parcours. Règles désormais :
    aucun créneau connu → on répond, point ; réservation en cours → on reprend sur ce qui
    manque, mais jamais deux fois de suite (si notre phrase précédente était déjà une
    question). Appliqué aux 2 copies de routed_turn.py (repo + runpod_deploy). 90 tests.
14. **Recherche voix & ton téléphone (2026-10-02, sourcée)** : AUCUNE des 9 voix CustomVoice
    n'est française native (5 zh, 2 en, 1 ja, 1 ko) → plafond d'accent structurel. Seules
    `ryan` (homme, en) et `vivian` (femme, zh) ont des retours francophones « tient le
    français d'un bout à l'autre » ; `serena` DÉRIVE vers l'anglais (cross-lingual, elle est
    chinoise) ; `ono_anna` : zéro retour positif en fr, un négatif → à retirer. Reco : ryan
    n°1, vivian n°2 (la chaleur perçue compte plus que le genre). TON AU TÉLÉPHONE : la
    bande étroite dégrade surtout la JOIE (80 %→58 % reconnue) → sobriété, pas d'enthousiasme
    marqué ; confirmation « chaleureuse et sûre, posée », intonation DESCENDANTE (crédibilité) ;
    excuses basses/lentes en assumant ; registre fr = retenue. `subtalker_dosample=True` est
    le RÉGLAGE OFFICIEL Qwen (greedy = « trames quasi-silencieuses répétées », non supporté)
    → garder fastS, borner la génération. Vraie voix fr native = FINE-TUNING d'un speaker
    CustomVoice (30-60 min audio consenti/LibriVox, lr 2e-6) qui GARDE `instruct` ; le clonage
    zero-shot Base perd `instruct` et est instable → pas pour la prod. Légal : consentement
    écrit (RGPD art. 9, C. pén. 226-8-1), marquage AI Act art. 50. Test décisif : 4 voix × 8
    phrases, rendu téléphone, WER via NOTRE STT + écoute aveugle → `/workspace/voice_lab/`.
    RÉSULTAT (64 synthèses fastS, instructions sobres, juge = STT prod :8801 sur audio tél.) :
    ryan WER 0,060 / 0 sur-gén / 4,1 st ; ono_anna 0,063 / 0 / 2,9 st (LA PLUS PLATE, −15 %
    au tél.) ; vivian 0,074 / 1 sur-gén / 4,4 st (LA PLUS EXPRESSIVE) ; serena 0,200 (rejetée
    — l'oreille du client avait raison). 0 dérive EN avec language=French. ono_anna n'est pas
    mal comprise : son défaut est la platitude. Choix = persona : féminine → vivian (+ garde-fou
    durée), sinon ryan. Les 7 instructions SOBRES EN donnent de meilleurs WER que les
    « enthousiastes » → les utiliser comme table EMOTIONS par défaut. Si la voix change :
    champ `speaker` optionnel sur l'API TTS + salutation cachée régénérée. Échantillons tél. :
    `/workspace/voice_lab/best/<voix>__<intention>_tel.wav`. (Perdus avec le pod, voir « Accès au pod ».)
15. **Base de connaissances déployée sur staging (2026-10-02, serveur 51.15.210.223)** :
    PR back #26 (`b82bc4ad`, Flyway V30 `knowledge_base_entries` + `vector(1024)` HNSW) et
    front #19 (`48f5958f`) fusionnées dans `staging` UNIQUEMENT (`main` intact). Côté serveur,
    la stack LIVE est `/opt/hikky` (`hikky-*`, DB `hikky-postgres-1`, 60 réservations) — PAS
    `callbot-*` (ancienne stack, 0 réservation). Postgres passé en `callbot-postgres:16-pgvector`
    (dumps `~/backup-avant-pgvector*.sql`, compose `docker-compose.yml.bak-avant-pgvector`).
    Ajouté à la main dans `/opt/hikky/docker-compose.yml` : `VOYAGE_API_KEY: ${VOYAGE_API_KEY:-}`
    sous `backend.environment` (sauvegarde `.bak-avant-voyage`) — le compose du repo ne pilote
    PAS cette stack. Secret GitHub `VOYAGE_API_KEY` posé (back démarre « Embeddings: Voyage AI »).
    Pièges : (a) `deploy.sh` lancé en root rend `images.env` root → le CI (`deploy`) échoue
    « Permission denied » → `chown deploy:deploy /opt/hikky/*` ; (b) `gh secret set` sans
    `--body` depuis le shell `!` enregistre une valeur VIDE ; (c) le workflow FRONT réécrit
    `/opt/hikky/.env` sans `VOYAGE_API_KEY` ni webhooks Discord (bug latent signalé à l'équipe) ;
    (d) la clé Voyage a transité en clair → à faire tourner.

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
