# Pipeline téléphonique temps réel — carte de lecture

Un client appelle le restaurant ; Telnyx nous passe sa voix par paquets de 20 ms ;
on écoute, on comprend, on répond en flux, et la réservation part au backend.
Trois diagrammes pour lire le code en dix minutes. Les frontières de **ce module**
sont en gras : ce qui entre, c'est la voix du client ; ce qui sort, c'est la voix du
bot et les appels aux services voisins (STT, TTS, LLM/domaine, backend).

## 1. Un appel, de bout en bout (séquence)

```mermaid
sequenceDiagram
    autonumber
    actor Client as Client (téléphone)
    participant Telnyx
    participant Server as server.py<br/>(inbound / WebSocket)
    participant Detector as turn_detector.py
    participant STT as STT :8801<br/>(faster-whisper)
    participant Domain as domaine hikky<br/>(routed_turn)
    participant Speaker as speaker.py
    participant TTS as TTS :8802<br/>(Qwen3-TTS, flux)
    participant Back as Backend<br/>(réservations, contexte)

    Client->>Telnyx: compose le numéro
    Telnyx->>Server: POST /telnyx/inbound (CallSid)
    Server-->>Telnyx: TeXML <Connect><Stream wss://…/telnyx/stream PCMU>
    Note over Server: inbound.py : un seul Stream par CallSid
    Telnyx->>Server: WS « start » (codec PCMA)
    Server->>Back: GET /api/calls/context
    Back-->>Server: horaires, attributs, règles
    Server->>Speaker: salutation (trames en cache)
    Speaker-->>Telnyx: « media » (µ-law 8 kHz)
    Telnyx-->>Client: « Bonjour, vous êtes au restaurant… »

    loop chaque paquet de 20 ms
        Telnyx->>Server: WS « media » (A-law base64)
        Server->>Detector: feed(pcm, bot_speaking)
    end
    Detector-->>Server: Utterance (pré-roll + phrase) après 700 ms de silence
    Server->>STT: POST /transcribe (16 kHz + 1 s de silence)
    STT-->>Server: texte
    Note over Server: stt_correction : « à l'al » → « halal »
    Server->>Domain: run_routed_turn(texte, état, speak=speaker.say)
    Domain->>Back: GET /availability, POST réservation (si confirmée)
    Domain-->>Server: réponse (texte) via speak()
    Server->>Speaker: say(texte)
    Speaker->>TTS: POST /synthesize_stream
    TTS-->>Speaker: morceaux PCM 24 kHz (taille sur 4 octets + données)
    Speaker-->>Telnyx: « media » toutes les ~1,05 s, trames de 160 octets
    Telnyx-->>Client: la voix du bot (premier son ≈ 0,65 s)

    opt le client coupe le bot (barge-in)
        Detector-->>Server: BargeIn (rms > 2500 pendant 240 ms)
        Server->>Speaker: interrupt()
        Speaker-->>Telnyx: « clear » (vide la file audio)
        Note over Speaker: l'historique ne garde que ce qui a été entendu
    end

    opt rien compris (3 fois)
        Server->>Speaker: say(repair_reply) — avancer, reformuler, puis clore
        Server-->>Telnyx: ws.close()
    end

    Domain-->>Server: should_end (réservation faite / au revoir)
    Server-->>Telnyx: ws.close()
    Telnyx-->>Client: fin d'appel
```

## 2. Les pièces et qui appelle qui (composants)

```mermaid
flowchart LR
    classDef mine fill:#eaf2fb,stroke:#0b5fa5,stroke-width:2px,color:#1c2230
    classDef ext fill:#f3f3f3,stroke:#7a8396,color:#1c2230,stroke-dasharray: 4 3
    classDef svc fill:#fff4e8,stroke:#c2410c,color:#1c2230

    Client([Client au téléphone]):::ext
    Telnyx[Telnyx<br/>numéro, TeXML, flux WebSocket]:::ext

    subgraph POD["Pod GPU (A40) — /workspace/services"]
        direction LR
        subgraph MINE["telnyx_pipeline/ — mon module (port 19123)"]
            direction TB
            server[server.py<br/>3 points d'entrée + boucle de l'appel]:::mine
            inbound[inbound.py<br/>TeXML, un Stream par CallSid]:::mine
            detector[turn_detector.py<br/>parle / fini / coupe le bot]:::mine
            state[call_state.py<br/>l'état d'un appel]:::mine
            speaker[speaker.py<br/>flux TTS → trames → Telnyx]:::mine
            repair[repair.py<br/>avancer, reformuler, clore]:::mine
            audio[audio.py<br/>codecs, cadences, trames]:::mine
            server --> inbound
            server --> detector
            server --> state
            server --> speaker
            server --> repair
            detector --> audio
            speaker --> audio
            speaker --> state
        end
        STT[stt_server.py :8801<br/>faster-whisper large-v3]:::svc
        TTS[tts_server.py :8802<br/>Qwen3-TTS + fast_tts.py]:::svc
        LLM[(Qwen2.5-32B<br/>llama.cpp, in-process)]:::svc
        Domain[domaine hikky<br/>routed_turn, question_router,<br/>stt_correction]:::svc
    end

    Back[Backend Spring Boot<br/>contexte, disponibilité,<br/>réservation, connaissances]:::ext
    Discord[Discord<br/>Alfred / Jarvis]:::ext

    Client <-->|voix 8 kHz| Telnyx
    Telnyx -->|POST /telnyx/inbound| inbound
    Telnyx <-->|WS /telnyx/stream<br/>media A-law ⇄ µ-law| server
    server -->|PCM 16 kHz| STT
    speaker -->|texte| TTS
    server -->|texte transcrit| Domain
    Domain --> LLM
    Domain -->|HTTP + clé API| Back
    server -->|contexte au début de l'appel| Back
    Back --> Discord
```

## 3. La vie d'un appel (états)

```mermaid
stateDiagram-v2
    [*] --> Sonne : POST /telnyx/inbound
    Sonne --> Salutation : WS start, contexte chargé
    Salutation --> Ecoute : salutation jouée

    state Ecoute {
        [*] --> Silence
        Silence --> Parole : rms ≥ 800
        Parole --> Parole : voix
        Parole --> Silence : 700 ms de silence → Utterance
    }

    Ecoute --> Transcription : Utterance (≥ 240 ms de parole)
    Transcription --> Reparation : texte vide
    Transcription --> Decision : texte (corrigé)
    Decision --> Parle : réponse du domaine
    Parle --> Ecoute : flux terminé
    Parle --> Interrompu : client parle fort 240 ms (barge-in)
    Interrompu --> Ecoute : « clear », historique tronqué
    Parle --> EnonceGarde : le client parle par-dessus (≥ 480 ms)
    EnonceGarde --> Transcription : bot fini, < 6 s, pas une répétition
    EnonceGarde --> Ecoute : périmé ou répétition → ignoré

    Reparation --> Parle : 1er échec « j'avance » / 2e « je reprends »
    Reparation --> Fin : 3e échec → au revoir poli
    Decision --> Fin : réservation créée ou au revoir (should_end)
    Fin --> [*] : ws.close()
```

### Où est chaque constante
| Constante | Valeur | Fichier | Pourquoi |
|---|---|---|---|
| seuil de parole | 800 | `turn_detector.py` | bruit de ligne + marge, lu dans les logs |
| fin de phrase | 35 trames = 700 ms | `turn_detector.py` | 500 ms coupait 2 phrases sur 10 |
| parole minimale | 12 trames = 240 ms | `turn_detector.py` | filtre les blips, garde « oui » |
| barge-in | rms > 2500 pendant 12 trames | `turn_detector.py` | un « mmh » ne coupe pas le bot |
| pré-roll | 6 400 octets = 400 ms | `turn_detector.py` | la première syllabe n'est plus rognée |
| énoncé gardé périmé | 6 s | `call_state.py` | le client a déjà avancé |
| espacement des envois | 1,05 s | `speaker.py` | contrainte Telnyx (1 message/s) |
| trame | 160 octets µ-law = 20 ms | `audio.py` | protocole Telnyx |
| échecs avant clôture | 3 | `repair.py` | règle « 3 no-match → humain » |

### Configuration : tout vient de l'environnement (`config.py`)
Aucune adresse ni chemin n'est écrit dans le code : `Settings.from_env()` lit ces variables
au démarrage, et une variable obligatoire absente arrête le processus avec son nom.

| Variable | Défaut | Rôle |
|---|---|---|
| `PUBLIC_HOST` | **obligatoire** | hôte public du pod, mis dans le TeXML rendu à Telnyx |
| `HIKKY_BACK_BASE_URL` | **obligatoire** | URL du backend (réservations, base de connaissances) |
| `HIKKY_BACK_API_KEY` | vide | clé API du backend |
| `TTS_STREAM` | vide (= synthèse complète) | `1` = voix envoyée en flux |
| `HIKKY_STT_URL` | `http://127.0.0.1:8801/transcribe` | service de reconnaissance vocale |
| `HIKKY_TTS_URL` | `http://127.0.0.1:8802/synthesize` | synthèse complète (repli, salutation) |
| `HIKKY_TTS_STREAM_URL` | `http://127.0.0.1:8802/synthesize_stream` | synthèse en flux |
| `HIKKY_RESTAURANT_PHONE` | `+33472100100` | numéro du restaurant (identifie la session) |
| `HIKKY_LLM_GGUF` | `/workspace/models/Qwen2.5-32B-Instruct-Q5_K_M.gguf` | modèle de langue |
| `HIKKY_GREETING` | « Bonjour, vous êtes au restaurant Le Petit Sud, je vous écoute. » | première phrase |
| `HIKKY_DEBUG_DIR` | `/workspace/debug` | WAV de chaque énoncé ; vide = pas d'enregistrement |
| `HIKKY_SRC_DIRS` | `/workspace/Call-bot/src:/workspace/Call-bot/scripts` | où trouver `hikky` (séparés par `:`) |
| `HIKKY_ENV_FILE` | `/workspace/bot_back.env` | fichier `KEY=VALUE` chargé au démarrage, sans écraser l'existant |
| `HIKKY_TIMEZONE` | `Europe/Paris` | fuseau des dates dites au client |

Tests : `tests/unit/telnyx_pipeline/` (36 tests, sans réseau ni GPU).
