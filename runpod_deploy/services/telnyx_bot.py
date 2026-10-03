"""Orchestrateur bot vocal Telnyx (venv-bot, port 19123).
- POST /telnyx/inbound : TeXML <Connect><Stream> (entree PCMA A-law reelle, sortie PCMU).
- WS /telnyx/stream : protocole Media Streaming Telnyx.
Reutilise le domaine Call-bot (run_routed_turn, reservation backend), LLM Qwen 32B in-process.
Reconstruit 2026-09-16 sur nouveau pod. Voix ona, codec A-law entrant, salutation cachee,
reponses courtes, re-prompt de confirmation."""
import os, sys, json, base64, asyncio, audioop, time, logging
try:  # reechantillonnage 8k->16k de qualite (polyphase). audioop.ratecv = interpolation
    import numpy as _np  # grossiere (et audioop disparait en Python 3.13).
    import soxr as _soxr
except Exception:  # noqa: BLE001 - repli silencieux, l'appel ne doit jamais casser
    _soxr = None
import httpx
from fastapi import FastAPI, Request, WebSocket
from fastapi.responses import Response

sys.path.insert(0, "/workspace/Call-bot/src")
sys.path.insert(0, "/workspace/Call-bot/scripts")

# --- env backend ---
for _line in open("/workspace/bot_back.env"):
    _line = _line.strip()
    if _line and not _line.startswith("#") and "=" in _line:
        _k, _v = _line.split("=", 1)
        os.environ.setdefault(_k, _v)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("telnyx")
conv = logging.getLogger("hikky.conversation")

PUBLIC_HOST = os.environ.get("PUBLIC_HOST", "uv9jklqxm6vwnj-19123.proxy.runpod.net")
STT_URL = "http://127.0.0.1:8801/transcribe"
TTS_URL = "http://127.0.0.1:8802/synthesize"
TTS_STREAM_URL = "http://127.0.0.1:8802/synthesize_stream"
STREAM = bool(os.environ.get("TTS_STREAM"))
RESTO_PHONE = os.environ.get("HIKKY_RESTAURANT_PHONE", "+33472100100")
GGUF = "/workspace/models/Qwen2.5-32B-Instruct-Q5_K_M.gguf"

THRESH = 800
SIL_FRAMES = 35          # 700 ms de silence (hangover) avant de cloturer un tour.
                         # 500 ms coupait 2 énoncés sur 10 en pleine phrase (appel du 2026-10-03 :
                         # « si le restaurant est à l'al… », « j'aimerais savoir si le… »).
MIN_SPEECH = 12          # min ~240 ms de parole : filtre les blips, garde 'oui'/'non'
BARGE_THRESH = 2500      # parole soutenue nettement au-dessus du bruit de ligne
BARGE_MIN_FRAMES = 12    # ~240 ms continus avant de couper le bot

STATE = {}
GREETING = "Bonjour, vous êtes au restaurant Le Petit Sud, je vous écoute."

app = FastAPI()


@app.on_event("startup")
async def _build():
    from hikky.adapters.voice.llama_cpp_llm import LlamaCppLLMAdapter
    from hikky.adapters.back.http_client import BackHttpClient
    from hikky.adapters.back.call_ingest_adapter import CallIngestAdapter
    from hikky.adapters.back.backend_restaurant_context import BackendRestaurantContextAdapter
    from hikky.domain.conversation_brain import QuestionAnswerer
    from hikky.domain.phraseur import Phraseur
    from hikky.pipeline.llm_slot_extractor import LLMSlotExtractor
    from poc_common import build_session_factory

    t = time.time()
    log.info("chargement LLM Qwen 32B...")
    llm = LlamaCppLLMAdapter(model_path=GGUF, n_ctx=4096, n_gpu_layers=-1)
    log.info("LLM charge en %.1fs", time.time() - t)
    client = BackHttpClient(base_url=os.environ["HIKKY_BACK_BASE_URL"],
                            api_key=os.environ.get("HIKKY_BACK_API_KEY", ""))
    STATE["reservation"] = CallIngestAdapter(client, restaurant_phone=RESTO_PHONE)
    STATE["context_port"] = BackendRestaurantContextAdapter(client)
    STATE["session_factory"] = build_session_factory(llm, reservation_port=STATE["reservation"])
    # Base de connaissances du restaurant (PR #3) : passages écrits par le
    # restaurateur pour les questions hors réservation ; les questions sans
    # réponse lui sont remontées. Le port ne lève jamais : backend injoignable
    # = base vide, l'appel continue.
    from hikky.adapters.back.knowledge_adapter import BackendKnowledgeAdapter
    knowledge = BackendKnowledgeAdapter(client, restaurant_phone=RESTO_PHONE)
    STATE["answerer"] = QuestionAnswerer(llm, knowledge=knowledge)
    log.info("base de connaissances branchée (%s)", RESTO_PHONE)
    STATE["phraseur"] = Phraseur(llm)
    STATE["extractor"] = LLMSlotExtractor(llm)
    STATE["http"] = httpx.AsyncClient(timeout=60.0)
    # Cache de la salutation (texte fixe) : synthetisee 1x -> frames mu-law pretes.
    try:
        _gp, _gsr = await synth_pcm(GREETING)
        STATE["greeting_frames"] = pcm_to_ulaw_frames(_gp, _gsr)
        log.info("salutation cachee: %d frames", len(STATE["greeting_frames"]))
    except Exception as _ge:
        STATE["greeting_frames"] = None
        log.warning("cache salutation echoue: %s", _ge)
    # Prechauffage LLM.
    try:
        t = time.time()
        await llm.complete([{"role": "user", "content": "Bonjour"}])
        log.info("LLM prechauffe en %.1fs", time.time() - t)
    except Exception as e:
        log.warning("prechauffage LLM echoue: %s", e)
    log.info("orchestrateur pret")


_STREAMED: dict = {}   # CallSid -> horodatage du <Stream> déjà rendu (dédoublonnage)


@app.post("/telnyx/inbound")
async def inbound(request: Request):
    # Observé en appel réel (2026-10-03) : DEUX POST pour UN appel, à 7 s d'écart (status
    # callback ou nouvelle tentative Telnyx). Chacun recevait un <Connect><Stream> -> deux
    # sessions sur le même appel : deux bots parlant l'un sur l'autre, LLM et TTS partagés
    # (phraseur « trop lent », TTFA 2,3 s), énoncés ignorés. Un seul Stream par CallSid.
    from urllib.parse import parse_qs
    form = {k: v[0] for k, v in parse_qs((await request.body()).decode(errors="replace")).items()}
    sid = form.get("CallSid") or form.get("CallSidLegacy") or ""
    log.info("inbound: CallSid=%s status=%s from=%s champs=%s", sid, form.get("CallStatus"),
             form.get("From"), sorted(form))
    now = time.time()
    for k in [k for k, ts in _STREAMED.items() if now - ts > 3600]:
        _STREAMED.pop(k, None)
    if sid and sid in _STREAMED:
        log.warning("inbound dupliqué pour CallSid=%s -> pas de second Stream", sid)
        return Response(content='<?xml version="1.0" encoding="UTF-8"?><Response/>',
                        media_type="application/xml")
    if sid:
        _STREAMED[sid] = now
    xml = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<Response><Connect><Stream url="wss://' + PUBLIC_HOST + '/telnyx/stream" '
        'bidirectionalMode="rtp" bidirectionalCodec="PCMU"/></Connect></Response>'
    )
    return Response(content=xml, media_type="application/xml")


def _to16k(pcm8k: bytes) -> bytes:
    # Entree STT : 8 kHz telephone -> 16 kHz. soxr HQ si dispo, sinon ratecv.
    if _soxr is not None:
        x = _np.frombuffer(pcm8k, dtype="<i2")
        return _soxr.resample(x, 8000, 16000, quality="HQ").astype("<i2").tobytes()
    out, _ = audioop.ratecv(pcm8k, 2, 1, 8000, 16000, None)
    return out


def pcm_to_ulaw_frames(pcm: bytes, sr: int):
    # 24k -> 8k puis mu-law sortant (Telnyx transcode vers l'appel A-law).
    pcm8, _ = audioop.ratecv(pcm, 2, 1, sr, 8000, None)
    mu = audioop.lin2ulaw(pcm8, 2)
    return [mu[i:i + 160] for i in range(0, len(mu), 160)]


async def synth_pcm(text: str):
    r = await STATE["http"].post(TTS_URL, json={"text": text})
    return r.content, int(r.headers.get("X-Sample-Rate", "24000"))


async def transcribe(pcm16k: bytes) -> str:
    r = await STATE["http"].post(STT_URL, content=pcm16k,
                                 headers={"content-type": "application/octet-stream"})
    return (r.json().get("text") or "").strip()


@app.websocket("/telnyx/stream")
async def stream(ws: WebSocket):
    await ws.accept()
    sess = None
    hist: list = []
    awaiting = False
    phr: set = set()
    speaking = False
    stop_flag = {"v": False}
    stream_id = None
    last_send_ts = 0.0
    alaw = True              # PCMA A-law par defaut (Europe) ; corrige depuis media_format
    utter = bytearray()
    silence = 0
    speech = 0
    bargein = 0
    preroll = bytearray()   # ~400ms d'audio REEL avant l'attaque (le VAD energie rogne le 1er mot)
    PREROLL_BYTES = 6400    # 0.4s @ 8kHz PCM16
    # Un seul tour à la fois : deux énoncés rapprochés (STT à froid, phrase coupée) lançaient
    # deux do_turn concurrents -> deux réponses parlées l'une sur l'autre.
    turn_lock = asyncio.Lock()
    call_tag = time.strftime("%H%M%S")
    utter_n = 0
    n_incompris = 0         # énoncés vides consécutifs (remis à zéro dès qu'on comprend)

    async def send_ulaw(frames):
        # Telnyx bufferise et joue lui-meme ; 1 message media / seconde max.
        # On envoie tout l enonce en UN message. Pas de stream_id en sortie.
        nonlocal speaking, last_send_ts
        if not frames:
            return
        speaking = True
        stop_flag["v"] = False
        try:
            loop = asyncio.get_event_loop()
            gap = 1.05 - (loop.time() - last_send_ts)
            if gap > 0:
                await asyncio.sleep(gap)
            if stop_flag["v"]:
                return
            payload = b"".join(frames)
            await ws.send_text(json.dumps({
                "event": "media",
                "media": {"payload": base64.b64encode(payload).decode()},
            }))
            last_send_ts = loop.time()
            end = last_send_ts + len(frames) * 0.02
            while loop.time() < end and not stop_flag["v"]:
                await asyncio.sleep(0.05)
        finally:
            speaking = False

    async def speak_stream(text: str):
        # Lit /synthesize_stream (framing longueur-prefixee) et envoie a Telnyx
        # au fil de l eau. Reechantillonnage a etat continu (pas de clic entre chunks).
        rate_state = None
        sent_any = False
        t0 = time.time()
        n_frames = 0
        try:
            async with STATE["http"].stream("POST", TTS_STREAM_URL, json={"text": text}) as resp:
                buf = bytearray()
                async for data in resp.aiter_bytes():
                    buf.extend(data)
                    while len(buf) >= 4:
                        ln = int.from_bytes(buf[:4], "big")
                        if len(buf) < 4 + ln:
                            break
                        chunk = bytes(buf[4:4 + ln]); del buf[:4 + ln]
                        if stop_flag["v"]:
                            return
                        pcm8, rate_state = audioop.ratecv(chunk, 2, 1, 24000, 8000, rate_state)
                        mu = audioop.lin2ulaw(pcm8, 2)
                        frames = [mu[i:i + 160] for i in range(0, len(mu), 160)]
                        if frames and not stop_flag["v"]:
                            sent_any = True
                            n_frames += len(frames)
                            await send_ulaw(frames)
        except Exception as e:
            log.warning("stream TTS echoue (%s)%s", e,
                        "" if sent_any else " -> fallback synth complet")
            if not sent_any and not stop_flag["v"]:
                pcm, sr = await synth_pcm(text)
                if not stop_flag["v"]:
                    await send_ulaw(pcm_to_ulaw_frames(pcm, sr))
        finally:
            log.info("stream fini: %d frames (%.1fs audio) en %.1fs%s", n_frames,
                     n_frames * 0.02, time.time() - t0,
                     " COUPE (barge-in)" if stop_flag["v"] else "")
        return n_frames

    async def speak(text: str):
        if not text:
            return
        conv.info("BOT    : %s", text)
        if STREAM:
            n_frames = await speak_stream(text)
            if stop_flag["v"] and n_frames is not None:
                # Coupé par le client : l'historique ne doit contenir que ce qu'il a
                # ENTENDU. Sinon le modèle croit avoir dit des choses jamais prononcées
                # et fabrique lui-même les malentendus suivants. ~14 caractères/s.
                entendu = int(n_frames * 0.02 * 14)
                if hist and hist[-1].get("role") == "assistant" and hist[-1].get("content") == text \
                        and entendu < len(text):
                    hist[-1]["content"] = text[:entendu].rstrip() + "… (coupé par le client)"
                    log.info("historique tronqué à %d caractères (barge-in)", entendu)
            return
        pcm, sr = await synth_pcm(text)
        if stop_flag["v"]:
            return
        await send_ulaw(pcm_to_ulaw_frames(pcm, sr))

    def _dump(buf: bytes, n: int) -> str:
        # Chaque énoncé en WAV 8 kHz pour écoute/mesure hors ligne (/workspace/debug).
        # C'est la seule façon de trancher entre « STT sourd » et « VAD qui coupe ».
        try:
            import wave
            os.makedirs("/workspace/debug", exist_ok=True)
            path = f"/workspace/debug/{call_tag}_{n:02d}.wav"
            with wave.open(path, "wb") as w:
                w.setnchannels(1); w.setsampwidth(2); w.setframerate(8000)
                w.writeframes(buf)
            return path
        except Exception as e:  # noqa: BLE001 - le debug ne doit jamais casser l'appel
            return f"(dump échoué: {e})"

    async def do_turn(buf: bytes):
        nonlocal awaiting, utter_n
        utter_n += 1
        n = utter_n
        path = _dump(buf, n)
        log.info("énoncé #%d : %.2fs, rms=%d, max=%d -> %s", n, len(buf) / 16000,
                 audioop.rms(buf, 2), audioop.max(buf, 2), path)
        async with turn_lock:
            await _do_turn_locked(buf)

    async def _incompris(n: int):
        # Réparation mesurée (Bohus & Rudnicky 2005, 8 278 tours) : « pouvez-vous
        # répéter ? » récupère 33,7 % des cas, AVANCER sur une autre question 64,4 %.
        # Et après deux échecs, un troisième reprompt identique ne sert à rien :
        # on clôt poliment (règle « 3 no-match → humain » de Google).
        from hikky.domain.question_router import SLOT_ORDER, phrase_for_slot
        from hikky.domain.routed_turn import build_recap
        if n >= 3:
            await speak("Je suis désolée, je vous entends très mal et je ne voudrais pas "
                        "noter une réservation erronée. N'hésitez pas à rappeler, l'équipe "
                        "du restaurant se fera un plaisir de vous répondre. Bonne journée !")
            try:
                await ws.close()
            except Exception:  # noqa: BLE001
                pass
            return
        prefixe = "Je vous entends mal. " if n == 1 else "La ligne est mauvaise, je reprends. "
        if awaiting:
            await speak(prefixe + "Vous confirmez la réservation ? Dites oui, ou non.")
            return
        manquants = sess.intent.missing_slots()
        slot = next((s for s in SLOT_ORDER if s in manquants), None)
        if slot is None:
            await speak(prefixe + build_recap(sess.intent))
            return
        phrase = phrase_for_slot(slot, phr)
        phr.add(phrase)
        await speak(prefixe + phrase)

    async def _do_turn_locked(buf: bytes):
        nonlocal awaiting, n_incompris
        buf16 = _to16k(buf)
        # Silence en QUEUE seulement : vide le contexte droit (lookahead) de Nemotron.
        # En tete c'est inutile - l'attaque est couverte par le pre-roll d'audio reel.
        buf16 = buf16 + b"\x00" * (16000 * 2)   # 1.0s @ 16kHz
        text = await transcribe(buf16)
        if not text:
            # STT vide : ne jamais rester muet -> réparer (avancer, pas répéter).
            n_incompris += 1
            conv.info("CLIENT : (inaudible) [%d]", n_incompris)
            await _incompris(n_incompris)
            return
        n_incompris = 0
        # Correction phonétique sur le lexique métier (« à l'al » -> « halal »).
        from hikky.pipeline.stt_correction import corriger_transcription
        text, changements = corriger_transcription(text, detail=True)
        if changements:
            log.info("STT corrigé : %s", changements)
        conv.info("CLIENT : %s", text)
        from hikky.domain.routed_turn import run_routed_turn
        out = await run_routed_turn(
            session=sess, user_text=text, customer_phone=None,
            history=hist, extractor=STATE["extractor"], answerer=STATE["answerer"],
            awaiting_confirmation=awaiting, speak=speak,
            recent_phrasings=phr, phraseur=STATE["phraseur"],
        )
        awaiting = out.awaiting_confirmation

    while True:
        try:
            raw = await ws.receive_text()
        except Exception:
            break
        msg = json.loads(raw)
        ev = msg.get("event")
        if ev == "start":
            stream_id = msg.get("stream_id") or msg.get("start", {}).get("stream_id")
            _enc = (msg.get("start", {}).get("media_format", {}) or {}).get("encoding", "PCMA")
            alaw = (_enc != "PCMU")
            log.info("CODEC entrant=%s -> alaw=%s", _enc, alaw)
            ctx = await STATE["context_port"].load(RESTO_PHONE)
            sess = STATE["session_factory"]("telnyx-" + str(stream_id), ctx)
            await sess.begin()
            log.info("appel demarre stream=%s", stream_id)
            if STATE.get("greeting_frames"):
                asyncio.create_task(send_ulaw(STATE["greeting_frames"]))
            else:
                asyncio.create_task(speak(GREETING))
        elif ev == "media" and sess is not None:
            _raw = base64.b64decode(msg["media"]["payload"])
            pcm = audioop.alaw2lin(_raw, 2) if alaw else audioop.ulaw2lin(_raw, 2)
            rms = audioop.rms(pcm, 2)
            if speaking and rms > BARGE_THRESH:
                bargein += 1
                if bargein >= BARGE_MIN_FRAMES:
                    log.info("barge-in : le client parle (rms=%d), bot coupé", rms)
                    stop_flag["v"] = True
                    await ws.send_text(json.dumps({"event": "clear"}))
                    bargein = 0
            else:
                bargein = 0
            if rms >= THRESH:
                if speech == 0:
                    utter += preroll        # attaque : injecter le pre-roll
                utter += pcm
                speech += 1
                silence = 0
            elif speech > 0:
                utter += pcm
                silence += 1
                if silence >= SIL_FRAMES:
                    buf = bytes(utter)
                    n = speech
                    utter = bytearray()
                    speech = 0
                    silence = 0
                    if n >= MIN_SPEECH and not speaking:
                        asyncio.create_task(do_turn(buf))
                    elif n >= MIN_SPEECH:
                        # Le client a parlé pendant que le bot parlait, sans franchir le seuil
                        # de barge-in : énoncé perdu. Tracé pour mesurer la fréquence du cas.
                        log.info("énoncé ignoré (bot en train de parler) : %d frames, %.1fs",
                                 n, len(buf) / 16000)
            preroll += pcm
            if len(preroll) > PREROLL_BYTES:
                del preroll[:len(preroll) - PREROLL_BYTES]
        elif ev == "stop":
            log.info("appel termine stream=%s", stream_id)
            break


@app.get("/health")
def health():
    return {"ok": STATE.get("session_factory") is not None}
