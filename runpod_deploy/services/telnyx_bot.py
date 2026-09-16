"""Orchestrateur bot vocal Telnyx (venv-bot, port 19123).
- POST /telnyx/inbound : TeXML <Connect><Stream> (entree PCMA A-law reelle, sortie PCMU).
- WS /telnyx/stream : protocole Media Streaming Telnyx.
Reutilise le domaine Call-bot (run_routed_turn, reservation backend), LLM Qwen 32B in-process.
Reconstruit 2026-09-16 sur nouveau pod. Voix ona, codec A-law entrant, salutation cachee,
reponses courtes, re-prompt de confirmation."""
import os, sys, json, base64, asyncio, audioop, time, logging
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
SIL_FRAMES = 25          # 800 ms de silence avant de cloturer un tour
MIN_SPEECH = 8           # min ~160 ms de parole pour un tour
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
    STATE["answerer"] = QuestionAnswerer(llm)
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


@app.post("/telnyx/inbound")
async def inbound(_: Request):
    xml = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<Response><Connect><Stream url="wss://' + PUBLIC_HOST + '/telnyx/stream" '
        'bidirectionalMode="rtp" bidirectionalCodec="PCMU"/></Connect></Response>'
    )
    return Response(content=xml, media_type="application/xml")


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
                            await send_ulaw(frames)
        except Exception as e:
            log.warning("stream TTS echoue (%s)%s", e,
                        "" if sent_any else " -> fallback synth complet")
            if not sent_any and not stop_flag["v"]:
                pcm, sr = await synth_pcm(text)
                if not stop_flag["v"]:
                    await send_ulaw(pcm_to_ulaw_frames(pcm, sr))

    async def speak(text: str):
        if not text:
            return
        conv.info("BOT    : %s", text)
        if STREAM:
            await speak_stream(text)
            return
        pcm, sr = await synth_pcm(text)
        if stop_flag["v"]:
            return
        await send_ulaw(pcm_to_ulaw_frames(pcm, sr))

    async def do_turn(buf: bytes):
        nonlocal awaiting
        buf16, _ = audioop.ratecv(buf, 2, 1, 8000, 16000, None)
        text = await transcribe(buf16)
        if not text:
            # STT vide : ne jamais rester muet -> redemander (adapte au contexte).
            conv.info("CLIENT : (inaudible)")
            if awaiting:
                await speak("Pardon, je n'ai pas bien saisi. Vous confirmez la réservation ? Dites oui, ou non.")
            else:
                await speak("Pardon, je n'ai pas compris, pouvez-vous répéter ?")
            return
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
                    stop_flag["v"] = True
                    await ws.send_text(json.dumps({"event": "clear"}))
                    bargein = 0
            else:
                bargein = 0
            if rms >= THRESH:
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
        elif ev == "stop":
            log.info("appel termine stream=%s", stream_id)
            break


@app.get("/health")
def health():
    return {"ok": STATE.get("session_factory") is not None}
