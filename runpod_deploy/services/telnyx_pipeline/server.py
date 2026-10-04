"""Les trois points d'entrée, et la boucle d'un appel.

    POST /telnyx/inbound   Telnyx : « ça sonne » -> on rend la consigne TeXML (inbound.py)
    WS   /telnyx/stream    le flux de l'appel : paquets de 20 ms dans les deux sens
    GET  /health           les services voisins sont-ils prêts ?

La boucle de l'appel (`handle_call`) ne fait que brancher les pièces : le
détecteur de parole dit « phrase finie », le tour de dialogue (domaine `hikky`)
décide quoi répondre, le haut-parleur (`speaker.py`) le prononce.

Lancement : `uvicorn telnyx_pipeline.server:app` depuis /workspace/services
(voir run_telnyx.sh). Toute la configuration vient de l'environnement : voir config.py.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import sys
import time

# Le pod est en UTC ; les dates du dialogue (« aujourd'hui », « demain ») sont celles
# du restaurant. À 0 h 29 à Paris un dimanche, le bot croyait être samedi.
os.environ.setdefault("TZ", os.environ.get("HIKKY_TIMEZONE", "Europe/Paris"))  # voir config.py
time.tzset()

from fastapi import FastAPI, Request, WebSocket  # noqa: E402
from fastapi.responses import Response  # noqa: E402

from . import audio  # noqa: E402
from .call_state import CallState  # noqa: E402
from .config import Settings, load_env_file  # noqa: E402
from .inbound import EMPTY_TEXML, StreamRegistry, parse_form, texml_connect  # noqa: E402
from .repair import repair_reply  # noqa: E402
from .speaker import TelnyxSpeaker  # noqa: E402
from .startup import make_lifespan  # noqa: E402
from .turn_detector import BargeIn, TurnDetector, Utterance  # noqa: E402

# ── configuration : tout vient de l'environnement (voir config.py) ─────────────
load_env_file(os.environ.get("HIKKY_ENV_FILE", "/workspace/bot_back.env"))
settings = Settings.from_env()
for _dir in settings.src_dirs:  # rend importables le domaine `hikky` et `poc_common`
    sys.path.insert(0, _dir)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("telnyx")
conv = logging.getLogger("hikky.conversation")

# Une fin de phrase nette pour le STT : la dernière syllabe n'est jamais coupée.
STT_TRAILING_SILENCE_S = 1.0

STATE: dict = {}
STREAMS = StreamRegistry()


app = FastAPI(lifespan=make_lifespan(settings, STATE))


# ── ça sonne ──────────────────────────────────────────────────────────────────
@app.post("/telnyx/inbound")
async def inbound(request: Request) -> Response:
    form = parse_form(await request.body())
    call_sid = form.get("CallSid") or form.get("CallSidLegacy") or ""
    log.info(
        "inbound: CallSid=%s status=%s from=%s champs=%s",
        call_sid,
        form.get("CallStatus"),
        form.get("From"),
        sorted(form),
    )
    if not STREAMS.accept(call_sid):
        log.warning("inbound dupliqué pour CallSid=%s -> pas de second Stream", call_sid)
        return Response(content=EMPTY_TEXML, media_type="application/xml")
    return Response(content=texml_connect(settings.public_host), media_type="application/xml")


@app.get("/health")
def health() -> dict:
    return {"ok": STATE.get("session_factory") is not None}


# ── l'appel ───────────────────────────────────────────────────────────────────
async def transcribe(pcm8k: bytes) -> str:
    """8 kHz -> 16 kHz + 1 s de silence en queue, puis le service STT."""
    pcm16k = audio.to_stt_rate(pcm8k) + b"\x00" * int(audio.STT_RATE * 2 * STT_TRAILING_SILENCE_S)
    r = await STATE["http"].post(
        settings.stt_url, content=pcm16k, headers={"content-type": "application/octet-stream"}
    )
    return (r.json().get("text") or "").strip()


@app.websocket("/telnyx/stream")
async def stream(ws: WebSocket) -> None:
    await ws.accept()
    await handle_call(ws)


async def handle_call(ws: WebSocket) -> None:
    state = CallState()
    detector = TurnDetector()
    speaker = TelnyxSpeaker(
        ws=ws,
        http=STATE["http"],
        state=state,
        tts_url=settings.tts_url,
        tts_stream_url=settings.tts_stream_url,
        streaming=settings.streaming,
    )
    turn_lock = asyncio.Lock()  # un seul tour à la fois : jamais deux réponses l'une sur l'autre

    async def on_utterance(pcm: bytes, *, replayed: bool = False) -> None:
        state.utterances += 1
        path = _dump(state, pcm)
        log.info(
            "énoncé #%d%s : %.2fs, rms=%d, max=%d -> %s",
            state.utterances,
            " (rejoué)" if replayed else "",
            len(pcm) / (audio.TELEPHONY_RATE * 2),
            audio.rms(pcm),
            audio.peak(pcm),
            path,
        )
        async with turn_lock:
            await run_turn(state, speaker, ws, pcm, replayed=replayed)
        # Ce que le client a dit pendant que le bot parlait est traité maintenant —
        # sauf s'il est périmé (plus de 6 s) ou s'il répète la question précédente.
        if not state.speaking:
            kept = state.pop_pending()
            if kept is not None:
                asyncio.create_task(on_utterance(kept, replayed=True))

    while True:
        try:
            raw = await ws.receive_text()
        except Exception:  # noqa: BLE001 — le client a raccroché
            break
        msg = json.loads(raw)
        event = msg.get("event")

        if event == "start":
            state.stream_id = msg.get("stream_id") or msg.get("start", {}).get("stream_id")
            encoding = (msg.get("start", {}).get("media_format", {}) or {}).get("encoding", "PCMA")
            state.alaw = encoding != "PCMU"
            log.info("CODEC entrant=%s -> alaw=%s", encoding, state.alaw)
            context = await STATE["context_port"].load(settings.restaurant_phone)
            state.session = STATE["session_factory"]("telnyx-" + str(state.stream_id), context)
            await state.session.begin()
            log.info("appel démarré stream=%s", state.stream_id)
            if STATE.get("greeting_frames"):
                asyncio.create_task(speaker.send_frames(STATE["greeting_frames"]))
            else:
                asyncio.create_task(speaker.say(settings.greeting))

        elif event == "media" and state.session is not None:
            pcm = audio.decode_telephony(base64.b64decode(msg["media"]["payload"]), alaw=state.alaw)
            for event in detector.feed(pcm, bot_speaking=state.speaking):
                if isinstance(event, BargeIn):
                    log.info("barge-in : le client parle (rms=%d), bot coupé", event.rms)
                    await speaker.interrupt()
                elif isinstance(event, Utterance) and event.during_bot_speech:
                    log.info(
                        "énoncé pendant la parole du bot : %d frames -> gardé", event.speech_frames
                    )
                    state.keep_pending(event.pcm)
                elif isinstance(event, Utterance):
                    asyncio.create_task(on_utterance(event.pcm))

        elif event == "stop":
            log.info("appel terminé stream=%s", state.stream_id)
            break


async def run_turn(
    state: CallState, speaker: TelnyxSpeaker, ws: WebSocket, pcm: bytes, *, replayed: bool
) -> None:
    """Un tour : transcrire, corriger, décider, parler. Le domaine `hikky` décide."""
    from hikky.domain.question_router import SLOT_ORDER, phrase_for_slot
    from hikky.domain.routed_turn import build_recap, run_routed_turn
    from hikky.pipeline.stt_correction import corriger_transcription

    text = await transcribe(pcm)
    if not text:
        state.misunderstood += 1
        conv.info("CLIENT : (inaudible) [%d]", state.misunderstood)
        missing = state.session.intent.missing_slots()
        slot = next((s for s in SLOT_ORDER if s in missing), None)
        question = None
        if slot is not None:
            question = phrase_for_slot(slot, state.recent_phrasings)
            state.recent_phrasings.add(question)
        recap = None if slot is not None else build_recap(state.session.intent)
        reply, end_call = repair_reply(
            state.misunderstood,
            awaiting_confirmation=state.awaiting_confirmation,
            next_question=question,
            recap=recap,
        )
        await speaker.say(reply)
        if end_call:
            await _hangup(ws)
        return

    state.misunderstood = 0
    text, changes = corriger_transcription(text, detail=True)  # « à l'al » -> « halal »
    if changes:
        log.info("STT corrigé : %s", changes)
    if state.is_repeat(text) and replayed:
        # Le client a répété sa question pendant que le bot y répondait : pas deux réponses.
        log.info("énoncé rejoué identique au précédent -> ignoré : %r", text)
        return
    conv.info("CLIENT : %s", text)
    outcome = await run_routed_turn(
        session=state.session,
        user_text=text,
        customer_phone=None,
        history=state.history,
        extractor=STATE["extractor"],
        answerer=STATE["answerer"],
        awaiting_confirmation=state.awaiting_confirmation,
        speak=speaker.say,
        recent_phrasings=state.recent_phrasings,
        phraseur=STATE["phraseur"],
    )
    state.awaiting_confirmation = outcome.awaiting_confirmation
    if outcome.should_end:
        log.info("fin d'appel demandée par le dialogue")
        await _hangup(ws)


async def _hangup(ws: WebSocket) -> None:
    try:
        await ws.close()
    except Exception:  # noqa: BLE001
        pass


def _dump(state: CallState, pcm: bytes) -> str:
    """Chaque énoncé en WAV pour l'écoute hors ligne ; ne doit jamais casser l'appel."""
    if not settings.debug_dir:
        return "(enregistrement désactivé)"
    try:
        return audio.write_wav(
            f"{settings.debug_dir}/{state.call_tag}_{state.utterances:02d}.wav", pcm
        )
    except Exception as e:  # noqa: BLE001
        return f"(dump échoué: {e})"
