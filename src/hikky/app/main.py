"""Application FastAPI — point d'entrée HTTP/WS du callbot Hikky.

Lancement local :

    uvicorn hikky.app.main:app --host 0.0.0.0 --port 8000

Pour exposer le port à Twilio en développement :

    ngrok http 8000

Côté Twilio, configurer le numéro pour qu'au décrochage il <Connect><Stream>
vers `wss://<votre-host-ngrok>/twilio/{{CallSid}}`. Le `CallSid` Twilio
sert d'identifiant d'appel côté Hikky.

Comportement actuel (walking skeleton) :
- À la réception de l'event `start`, le serveur **joue un audio d'accueil canné**
  (silence pour l'instant — le vrai TTS arrive en plan D).
- Chaque chunk d'audio `inbound` est renvoyé tel quel à l'appelant (écho).
- Les media `outbound` (réflexion éventuelle) sont ignorés pour éviter toute boucle.
"""

import json
import logging

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from hikky.adapters.telephony.twilio_adapter import TwilioMediaStreamsAdapter
from hikky.adapters.telephony.twilio_protocol import (
    MediaFrame,
    StartFrame,
    StopFrame,
    decode_inbound,
)

logger = logging.getLogger("hikky.app")

# Greeting audio canné — 320 octets μ-law 8 kHz = ~40 ms de silence.
# Sera remplacé par le vrai TTS en plan D.
GREETING_AUDIO: bytes = b"\xff" * 320


def create_app(
    telephony: TwilioMediaStreamsAdapter | None = None,
    greeting_audio: bytes = GREETING_AUDIO,
) -> FastAPI:
    app = FastAPI(title="Hikky IA")
    telephony_adapter = telephony or TwilioMediaStreamsAdapter()
    app.state.telephony = telephony_adapter

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.websocket("/twilio/{call_sid}")
    async def twilio_stream(websocket: WebSocket, call_sid: str) -> None:
        await websocket.accept()
        bound = False
        try:
            while True:
                raw = await websocket.receive_text()
                payload = json.loads(raw)
                frame = decode_inbound(payload)

                if isinstance(frame, StartFrame):
                    telephony_adapter.bind_call(call_sid, websocket, frame.stream_sid)
                    bound = True
                    await telephony_adapter.send_audio(call_sid, greeting_audio)

                elif isinstance(frame, MediaFrame):
                    if not bound:
                        logger.warning(
                            "Dropping media frame received before start (call_sid=%s)",
                            call_sid,
                        )
                        continue
                    if frame.track != "inbound":
                        logger.debug(
                            "Ignoring media on track=%s (call_sid=%s)",
                            frame.track,
                            call_sid,
                        )
                        continue
                    await telephony_adapter.send_audio(call_sid, frame.audio)

                elif isinstance(frame, StopFrame):
                    break

        except WebSocketDisconnect:
            pass
        finally:
            telephony_adapter.unbind_call(call_sid)

    return app


app = create_app()
