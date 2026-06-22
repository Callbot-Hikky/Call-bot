"""Application FastAPI — point d'entrée HTTP/WS du callbot Hikky.

Lancement local :

    uvicorn hikky.app.main:app --host 0.0.0.0 --port 8000

Pour exposer le port à Twilio en développement :

    ngrok http 8000

Côté Twilio, configurer le numéro pour qu'au décrochage il <Connect><Stream>
vers `wss://<votre-host-ngrok>/twilio/{{CallSid}}`. Le `CallSid` Twilio
sert d'identifiant d'appel côté Hikky.

Pour l'instant, l'app fait juste de l'écho : tout chunk audio reçu est
renvoyé à l'identique. C'est le walking skeleton — la pipeline réelle
(STT, dialogue, TTS) viendra dans les plans D et E.
"""

import json

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from hikky.adapters.telephony.twilio_adapter import TwilioMediaStreamsAdapter
from hikky.adapters.telephony.twilio_protocol import (
    MediaFrame,
    StartFrame,
    StopFrame,
    decode_inbound,
)


def create_app(telephony: TwilioMediaStreamsAdapter | None = None) -> FastAPI:
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

                elif isinstance(frame, MediaFrame):
                    # Walking skeleton : on renvoie l'audio tel quel
                    if bound:
                        await telephony_adapter.send_audio(call_sid, frame.audio)

                elif isinstance(frame, StopFrame):
                    break

        except WebSocketDisconnect:
            pass
        finally:
            telephony_adapter.unbind_call(call_sid)

    return app


app = create_app()
