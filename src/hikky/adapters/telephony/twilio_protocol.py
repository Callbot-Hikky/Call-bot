"""Parseur/sérializer du protocole Twilio Media Streams.

Référence : https://www.twilio.com/docs/voice/twiml/stream

Twilio envoie des événements JSON sur la WebSocket :
- `connected` : handshake initial
- `start` : début du stream (contient streamSid, callSid, customParameters)
- `media` : chunk audio μ-law 8 kHz mono, encodé base64
- `stop` : fin du stream

Pour parler à Twilio en retour, on lui envoie des événements `media` avec
le même format. Ce module ne s'occupe que de la (dé)sérialisation — la
gestion de la WebSocket vit dans `twilio_adapter.py`.
"""

import base64
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ConnectedFrame:
    pass


@dataclass(frozen=True, slots=True)
class StartFrame:
    stream_sid: str
    call_sid: str
    custom_parameters: dict[str, str]


@dataclass(frozen=True, slots=True)
class MediaFrame:
    stream_sid: str
    audio: bytes  # μ-law 8 kHz mono


@dataclass(frozen=True, slots=True)
class StopFrame:
    stream_sid: str


InboundFrame = ConnectedFrame | StartFrame | MediaFrame | StopFrame


def decode_inbound(payload: dict) -> InboundFrame:
    event = payload.get("event")
    if event == "connected":
        return ConnectedFrame()
    if event == "start":
        start = payload.get("start", {})
        return StartFrame(
            stream_sid=payload.get("streamSid") or start.get("streamSid", ""),
            call_sid=start.get("callSid", ""),
            custom_parameters=start.get("customParameters", {}) or {},
        )
    if event == "media":
        media = payload.get("media", {})
        audio = base64.b64decode(media.get("payload", ""))
        return MediaFrame(stream_sid=payload.get("streamSid", ""), audio=audio)
    if event == "stop":
        return StopFrame(stream_sid=payload.get("streamSid", ""))
    raise ValueError(f"Unknown Twilio Media Streams event: {event!r}")


def encode_outbound_media(stream_sid: str, audio: bytes) -> dict:
    return {
        "event": "media",
        "streamSid": stream_sid,
        "media": {"payload": base64.b64encode(audio).decode()},
    }


def encode_clear(stream_sid: str) -> dict:
    """Demande à Twilio de purger le buffer audio en cours (utile au barge-in)."""

    return {"event": "clear", "streamSid": stream_sid}
