import base64
import json

from fastapi.testclient import TestClient

from hikky.app.main import create_app


def test_health_endpoint():
    client = TestClient(create_app())
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ws_echoes_inbound_media_back_to_client():
    client = TestClient(create_app())
    audio = b"\x10\x20\x30\x40"
    encoded = base64.b64encode(audio).decode()

    with client.websocket_connect("/twilio/CAxxx") as ws:
        ws.send_text(
            json.dumps(
                {
                    "event": "start",
                    "sequenceNumber": "1",
                    "start": {
                        "streamSid": "MZxxx",
                        "callSid": "CAxxx",
                        "tracks": ["inbound"],
                        "customParameters": {},
                        "mediaFormat": {
                            "encoding": "audio/x-mulaw",
                            "sampleRate": 8000,
                            "channels": 1,
                        },
                    },
                    "streamSid": "MZxxx",
                }
            )
        )
        # Le greeting est envoyé immédiatement après start ; on le consomme
        _ = ws.receive_text()

        ws.send_text(
            json.dumps(
                {
                    "event": "media",
                    "sequenceNumber": "2",
                    "media": {
                        "track": "inbound",
                        "chunk": "1",
                        "timestamp": "100",
                        "payload": encoded,
                    },
                    "streamSid": "MZxxx",
                }
            )
        )
        # On doit recevoir un media outbound avec le même audio
        echoed = json.loads(ws.receive_text())
        assert echoed["event"] == "media"
        assert echoed["streamSid"] == "MZxxx"
        assert base64.b64decode(echoed["media"]["payload"]) == audio

        # Terminer proprement
        ws.send_text(
            json.dumps(
                {
                    "event": "stop",
                    "sequenceNumber": "3",
                    "stop": {"accountSid": "ACxxx", "callSid": "CAxxx"},
                    "streamSid": "MZxxx",
                }
            )
        )


def test_ws_sends_greeting_audio_right_after_start():
    """Au décrochage, le bot doit envoyer immédiatement un message d'accueil
    audio. C'est le critère de sortie du walking skeleton : « ça décroche,
    ça parle, ça raccroche »."""

    client = TestClient(create_app())
    with client.websocket_connect("/twilio/CAxxx") as ws:
        ws.send_text(
            json.dumps(
                {
                    "event": "start",
                    "start": {
                        "streamSid": "MZxxx",
                        "callSid": "CAxxx",
                        "tracks": ["inbound"],
                        "customParameters": {},
                        "mediaFormat": {
                            "encoding": "audio/x-mulaw",
                            "sampleRate": 8000,
                            "channels": 1,
                        },
                    },
                    "streamSid": "MZxxx",
                }
            )
        )
        # Premier message reçu après start = greeting
        first = json.loads(ws.receive_text())
        assert first["event"] == "media"
        assert first["streamSid"] == "MZxxx"
        greeting_audio = base64.b64decode(first["media"]["payload"])
        assert len(greeting_audio) > 0

        ws.send_text(json.dumps({"event": "stop", "streamSid": "MZxxx"}))


def test_ws_ignores_outbound_track_to_avoid_feedback():
    client = TestClient(create_app())
    inbound_audio = b"\x10\x20"
    outbound_audio = b"\xfe\xff"

    with client.websocket_connect("/twilio/CAxxx") as ws:
        ws.send_text(
            json.dumps(
                {
                    "event": "start",
                    "start": {
                        "streamSid": "MZxxx",
                        "callSid": "CAxxx",
                        "tracks": ["inbound"],
                        "customParameters": {},
                        "mediaFormat": {
                            "encoding": "audio/x-mulaw",
                            "sampleRate": 8000,
                            "channels": 1,
                        },
                    },
                    "streamSid": "MZxxx",
                }
            )
        )
        # Greeting est attendu
        _ = ws.receive_text()

        # On envoie un media avec track=outbound — il doit être ignoré
        ws.send_text(
            json.dumps(
                {
                    "event": "media",
                    "media": {
                        "track": "outbound",
                        "payload": base64.b64encode(outbound_audio).decode(),
                    },
                    "streamSid": "MZxxx",
                }
            )
        )
        # Puis un inbound — qui doit être écho
        ws.send_text(
            json.dumps(
                {
                    "event": "media",
                    "media": {
                        "track": "inbound",
                        "payload": base64.b64encode(inbound_audio).decode(),
                    },
                    "streamSid": "MZxxx",
                }
            )
        )
        # On doit recevoir EXACTEMENT l'audio inbound, pas l'outbound
        echoed = json.loads(ws.receive_text())
        assert base64.b64decode(echoed["media"]["payload"]) == inbound_audio

        ws.send_text(json.dumps({"event": "stop", "streamSid": "MZxxx"}))


def test_ws_drops_media_received_before_start_without_crashing():
    client = TestClient(create_app())
    encoded = base64.b64encode(b"\x00").decode()

    with client.websocket_connect("/twilio/CAxxx") as ws:
        # On envoie un media avant le start — l'app ne doit pas planter ni renvoyer
        ws.send_text(
            json.dumps(
                {
                    "event": "media",
                    "media": {"payload": encoded},
                    "streamSid": "MZxxx",
                }
            )
        )
        # Puis stop, pour fermer
        ws.send_text(json.dumps({"event": "stop", "streamSid": "MZxxx"}))
