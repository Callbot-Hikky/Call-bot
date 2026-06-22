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
