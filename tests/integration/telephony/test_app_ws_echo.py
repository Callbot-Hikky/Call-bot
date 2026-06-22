"""Smoke tests de l'app FastAPI.

Les tests d'écho du walking skeleton (Plan C) ont été retirés : l'endpoint
`/twilio/{call_sid}` est maintenant câblé sur la Pipeline Pipecat
(plan E) qui exige des modèles voix et de l'audio réel — non testable
avec TestClient. La validation E2E vivre sur la machine GPU.

On garde ici :
- un test que `/health` répond,
- un test que `/twilio/{call_sid}` accepte la connexion et se ferme
  proprement quand aucune dépendance n'est configurée (cas par défaut).
"""

from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from hikky.app.main import create_app


def test_health_endpoint():
    client = TestClient(create_app())
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ws_accepts_connection_and_closes_when_no_deps_configured():
    client = TestClient(create_app(deps=None))
    with client.websocket_connect("/twilio/CAxxx") as ws:
        try:
            ws.receive_text()
        except WebSocketDisconnect:
            pass  # attendu : close(1011) puisque deps est None
