"""Scénario E2E : CallSession câblée avec les vrais BackHttpAdapter,
le Back étant mocké via httpx_mock. Le LanguageModel reste un fake.

Vérifie que le flux complet (start, check, create, confirm, end) déclenche
les bonnes requêtes HTTP dans le bon ordre.
"""

import json
from datetime import datetime

from pytest_httpx import HTTPXMock

from hikky.adapters.back.call_log_adapter import BackHttpCallLogAdapter
from hikky.adapters.back.http_client import BackHttpClient
from hikky.adapters.back.notification_adapter import BackHttpNotificationAdapter
from hikky.adapters.back.reservation_adapter import BackHttpReservationAdapter
from hikky.adapters.back.restaurant_context_adapter import (
    BackHttpRestaurantContextAdapter,
)
from hikky.domain.call_session import CallSession
from hikky.domain.dialogue_engine import DialogueEngine
from hikky.domain.fallback_policy import FallbackPolicy
from hikky.domain.outcomes import CallOutcome
from tests.fakes.fake_language_model import FakeLanguageModel


def _restaurant_payload() -> dict:
    return {
        "id": "r-1",
        "name": "Chez Test",
        "greeting": "Bonjour.",
        "opening_hours": [
            {"weekday": d, "opens": "00:00:00", "closes": "23:59:00"} for d in range(7)
        ],
        "total_capacity": 40,
        "rules": {"max_group_size": 8, "reservation_duration_minutes": 90},
        "transfer_number": None,
        "fallback_message": "…",
    }


async def test_full_reservation_flow_hits_back_endpoints(
    httpx_mock: HTTPXMock, back_client: BackHttpClient, base_url: str
):
    # Mocks (dans l'ordre attendu)
    httpx_mock.add_response(
        url=f"{base_url}/restaurants/by-phone/+33100000001", json=_restaurant_payload()
    )
    httpx_mock.add_response(
        url=f"{base_url}/calls/c-1/start", method="POST", status_code=204
    )
    httpx_mock.add_response(
        url=f"{base_url}/restaurants/r-1/reservations",
        method="POST",
        json={"reservation_id": "res-abc"},
    )
    httpx_mock.add_response(
        url=f"{base_url}/reservations/res-abc/confirmation",
        method="POST",
        status_code=204,
    )
    httpx_mock.add_response(
        url=f"{base_url}/calls/c-1/end", method="POST", status_code=204
    )

    # Adapters réels
    restaurant_adapter = BackHttpRestaurantContextAdapter(back_client)
    reservation_adapter = BackHttpReservationAdapter(back_client)
    call_log_adapter = BackHttpCallLogAdapter(back_client)
    notification_adapter = BackHttpNotificationAdapter(back_client)

    # Le flux d'identification du restaurant se fait avant la CallSession
    ctx = await restaurant_adapter.load("+33100000001")
    assert ctx.id == "r-1"

    # Câblage de la session avec les adapters réels (sauf le LLM qui reste fake)
    session = CallSession(
        call_id="c-1",
        context=ctx,
        dialogue_engine=DialogueEngine(
            FakeLanguageModel(replies=["Quand ?", "Combien ?", "Nom ?", "Confirmé."])
        ),
        fallback_policy=FallbackPolicy(),
        reservation_port=reservation_adapter,
        call_log=call_log_adapter,
        notification=notification_adapter,
        clock=lambda: datetime(2026, 7, 1, 20, 0),
    )
    await session.begin()

    await session.process_user_turn("Réserver", slot_updates={})
    await session.process_user_turn(
        "Demain 20h", slot_updates={"date_time": datetime(2026, 7, 2, 20)}
    )
    await session.process_user_turn("4", slot_updates={"party_size": 4})
    await session.process_user_turn("Dupont", slot_updates={"customer_name": "Dupont"})

    outcome = await session.finalize_if_complete(customer_phone="+33600000000")
    assert outcome == CallOutcome.RESERVATION_CREATED

    # Vérification des requêtes émises dans l'ordre
    paths = [str(r.url.path) for r in httpx_mock.get_requests()]
    assert paths == [
        "/restaurants/by-phone/+33100000001",
        "/calls/c-1/start",
        "/restaurants/r-1/reservations",
        "/reservations/res-abc/confirmation",
        "/calls/c-1/end",
    ]

    # La requête /calls/c-1/end doit porter l'outcome
    end_body = json.loads(httpx_mock.get_requests()[-1].read())
    assert end_body["outcome"] == "reservation_created"
