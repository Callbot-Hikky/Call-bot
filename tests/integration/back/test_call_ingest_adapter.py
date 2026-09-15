"""Adapter vers le vrai backend Spring Boot.

Le backend n'expose pas les routes que mes premiers adapters appelaient.
Contrat reel, releve dans les controleurs :

    POST /api/calls/ingest        clé d'idempotence twilioCallSid
    GET  /api/tables/available    restaurantId, startsAt, endsAt, partySize
    Auth : en-tete X-Api-Key

L'ingestion est concue pour l'IA : elle recoit l'appel complet en une
fois — client + reservation — et non des creations incrementales.
"""

from datetime import datetime

import pytest
from pytest_httpx import HTTPXMock

from hikky.adapters.back.call_ingest_adapter import CallIngestAdapter
from hikky.exceptions import BackUnavailable


@pytest.fixture
def adapter(back_client):
    return CallIngestAdapter(back_client, restaurant_phone="+33100000000")


# ── Disponibilite ───────────────────────────────────────────────────────


async def test_no_free_table_means_unavailable(
    httpx_mock: HTTPXMock, adapter, base_url: str
):
    httpx_mock.add_response(
        method="GET", json=[]
    )
    assert await adapter.check_availability("r-1", datetime(2026, 7, 22, 20), 4) is False








# ── Ingestion de l'appel ────────────────────────────────────────────────


async def test_create_posts_to_the_ingest_endpoint(
    httpx_mock: HTTPXMock, adapter, base_url: str
):
    httpx_mock.add_response(
        method="GET",
        json={"available": True, "tableId": "tbl-1", "alternatives": []},
        is_reusable=True,
    )
    httpx_mock.add_response(
        url=f"{base_url}/api/calls/ingest",
        method="POST",
        json={"callId": "c-1", "customerId": "cu-1",
              "reservation": {"id": "res-42"}, "alreadyProcessed": False},
    )
    rid = await adapter.create(
        "r-1", datetime(2026, 7, 22, 20), 4, "Dupont", "+33600000000"
    )
    assert rid == "res-42"


async def test_ingest_body_matches_the_backend_contract(
    httpx_mock: HTTPXMock, adapter, base_url: str
):
    import json

    httpx_mock.add_response(
        method="GET",
        json={"available": True, "tableId": "tbl-1", "alternatives": []},
        is_reusable=True,
    )
    httpx_mock.add_response(
        url=f"{base_url}/api/calls/ingest", method="POST",
        json={"reservation": {"id": "res-1"}},
    )
    await adapter.create("r-1", datetime(2026, 7, 22, 20), 4, "Dupont", "+33600000000")
    post = next(r for r in httpx_mock.get_requests() if r.method == "POST")
    body = json.loads(post.read())

    assert body["restaurantPhone"] == "+33100000000"
    assert body["customer"]["phone"] == "+33600000000"
    assert body["customer"]["lastName"] == "Dupont"
    assert body["reservation"]["partySize"] == 4
    assert body["reservation"]["startsAt"].startswith("2026-07-22T20:00")
    assert body["reservation"]["endsAt"].startswith("2026-07-22T21:30")


async def test_call_id_is_used_as_the_idempotency_key(
    httpx_mock: HTTPXMock, adapter, base_url: str
):
    import json

    from hikky.observability.logging import clear_call_context, set_call_context

    httpx_mock.add_response(
        method="GET",
        json={"available": True, "tableId": "tbl-1", "alternatives": []},
        is_reusable=True,
    )
    httpx_mock.add_response(
        url=f"{base_url}/api/calls/ingest", method="POST",
        json={"reservation": {"id": "res-1"}},
    )
    set_call_context(call_id="appel-77", restaurant_id="r-1")
    try:
        await adapter.create("r-1", datetime(2026, 7, 22, 20), 4, "Dupont", None)
    finally:
        clear_call_context()
    post = next(r for r in httpx_mock.get_requests() if r.method == "POST")
    body = json.loads(post.read())
    assert body["twilioCallSid"] == "appel-77"


async def test_ingest_ref_rend_la_cle_unique_meme_si_le_call_id_est_constant(
    httpx_mock: HTTPXMock, adapter, base_url: str
):
    """Bug réel : le dialplan Asterisk renvoyait un UUID AudioSocket FIGÉ, donc
    `call_id` (→ twilioCallSid) était identique à chaque appel ; la 2ᵉ
    réservation d'un même créneau repartait sur la même clé d'idempotence et
    le backend renvoyait 409 (jamais créée). Un `ingest_ref` unique par
    connexion doit garantir l'unicité MÊME quand `call_id` se répète."""
    import json

    from hikky.observability.logging import clear_call_context, set_call_context

    httpx_mock.add_response(
        method="GET",
        json={"available": True, "tableId": "tbl-1", "alternatives": []},
        is_reusable=True,
    )
    httpx_mock.add_response(
        url=f"{base_url}/api/calls/ingest", method="POST",
        json={"reservation": {"id": "res"}}, is_reusable=True,
    )
    sids = []
    for ref in ("uniq-1", "uniq-2"):
        set_call_context(call_id="UUID-FIGE", ingest_ref=ref)
        try:
            await adapter.create("r-1", datetime(2026, 7, 22, 20), 4, "Dupont", None)
        finally:
            clear_call_context()
    for req in httpx_mock.get_requests():
        if req.method == "POST":
            sids.append(json.loads(req.read())["twilioCallSid"])
    assert sids[0] != sids[1]  # clés distinctes malgré le même call_id


async def test_table_overlap_reessaie_avec_une_autre_table_et_reussit(
    httpx_mock: HTTPXMock, adapter, base_url: str
):
    """Course check-then-act : `availability` renvoie une table, mais elle est
    prise au moment de l'`ingest` → 409 `table_overlap`. Le bot doit
    re-demander une table libre et réessayer, plutôt que de refuser à tort un
    créneau qui a d'autres tables libres."""
    import json

    # availability renvoie d'abord tbl-1 (prise), puis tbl-2 (libre).
    httpx_mock.add_response(
        method="GET",
        json={"available": True, "tableId": "tbl-1", "alternatives": []},
    )
    httpx_mock.add_response(
        method="GET",
        json={"available": True, "tableId": "tbl-2", "alternatives": []},
    )
    # 1er ingest : table prise ; 2e : succès.
    httpx_mock.add_response(
        url=f"{base_url}/api/calls/ingest", method="POST", status_code=409,
        json={"error": "table_overlap", "message": "already booked"},
    )
    httpx_mock.add_response(
        url=f"{base_url}/api/calls/ingest", method="POST",
        json={"reservation": {"id": "res-ok"}},
    )

    rid = await adapter.create("r-1", datetime(2026, 7, 22, 20), 4, "Dupont", None)
    assert rid == "res-ok"
    posts = [json.loads(r.read()) for r in httpx_mock.get_requests() if r.method == "POST"]
    assert posts[0]["reservation"]["tableId"] == "tbl-1"
    assert posts[1]["reservation"]["tableId"] == "tbl-2"  # nouvelle table au retry


async def test_table_overlap_persistant_finit_en_reservation_conflict(
    httpx_mock: HTTPXMock, adapter, base_url: str
):
    """Si toutes les tables restent prises (overlap à chaque essai), on renonce
    proprement en `ReservationConflict` (le bot dira « créneau indisponible »)."""
    from hikky.exceptions import ReservationConflict

    httpx_mock.add_response(
        method="GET",
        json={"available": True, "tableId": "tbl-1", "alternatives": []},
        is_reusable=True,
    )
    httpx_mock.add_response(
        url=f"{base_url}/api/calls/ingest", method="POST", status_code=409,
        json={"error": "table_overlap", "message": "already booked"},
        is_reusable=True,
    )
    with pytest.raises(ReservationConflict):
        await adapter.create("r-1", datetime(2026, 7, 22, 20), 4, "Dupont", None)


async def test_un_conflit_409_leve_reservation_conflict(
    httpx_mock: HTTPXMock, adapter, base_url: str
):
    """La clé d'idempotence étant désormais unique par appel, un 409 n'est
    plus une collision de clé : c'est un vrai conflit métier. L'adapter lève
    `ReservationConflict` (que la couche domaine traduira en « créneau
    indisponible »), et NON un faux succès — le bot ne doit pas annoncer une
    réservation qui n'existe pas. Ce n'est pas non plus un `BackUnavailable`
    qui planterait l'appel."""
    from hikky.exceptions import ReservationConflict

    httpx_mock.add_response(
        method="GET",
        json={"available": True, "tableId": "tbl-1", "alternatives": []},
        is_reusable=True,
    )
    httpx_mock.add_response(
        url=f"{base_url}/api/calls/ingest", method="POST", status_code=409,
    )
    with pytest.raises(ReservationConflict):
        await adapter.create("r-1", datetime(2026, 7, 22, 20), 4, "Dupont", None)


async def test_missing_phone_is_replaced_by_a_placeholder(
    httpx_mock: HTTPXMock, adapter, base_url: str
):
    """Le backend exige un telephone client ; un appel sans presentation
    du numero ne doit pas faire echouer la reservation."""
    import json

    httpx_mock.add_response(
        method="GET",
        json={"available": True, "tableId": "tbl-1", "alternatives": []},
        is_reusable=True,
    )
    httpx_mock.add_response(
        url=f"{base_url}/api/calls/ingest", method="POST",
        json={"reservation": {"id": "res-1"}},
    )
    await adapter.create("r-1", datetime(2026, 7, 22, 20), 4, "Dupont", None)
    post = next(r for r in httpx_mock.get_requests() if r.method == "POST")
    body = json.loads(post.read())
    assert body["customer"]["phone"]


async def test_le_placeholder_de_telephone_tient_dans_varchar_32(
    httpx_mock: HTTPXMock, adapter, base_url: str
):
    """La colonne `customers.phone` est un varchar(32). Le placeholder anonyme
    doit tenir dedans : `inconnu-<ingest_ref 32 hex>` faisait 40 caractères et
    le backend rejetait CHAQUE réservation anonyme (« value too long », mappé en
    409). Le placeholder doit rester ≤ 32."""
    import json

    from hikky.observability.logging import clear_call_context, set_call_context

    httpx_mock.add_response(
        method="GET",
        json={"available": True, "tableId": "tbl-1", "alternatives": []},
        is_reusable=True,
    )
    httpx_mock.add_response(
        url=f"{base_url}/api/calls/ingest", method="POST",
        json={"reservation": {"id": "res-1"}},
    )
    set_call_context(call_id="c", ingest_ref="a" * 32)  # ingest_ref = uuid4().hex
    try:
        await adapter.create("r-1", datetime(2026, 7, 22, 20), 4, "Dupont", None)
    finally:
        clear_call_context()
    post = next(r for r in httpx_mock.get_requests() if r.method == "POST")
    phone = json.loads(post.read())["customer"]["phone"]
    assert len(phone) <= 32, f"{phone!r} fait {len(phone)} > 32"


async def test_already_processed_returns_the_existing_reservation(
    httpx_mock: HTTPXMock, adapter, base_url: str
):
    httpx_mock.add_response(
        method="GET",
        json={"available": True, "tableId": "tbl-1", "alternatives": []},
        is_reusable=True,
    )
    httpx_mock.add_response(
        url=f"{base_url}/api/calls/ingest", method="POST",
        json={"reservation": {"id": "res-deja"}, "alreadyProcessed": True},
    )
    rid = await adapter.create("r-1", datetime(2026, 7, 22, 20), 4, "Dupont", None)
    assert rid == "res-deja"


async def test_persistent_failure_raises_back_unavailable(
    httpx_mock: HTTPXMock, adapter, base_url: str
):
    httpx_mock.add_response(
        method="GET",
        json={"available": True, "tableId": "tbl-1", "alternatives": []},
        is_reusable=True,
    )
    httpx_mock.add_response(
        url=f"{base_url}/api/calls/ingest", method="POST", status_code=503,
        is_reusable=True,
    )
    with pytest.raises(BackUnavailable):
        await adapter.create("r-1", datetime(2026, 7, 22, 20), 4, "Dupont", None)


# ── Rappels : non supportes par ce backend ──────────────────────────────


async def test_callback_request_is_reported_as_unsupported(adapter):
    """Le backend n'expose aucune route de rappel. On journalise plutot
    que d'appeler une URL inexistante."""
    rid = await adapter.create_callback_request(
        "r-1", "+33600000000", "ce soir", "trois échecs"
    )
    assert rid is None


# ── Contexte restaurant charge depuis le backend ────────────────────────
#
# Le contexte factice inventait l'identifiant « r-poc ». Spring echoue a
# le convertir en UUID et repond 401 — ce qui m'a fait diagnostiquer a
# tort un probleme d'authentification. L'identifiant doit venir du
# backend, pas d'une constante.

async def test_restaurant_is_resolved_by_phone_number(
    httpx_mock: HTTPXMock, back_client, base_url: str
):
    from hikky.adapters.back.backend_restaurant_context import (
        BackendRestaurantContextAdapter,
    )

    httpx_mock.add_response(
        method="GET",
        json={
            "restaurant": {
                "id": "aaaa-1111",
                "name": "Le Petit Sud",
                "phoneNumber": "+33100000000",
            },
            "hours": [],
            "policies": {"maxPartySize": 4, "defaultDurationMinutes": 90},
        },
    )
    adapter = BackendRestaurantContextAdapter(back_client)
    ctx = await adapter.load("+33100000000")
    assert ctx.id == "aaaa-1111"
    assert ctx.name == "Le Petit Sud"


async def test_unknown_phone_raises_unknown_restaurant(
    httpx_mock: HTTPXMock, back_client, base_url: str
):
    from hikky.adapters.back.backend_restaurant_context import (
        BackendRestaurantContextAdapter,
    )
    from hikky.exceptions import UnknownRestaurant

    httpx_mock.add_response(method="GET", json={"restaurant": None})
    with pytest.raises(UnknownRestaurant):
        await BackendRestaurantContextAdapter(back_client).load("+33999999999")


async def test_loaded_context_has_usable_opening_hours(
    httpx_mock: HTTPXMock, back_client, base_url: str
):
    from hikky.adapters.back.backend_restaurant_context import (
        BackendRestaurantContextAdapter,
    )

    # La base ne contient aucun horaire : le contexte doit rester
    # exploitable malgre tout, sinon l'appel echoue au decroche.
    httpx_mock.add_response(
        method="GET",
        json={
            "restaurant": {"id": "aaaa-1111", "name": "X"},
            "hours": [],
            "policies": {},
        },
    )
    ctx = await BackendRestaurantContextAdapter(back_client).load("+33100000000")
    assert len(ctx.opening_hours) >= 1
    assert ctx.greeting


# ── Disponibilite sur la branche staging ────────────────────────────────
#
# `/api/tables/available` a ete supprime (commit « remove unused
# reservation routes »). Il faut donc calculer le chevauchement cote bot :
# lister les tables du restaurant, lister les reservations, et comparer.


# ── Groupe reparti sur plusieurs tables ─────────────────────────────────


async def test_availability_detail_reads_all_retained_tables(
    httpx_mock: HTTPXMock, adapter, base_url: str
):
    """Le backend peut retenir plusieurs tables (groupe reparti)."""
    httpx_mock.add_response(
        method="GET",
        json={
            "available": True,
            "tableId": "tbl-8",
            "tableIds": ["tbl-8", "tbl-4"],
            "alternatives": [],
        },
    )
    verdict = await adapter.availability_detail("r-1", datetime(2026, 7, 22, 20), 10)
    assert verdict.available is True
    assert verdict.table_id == "tbl-8"
    assert verdict.table_ids == ["tbl-8", "tbl-4"]


async def test_ingest_body_carries_all_tables_for_a_split_group(
    httpx_mock: HTTPXMock, adapter, base_url: str
):
    """La reservation envoyee doit porter TOUTES les tables du groupe."""
    import json

    httpx_mock.add_response(
        method="GET",
        json={
            "available": True,
            "tableId": "tbl-8",
            "tableIds": ["tbl-8", "tbl-4"],
            "alternatives": [],
        },
        is_reusable=True,
    )
    httpx_mock.add_response(
        url=f"{base_url}/api/calls/ingest", method="POST",
        json={"reservation": {"id": "res-1"}},
    )
    await adapter.create("r-1", datetime(2026, 7, 22, 20), 10, "Dupont", "+33600000000")
    post = next(r for r in httpx_mock.get_requests() if r.method == "POST")
    body = json.loads(post.read())
    assert body["reservation"]["tableId"] == "tbl-8"
    assert body["reservation"]["tableIds"] == ["tbl-8", "tbl-4"]


async def test_deux_appels_anonymes_ne_partagent_pas_le_client(
    httpx_mock: HTTPXMock, adapter, base_url: str
):
    """Sans numéro appelant, chaque appel doit rester un client distinct
    (sinon renommer réécrit le nom de toutes les réservations passées)."""
    import json

    from hikky.observability.logging import clear_call_context, set_call_context

    httpx_mock.add_response(
        method="GET",
        json={"available": True, "tableId": "tbl-1", "alternatives": []},
        is_reusable=True,
    )
    httpx_mock.add_response(
        url=f"{base_url}/api/calls/ingest", method="POST",
        json={"reservation": {"id": "res"}}, is_reusable=True,
    )
    phones = []
    for call_id in ("appel-A", "appel-B"):
        set_call_context(call_id=call_id)
        try:
            await adapter.create(
                "r-1", datetime(2026, 7, 22, 20), 4, "Dupont", None
            )
        finally:
            clear_call_context()
    for req in httpx_mock.get_requests():
        if req.method == "POST":
            phones.append(json.loads(req.read())["customer"]["phone"])
    assert phones[0] != phones[1]  # deux clients distincts
