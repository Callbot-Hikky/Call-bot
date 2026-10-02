"""Base de connaissances lue sur les routes du callbot.

Le contrat tient en deux routes. Ce qui compte surtout : aucune panne de
ces routes ne doit remonter jusqu'à l'appel en cours.
"""

import json

import httpx
import pytest

from hikky.adapters.back.knowledge_adapter import BackendKnowledgeAdapter

TEL = "+33100000000"


@pytest.fixture
def adapter(back_client) -> BackendKnowledgeAdapter:
    return BackendKnowledgeAdapter(back_client, restaurant_phone=TEL)


async def test_search_envoie_le_numero_et_la_question(adapter, httpx_mock, base_url):
    httpx_mock.add_response(
        json={
            "question": "un menu enfant ?",
            "matches": [
                {"id": "1", "title": "Menu enfant", "content": " Neuf euros. ", "score": 0.83},
                {"id": "2", "title": "Parking", "content": "Jaurès.", "score": 0.2},
            ],
        }
    )

    passages = await adapter.search("un menu enfant ?")

    requete = httpx_mock.get_request()
    assert requete.url.path == "/api/calls/knowledge"
    assert requete.url.params["restaurantPhone"] == TEL
    assert requete.url.params["question"] == "un menu enfant ?"
    assert requete.url.params["limit"] == "3"
    assert requete.headers["X-Api-Key"] == "test-key"
    assert [(p.title, p.content, p.score) for p in passages] == [
        ("Menu enfant", "Neuf euros.", 0.83),
        ("Parking", "Jaurès.", 0.2),
    ]


async def test_search_sans_resultat_renvoie_une_liste_vide(adapter, httpx_mock):
    httpx_mock.add_response(json={"question": "x", "matches": []})

    assert await adapter.search("x") == []


async def test_search_ignore_les_entrees_illisibles(adapter, httpx_mock):
    httpx_mock.add_response(
        json={"matches": [{"title": "Vide", "content": ""}, "n'importe quoi", {"content": "Ok."}]}
    )

    passages = await adapter.search("x")

    assert [(p.title, p.content, p.score) for p in passages] == [("", "Ok.", 0.0)]


async def test_search_ne_leve_pas_quand_le_back_tombe(adapter, httpx_mock):
    httpx_mock.add_exception(httpx.ConnectError("down"), is_reusable=True)

    assert await adapter.search("x") == []


async def test_search_ne_leve_pas_sur_restaurant_inconnu(adapter, httpx_mock):
    httpx_mock.add_response(status_code=404)

    assert await adapter.search("x") == []


async def test_report_poste_la_question(adapter, httpx_mock):
    httpx_mock.add_response(status_code=204)

    await adapter.report_unanswered("On peut venir avec un chien ?")

    requete = httpx_mock.get_request()
    assert requete.method == "POST"
    assert requete.url.path == "/api/calls/unanswered"
    assert json.loads(requete.content) == {
        "restaurantPhone": TEL,
        "question": "On peut venir avec un chien ?",
    }


async def test_report_ne_leve_pas_quand_le_back_tombe(adapter, httpx_mock):
    httpx_mock.add_response(status_code=503, is_reusable=True)

    await adapter.report_unanswered("x")
