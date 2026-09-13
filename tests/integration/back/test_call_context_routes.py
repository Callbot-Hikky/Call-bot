"""Le backend expose enfin des routes concues pour le callbot.

Jusqu'ici l'adapter reconstituait la disponibilite a la main : lister
`/api/tables`, lister `/api/reservations`, calculer les chevauchements.
Cette logique vivait du mauvais cote — le callbot ne connait ni les
tables desactivees, ni les horaires reels, ni les statuts qui liberent
un creneau.

`staging` fournit desormais :

    GET /api/calls/context?restaurantPhone=...
    GET /api/calls/availability?restaurantPhone=...&startsAt=...&partySize=...

La seconde repond `available`, un `reason` (`closed` / `no_table`) et
surtout des `alternatives`. C'est ce dernier champ qui change la
conversation : au lieu d'un « non » sec, le bot peut proposer un autre
creneau.
"""

from datetime import UTC, datetime

from hikky.adapters.back.call_ingest_adapter import CallIngestAdapter

PHONE = "+33123456789"
QUAND = datetime(2026, 7, 22, 12, 0, tzinfo=UTC)


class _ClientScripte:
    """Backend simule : repond selon le chemin, enregistre les appels."""

    def __init__(self, reponses: dict) -> None:
        self._reponses = reponses
        self.appels: list[tuple[str, dict]] = []

    async def get(self, path: str, params: dict | None = None):
        self.appels.append((path, params or {}))
        if path not in self._reponses:
            raise AssertionError(f"chemin inattendu : {path}")
        return self._reponses[path]

    async def post(self, path: str, json=None, headers=None):
        self.appels.append((path, json or {}))
        return self._reponses.get(path, {})

    def chemins(self) -> list[str]:
        return [p for p, _ in self.appels]


def _adapter(reponses: dict) -> tuple[CallIngestAdapter, _ClientScripte]:
    client = _ClientScripte(reponses)
    return CallIngestAdapter(client, restaurant_phone=PHONE), client


# ── Disponibilite ───────────────────────────────────────────────────────


async def test_la_disponibilite_interroge_la_route_dediee():
    """Plus de reconstruction : une seule question au backend."""
    adapter, client = _adapter(
        {"/api/calls/availability": {"available": True, "reason": None, "alternatives": []}}
    )

    assert await adapter.check_availability("r-1", QUAND, 4) is True
    assert client.chemins() == ["/api/calls/availability"]

    _, params = client.appels[0]
    assert params["restaurantPhone"] == PHONE
    assert params["partySize"] == 4
    assert params["startsAt"].startswith("2026-07-22T12:00")


async def test_une_table_indisponible_renvoie_faux():
    adapter, _ = _adapter(
        {
            "/api/calls/availability": {
                "available": False,
                "reason": "no_table",
                "alternatives": [],
            }
        }
    )
    assert await adapter.check_availability("r-1", QUAND, 4) is False


async def test_les_alternatives_sont_remontees_au_dialogue():
    """Sans ce detail, le bot ne peut que refuser — il ne propose rien."""
    adapter, _ = _adapter(
        {
            "/api/calls/availability": {
                "available": False,
                "reason": "no_table",
                "alternatives": [
                    {"startsAt": "2026-07-22T13:30:00Z", "capacity": 4},
                    {"startsAt": "2026-07-22T14:00:00Z", "capacity": 6},
                ],
            }
        }
    )

    verdict = await adapter.availability_detail("r-1", QUAND, 4)

    assert verdict.available is False
    assert verdict.reason == "no_table"
    assert len(verdict.alternatives) == 2
    assert verdict.alternatives[0].hour == 13
    assert verdict.alternatives[0].minute == 30


async def test_un_restaurant_ferme_est_distingue_d_une_table_prise():
    """« On est ferme » et « c'est complet » n'appellent pas la meme reponse."""
    adapter, _ = _adapter(
        {
            "/api/calls/availability": {
                "available": False,
                "reason": "closed",
                "alternatives": [],
            }
        }
    )
    verdict = await adapter.availability_detail("r-1", QUAND, 4)
    assert verdict.reason == "closed"
    assert verdict.is_closed is True


async def test_une_reponse_vide_ne_fait_pas_promettre_une_table():
    """En cas de reponse inexploitable, refuser vaut mieux qu'engager."""
    adapter, _ = _adapter({"/api/calls/availability": None})
    assert await adapter.check_availability("r-1", QUAND, 4) is False


# ── Fuseau horaire ──────────────────────────────────────────────────────


async def test_un_horaire_naif_est_interprete_dans_le_fuseau_du_restaurant():
    """« Midi » veut dire midi a Paris, pas midi UTC.

    Le domaine raisonne en heure locale naive. Attacher UTC a ces
    instants decalait chaque reservation de deux heures en ete : le
    client demandait midi et la table etait reservee a 14h.
    """
    from datetime import datetime as dt

    adapter, client = _adapter(
        {"/api/calls/availability": {"available": True, "reason": None, "alternatives": []}}
    )
    await adapter.check_availability("r-1", dt(2026, 7, 22, 12, 0), 4)

    _, params = client.appels[0]
    assert params["startsAt"] == "2026-07-22T12:00:00+02:00"


async def test_un_horaire_deja_date_est_respecte():
    """Un instant explicitement date ne doit pas etre re-interprete."""
    adapter, client = _adapter(
        {"/api/calls/availability": {"available": True, "reason": None, "alternatives": []}}
    )
    await adapter.check_availability("r-1", QUAND, 4)

    _, params = client.appels[0]
    assert params["startsAt"].endswith("+00:00")


# ── Assignation de table ────────────────────────────────────────────────


async def test_la_disponibilite_remonte_la_table_libre():
    """Sans le tableId, la reservation reste sans table assignee."""
    adapter, _ = _adapter(
        {
            "/api/calls/availability": {
                "available": True,
                "reason": None,
                "tableId": "9a9d4bf5-8c67-47ed-a09c-8c55de07f80b",
                "alternatives": [],
            }
        }
    )
    verdict = await adapter.availability_detail("r-1", QUAND, 4)
    assert verdict.table_id == "9a9d4bf5-8c67-47ed-a09c-8c55de07f80b"


async def test_la_reservation_est_assignee_a_une_table():
    """Le plan de salle et la simulation de soiree raisonnent par table :
    sans assignation, la reservation existe mais n'apparait nulle part."""
    adapter, client = _adapter(
        {
            "/api/calls/availability": {
                "available": True,
                "reason": None,
                "tableId": "9a9d4bf5-8c67-47ed-a09c-8c55de07f80b",
                "alternatives": [],
            },
            "/api/calls/ingest": {"reservation": {"id": "resa-1"}},
        }
    )

    await adapter.create("r-1", QUAND, 4, "Dupont", "+33600000000")

    corps = next(c for p, c in client.appels if p == "/api/calls/ingest")
    assert corps["reservation"]["tableId"] == "9a9d4bf5-8c67-47ed-a09c-8c55de07f80b"


async def test_une_table_prise_entre_temps_n_est_pas_reservee():
    """Entre la verification et l'enregistrement, la table peut partir.
    Mieux vaut une reservation sans table qu'une table double-reservee."""
    adapter, client = _adapter(
        {
            "/api/calls/availability": {
                "available": False,
                "reason": "no_table",
                "tableId": None,
                "alternatives": [],
            },
            "/api/calls/ingest": {"reservation": {"id": "resa-1"}},
        }
    )

    await adapter.create("r-1", QUAND, 4, "Dupont", "+33600000000")

    corps = next(c for p, c in client.appels if p == "/api/calls/ingest")
    assert corps["reservation"]["tableId"] is None
