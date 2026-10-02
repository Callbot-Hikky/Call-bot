"""Le callbot lit le contexte sur la route qui lui est destinee.

Jusqu'ici il listait `/api/restaurants` et filtrait par telephone cote
client, puis **inventait** le reste : horaires 9h-23h sept jours sur
sept, et une taille de groupe maximale de 12.

Or la politique reelle du restaurant plafonne a 4 couverts. Le bot
acceptait donc un groupe de 8, collectait toute la reservation, puis
butait sur « pas de table » sans jamais pouvoir expliquer pourquoi — il
tournait en rond sur « souhaitez-vous une autre heure ? ».

`/api/calls/context` porte ces informations. Un point de vigilance : la
base ne contient aujourd'hui **aucun horaire**. Prendre la liste vide
telle quelle supprimerait tout controle d'ouverture, ce qui serait pire
que les valeurs par defaut. On retombe donc dessus explicitement.
"""

from datetime import time

import pytest

from hikky.adapters.back.backend_restaurant_context import (
    DEFAULT_CLOSES,
    DEFAULT_OPENS,
    BackendRestaurantContextAdapter,
)
from hikky.exceptions import UnknownRestaurant

TEL = "+33100000000"


def _contexte(**surcharges):
    base = {
        "restaurant": {
            "id": "22b60047-3341-4b71-bed8-e22bc08c3603",
            "name": "Le Bistrot du Coin",
            "phoneNumber": TEL,
            "timezone": "Europe/Paris",
        },
        "hours": [],
        "attributes": {},
        "policies": {"maxPartySize": 4, "defaultDurationMinutes": 90},
    }
    base.update(surcharges)
    return base


class _Client:
    def __init__(self, reponse) -> None:
        self.reponse = reponse
        self.appels: list[tuple[str, dict]] = []

    async def get(self, path: str, params: dict | None = None):
        self.appels.append((path, params or {}))
        if isinstance(self.reponse, Exception):
            raise self.reponse
        return self.reponse


async def _charger(reponse):
    client = _Client(reponse)
    contexte = await BackendRestaurantContextAdapter(client).load(TEL)
    return contexte, client


# ── La bonne route, le bon parametre ────────────────────────────────────


async def test_le_contexte_vient_de_la_route_dediee():
    _, client = await _charger(_contexte())

    chemin, params = client.appels[0]
    assert chemin == "/api/calls/context"
    assert params["restaurantPhone"] == TEL
    assert len(client.appels) == 1, "une seule requete au decroche"


async def test_l_identite_du_restaurant_est_reprise():
    contexte, _ = await _charger(_contexte())

    assert contexte.id == "22b60047-3341-4b71-bed8-e22bc08c3603"
    assert contexte.name == "Le Bistrot du Coin"
    assert "Le Bistrot du Coin" in contexte.greeting


# ── Les politiques, qui evitaient au bot de tourner en rond ─────────────


async def test_la_taille_de_groupe_maximale_vient_du_backend():
    """Le defaut a 12 faisait accepter des groupes que le resto refuse."""
    contexte, _ = await _charger(_contexte())

    assert contexte.rules.max_group_size == 4


async def test_la_duree_de_table_vient_du_backend():
    contexte, _ = await _charger(
        _contexte(policies={"maxPartySize": 6, "defaultDurationMinutes": 120})
    )

    assert contexte.rules.reservation_duration_minutes == 120
    assert contexte.rules.max_group_size == 6


async def test_des_politiques_absentes_ne_font_pas_echouer_l_appel():
    """Un backend incomplet ne doit pas empecher de repondre au telephone."""
    contexte, _ = await _charger(_contexte(policies={}))

    assert contexte.rules.max_group_size > 0
    assert contexte.rules.reservation_duration_minutes > 0


# ── Les horaires ────────────────────────────────────────────────────────


async def test_les_horaires_reels_sont_repris():
    contexte, _ = await _charger(
        _contexte(
            hours=[
                {"dayOfWeek": 0, "opensAt": "12:00:00", "closesAt": "14:30:00"},
                {"dayOfWeek": 0, "opensAt": "19:00:00", "closesAt": "22:30:00"},
                {"dayOfWeek": 5, "opensAt": "19:00:00", "closesAt": "23:00:00"},
            ]
        )
    )

    lundi = [h for h in contexte.opening_hours if h.weekday == 0]
    assert len(lundi) == 2
    assert lundi[0].opens == time(12, 0)
    assert lundi[0].closes == time(14, 30)


async def test_un_jour_ferme_est_ecarte():
    """`isClosed` marque une fermeture : la garder ouvrirait le restaurant."""
    contexte, _ = await _charger(
        _contexte(
            hours=[
                {"dayOfWeek": 0, "opensAt": "12:00:00", "closesAt": "22:00:00"},
                {
                    "dayOfWeek": 6,
                    "opensAt": "12:00:00",
                    "closesAt": "22:00:00",
                    "isClosed": True,
                },
            ]
        )
    )

    jours = {h.weekday for h in contexte.opening_hours}
    assert 0 in jours
    assert 6 not in jours, "le dimanche est ferme"


async def test_sans_horaire_en_base_on_retombe_sur_les_valeurs_par_defaut():
    """La base n'en contient aucun : une liste vide otecait tout controle.

    `RestaurantContext` refuse d'ailleurs une liste vide — l'appel
    echouerait au decroche.
    """
    contexte, _ = await _charger(_contexte(hours=[]))

    assert len(contexte.opening_hours) == 7
    assert contexte.opening_hours[0].opens == DEFAULT_OPENS
    assert contexte.opening_hours[0].closes == DEFAULT_CLOSES


async def test_un_horaire_illisible_n_empeche_pas_de_repondre():
    contexte, _ = await _charger(
        _contexte(
            hours=[
                {"dayOfWeek": 0, "opensAt": "midi", "closesAt": "22:00:00"},
                {"dayOfWeek": 1, "opensAt": "12:00:00", "closesAt": "22:00:00"},
            ]
        )
    )

    jours = {h.weekday for h in contexte.opening_hours}
    assert jours == {1}


# ── Robustesse ──────────────────────────────────────────────────────────


async def test_un_numero_inconnu_est_signale():
    """Le backend renvoie un contexte sans restaurant : on refuse l'appel
    plutot que de repondre au nom d'un etablissement inexistant."""
    with pytest.raises(UnknownRestaurant):
        await _charger({"restaurant": None, "hours": [], "policies": {}})


async def test_une_reponse_vide_est_signalee():
    with pytest.raises(UnknownRestaurant):
        await _charger(None)


async def test_une_panne_du_backend_remonte_telle_quelle():
    """Une panne reseau ne doit pas etre confondue avec un numero inconnu :
    le diagnostic serait faux."""
    from hikky.exceptions import BackUnavailable

    with pytest.raises(BackUnavailable):
        await _charger(BackUnavailable("GET /api/calls/context: HTTP 503"))


async def test_les_attributs_du_restaurant_sont_conserves():
    # Le backend les envoyait déjà ; le bot les jetait et ne pouvait pas
    # dire s'il y avait une terrasse.
    contexte, _ = await _charger(_contexte(attributes={"terrasse": True, "halal": False}))

    assert contexte.attributes == {"terrasse": True, "halal": False}


async def test_des_attributs_absents_ou_illisibles_donnent_un_dictionnaire_vide():
    contexte, _ = await _charger(_contexte(attributes=None))

    assert contexte.attributes == {}
