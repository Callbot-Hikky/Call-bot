"""`/api/calls/context` renvoie `attributes` : l'adaptateur doit les garder.

Charge utile reelle du backend (Le Petit Sud, 2026-10-02) : halal, terrasse,
paiements… etaient renvoyes et jetes par `_to_context`. Le bot repondait
« je n'ai pas cette information » a une question dont il avait la reponse.
"""

from hikky.adapters.back.backend_restaurant_context import _to_context


def _payload(**extra):
    base = {
        "restaurant": {
            "id": "r-1",
            "name": "Le Petit Sud",
            "address": "12 rue de la République",
            "city": "Lyon",
            "postalCode": "69002",
        },
        "hours": [],
        "attributes": {
            "dietary": {"halal": True},
            "equipments": {"terrace": True, "pets_allowed": False},
        },
        "policies": {"maxPartySize": 15},
    }
    base.update(extra)
    return base


def test_attributes_are_kept_as_is():
    ctx = _to_context(_payload())
    assert ctx.attributes["dietary"]["halal"] is True
    assert ctx.attributes["equipments"]["terrace"] is True


def test_address_is_assembled_for_the_answerer():
    assert _to_context(_payload()).address == "12 rue de la République, 69002 Lyon"


def test_missing_or_malformed_attributes_fall_back_to_empty():
    assert _to_context(_payload(attributes=None)).attributes == {}
    assert _to_context(_payload(attributes="n/a")).attributes == {}
    sans = _payload()
    del sans["attributes"]
    assert _to_context(sans).attributes == {}


def test_missing_address_is_none():
    ctx = _to_context({"restaurant": {"id": "r-1", "name": "X"}})
    assert ctx.address is None
