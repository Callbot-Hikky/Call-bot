"""Charge le `RestaurantContext` depuis la route dédiée au callbot.

Le backend expose `/api/calls/context`, qui résout le restaurant par son
numéro d'appel et renvoie identité, horaires et politiques en une seule
requête — exactement ce dont on a besoin au décroché.

Ce que cette route a corrigé : le callbot listait `/api/restaurants` puis
**inventait** le reste. Horaires 9h-23h sept jours sur sept, et surtout
une taille de groupe maximale de 12 alors que la politique réelle
plafonne à 4. Le bot acceptait donc un groupe de 8, collectait toute la
réservation, puis butait sur « pas de table » sans pouvoir expliquer
pourquoi — il tournait en rond sur « souhaitez-vous une autre heure ? ».

Point de vigilance : la base ne contient aujourd'hui aucun horaire. Une
liste vide supprimerait tout contrôle d'ouverture — et `RestaurantContext`
la refuse de toute façon, ce qui ferait échouer l'appel au décroché. On
retombe donc explicitement sur des valeurs par défaut.
"""

from __future__ import annotations

import logging
from datetime import time
from typing import Any

from hikky.domain.restaurant_context import (
    OpeningHours,
    RestaurantContext,
    RestaurantRules,
)
from hikky.exceptions import UnknownRestaurant
from hikky.ports.restaurant_context import RestaurantContextPort

logger = logging.getLogger("hikky.back.restaurant")

# Repli tant qu'aucun horaire n'est saisi en base. Le moteur s'en sert
# pour répondre « êtes-vous ouverts le dimanche ? » : mieux vaut une
# valeur explicite qu'un contexte vide.
DEFAULT_OPENS = time(9, 0)
DEFAULT_CLOSES = time(23, 0)


class BackendRestaurantContextAdapter(RestaurantContextPort):
    def __init__(self, client: Any) -> None:
        self._client = client

    async def load(self, called_number: str) -> RestaurantContext:
        payload = await self._client.get(
            "/api/calls/context", params={"restaurantPhone": called_number}
        )
        if not isinstance(payload, dict) or not payload.get("restaurant"):
            logger.warning("aucun restaurant pour %s", called_number)
            raise UnknownRestaurant(called_number)
        return _to_context(payload)


def _to_context(payload: dict[str, Any]) -> RestaurantContext:
    restaurant = payload.get("restaurant") or {}
    name = restaurant.get("name") or "notre restaurant"
    regles = _rules(payload.get("policies") or {})
    horaires = _opening_hours(payload.get("hours") or [])
    # `attributes` (halal, terrasse, paiements…) était renvoyé par le backend
    # depuis la V9 mais jamais lu : le bot répondait « je n'ai pas cette
    # information » à des questions dont la réponse était dans la charge utile.
    attributs = payload.get("attributes")
    if not isinstance(attributs, dict):
        attributs = {}

    logger.info(
        "contexte chargé : %s, %d plage(s) d'ouverture, %d couverts max, %d groupe(s) d'attributs",
        name, len(horaires), regles.max_group_size, len(attributs),
    )

    return RestaurantContext(
        id=str(restaurant.get("id")),
        name=name,
        # L'accueil situe l'appelant, annonce honnêtement qu'il parle à un
        # assistant vocal, puis l'invite à parler. Le découvrir en cours de
        # conversation met le client mal à l'aise. La longueur est
        # contrainte : XTTS synthétise ~6 caractères par 100 ms, et
        # l'appelant patiente avant d'avoir pu dire un mot.
        greeting=(
            f"Bonjour, vous êtes bien au restaurant {name}. "
            "Notre équipe est occupée en salle, je suis son assistant vocal "
            "et je prends les réservations. Que puis-je faire pour vous ?"
        ),
        opening_hours=horaires,
        total_capacity=restaurant.get("capacity") or 40,
        rules=regles,
        transfer_number=None,
        fallback_message="Je vous rappelle au plus vite, merci de votre appel.",
        address=_adresse(restaurant),
        attributes=attributs,
    )


def _adresse(restaurant: dict[str, Any]) -> str | None:
    """« 12 rue de la République, 69002 Lyon » — ou None si rien n'est saisi."""
    rue = (restaurant.get("address") or "").strip()
    ville = " ".join(
        p for p in ((restaurant.get("postalCode") or "").strip(), (restaurant.get("city") or "").strip()) if p
    )
    morceaux = [p for p in (rue, ville) if p]
    return ", ".join(morceaux) or None


def _rules(policies: dict[str, Any]) -> RestaurantRules:
    """Politiques du restaurant, avec repli sur les valeurs du domaine.

    Un backend incomplet ne doit pas empêcher de répondre au téléphone :
    on garde alors les valeurs par défaut plutôt que d'échouer.
    """
    defauts = RestaurantRules()
    return RestaurantRules(
        max_group_size=policies.get("maxPartySize") or defauts.max_group_size,
        reservation_duration_minutes=(
            policies.get("defaultDurationMinutes")
            or defauts.reservation_duration_minutes
        ),
    )


def _opening_hours(hours: list[Any]) -> list[OpeningHours]:
    plages: list[OpeningHours] = []
    for entree in hours:
        if not isinstance(entree, dict) or entree.get("isClosed"):
            continue
        jour = entree.get("dayOfWeek")
        ouvre = _heure(entree.get("opensAt"))
        ferme = _heure(entree.get("closesAt"))
        if jour is None or ouvre is None or ferme is None:
            logger.warning("plage d'ouverture illisible : %r", entree)
            continue
        plages.append(OpeningHours(weekday=int(jour), opens=ouvre, closes=ferme))

    if plages:
        return plages

    # Aucune plage exploitable : sans repli, `RestaurantContext` refuse la
    # liste vide et l'appel échouerait au décroché.
    logger.info("aucun horaire en base — repli sur %s-%s", DEFAULT_OPENS, DEFAULT_CLOSES)
    return [
        OpeningHours(weekday=jour, opens=DEFAULT_OPENS, closes=DEFAULT_CLOSES)
        for jour in range(7)
    ]


def _heure(valeur: Any) -> time | None:
    if not isinstance(valeur, str):
        return None
    try:
        return time.fromisoformat(valeur)
    except ValueError:
        return None
