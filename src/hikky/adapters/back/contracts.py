"""Contrat REST proposé pour l'API Back de Hikky.

Ce module regroupe **tous** les détails du contrat HTTP IA ↔ Back :
- les chemins d'endpoints,
- les schémas Pydantic des payloads et des réponses.

L'équipe Back doit valider / négocier ce contrat. Si une modification est
nécessaire, elle se fait ici — les adapters n'ont pas à être réécrits.
"""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from hikky.domain.outcomes import CallOutcome
from hikky.domain.restaurant_context import (
    OpeningHours,
    RestaurantRules,
)


class Endpoints:
    """Chemins d'endpoints relatifs (à concaténer à la base URL du Back)."""

    RESTAURANT_BY_PHONE = "/restaurants/by-phone/{phone_number}"
    AVAILABILITY = "/restaurants/{restaurant_id}/availability"
    RESERVATIONS = "/restaurants/{restaurant_id}/reservations"
    CALLBACK_REQUESTS = "/restaurants/{restaurant_id}/callback-requests"
    CALL_START = "/calls/{call_id}/start"
    CALL_END = "/calls/{call_id}/end"
    NOTIFICATION_CONFIRMATION = "/reservations/{reservation_id}/confirmation"


# ----- Restaurant -----


class RestaurantContextResponse(BaseModel):
    """Réponse de GET /restaurants/by-phone/{phone}.

    Le shape correspond exactement aux champs de `RestaurantContext` du domain.
    Si le Back préfère un autre nommage, on adapte la traduction ici.
    """

    model_config = ConfigDict(extra="ignore")

    id: str
    name: str
    greeting: str
    opening_hours: list[OpeningHours]
    total_capacity: int = Field(ge=0)
    rules: RestaurantRules
    transfer_number: str | None = None
    fallback_message: str


# ----- Reservation -----


class AvailabilityRequest(BaseModel):
    date_time: datetime
    party_size: int = Field(gt=0)


class AvailabilityResponse(BaseModel):
    available: bool


class ReservationCreateRequest(BaseModel):
    date_time: datetime
    party_size: int = Field(gt=0)
    customer_name: str
    customer_phone: str | None = None


class ReservationCreateResponse(BaseModel):
    reservation_id: str


class CallbackRequestPayload(BaseModel):
    customer_phone: str
    preferred_slot: str | None = None
    note: str


class CallbackRequestResponse(BaseModel):
    callback_request_id: str


# ----- Call log -----


class CallStartRequest(BaseModel):
    restaurant_id: str | None
    started_at: datetime


class CallEndRequest(BaseModel):
    outcome: CallOutcome
    duration_seconds: float


# ----- Idempotence -----

IDEMPOTENCY_KEY_HEADER = "Idempotency-Key"


def build_reservation_idempotency_key(
    *,
    call_id: str,
    restaurant_id: str,
    date_time: datetime,
    party_size: int,
) -> str:
    """Clé d'idempotence pour POST /restaurants/{rid}/reservations.

    Le Back DOIT renvoyer la même `reservation_id` pour des appels
    successifs avec la même clé pendant une fenêtre raisonnable
    (≥30 min recommandé). Sans cette clé, un retry réseau après commit
    serveur créerait une seconde réservation — table perdue, client
    mécontent.

    La clé combine `call_id` (unique par appel Twilio) et le « slot »
    (restaurant + créneau + couverts). Le nom du client n'y est PAS
    inclus : si l'utilisateur corrige son nom en cours d'appel, on
    veut la même résa.
    """
    return f"{call_id}:{restaurant_id}:{date_time.isoformat()}:{party_size}"
