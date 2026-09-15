"""Adapter vers le backend Spring Boot réel.

Le backend expose trois routes **conçues pour le callbot** :

    /api/calls/context       resto, horaires, politiques — lu une fois
    /api/calls/availability  « avez-vous une table ? », avec alternatives
    /api/calls/ingest        appel complet en une transaction

`ingest` reçoit client et réservation ensemble, avec `twilioCallSid`
comme clé d'idempotence : un même appel rejoué ne crée pas de doublon.

`availability` a remplacé une reconstruction côté callbot (lister les
tables, lister les réservations, calculer les chevauchements) qui
ignorait les tables désactivées et les horaires réels.

Authentification machine : en-tête `X-Api-Key`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

from hikky.adapters.back.http_client import Conflict
from hikky.exceptions import ReservationConflict
from hikky.observability.logging import get_call_context
from hikky.ports.reservation import ReservationPort

logger = logging.getLogger("hikky.back.ingest")

# Durée d'occupation d'une table, alignée sur `reservation_duration_minutes`
# du domaine. Le backend exige un `endsAt` explicite.
DEFAULT_DURATION_MINUTES = 90

# Le backend valide `customer.phone` comme non vide. En téléphonie, le
# numéro appelant peut être masqué : on envoie un marqueur explicite
# plutôt que de laisser échouer une réservation par ailleurs valide.
UNKNOWN_PHONE = "inconnu"

# Fuseau du restaurant, tel que renvoyé par /api/calls/context.
DEFAULT_TIMEZONE = "Europe/Paris"

# Nombre de nouvelles tentatives sur un conflit d'écriture transitoire (course
# entre `availability`/`ingest`). À chaque retry, on redemande une table libre
# et on relance ; au-delà, on considère le créneau réellement indisponible.
_MAX_TABLE_RETRIES = 3

# Codes de 409 dus à une course (pas à un vrai « complet »), donc à réessayer :
# - table_overlap  : la table proposée vient d'être prise → en redemander une.
# - duplicate_phone: deux upserts concurrents du même client anonyme → le retry
#   retrouve le client désormais existant.
# Le code générique `conflict` n'est PAS rejouable : il recouvre des erreurs non
# transitoires (ex. « value too long »), qu'un retry ne fait que marteler.
_RETRYABLE_CONFLICT_CODES = frozenset({"table_overlap", "duplicate_phone"})


@dataclass(frozen=True)
class Disponibilite:
    """Verdict du backend sur un créneau.

    Le booléen seul appauvrissait la conversation : le bot disait « non »
    sans pouvoir expliquer ni proposer. `reason` distingue « on est
    fermé » de « c'est complet », deux réponses très différentes au
    téléphone, et `alternatives` permet d'enchaîner sur une proposition.
    """

    available: bool
    reason: str | None
    alternatives: list[datetime]
    # Table principale retenue par le backend (première de `table_ids`).
    # Sans elle, la réservation existe mais n'apparaît ni au plan de salle ni
    # dans la simulation de soirée, qui raisonnent tous deux par table.
    table_id: str | None = None
    # Toutes les tables retenues : plusieurs quand le groupe est réparti sur
    # plusieurs tables (ex. 15 pers. = 8 + 4 + 4).
    table_ids: list[str] = field(default_factory=list)

    @property
    def is_closed(self) -> bool:
        return self.reason == "closed"

    @property
    def is_party_too_large(self) -> bool:
        return self.reason == "party_too_large"


def _parse_iso(valeur: Any) -> datetime | None:
    if not isinstance(valeur, str):
        return None
    try:
        return datetime.fromisoformat(valeur.replace("Z", "+00:00"))
    except ValueError:
        logger.warning("horodatage illisible : %r", valeur)
        return None


class CallIngestAdapter(ReservationPort):
    def __init__(
        self,
        client: Any,
        *,
        restaurant_phone: str,
        duration_minutes: int = DEFAULT_DURATION_MINUTES,
        timezone: str = DEFAULT_TIMEZONE,
    ) -> None:
        self._client = client
        self._restaurant_phone = restaurant_phone
        self._duration = timedelta(minutes=duration_minutes)
        self._fuseau = ZoneInfo(timezone)

    async def availability_detail(
        self, restaurant_id: str, date_time: datetime, party_size: int
    ) -> Disponibilite:
        """Interroge la route que le backend expose pour le callbot.

        Cette route sait ce que le callbot ignore : les tables désactivées,
        les horaires réels du jour, les statuts qui libèrent un créneau.
        Elle renvoie surtout des **alternatives**, sans lesquelles le bot
        ne peut que refuser — il ne propose rien.
        """
        payload = await self._client.get(
            "/api/calls/availability",
            params={
                "restaurantPhone": self._restaurant_phone,
                "startsAt": _iso(date_time, self._fuseau),
                "endsAt": _iso(date_time + self._duration, self._fuseau),
                "partySize": party_size,
            },
        )
        if not isinstance(payload, dict):
            # Réponse inexploitable : refuser vaut mieux qu'engager le
            # restaurant sur une table dont on ne sait rien.
            logger.warning("réponse de disponibilité inexploitable")
            return Disponibilite(
                available=False, reason="inconnu", alternatives=[], table_id=None
            )

        alternatives = []
        for slot in payload.get("alternatives") or []:
            moment = _parse_iso(slot.get("startsAt"))
            if moment is not None:
                alternatives.append(moment)

        raw_ids = payload.get("tableIds") or []
        table_ids = [str(t) for t in raw_ids if t]
        table_id = payload.get("tableId")
        primary = str(table_id) if table_id else (table_ids[0] if table_ids else None)
        verdict = Disponibilite(
            available=bool(payload.get("available")),
            reason=payload.get("reason"),
            alternatives=alternatives,
            table_id=primary,
            table_ids=table_ids or ([primary] if primary else []),
        )
        logger.info(
            "disponibilité : %s (%s), %d alternative(s)",
            verdict.available, verdict.reason or "-", len(verdict.alternatives),
        )
        return verdict

    async def check_availability(
        self, restaurant_id: str, date_time: datetime, party_size: int
    ) -> bool:
        verdict = await self.availability_detail(restaurant_id, date_time, party_size)
        return verdict.available

    async def create(
        self,
        restaurant_id: str,
        date_time: datetime,
        party_size: int,
        customer_name: str,
        customer_phone: str | None,
    ) -> str:
        """Ingère l'appel : crée client et réservation en une transaction.

        La table est redemandée juste avant d'enregistrer, et non reprise
        d'une vérification antérieure : entre les deux, le créneau a pu
        partir. Et si la table proposée est prise à l'instant de l'écriture
        (course entre `availability` et `ingest`), on en redemande une libre
        et on réessaie — le backend fait confiance au `tableId` fourni et
        rejette sinon en 409 `table_overlap`, alors que d'autres tables sont
        libres.
        """
        # Clé d'idempotence : on préfère `ingest_ref` (unique par appel, minté
        # à l'ouverture de connexion), sinon `call_id`, sinon un repli daté.
        ctx = get_call_context()
        call_ref = (
            ctx.get("ingest_ref")
            or ctx.get("call_id")
            or f"hikky-{date_time.isoformat()}"
        )
        # Numéro appelant inconnu (softphone sans présentation du numéro) : on
        # ne le fusionne PAS avec les autres appels anonymes (sinon renommer
        # réécrirait le nom de toutes les réservations de ce faux client). La
        # colonne `customers.phone` est un varchar(32) : `inconnu-<ref 32 hex>`
        # faisait 40 caractères et le backend rejetait CHAQUE réservation
        # anonyme (« value too long » → 409). On tronque donc à 32 tout en
        # gardant l'unicité (le suffixe de `call_ref` reste distinctif).
        phone = customer_phone or f"{UNKNOWN_PHONE}-{call_ref}"[:32]

        payload: dict[str, Any] | None = None
        for tentative in range(_MAX_TABLE_RETRIES + 1):
            # Table redemandée à CHAQUE essai : au retry, `availability` exclut
            # la table qui vient d'être prise et en propose une autre.
            verdict = await self.availability_detail(
                restaurant_id, date_time, party_size
            )
            if verdict.table_id is None:
                logger.warning(
                    "aucune table assignée — réservation sans place attribuée"
                )
            body = {
                "twilioCallSid": call_ref,
                "restaurantPhone": self._restaurant_phone,
                "fromNumber": customer_phone,
                "customer": {
                    "phone": phone,
                    "lastName": customer_name,
                },
                "reservation": {
                    "tableId": verdict.table_id,
                    "tableIds": verdict.table_ids,
                    "startsAt": _iso(date_time, self._fuseau),
                    "endsAt": _iso(date_time + self._duration, self._fuseau),
                    "partySize": party_size,
                    "notes": "Réservation prise par l'assistant vocal Hikky",
                },
            }
            try:
                payload = await self._client.post("/api/calls/ingest", json=body)
                break
            except Conflict as exc:
                # Conflit transitoire dû à une course (table prise entre-temps,
                # upsert client concurrent…) : on redemande une table libre et
                # on réessaie — il ne doit PAS faire refuser un créneau réservable.
                if exc.code in _RETRYABLE_CONFLICT_CODES and tentative < _MAX_TABLE_RETRIES:
                    logger.warning(
                        "conflit '%s' — nouvelle tentative (%d/%d)",
                        exc.code, tentative + 1, _MAX_TABLE_RETRIES,
                    )
                    continue
                # Conflit persistant ou non rejouable : vrai refus. On NE
                # l'annonce PAS comme un succès — on lève une erreur de domaine
                # que `finalize_if_complete` traduit en « créneau indisponible »,
                # sans planter l'appel. Le code est journalisé pour diagnostic.
                logger.warning(
                    "réservation refusée par le Back (409, code=%s) : %s",
                    exc.code, exc,
                )
                raise ReservationConflict(str(exc)) from exc

        assert payload is not None  # la boucle sort par break ou par raise
        if payload.get("alreadyProcessed"):
            logger.info("appel déjà ingéré — réservation existante renvoyée")
        reservation = payload.get("reservation") or {}
        return str(reservation.get("id", ""))

    async def create_callback_request(
        self,
        restaurant_id: str,
        customer_phone: str,
        preferred_slot: str | None,
        note: str,
    ) -> str | None:
        """Le backend n'expose aucune route de rappel.

        On journalise l'intention plutôt que d'appeler une URL inexistante,
        qui remonterait en `BackUnavailable` et masquerait la vraie cause.
        """
        logger.warning(
            "demande de rappel non transmise — le backend n'expose pas cette route "
            "(client %s, créneau %s, motif %s)",
            customer_phone,
            preferred_slot,
            note,
        )
        return None


def _iso(moment: datetime, fuseau: ZoneInfo) -> str:
    """Date un instant avant de l'envoyer au backend.

    Le domaine raisonne en heure locale naïve : « midi » veut dire midi
    au restaurant. Y attacher UTC décalait chaque réservation de deux
    heures en été — le client demandait midi, la table était prise à 14h.
    """
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=fuseau)
    return moment.isoformat()
