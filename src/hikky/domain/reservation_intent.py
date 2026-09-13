"""Intention de réservation en cours de construction.

Date et heure sont stockées **séparément**. Un `date_time` monolithique
ne sait pas exprimer « je connais le jour, pas l'heure » — l'état le plus
banal d'une prise de réservation. En appel réel, ça produisait ceci :

    CLIENT : demain matin
    BOT    : Pouvez-vous me donner la date exacte ?
    CLIENT : Je t'ai dit demain matin.

Le jour était perdu faute de pouvoir être stocké seul, et le bot
réclamait une information déjà donnée jusqu'à épuiser la patience du
client puis déclencher le repli.
"""

from dataclasses import dataclass, replace
from datetime import date as _date
from datetime import datetime
from datetime import time as _time


@dataclass(frozen=True, slots=True)
class ReservationIntent:
    date: _date | None = None
    time: _time | None = None
    party_size: int | None = None
    customer_name: str | None = None

    @property
    def date_time(self) -> datetime | None:
        """Composition des deux moitiés, ou `None` s'il en manque une."""
        if self.date is None or self.time is None:
            return None
        return datetime.combine(self.date, self.time)

    def with_date(self, day: _date) -> "ReservationIntent":
        return replace(self, date=day)

    def with_time(self, hour: _time) -> "ReservationIntent":
        return replace(self, time=hour)

    def with_date_time(self, dt: datetime) -> "ReservationIntent":
        return replace(self, date=dt.date(), time=dt.time())

    def with_party_size(self, n: int) -> "ReservationIntent":
        if n <= 0:
            raise ValueError("party_size must be > 0")
        return replace(self, party_size=n)

    def with_customer_name(self, name: str) -> "ReservationIntent":
        if not name or not name.strip():
            raise ValueError("customer_name must not be blank")
        return replace(self, customer_name=name.strip())

    def without(self, slot: str) -> "ReservationIntent":
        """Efface un slot contesté par le client.

        Indispensable : une erreur de transcription remplit un slot, et
        sans moyen de l'annuler elle devient définitive. En appel réel le
        bot a appelé le client « Monsieur Medica » jusqu'à ce qu'il
        raccroche, sans jamais pouvoir revenir en arrière.
        """
        if slot not in {"date", "time", "party_size", "customer_name"}:
            return self
        return replace(self, **{slot: None})

    def missing_slots(self) -> set[str]:
        missing = set()
        if self.date is None:
            missing.add("date")
        if self.time is None:
            missing.add("time")
        if self.party_size is None:
            missing.add("party_size")
        if self.customer_name is None:
            missing.add("customer_name")
        return missing

    def is_complete(self) -> bool:
        return not self.missing_slots()
