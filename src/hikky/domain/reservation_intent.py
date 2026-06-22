from dataclasses import dataclass, replace
from datetime import datetime


@dataclass(frozen=True, slots=True)
class ReservationIntent:
    date_time: datetime | None = None
    party_size: int | None = None
    customer_name: str | None = None

    def with_date_time(self, dt: datetime) -> "ReservationIntent":
        return replace(self, date_time=dt)

    def with_party_size(self, n: int) -> "ReservationIntent":
        if n <= 0:
            raise ValueError("party_size must be > 0")
        return replace(self, party_size=n)

    def with_customer_name(self, name: str) -> "ReservationIntent":
        if not name or not name.strip():
            raise ValueError("customer_name must not be blank")
        return replace(self, customer_name=name.strip())

    def missing_slots(self) -> set[str]:
        missing = set()
        if self.date_time is None:
            missing.add("date_time")
        if self.party_size is None:
            missing.add("party_size")
        if self.customer_name is None:
            missing.add("customer_name")
        return missing

    def is_complete(self) -> bool:
        return not self.missing_slots()
