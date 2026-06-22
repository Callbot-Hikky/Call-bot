from abc import ABC, abstractmethod
from datetime import datetime


class ReservationPort(ABC):
    @abstractmethod
    async def check_availability(
        self, restaurant_id: str, date_time: datetime, party_size: int
    ) -> bool: ...

    @abstractmethod
    async def create(
        self,
        restaurant_id: str,
        date_time: datetime,
        party_size: int,
        customer_name: str,
        customer_phone: str | None,
    ) -> str:
        """Renvoie l'identifiant de la réservation créée."""

    @abstractmethod
    async def create_callback_request(
        self,
        restaurant_id: str,
        customer_phone: str,
        preferred_slot: str | None,
        note: str,
    ) -> str:
        """Persiste une demande de rappel. Renvoie l'identifiant."""
