from datetime import datetime
from uuid import uuid4

from hikky.ports.reservation import ReservationPort


class FakeReservation(ReservationPort):
    def __init__(self) -> None:
        self._availability: dict[str, bool] = {}
        self.reservations: dict[str, dict] = {}
        self.callback_requests: dict[str, dict] = {}

    def set_availability(self, restaurant_id: str, available: bool) -> None:
        self._availability[restaurant_id] = available

    async def check_availability(
        self, restaurant_id: str, date_time: datetime, party_size: int
    ) -> bool:
        return self._availability.get(restaurant_id, False)

    async def create(
        self,
        restaurant_id: str,
        date_time: datetime,
        party_size: int,
        customer_name: str,
        customer_phone: str | None,
    ) -> str:
        rid = f"res-{uuid4().hex[:8]}"
        self.reservations[rid] = {
            "restaurant_id": restaurant_id,
            "date_time": date_time,
            "party_size": party_size,
            "customer_name": customer_name,
            "customer_phone": customer_phone,
        }
        return rid

    async def create_callback_request(
        self,
        restaurant_id: str,
        customer_phone: str,
        preferred_slot: str | None,
        note: str,
    ) -> str:
        cid = f"cb-{uuid4().hex[:8]}"
        self.callback_requests[cid] = {
            "restaurant_id": restaurant_id,
            "customer_phone": customer_phone,
            "preferred_slot": preferred_slot,
            "note": note,
        }
        return cid
