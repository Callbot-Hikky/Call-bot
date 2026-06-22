from datetime import datetime

from hikky.adapters.back.contracts import (
    AvailabilityRequest,
    AvailabilityResponse,
    CallbackRequestPayload,
    CallbackRequestResponse,
    Endpoints,
    ReservationCreateRequest,
    ReservationCreateResponse,
)
from hikky.adapters.back.http_client import BackHttpClient
from hikky.ports.reservation import ReservationPort


class BackHttpReservationAdapter(ReservationPort):
    def __init__(self, client: BackHttpClient) -> None:
        self._client = client

    async def check_availability(
        self, restaurant_id: str, date_time: datetime, party_size: int
    ) -> bool:
        body = AvailabilityRequest(date_time=date_time, party_size=party_size).model_dump(
            mode="json"
        )
        path = Endpoints.AVAILABILITY.format(restaurant_id=restaurant_id)
        payload = await self._client.post(path, json=body)
        return AvailabilityResponse.model_validate(payload).available

    async def create(
        self,
        restaurant_id: str,
        date_time: datetime,
        party_size: int,
        customer_name: str,
        customer_phone: str | None,
    ) -> str:
        body = ReservationCreateRequest(
            date_time=date_time,
            party_size=party_size,
            customer_name=customer_name,
            customer_phone=customer_phone,
        ).model_dump(mode="json")
        path = Endpoints.RESERVATIONS.format(restaurant_id=restaurant_id)
        payload = await self._client.post(path, json=body)
        return ReservationCreateResponse.model_validate(payload).reservation_id

    async def create_callback_request(
        self,
        restaurant_id: str,
        customer_phone: str,
        preferred_slot: str | None,
        note: str,
    ) -> str:
        body = CallbackRequestPayload(
            customer_phone=customer_phone, preferred_slot=preferred_slot, note=note
        ).model_dump(mode="json")
        path = Endpoints.CALLBACK_REQUESTS.format(restaurant_id=restaurant_id)
        payload = await self._client.post(path, json=body)
        return CallbackRequestResponse.model_validate(payload).callback_request_id
