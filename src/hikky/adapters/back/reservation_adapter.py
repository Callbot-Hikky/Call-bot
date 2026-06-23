from datetime import datetime

from hikky.adapters.back.contracts import (
    IDEMPOTENCY_KEY_HEADER,
    AvailabilityRequest,
    AvailabilityResponse,
    CallbackRequestPayload,
    CallbackRequestResponse,
    Endpoints,
    ReservationCreateRequest,
    ReservationCreateResponse,
    build_reservation_idempotency_key,
)
from hikky.adapters.back.http_client import BackHttpClient
from hikky.observability.logging import get_call_context
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

        # Idempotence : si on connaît le call_id en cours via le ContextVar
        # d'observabilité, on en dérive une clé que le Back utilise pour
        # dédupliquer les retries (cf. contracts.build_reservation_idempotency_key).
        headers: dict[str, str] | None = None
        call_id = get_call_context().get("call_id")
        if call_id:
            headers = {
                IDEMPOTENCY_KEY_HEADER: build_reservation_idempotency_key(
                    call_id=call_id,
                    restaurant_id=restaurant_id,
                    date_time=date_time,
                    party_size=party_size,
                )
            }

        payload = await self._client.post(path, json=body, headers=headers)
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
