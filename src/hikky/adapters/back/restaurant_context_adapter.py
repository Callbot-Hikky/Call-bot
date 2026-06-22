from hikky.adapters.back.contracts import Endpoints, RestaurantContextResponse
from hikky.adapters.back.http_client import BackHttpClient, NotFound
from hikky.domain.restaurant_context import RestaurantContext
from hikky.exceptions import UnknownRestaurant
from hikky.ports.restaurant_context import RestaurantContextPort


class BackHttpRestaurantContextAdapter(RestaurantContextPort):
    def __init__(self, client: BackHttpClient) -> None:
        self._client = client

    async def load(self, called_number: str) -> RestaurantContext:
        path = Endpoints.RESTAURANT_BY_PHONE.format(phone_number=called_number)
        try:
            payload = await self._client.get(path)
        except NotFound as exc:
            raise UnknownRestaurant(called_number) from exc
        parsed = RestaurantContextResponse.model_validate(payload)
        return RestaurantContext(
            id=parsed.id,
            name=parsed.name,
            greeting=parsed.greeting,
            opening_hours=parsed.opening_hours,
            total_capacity=parsed.total_capacity,
            rules=parsed.rules,
            transfer_number=parsed.transfer_number,
            fallback_message=parsed.fallback_message,
        )
