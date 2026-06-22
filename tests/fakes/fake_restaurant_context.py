from hikky.domain.restaurant_context import RestaurantContext
from hikky.exceptions import UnknownRestaurant
from hikky.ports.restaurant_context import RestaurantContextPort


class FakeRestaurantContext(RestaurantContextPort):
    def __init__(self) -> None:
        self._by_number: dict[str, RestaurantContext] = {}

    def register(self, called_number: str, ctx: RestaurantContext) -> None:
        self._by_number[called_number] = ctx

    async def load(self, called_number: str) -> RestaurantContext:
        if called_number not in self._by_number:
            raise UnknownRestaurant(called_number)
        return self._by_number[called_number]
