from abc import ABC, abstractmethod

from hikky.domain.restaurant_context import RestaurantContext


class RestaurantContextPort(ABC):
    @abstractmethod
    async def load(self, called_number: str) -> RestaurantContext:
        """Renvoie le contexte pour le numéro appelé. Lève UnknownRestaurant si non trouvé."""
