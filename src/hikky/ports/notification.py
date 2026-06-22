from abc import ABC, abstractmethod


class NotificationPort(ABC):
    @abstractmethod
    async def send_confirmation(self, reservation_id: str) -> None: ...
