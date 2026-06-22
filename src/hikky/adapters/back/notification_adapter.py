from hikky.adapters.back.contracts import Endpoints
from hikky.adapters.back.http_client import BackHttpClient
from hikky.ports.notification import NotificationPort


class BackHttpNotificationAdapter(NotificationPort):
    def __init__(self, client: BackHttpClient) -> None:
        self._client = client

    async def send_confirmation(self, reservation_id: str) -> None:
        path = Endpoints.NOTIFICATION_CONFIRMATION.format(reservation_id=reservation_id)
        await self._client.post(path)
