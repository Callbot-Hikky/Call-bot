from hikky.ports.notification import NotificationPort


class FakeNotification(NotificationPort):
    def __init__(self) -> None:
        self.sent: list[str] = []

    async def send_confirmation(self, reservation_id: str) -> None:
        self.sent.append(reservation_id)
