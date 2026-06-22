from datetime import datetime

from hikky.domain.outcomes import CallOutcome
from hikky.ports.call_log import CallLogPort


class FakeCallLog(CallLogPort):
    def __init__(self) -> None:
        self.entries: list[tuple] = []

    async def start(
        self, call_id: str, restaurant_id: str | None, started_at: datetime
    ) -> None:
        self.entries.append(("start", call_id, restaurant_id, started_at))

    async def end(
        self, call_id: str, outcome: CallOutcome, duration_seconds: float
    ) -> None:
        self.entries.append(("end", call_id, outcome, duration_seconds))
