from abc import ABC, abstractmethod
from datetime import datetime

from hikky.domain.outcomes import CallOutcome


class CallLogPort(ABC):
    @abstractmethod
    async def start(
        self, call_id: str, restaurant_id: str | None, started_at: datetime
    ) -> None: ...

    @abstractmethod
    async def end(
        self, call_id: str, outcome: CallOutcome, duration_seconds: float
    ) -> None: ...
