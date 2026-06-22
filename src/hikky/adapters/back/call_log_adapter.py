from datetime import datetime

from hikky.adapters.back.contracts import CallEndRequest, CallStartRequest, Endpoints
from hikky.adapters.back.http_client import BackHttpClient
from hikky.domain.outcomes import CallOutcome
from hikky.ports.call_log import CallLogPort


class BackHttpCallLogAdapter(CallLogPort):
    def __init__(self, client: BackHttpClient) -> None:
        self._client = client

    async def start(
        self, call_id: str, restaurant_id: str | None, started_at: datetime
    ) -> None:
        body = CallStartRequest(
            restaurant_id=restaurant_id, started_at=started_at
        ).model_dump(mode="json")
        path = Endpoints.CALL_START.format(call_id=call_id)
        await self._client.post(path, json=body)

    async def end(
        self, call_id: str, outcome: CallOutcome, duration_seconds: float
    ) -> None:
        body = CallEndRequest(outcome=outcome, duration_seconds=duration_seconds).model_dump(
            mode="json"
        )
        path = Endpoints.CALL_END.format(call_id=call_id)
        await self._client.post(path, json=body)
