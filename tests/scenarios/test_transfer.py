from datetime import time

from hikky.domain.outcomes import CallOutcome
from hikky.domain.restaurant_context import (
    OpeningHours,
    RestaurantContext,
    RestaurantRules,
)
from tests.replay.conversation_replay import (
    ConversationReplay,
    ExpectOutcome,
    UserSays,
)


def _ctx(transfer_number: str | None) -> RestaurantContext:
    return RestaurantContext(
        id="r-1",
        name="Chez Test",
        greeting="Bonjour.",
        opening_hours=[
            OpeningHours(weekday=d, opens=time(0, 0), closes=time(23, 59))
            for d in range(7)
        ],
        total_capacity=40,
        rules=RestaurantRules(),
        transfer_number=transfer_number,
        fallback_message="…",
    )


async def test_human_requested_with_transfer_configured():
    replay = (
        ConversationReplay()
        .with_context(_ctx(transfer_number="+33100000099"))
        .with_llm_replies(["Bien sûr, je transfère."])
    )
    await replay.run(
        [
            UserSays("Je veux parler à un humain", user_requested_human=True),
            ExpectOutcome(CallOutcome.TRANSFERRED),
        ]
    )


async def test_human_requested_without_transfer_falls_back_to_callback():
    replay = (
        ConversationReplay()
        .with_context(_ctx(transfer_number=None))
        .with_llm_replies(["Personne n'est dispo, je note un rappel."])
    )
    await replay.run(
        [
            UserSays("Je veux parler à un humain", user_requested_human=True),
            ExpectOutcome(CallOutcome.CALLBACK_REQUESTED),
        ],
        customer_phone="+33600000000",
    )
