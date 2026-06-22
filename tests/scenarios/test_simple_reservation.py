from datetime import datetime

from hikky.domain.outcomes import CallOutcome
from tests.replay.conversation_replay import (
    ConversationReplay,
    ExpectBackCalled,
    ExpectOutcome,
    UserSays,
)


async def test_direct_simple_reservation_completes():
    replay = (
        ConversationReplay()
        .with_default_context()
        .with_availability(True)
        .with_llm_replies(
            ["Pour quand ?", "Combien de personnes ?", "À quel nom ?", "Confirmé."]
        )
    )
    await replay.run(
        [
            UserSays("Je voudrais réserver"),
            UserSays("Demain 20h", slot_updates={"date_time": datetime(2026, 7, 2, 20)}),
            UserSays("Pour 4", slot_updates={"party_size": 4}),
            UserSays("Dupont", slot_updates={"customer_name": "Dupont"}),
            ExpectBackCalled("create"),
            ExpectOutcome(CallOutcome.RESERVATION_CREATED),
        ],
        customer_phone="+33600000000",
    )
