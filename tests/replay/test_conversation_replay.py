from datetime import datetime

from hikky.domain.outcomes import CallOutcome
from tests.replay.conversation_replay import (
    ConversationReplay,
    ExpectBackCalled,
    ExpectBotContains,
    ExpectOutcome,
    UserSays,
)


async def test_replay_runs_simple_reservation():
    replay = ConversationReplay()
    replay.with_default_context()
    replay.with_availability(True)
    replay.with_llm_replies(
        ["À quelle heure ?", "Combien de personnes ?", "À quel nom ?", "Confirmé."]
    )
    await replay.run(
        [
            UserSays("Je veux réserver"),
            ExpectBotContains("heure"),
            UserSays(
                "Demain à 20h",
                slot_updates={"date_time": datetime(2026, 7, 2, 20)},
            ),
            UserSays("Pour 4", slot_updates={"party_size": 4}),
            UserSays("Dupont", slot_updates={"customer_name": "Dupont"}),
            ExpectBackCalled("create"),
            ExpectOutcome(CallOutcome.RESERVATION_CREATED),
        ],
        customer_phone="+33600000000",
    )
