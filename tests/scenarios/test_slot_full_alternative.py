from datetime import datetime

from hikky.domain.outcomes import CallOutcome
from tests.replay.conversation_replay import (
    ConversationReplay,
    ExpectBackCalled,
    ExpectBotContains,
    ExpectOutcome,
    UserSays,
)


async def test_full_slot_then_alternative_accepted():
    replay = (
        ConversationReplay()
        .with_default_context()
        .with_availability(False)
        .with_llm_replies(
            [
                "Pour quand ?",
                "Combien de personnes ?",
                "Désolé, 20h est complet, je peux vous proposer 21h ?",
                "À quel nom ?",
                "Confirmé.",
            ]
        )
    )
    await replay.run(
        [
            UserSays("Je voudrais réserver"),
            UserSays("Demain 20h", slot_updates={"date_time": datetime(2026, 7, 2, 20)}),
            UserSays("Pour 4", slot_updates={"party_size": 4}),
            ExpectBotContains("21h"),
        ],
    )
    replay.with_availability(True)
    replay.reservation.set_availability("r-1", True)

    await replay.run(
        [
            UserSays(
                "Va pour 21h",
                slot_updates={"date_time": datetime(2026, 7, 2, 21), "party_size": 4},
            ),
            UserSays("Dupont", slot_updates={"customer_name": "Dupont"}),
            ExpectBackCalled("create"),
            ExpectOutcome(CallOutcome.RESERVATION_CREATED),
        ],
        call_id="c-2",
        customer_phone="+33600000000",
    )
