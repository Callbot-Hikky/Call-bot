from datetime import datetime, time

from hikky.domain.outcomes import CallOutcome
from hikky.domain.restaurant_context import (
    OpeningHours,
    RestaurantContext,
    RestaurantRules,
)
from tests.replay.conversation_replay import (
    ConversationReplay,
    ExpectBackCalled,
    ExpectOutcome,
    UserSays,
)


def _ctx(rest_id: str, name: str) -> RestaurantContext:
    return RestaurantContext(
        id=rest_id,
        name=name,
        greeting=f"Bonjour, {name}.",
        opening_hours=[
            OpeningHours(weekday=d, opens=time(0, 0), closes=time(23, 59))
            for d in range(7)
        ],
        total_capacity=40,
        rules=RestaurantRules(max_group_size=8),
        transfer_number=None,
        fallback_message="…",
    )


async def test_two_parallel_sessions_do_not_leak_context():
    replay_a = (
        ConversationReplay()
        .with_context(_ctx("r-A", "Chez A"))
        .with_availability(True)
        .with_llm_replies(["Pour quand ?", "Combien ?", "À quel nom ?", "Confirmé chez A."])
    )
    replay_b = (
        ConversationReplay()
        .with_context(_ctx("r-B", "Chez B"))
        .with_availability(True)
        .with_llm_replies(["Pour quand ?", "Combien ?", "À quel nom ?", "Confirmé chez B."])
    )

    await replay_a.run(
        [
            UserSays("Réserver"),
            UserSays("20h", slot_updates={"date_time": datetime(2026, 7, 2, 20)}),
            UserSays("4", slot_updates={"party_size": 4}),
            UserSays("Alice", slot_updates={"customer_name": "Alice"}),
            ExpectBackCalled("create"),
            ExpectOutcome(CallOutcome.RESERVATION_CREATED),
        ],
        call_id="c-A",
    )
    await replay_b.run(
        [
            UserSays("Réserver"),
            UserSays("21h", slot_updates={"date_time": datetime(2026, 7, 2, 21)}),
            UserSays("2", slot_updates={"party_size": 2}),
            UserSays("Bob", slot_updates={"customer_name": "Bob"}),
            ExpectBackCalled("create"),
            ExpectOutcome(CallOutcome.RESERVATION_CREATED),
        ],
        call_id="c-B",
    )

    a_reservations = [
        r for r in replay_a.reservation.reservations.values() if r["restaurant_id"] == "r-A"
    ]
    b_reservations = [
        r for r in replay_b.reservation.reservations.values() if r["restaurant_id"] == "r-B"
    ]
    assert len(a_reservations) == 1
    assert a_reservations[0]["customer_name"] == "Alice"
    assert len(b_reservations) == 1
    assert b_reservations[0]["customer_name"] == "Bob"
    assert all(r["customer_name"] != "Bob" for r in a_reservations)
    assert all(r["customer_name"] != "Alice" for r in b_reservations)
