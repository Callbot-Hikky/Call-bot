from datetime import datetime

from hikky.domain.dialogue_engine import DialogueEngine, TurnResult
from hikky.domain.reservation_intent import ReservationIntent
from tests.fakes.fake_language_model import FakeLanguageModel


async def test_engine_asks_for_missing_slot_when_intent_incomplete():
    llm = FakeLanguageModel(replies=["Pour quelle date ?"])
    engine = DialogueEngine(llm)
    result = await engine.handle_turn(
        user_text="Je voudrais réserver",
        current_intent=ReservationIntent(),
        slot_updates={},
    )
    assert isinstance(result, TurnResult)
    assert result.bot_says == "Pour quelle date ?"
    assert result.requires_availability_check is False
    assert result.intent_complete is False


async def test_engine_marks_complete_when_all_slots_present():
    llm = FakeLanguageModel(replies=["Très bien, je confirme votre réservation."])
    engine = DialogueEngine(llm)
    result = await engine.handle_turn(
        user_text="Au nom de Dupont",
        current_intent=ReservationIntent(
            date_time=datetime(2026, 7, 1, 20),
            party_size=4,
            customer_name=None,
        ),
        slot_updates={"customer_name": "Dupont"},
    )
    assert result.intent_complete is True
    assert result.updated_intent.customer_name == "Dupont"


async def test_engine_requires_availability_check_when_dt_and_size_arrive():
    llm = FakeLanguageModel(replies=["Je vérifie un instant."])
    engine = DialogueEngine(llm)
    result = await engine.handle_turn(
        user_text="Pour 4 personnes demain à 20h",
        current_intent=ReservationIntent(),
        slot_updates={"date_time": datetime(2026, 7, 1, 20), "party_size": 4},
    )
    assert result.requires_availability_check is True
