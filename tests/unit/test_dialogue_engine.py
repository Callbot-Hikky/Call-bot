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
            date=datetime(2026, 7, 1, 20).date(),
            time=datetime(2026, 7, 1, 20).time(),
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


class CapturingLLM:
    def __init__(self, replies=None):
        self.replies = list(replies or ["ok"])
        self.prompts = []

    async def complete(self, messages):
        self.prompts.append(messages)
        return self.replies.pop(0) if self.replies else "ok"


def _ctx():
    from datetime import time

    from hikky.domain.restaurant_context import (
        OpeningHours,
        RestaurantContext,
        RestaurantRules,
    )

    return RestaurantContext(
        id="r-1",
        name="Le Petit Sud",
        greeting="Bonjour",
        opening_hours=[OpeningHours(weekday=i, opens=time(9), closes=time(23)) for i in range(7)],
        total_capacity=40,
        rules=RestaurantRules(),
        transfer_number=None,
        fallback_message="Au revoir",
    )


async def test_second_turn_includes_first_turn_in_prompt():
    """Sans historique, le LLM repose les mêmes questions — c'est ce qui
    le fait passer pour bête en conversation."""
    llm = CapturingLLM(["Pour combien de personnes ?", "Quel jour ?"])
    engine = DialogueEngine(llm)
    await engine.handle_turn(
        user_text="Bonjour je veux reserver", current_intent=ReservationIntent(), slot_updates={}
    )
    await engine.handle_turn(
        user_text="Quatre", current_intent=ReservationIntent(), slot_updates={}
    )
    flat = " ".join(m["content"] for m in llm.prompts[1])
    assert "Bonjour je veux reserver" in flat
    assert "Pour combien de personnes ?" in flat


async def test_prompt_includes_restaurant_name_when_context_given():
    llm = CapturingLLM()
    engine = DialogueEngine(llm)
    await engine.handle_turn(
        user_text="bonjour",
        current_intent=ReservationIntent(),
        slot_updates={},
        context=_ctx(),
    )
    assert "Le Petit Sud" in llm.prompts[0][0]["content"]


async def test_prompt_lists_already_known_slots():
    llm = CapturingLLM()
    engine = DialogueEngine(llm)
    await engine.handle_turn(
        user_text="on serait quatre",
        current_intent=ReservationIntent(),
        slot_updates={"party_size": 4},
    )
    system = llm.prompts[0][0]["content"]
    assert "4" in system


async def test_prompt_instructs_short_spoken_answers():
    llm = CapturingLLM()
    engine = DialogueEngine(llm)
    await engine.handle_turn(
        user_text="bonjour", current_intent=ReservationIntent(), slot_updates={}
    )
    system = llm.prompts[0][0]["content"].lower()
    assert "phrase" in system or "court" in system or "bref" in system


async def test_history_is_bounded():
    llm = CapturingLLM(["r"] * 30)
    engine = DialogueEngine(llm)
    for i in range(15):
        await engine.handle_turn(
            user_text=f"message {i}", current_intent=ReservationIntent(), slot_updates={}
        )
    assert len(llm.prompts[-1]) <= 15


async def test_date_alone_is_applied_to_the_intent():
    """Sans ca, l'extracteur emet « date » et le moteur la jette.

    C'est le maillon qui manquait dans le scenario reel : le bot
    reclamait « la date exacte » alors qu'elle venait d'etre donnee.
    """
    from datetime import date

    llm = CapturingLLM(["À quelle heure ?"])
    engine = DialogueEngine(llm)
    result = await engine.handle_turn(
        user_text="demain matin",
        current_intent=ReservationIntent(),
        slot_updates={"date": date(2026, 7, 21)},
    )
    assert result.updated_intent.date == date(2026, 7, 21)
    assert result.updated_intent.time is None


async def test_time_alone_is_applied_to_the_intent():
    from datetime import time

    llm = CapturingLLM(["Très bien."])
    engine = DialogueEngine(llm)
    result = await engine.handle_turn(
        user_text="à vingt heures",
        current_intent=ReservationIntent(),
        slot_updates={"time": time(20, 0)},
    )
    assert result.updated_intent.time == time(20, 0)


async def test_prompt_lists_the_known_day_and_the_missing_hour():
    from datetime import date

    llm = CapturingLLM()
    engine = DialogueEngine(llm)
    await engine.handle_turn(
        user_text="demain matin",
        current_intent=ReservationIntent(),
        slot_updates={"date": date(2026, 7, 21)},
    )
    system = llm.prompts[0][0]["content"]
    assert "21" in system, "le jour connu doit apparaitre"
    assert "time" in system or "heure" in system.lower()


async def test_prompt_never_leaks_internal_slot_identifiers():
    """Appel reel : le bot a dit « Customer name, s'il vous plait ».

    Il recitait l'identifiant technique liste dans le prompt.
    """
    llm = CapturingLLM()
    engine = DialogueEngine(llm)
    await engine.handle_turn(
        user_text="bonjour", current_intent=ReservationIntent(), slot_updates={}
    )
    system = llm.prompts[0][0]["content"]
    for identifier in ("customer_name", "party_size", "date_time"):
        assert identifier not in system, f"identifiant technique fuite: {identifier}"
