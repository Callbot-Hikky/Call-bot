from datetime import UTC, datetime

from hikky.pipeline.llm_slot_extractor import LLMSlotExtractor
from tests.fakes.fake_language_model import FakeLanguageModel


def _clock_today() -> datetime:
    return datetime(2026, 7, 1, 18, 30, tzinfo=UTC)  # mercredi


async def test_extract_returns_empty_when_user_text_blank():
    llm = FakeLanguageModel(replies=["{}"])
    extractor = LLMSlotExtractor(llm, clock=_clock_today)
    assert await extractor.extract("") == {}
    assert await extractor.extract("   ") == {}
    # Le LLM ne doit pas être appelé pour du texte vide
    assert llm.calls == []


async def test_extract_returns_empty_when_llm_returns_empty_object():
    llm = FakeLanguageModel(replies=["{}"])
    extractor = LLMSlotExtractor(llm, clock=_clock_today)
    assert await extractor.extract("bonjour") == {}


async def test_extract_returns_empty_when_llm_returns_non_json():
    llm = FakeLanguageModel(replies=["Pas de slots ici."])
    extractor = LLMSlotExtractor(llm, clock=_clock_today)
    assert await extractor.extract("bonjour") == {}


async def test_extract_parses_date_time():
    llm = FakeLanguageModel(replies=['{"date_time": "2026-07-02T20:00"}'])
    extractor = LLMSlotExtractor(llm, clock=_clock_today)
    result = await extractor.extract("demain 20h")
    assert result == {"date_time": datetime(2026, 7, 2, 20, 0)}


async def test_extract_parses_party_size():
    llm = FakeLanguageModel(replies=['{"party_size": 4}'])
    extractor = LLMSlotExtractor(llm, clock=_clock_today)
    result = await extractor.extract("pour 4")
    assert result == {"party_size": 4}


async def test_extract_parses_customer_name():
    llm = FakeLanguageModel(replies=['{"customer_name": "Dupont"}'])
    extractor = LLMSlotExtractor(llm, clock=_clock_today)
    result = await extractor.extract("au nom de Dupont")
    assert result == {"customer_name": "Dupont"}


async def test_extract_parses_combined_slots():
    llm = FakeLanguageModel(
        replies=['{"date_time": "2026-07-02T20:00", "party_size": 4, "customer_name": "X"}']
    )
    extractor = LLMSlotExtractor(llm, clock=_clock_today)
    result = await extractor.extract("Demain 20h pour 4 au nom de X")
    assert result == {
        "date_time": datetime(2026, 7, 2, 20, 0),
        "party_size": 4,
        "customer_name": "X",
    }


async def test_extract_ignores_invalid_party_size():
    llm = FakeLanguageModel(replies=['{"party_size": 0}'])
    extractor = LLMSlotExtractor(llm, clock=_clock_today)
    assert await extractor.extract("aucun") == {}


async def test_extract_ignores_malformed_date_time():
    llm = FakeLanguageModel(replies=['{"date_time": "demain à 20h"}'])
    extractor = LLMSlotExtractor(llm, clock=_clock_today)
    assert await extractor.extract("demain") == {}


async def test_extract_ignores_blank_customer_name():
    llm = FakeLanguageModel(replies=['{"customer_name": "   "}'])
    extractor = LLMSlotExtractor(llm, clock=_clock_today)
    assert await extractor.extract("nom vide") == {}


async def test_extract_returns_empty_when_llm_raises():
    class _BoomLLM:
        async def complete(self, messages):
            raise RuntimeError("boom")

    extractor = LLMSlotExtractor(_BoomLLM(), clock=_clock_today)
    assert await extractor.extract("bonjour") == {}


async def test_extract_strips_extra_text_around_json():
    llm = FakeLanguageModel(replies=['Voici : {"party_size": 3} merci.'])
    extractor = LLMSlotExtractor(llm, clock=_clock_today)
    result = await extractor.extract("on est 3")
    assert result == {"party_size": 3}


async def test_extract_prompt_includes_today_and_weekday():
    llm = FakeLanguageModel(replies=["{}"])
    extractor = LLMSlotExtractor(llm, clock=_clock_today)
    await extractor.extract("salut")
    sent_system = llm.calls[0][0]["content"]
    assert "2026-07-01" in sent_system
    assert "mercredi" in sent_system
    # Le prompt doit aussi citer demain (2026-07-02) en exemple
    assert "2026-07-02" in sent_system
