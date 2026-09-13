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


class _PromptCapturingLLM:
    def __init__(self, reply="{}"):
        self.reply = reply
        self.prompts = []

    async def complete(self, messages):
        self.prompts.append(messages)
        return self.reply


async def test_prompt_forbids_inventing_an_unstated_time():
    """« demain matin » ne doit pas devenir « demain 9h00 ».

    Bug constate en appel reel : le bot a reserve a 9h alors que le
    client avait seulement dit « demain matin », puis a raccroche.
    """
    from hikky.pipeline.llm_slot_extractor import LLMSlotExtractor

    llm = _PromptCapturingLLM()
    await LLMSlotExtractor(llm).extract("demain matin")
    system = llm.prompts[0][0]["content"].lower()
    assert "matin" in system, "le prompt doit traiter explicitement les moments vagues"
    assert "n'invente" in system or "invente" in system


async def test_prompt_gives_a_vague_time_counterexample():
    from hikky.pipeline.llm_slot_extractor import LLMSlotExtractor

    llm = _PromptCapturingLLM()
    await LLMSlotExtractor(llm).extract("x")
    system = llm.prompts[0][0]["content"]
    assert "demain matin" in system and "{}" in system


async def test_vague_time_yields_no_date_time(monkeypatch):
    """Contrat de sortie : sur une reponse vide du LLM, aucun slot."""
    from hikky.pipeline.llm_slot_extractor import LLMSlotExtractor

    llm = _PromptCapturingLLM(reply="{}")
    assert await LLMSlotExtractor(llm).extract("demain matin") == {}


async def test_vague_time_still_captures_the_day():
    """« demain matin » doit donner le jour, sans inventer l'heure.

    Regression du scenario reel ou le bot, ayant tout perdu, reclamait
    « la date exacte » que le client venait de donner.
    """
    # Date calculee, jamais ecrite en dur : une date figee finit par
    # passer, et l'extracteur la reporte alors sur l'annee suivante —
    # le test tombait du jour au lendemain, sans qu'aucun code ne bouge.
    from datetime import date, timedelta

    from hikky.pipeline.llm_slot_extractor import LLMSlotExtractor

    demain = date.today() + timedelta(days=1)
    llm = _PromptCapturingLLM(reply=f'{{"date": "{demain.isoformat()}"}}')
    slots = await LLMSlotExtractor(llm).extract("demain matin")

    assert slots == {"date": demain}


async def test_explicit_time_yields_both_parts():
    from datetime import date, time, timedelta

    from hikky.pipeline.llm_slot_extractor import LLMSlotExtractor

    demain = date.today() + timedelta(days=1)
    llm = _PromptCapturingLLM(
        reply=f'{{"date": "{demain.isoformat()}", "time": "20:00"}}'
    )
    slots = await LLMSlotExtractor(llm).extract("demain à vingt heures")
    assert slots == {"date": demain, "time": time(20, 0)}


async def test_prompt_asks_for_separate_date_and_time():
    from hikky.pipeline.llm_slot_extractor import LLMSlotExtractor

    llm = _PromptCapturingLLM()
    await LLMSlotExtractor(llm).extract("x")
    system = llm.prompts[0][0]["content"]
    assert '"date"' in system and '"time"' in system


# ── Garde-fou sur l'annee ───────────────────────────────────────────────
#
# Le modele renvoie parfois son annee d'entrainement : sur « demain
# 12h45 » il a produit 2023-07-22 alors qu'on etait en 2026. La
# reservation serait partie trois ans dans le passe.

async def test_past_year_is_reinterpreted_as_the_next_occurrence():
    from datetime import date, datetime

    from hikky.pipeline.llm_slot_extractor import LLMSlotExtractor

    llm = _PromptCapturingLLM(reply='{"date": "2023-07-22"}')
    ex = LLMSlotExtractor(llm, clock=lambda: datetime(2026, 7, 21, 12, 0))
    assert (await ex.extract("demain"))["date"] == date(2026, 7, 22)


async def test_a_past_date_this_year_rolls_to_next_year():
    from datetime import date, datetime

    from hikky.pipeline.llm_slot_extractor import LLMSlotExtractor

    llm = _PromptCapturingLLM(reply='{"date": "2026-01-05"}')
    ex = LLMSlotExtractor(llm, clock=lambda: datetime(2026, 7, 21, 12, 0))
    assert (await ex.extract("le 5 janvier"))["date"] == date(2027, 1, 5)


async def test_today_is_kept_as_is():
    from datetime import date, datetime

    from hikky.pipeline.llm_slot_extractor import LLMSlotExtractor

    llm = _PromptCapturingLLM(reply='{"date": "2026-07-21"}')
    ex = LLMSlotExtractor(llm, clock=lambda: datetime(2026, 7, 21, 12, 0))
    assert (await ex.extract("aujourd'hui"))["date"] == date(2026, 7, 21)


async def test_absurdly_far_dates_are_dropped():
    from datetime import datetime

    from hikky.pipeline.llm_slot_extractor import LLMSlotExtractor

    llm = _PromptCapturingLLM(reply='{"date": "2099-01-01"}')
    ex = LLMSlotExtractor(llm, clock=lambda: datetime(2026, 7, 21, 12, 0))
    assert "date" not in await ex.extract("x")
