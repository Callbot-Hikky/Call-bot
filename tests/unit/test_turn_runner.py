"""Le tour de dialogue ne doit jamais reserver sans confirmation.

Bug constate en appel reel : des que les trois slots etaient remplis
(certains par invention du LLM), la reservation etait creee en base et
l'appel raccroche, sans jamais recapituler ni demander l'accord du
client.
"""

from dataclasses import dataclass
from datetime import datetime, time

from hikky.domain.outcomes import CallOutcome
from hikky.domain.restaurant_context import (
    OpeningHours,
    RestaurantContext,
    RestaurantRules,
)
from hikky.domain.turn_runner import run_turn


def _ctx():
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


@dataclass
class _Turn:
    bot_says: str
    updated_intent: object = None


class _Intent:
    def __init__(self, complete: bool) -> None:
        self._complete = complete
        self.date_time = datetime(2026, 7, 21, 20, 0)
        self.party_size = 4
        self.customer_name = "Dupont"

    def is_complete(self) -> bool:
        return self._complete


class _Session:
    def __init__(self, complete: bool = True) -> None:
        self.context = _ctx()
        self.finalize_calls = 0
        self.ended_with = None
        self.turns = []
        self._intent = _Intent(complete)

    @property
    def intent(self):
        return self._intent

    async def process_user_turn(self, user_text, slot_updates):
        self.turns.append((user_text, slot_updates))
        return _Turn(bot_says="d'accord")

    def check_fallback(self, **kw):
        return None

    async def finalize_if_complete(self, phone):
        self.finalize_calls += 1
        return CallOutcome.RESERVATION_CREATED

    async def end_with(self, outcome):
        self.ended_with = outcome


async def _run(session, text, spoken, confirmed=None):
    return await run_turn(
        session=session,
        slot_extractor=None,
        user_text=text,
        customer_phone=None,
        speak=lambda t: spoken.append(t) or _noop(),
    )


async def _noop():
    return None


async def test_complete_intent_asks_for_confirmation_instead_of_booking():
    session = _Session(complete=True)
    spoken: list[str] = []
    decision = await _run(session, "demain matin", spoken)
    assert session.finalize_calls == 0, "la reservation ne doit pas etre creee sans accord"
    assert decision.should_end is False, "l'appel ne doit pas raccrocher"


async def test_confirmation_question_recaps_the_reservation():
    session = _Session(complete=True)
    spoken: list[str] = []
    await _run(session, "demain matin", spoken)
    recap = " ".join(spoken).lower()
    assert "4" in recap
    assert "dupont" in recap.lower()


async def test_booking_happens_only_after_the_customer_agrees():
    session = _Session(complete=True)
    spoken: list[str] = []
    await _run(session, "demain a vingt heures", spoken)
    decision = await _run(session, "oui c'est ca", spoken)
    assert session.finalize_calls == 1
    assert decision.should_end is True


async def test_refusal_keeps_the_call_open_without_booking():
    session = _Session(complete=True)
    spoken: list[str] = []
    await _run(session, "demain a vingt heures", spoken)
    decision = await _run(session, "non pas du tout", spoken)
    assert session.finalize_calls == 0
    assert decision.should_end is False


async def test_incomplete_intent_never_asks_for_confirmation():
    session = _Session(complete=False)
    spoken: list[str] = []
    decision = await _run(session, "bonjour", spoken)
    assert session.finalize_calls == 0
    assert decision.should_end is False
    assert "confirmer" not in " ".join(spoken).lower()


# ── Restitution orale ───────────────────────────────────────────────────
#
# Appel reel : le recapitulatif annoncait « le 20/07 », le client a
# entendu « 27 » et a du corriger une date pourtant juste au depart.

def _session_with(day, hour, size=2, name="Ryan"):
    from datetime import date as _d
    from datetime import time as _t

    s = _Session(complete=True)
    s._intent.date = _d(*day)
    s._intent.time = _t(*hour)
    s._intent.date_time = __import__("datetime").datetime(*day, *hour)
    s._intent.party_size = size
    s._intent.customer_name = name
    return s


async def test_recap_spells_the_date_in_words_not_digits():
    from hikky.domain.turn_runner import _recap

    session = _session_with((2026, 7, 21), (12, 30))
    recap = _recap(session)
    assert "/" not in recap, f"format numerique illisible a l'oral: {recap}"
    assert "juillet" in recap


async def test_recap_says_the_weekday():
    from hikky.domain.turn_runner import _recap

    session = _session_with((2026, 7, 21), (12, 30))
    assert "mardi" in _recap(session).lower()


async def test_recap_uses_singular_for_one_person():
    from hikky.domain.turn_runner import _recap

    session = _session_with((2026, 7, 21), (20, 0), size=1)
    recap = _recap(session)
    assert "1 personne," in recap or "une personne" in recap
