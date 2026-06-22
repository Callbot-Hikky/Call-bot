"""Tests du DialogueProcessor.

On utilise la mécanique Pipecat la plus minimale : on capture les frames
émises via push_frame en patchant la méthode. Pas besoin de monter une
Pipeline complète.
"""

from datetime import UTC, datetime, time
from typing import Any

import pytest
from pipecat.frames.frames import (
    EndFrame,
    StartFrame,
    TextFrame,
    TranscriptionFrame,
)
from pipecat.processors.frame_processor import FrameDirection

from hikky.domain.call_session import CallSession
from hikky.domain.dialogue_engine import DialogueEngine
from hikky.domain.fallback_policy import FallbackPolicy
from hikky.domain.restaurant_context import (
    OpeningHours,
    RestaurantContext,
    RestaurantRules,
)
from hikky.pipeline.dialogue_processor import DialogueProcessor, SlotExtractor
from tests.fakes.fake_call_log import FakeCallLog
from tests.fakes.fake_language_model import FakeLanguageModel
from tests.fakes.fake_notification import FakeNotification
from tests.fakes.fake_reservation import FakeReservation


def _ctx(transfer_number: str | None = None) -> RestaurantContext:
    return RestaurantContext(
        id="r-1",
        name="Chez Test",
        greeting="Que puis-je pour vous ?",
        opening_hours=[
            OpeningHours(weekday=d, opens=time(0, 0), closes=time(23, 59))
            for d in range(7)
        ],
        total_capacity=40,
        rules=RestaurantRules(max_group_size=8),
        transfer_number=transfer_number,
        fallback_message="Désolé, un membre de l'équipe vous rappellera.",
    )


def _make_session(
    llm_replies: list[str],
    *,
    transfer_number: str | None = None,
) -> tuple[CallSession, FakeReservation, FakeCallLog, FakeNotification]:
    log = FakeCallLog()
    reservation = FakeReservation()
    notif = FakeNotification()
    session = CallSession(
        call_id="c-1",
        context=_ctx(transfer_number=transfer_number),
        dialogue_engine=DialogueEngine(FakeLanguageModel(replies=llm_replies)),
        fallback_policy=FallbackPolicy(),
        reservation_port=reservation,
        call_log=log,
        notification=notif,
        clock=lambda: datetime(2026, 7, 1, 20, 0, tzinfo=UTC),
    )
    return session, reservation, log, notif


def _captured_processor(processor: DialogueProcessor) -> list:
    captured: list = []

    async def _capture(frame, direction=FrameDirection.DOWNSTREAM):
        captured.append((frame, direction))

    processor.push_frame = _capture  # type: ignore[method-assign]
    return captured


class _FakeSlotExtractor(SlotExtractor):
    def __init__(self, by_text: dict[str, dict[str, Any]]) -> None:
        self._by_text = by_text

    async def extract(self, user_text: str) -> dict[str, Any]:
        return self._by_text.get(user_text, {})


async def test_start_frame_triggers_rgpd_announcement_and_session_begin():
    session, _, log, _ = _make_session(llm_replies=[])
    processor = DialogueProcessor(session=session)
    captured = _captured_processor(processor)

    await processor._handle(StartFrame(), FrameDirection.DOWNSTREAM)

    # Logs : début d'appel enregistré
    assert log.entries[0][0] == "start"
    # Frames poussées : StartFrame propagée, puis TextFrame d'annonce
    assert isinstance(captured[0][0], StartFrame)
    text_frame = captured[1][0]
    assert isinstance(text_frame, TextFrame)
    assert "Cet appel peut être enregistré" in text_frame.text
    assert "Chez Test" in text_frame.text
    assert "Que puis-je" in text_frame.text  # greeting


async def test_transcription_frame_runs_a_turn_and_emits_llm_reply():
    session, _, _, _ = _make_session(llm_replies=["Pour quelle date ?"])
    processor = DialogueProcessor(session=session)
    await processor._handle(StartFrame(), FrameDirection.DOWNSTREAM)
    captured = _captured_processor(processor)

    await processor._handle(
        TranscriptionFrame(
            text="je veux réserver",
            user_id="caller",
            timestamp="now",
            finalized=True,
        ),
        FrameDirection.DOWNSTREAM,
    )

    text_frames = [f for f, _d in captured if isinstance(f, TextFrame)]
    assert len(text_frames) == 1
    assert text_frames[0].text == "Pour quelle date ?"


async def test_finalize_on_complete_intent_emits_confirmation_and_end():
    extractor = _FakeSlotExtractor(
        {
            "demain 20h pour 4": {
                "date_time": datetime(2026, 7, 2, 20, tzinfo=UTC).replace(tzinfo=None),
                "party_size": 4,
            },
            "dupont": {"customer_name": "Dupont"},
        }
    )
    session, reservation, _, notif = _make_session(
        llm_replies=["Pour qui ?", "Confirmé."]
    )
    reservation.set_availability("r-1", available=True)
    processor = DialogueProcessor(
        session=session, customer_phone="+33600000000", slot_extractor=extractor
    )
    await processor._handle(StartFrame(), FrameDirection.DOWNSTREAM)
    captured = _captured_processor(processor)

    await processor._handle(
        TranscriptionFrame(
            text="demain 20h pour 4", user_id="x", timestamp="t", finalized=True
        ),
        FrameDirection.DOWNSTREAM,
    )
    await processor._handle(
        TranscriptionFrame(text="dupont", user_id="x", timestamp="t", finalized=True),
        FrameDirection.DOWNSTREAM,
    )

    text_frames = [f.text for f, _d in captured if isinstance(f, TextFrame)]
    assert "Confirmé." in text_frames
    assert any(isinstance(f, EndFrame) for f, _d in captured)
    assert len(reservation.reservations) == 1
    assert notif.sent


async def test_oversized_group_triggers_fallback_and_end():
    extractor = _FakeSlotExtractor({"on est 30": {"party_size": 30}})
    session, reservation, _, _ = _make_session(llm_replies=["Combien êtes-vous ?"])
    processor = DialogueProcessor(
        session=session, customer_phone="+33600000000", slot_extractor=extractor
    )
    await processor._handle(StartFrame(), FrameDirection.DOWNSTREAM)
    captured = _captured_processor(processor)

    await processor._handle(
        TranscriptionFrame(text="on est 30", user_id="x", timestamp="t", finalized=True),
        FrameDirection.DOWNSTREAM,
    )

    text_frames = [f.text for f, _d in captured if type(f) is TextFrame]
    assert any("rappellera" in t for t in text_frames)
    assert any(isinstance(f, EndFrame) for f, _d in captured)
    assert len(reservation.callback_requests) == 1


async def test_non_finalized_transcription_is_ignored():
    session, _, _, _ = _make_session(llm_replies=["—"])
    processor = DialogueProcessor(session=session)
    await processor._handle(StartFrame(), FrameDirection.DOWNSTREAM)
    captured = _captured_processor(processor)

    await processor._handle(
        TranscriptionFrame(text="...", user_id="x", timestamp="t", finalized=False),
        FrameDirection.DOWNSTREAM,
    )
    # La frame doit être propagée mais aucun tour engagé → pas de TextFrame "réponse bot"
    # (on filtre par type exact car TranscriptionFrame hérite de TextFrame)
    pure_text_frames = [f for f, _d in captured if type(f) is TextFrame]
    assert pure_text_frames == []


async def test_processor_is_idempotent_after_end():
    session, reservation, _, _ = _make_session(llm_replies=["a", "b", "c", "d"])
    processor = DialogueProcessor(session=session, customer_phone="+33600000000")
    await processor._handle(StartFrame(), FrameDirection.DOWNSTREAM)
    # Forcer un fallback en simulant 3 tours sans avancée
    captured = _captured_processor(processor)
    for _ in range(3):
        await processor._handle(
            TranscriptionFrame(text="?", user_id="x", timestamp="t", finalized=True),
            FrameDirection.DOWNSTREAM,
        )
    # Une fois closed, plus rien ne se passe
    captured.clear()
    await processor._handle(
        TranscriptionFrame(text="encore", user_id="x", timestamp="t", finalized=True),
        FrameDirection.DOWNSTREAM,
    )
    assert captured == []


@pytest.fixture(autouse=True)
def _silence_fake_llm_exhausted(monkeypatch):
    """Empêche FakeLanguageModel de lever quand le test n'a pas calé les réponses
    à la perfection — on garde un fallback explicite."""
    # noop, juste pour documenter ; chaque test gère ses replies
    yield
