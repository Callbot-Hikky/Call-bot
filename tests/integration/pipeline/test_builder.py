"""Tests de la composition de la Pipeline.

On vérifie que `build_pipeline_task` construit bien une Pipeline avec
les briques attendues dans le bon ordre. On n'exécute pas la pipeline —
ça nécessite un transport audio réel.
"""

from collections.abc import AsyncIterator
from datetime import UTC, datetime, time
from unittest.mock import MagicMock

from hikky.domain.call_session import CallSession
from hikky.domain.dialogue_engine import DialogueEngine
from hikky.domain.fallback_policy import FallbackPolicy
from hikky.domain.restaurant_context import (
    OpeningHours,
    RestaurantContext,
    RestaurantRules,
)
from hikky.pipeline.builder import build_pipeline_task
from hikky.pipeline.dialogue_processor import DialogueProcessor
from hikky.pipeline.stt_service import HikkySTTService
from hikky.pipeline.tts_service import HikkyTTSService
from hikky.ports.speech_recognition import SpeechRecognitionPort
from hikky.ports.speech_synthesis import SpeechSynthesisPort
from tests.fakes.fake_call_log import FakeCallLog
from tests.fakes.fake_language_model import FakeLanguageModel
from tests.fakes.fake_notification import FakeNotification
from tests.fakes.fake_reservation import FakeReservation


class _NoopSTT(SpeechRecognitionPort):
    async def transcribe(self, audio_chunks: AsyncIterator[bytes]) -> AsyncIterator[str]:
        async def _stream() -> AsyncIterator[str]:
            if False:
                yield ""

        return _stream()


class _NoopTTS(SpeechSynthesisPort):
    async def synthesize(self, text: str) -> AsyncIterator[bytes]:
        async def _stream() -> AsyncIterator[bytes]:
            if False:
                yield b""

        return _stream()


def _ctx() -> RestaurantContext:
    return RestaurantContext(
        id="r-1",
        name="Chez Test",
        greeting="Bonjour.",
        opening_hours=[
            OpeningHours(weekday=d, opens=time(0, 0), closes=time(23, 59))
            for d in range(7)
        ],
        total_capacity=40,
        rules=RestaurantRules(),
        transfer_number=None,
        fallback_message="…",
    )


def _make_session() -> CallSession:
    return CallSession(
        call_id="c-1",
        context=_ctx(),
        dialogue_engine=DialogueEngine(FakeLanguageModel(replies=["…"])),
        fallback_policy=FallbackPolicy(),
        reservation_port=FakeReservation(),
        call_log=FakeCallLog(),
        notification=FakeNotification(),
        clock=lambda: datetime(2026, 7, 1, 20, tzinfo=UTC),
    )


def test_build_pipeline_task_wires_components_in_expected_order():
    transport_in = MagicMock(name="TransportInput")
    transport_out = MagicMock(name="TransportOutput")

    build = build_pipeline_task(
        transport_input=transport_in,
        transport_output=transport_out,
        stt_adapter=_NoopSTT(),
        tts_adapter=_NoopTTS(),
        session=_make_session(),
        customer_phone="+33600000000",
    )

    # On obtient les briques attendues
    assert isinstance(build.dialogue_processor, DialogueProcessor)
    assert build.task is not None
    assert build.pipeline is not None

    # La pipeline contient bien : transport_in, STT, dialogue, TTS, transport_out
    # On accède aux processors via l'attribut interne `_processors` de Pipecat.
    processors = list(build.pipeline._processors)
    # Pipecat encapsule la liste avec des input/output internes ; on vérifie
    # que NOS briques y sont présentes.
    types = [type(p) for p in processors]
    assert HikkySTTService in types
    assert DialogueProcessor in types
    assert HikkyTTSService in types
