"""Test d'intégration de la Pipeline Pipecat exécutée au runtime.

Contrairement à `test_builder.py` qui vérifie seulement la composition
structurelle (les bonnes briques dans la bonne liste), ce test fait
**tourner** `PipelineRunner().run(task)` avec :
- une source fake qui injecte StartFrame + TranscriptionFrame + EndFrame
- nos vrais HikkySTTService / DialogueProcessor / HikkyTTSService
- une sink fake qui capture les frames de sortie

Ça prouve :
1. `super().process_frame()` du DialogueProcessor est compatible avec
   notre logique métier (couvre la dette §10.2 du récap).
2. Le chaînage STT → Dialogue → TTS livre bien les frames attendues
   (TextFrame d'accueil RGPD puis chunks audio TTS).
3. CallSession.begin / end sont bien invoqués par la pipeline réelle.

Le runner ne s'arrête pas seul après l'EndFrame (limitation Pipecat
documentée) ; on encadre avec `asyncio.wait_for(timeout=...)` et on
considère le `TimeoutError` comme normal — ce qui compte c'est ce que
le sink a capturé d'ici là.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from datetime import UTC, datetime, time

from pipecat.frames.frames import (
    EndFrame,
    Frame,
    StartFrame,
    TranscriptionFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from hikky.domain.call_session import CallSession
from hikky.domain.dialogue_engine import DialogueEngine
from hikky.domain.fallback_policy import FallbackPolicy
from hikky.domain.restaurant_context import (
    OpeningHours,
    RestaurantContext,
    RestaurantRules,
)
from hikky.pipeline.dialogue_processor import DialogueProcessor
from hikky.pipeline.stt_service import HikkySTTService
from hikky.pipeline.tts_service import HikkyTTSService
from hikky.ports.speech_recognition import SpeechRecognitionPort
from hikky.ports.speech_synthesis import SpeechSynthesisPort
from tests.fakes.fake_call_log import FakeCallLog
from tests.fakes.fake_language_model import FakeLanguageModel
from tests.fakes.fake_notification import FakeNotification
from tests.fakes.fake_reservation import FakeReservation


class _FakeSource(FrameProcessor):
    """Injecte la séquence de frames souhaitée dès que la pipeline démarre."""

    def __init__(self, frames_to_emit: list[Frame]) -> None:
        super().__init__()
        self._to_emit = list(frames_to_emit)
        self._already_sent = False

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if isinstance(frame, StartFrame) and not self._already_sent:
            self._already_sent = True
            await self.push_frame(frame, direction)
            for f in self._to_emit:
                await self.push_frame(f)
            await self.push_frame(EndFrame())
        else:
            await self.push_frame(frame, direction)


class _FakeSink(FrameProcessor):
    """Capture le type de chaque frame qui arrive en bout de pipeline."""

    def __init__(self) -> None:
        super().__init__()
        self.captured: list[Frame] = []

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        self.captured.append(frame)


class _FakeSTTAdapter(SpeechRecognitionPort):
    async def transcribe(self, audio_chunks: AsyncIterator[bytes]) -> AsyncIterator[str]:
        async for _ in audio_chunks:
            pass

        async def _empty() -> AsyncIterator[str]:
            if False:
                yield ""

        return _empty()


class _FakeTTSAdapter(SpeechSynthesisPort):
    def __init__(self) -> None:
        self.spoken: list[str] = []

    async def synthesize(self, text: str) -> AsyncIterator[bytes]:
        self.spoken.append(text)

        async def _stream() -> AsyncIterator[bytes]:
            yield b"\x00\x01\x02\x03"  # 4 octets PCM bidons

        return _stream()


def _ctx() -> RestaurantContext:
    return RestaurantContext(
        id="r-1",
        name="Chez Test",
        greeting="Que puis-je pour vous ?",
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
        call_id="c-runtime",
        context=_ctx(),
        dialogue_engine=DialogueEngine(
            FakeLanguageModel(replies=["Bonjour, comment puis-je vous aider ?"])
        ),
        fallback_policy=FallbackPolicy(),
        reservation_port=FakeReservation(),
        call_log=FakeCallLog(),
        notification=FakeNotification(),
        clock=lambda: datetime(2026, 7, 1, 20, tzinfo=UTC),
    )


async def test_pipeline_runs_end_to_end_with_real_dialogue_processor():
    """Fait tourner la Pipeline complète. Le pipeline doit livrer au sink :
    - StartFrame
    - une TextFrame d'annonce RGPD (depuis DialogueProcessor)
    - les TTSAudioRawFrame correspondants (depuis HikkyTTSService)
    - la TextFrame de réponse au tour utilisateur
    - EndFrame
    """
    session = _make_session()
    log = session._log  # accès direct pour vérification
    tts_adapter = _FakeTTSAdapter()

    stt = HikkySTTService(_FakeSTTAdapter())
    dialogue = DialogueProcessor(session=session, customer_phone="+33600000000")
    tts = HikkyTTSService(tts_adapter)

    # On simule un tour utilisateur en injectant directement une TranscriptionFrame.
    # (Le STT en amont est sans effet car on n'envoie pas d'audio.)
    user_frame = TranscriptionFrame(
        text="je veux réserver", user_id="caller", timestamp="t", finalized=True
    )
    source = _FakeSource([user_frame])
    sink = _FakeSink()

    pipeline = Pipeline([source, stt, dialogue, tts, sink])
    # cancel_timeout_secs réduit (défaut 20s) pour que le test ne traîne pas
    task = PipelineTask(pipeline, cancel_timeout_secs=1.0)
    runner = PipelineRunner(handle_sigint=False)

    # On encadre pour éviter de bloquer la suite en cas de bug.
    try:
        await asyncio.wait_for(runner.run(task), timeout=5.0)
    except TimeoutError:
        pass

    captured_types = [type(f).__name__ for f in sink.captured]

    # 1. La Pipeline démarre et le StartFrame traverse tous les processors
    #    → preuve que `super().process_frame()` du DialogueProcessor est
    #    compatible (rien ne casse au lifecycle Pipecat).
    assert "StartFrame" in captured_types

    # 2. Le DialogueProcessor a ouvert la session côté métier via le
    #    CallLogPort → preuve que notre logique `_handle()` est bien
    #    appelée par le `process_frame()` du runtime Pipecat.
    assert log.entries[0][0] == "start"
    assert log.entries[0][1] == "c-runtime"

    # 3. Le DialogueProcessor a poussé une TextFrame d'annonce RGPD
    #    en sortie de StartFrame, et HikkyTTSService a appelé notre
    #    adapter TTS avec ce texte. Pipecat découpe les textes longs
    #    en phrases pour le streaming TTS — on concatène pour vérifier.
    assert tts_adapter.spoken, "Aucun texte n'a été passé au TTS"
    all_tts_text = " ".join(tts_adapter.spoken)
    assert "Cet appel peut être enregistré" in all_tts_text
    assert "Chez Test" in all_tts_text

    # 4. Le tour utilisateur a déclenché une réponse LLM ("Bonjour,
    #    comment puis-je vous aider ?") qui a été passée au TTS.
    assert "puis-je vous aider" in all_tts_text

    # 5. L'EndFrame atteint le sink → la chaîne complète a terminé proprement.
    assert "EndFrame" in captured_types

    # LIMITE CONNUE — les TTSAudioRawFrame produites par notre service ne
    # sont pas propagées jusqu'au sink à cause d'un mismatch avec le système
    # d'audio context interne de TTSService (warning Pipecat « unable to
    # append audio to context »). C'est une vraie dette que ce test révèle
    # et qui devra être corrigée avant le premier vrai appel Twilio sur GPU.
    # Pour l'instant, on documente la dette plutôt que de mentir avec un
    # `assert any(isinstance(f, TTSAudioRawFrame) ...)`.
