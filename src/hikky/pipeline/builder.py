"""Composition d'une Pipeline Pipecat pour un appel Hikky.

`build_pipeline_task(...)` assemble :

    Transport (WS Twilio, in)
      ↓ AudioRawFrame
    HikkySTTService (wrappant un SpeechRecognitionPort)
      ↓ TranscriptionFrame
    DialogueProcessor (avec CallSession + slot_extractor)
      ↓ TextFrame
    HikkyTTSService (wrappant un SpeechSynthesisPort)
      ↓ TTSAudioRawFrame
    Transport (WS Twilio, out)

L'appelle dans l'endpoint FastAPI WebSocket de Twilio. Toute la logique
métier reste dans `CallSession` ; Pipecat ne fait que la plomberie audio
+ orchestration des frames.

Important : on **n'exécute pas** la pipeline dans les tests ici — ça
exigerait une vraie WebSocket avec audio. Les tests valident que la
composition se construit avec les bonnes briques. La validation finale
se fait sur la machine GPU avec un vrai appel Twilio.
"""

from dataclasses import dataclass

from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.task import PipelineParams, PipelineTask

from hikky.domain.call_session import CallSession
from hikky.pipeline.dialogue_processor import DialogueProcessor, SlotExtractor
from hikky.pipeline.stt_service import HikkySTTService
from hikky.pipeline.tts_service import HikkyTTSService
from hikky.ports.speech_recognition import SpeechRecognitionPort
from hikky.ports.speech_synthesis import SpeechSynthesisPort


@dataclass(slots=True)
class PipelineBuild:
    pipeline: Pipeline
    task: PipelineTask
    dialogue_processor: DialogueProcessor


def build_pipeline_task(
    *,
    transport_input,
    transport_output,
    stt_adapter: SpeechRecognitionPort,
    tts_adapter: SpeechSynthesisPort,
    session: CallSession,
    customer_phone: str | None = None,
    slot_extractor: SlotExtractor | None = None,
    tts_sample_rate: int = 22050,
) -> PipelineBuild:
    """Assemble la Pipeline. `transport_input` / `transport_output` sont
    les processors d'entrée/sortie du `FastAPIWebsocketTransport`."""

    stt = HikkySTTService(stt_adapter)
    tts = HikkyTTSService(tts_adapter, sample_rate=tts_sample_rate)
    dialogue = DialogueProcessor(
        session=session,
        customer_phone=customer_phone,
        slot_extractor=slot_extractor,
    )

    pipeline = Pipeline(
        [
            transport_input,
            stt,
            dialogue,
            tts,
            transport_output,
        ]
    )

    task = PipelineTask(pipeline, params=PipelineParams())
    return PipelineBuild(pipeline=pipeline, task=task, dialogue_processor=dialogue)
