"""`DialogueProcessor` — le pont entre Pipecat et notre `CallSession`.

Position dans la pipeline :

    STT  →  DialogueProcessor  →  TTS

À la réception d'une `StartFrame` (début du run), il :
- ouvre la session (`session.begin()`),
- émet un `TextFrame` contenant l'**annonce RGPD + greeting** ; il sera
  consommé par le TTS.

À chaque `TranscriptionFrame` finalisée :
- appelle `CallSession.process_user_turn(text)`,
- vérifie `check_fallback(...)` ; si déclenché → émet le message de repli
  + `EndFrame`,
- sinon, tente `finalize_if_complete(...)` ; si la réservation est
  créée → émet la confirmation + `EndFrame`,
- sinon, émet la réponse du LLM (récupérée via `TurnResult.bot_says`)
  pour le tour suivant.

Toute autre `Frame` est propagée telle quelle.

L'extraction des slots (date_time, party_size, customer_name) à partir
du texte transcrit n'est PAS faite ici — c'est le rôle du LLM via
function-calling dans une future itération. Pour le POC, on passe
`slot_updates={}` ; le `DialogueEngine` se contente de relayer la
conversation et le LLM en dialogue ouvert. Les tests injectent les
slots quand ils veulent simuler une extraction.
"""

import logging

from pipecat.frames.frames import (
    EndFrame,
    Frame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    StartFrame,
    TextFrame,
    TranscriptionFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from hikky.domain.call_session import CallSession
from hikky.domain.turn_runner import NoOpSlotExtractor, SlotExtractor, run_turn

logger = logging.getLogger("hikky.dialogue")

RGPD_ANNOUNCEMENT_TEMPLATE = (
    "Bonjour, vous êtes en relation avec l'assistant vocal du restaurant "
    "{name}. Cet appel peut être enregistré pour améliorer le service. "
)


class DialogueProcessor(FrameProcessor):
    def __init__(
        self,
        *,
        session: CallSession,
        customer_phone: str | None = None,
        slot_extractor: "SlotExtractor | None" = None,
    ) -> None:
        super().__init__()
        self._session = session
        self._customer_phone = customer_phone
        self._slot_extractor = slot_extractor or _NoOpSlotExtractor()
        self._opened = False
        self._closed = False

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        await self._handle(frame, direction)

    async def _handle(self, frame: Frame, direction: FrameDirection) -> None:
        """Logique métier, testable sans la machinerie Pipecat (TaskManager)."""
        if isinstance(frame, StartFrame) and not self._opened:
            await self._open_call(frame, direction)
            return

        if isinstance(frame, TranscriptionFrame) and frame.finalized:
            await self._handle_user_turn(frame.text, direction)
            return

        # Tout le reste est propagé tel quel.
        await self.push_frame(frame, direction)

    async def _open_call(self, frame: StartFrame, direction: FrameDirection) -> None:
        await self.push_frame(frame, direction)  # propage la StartFrame d'abord
        await self._session.begin()
        self._opened = True
        announcement = (
            RGPD_ANNOUNCEMENT_TEMPLATE.format(name=self._session.context.name)
            + self._session.context.greeting
        )
        await self._emit_speech(announcement)

    async def _emit_speech(self, text: str) -> None:
        """Émet une séquence de frames consommable par Pipecat `TTSService`.

        `TTSService` n'initialise son `audio_context` qu'à la réception d'une
        `LLMFullResponseStartFrame` ; sans ça, les `TTSAudioRawFrame` produites
        par notre wrapper sont silencieusement droppées (« unable to append
        audio to context »). On encadre donc chaque TextFrame à prononcer
        par les frames de cycle d'une « réponse LLM » — ce qui revient à dire
        à Pipecat : voici un tour de parole du bot.
        """
        await self.push_frame(LLMFullResponseStartFrame())
        await self.push_frame(TextFrame(text=text))
        await self.push_frame(LLMFullResponseEndFrame())

    async def _handle_user_turn(self, user_text: str, direction: FrameDirection) -> None:
        if self._closed:
            return

        decision = await run_turn(
            session=self._session,
            slot_extractor=self._slot_extractor,
            user_text=user_text,
            customer_phone=self._customer_phone,
            speak=self._emit_speech,
        )
        if decision.should_end:
            await self.push_frame(EndFrame())
            self._closed = True


# Réexportés pour compatibilité : `SlotExtractor` vit désormais dans le
# domaine, pour que le chemin AudioSocket puisse l'utiliser sans Pipecat.
_NoOpSlotExtractor = NoOpSlotExtractor

__all__ = ["DialogueProcessor", "NoOpSlotExtractor", "SlotExtractor"]
