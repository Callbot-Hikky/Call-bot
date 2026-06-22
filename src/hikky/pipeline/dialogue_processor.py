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
from typing import Any

from pipecat.frames.frames import (
    EndFrame,
    Frame,
    StartFrame,
    TextFrame,
    TranscriptionFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from hikky.domain.call_session import CallSession
from hikky.domain.outcomes import CallOutcome
from hikky.observability.latency import measure_latency

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
        await self.push_frame(TextFrame(text=announcement))

    async def _handle_user_turn(self, user_text: str, direction: FrameDirection) -> None:
        if self._closed:
            return

        async with measure_latency("dialogue_turn"):
            slot_updates = await self._slot_extractor.extract(user_text)
            result = await self._session.process_user_turn(user_text, slot_updates)

        fallback = self._session.check_fallback(
            user_requested_human=False,  # détection texte → futur travail
            group_size=slot_updates.get("party_size"),
        )
        if fallback is not None:
            logger.info(
                "fallback triggered",
                extra={"outcome": str(fallback.outcome), "reason": fallback.reason},
            )
            if (
                fallback.outcome == CallOutcome.CALLBACK_REQUESTED
                and self._customer_phone is not None
            ):
                await self._session.request_callback(
                    customer_phone=self._customer_phone,
                    preferred_slot=None,
                    note=fallback.reason,
                )
            await self.push_frame(
                TextFrame(text=self._session.context.fallback_message)
            )
            await self._session.end_with(fallback.outcome)
            await self.push_frame(EndFrame())
            self._closed = True
            return

        outcome = await self._session.finalize_if_complete(self._customer_phone)
        if outcome is not None:
            logger.info("call finalized", extra={"outcome": str(outcome)})
            await self.push_frame(TextFrame(text=result.bot_says))
            await self.push_frame(EndFrame())
            self._closed = True
            return

        # Tour normal : on émet la réponse du LLM
        await self.push_frame(TextFrame(text=result.bot_says))


class SlotExtractor:
    """Interface async : extrait des slots structurés d'un tour de parole."""

    async def extract(self, user_text: str) -> dict[str, Any]:  # pragma: no cover
        raise NotImplementedError


class _NoOpSlotExtractor(SlotExtractor):
    async def extract(self, user_text: str) -> dict[str, Any]:
        return {}
