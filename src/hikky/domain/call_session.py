from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from hikky.domain.dialogue_engine import DialogueEngine, TurnResult
from hikky.domain.fallback_policy import (
    FallbackDecision,
    FallbackPolicy,
    is_clarification_request,
    is_correction,
)
from hikky.domain.outcomes import CallOutcome
from hikky.domain.reservation_intent import ReservationIntent
from hikky.domain.restaurant_context import RestaurantContext
from hikky.ports.call_log import CallLogPort
from hikky.ports.notification import NotificationPort
from hikky.ports.reservation import ReservationPort


@dataclass(slots=True)
class _SessionState:
    intent: ReservationIntent = field(default_factory=ReservationIntent)
    last_bot_message: str = ""
    consecutive_no_progress_turns: int = 0
    started_at: datetime | None = None
    ended: bool = False
    last_availability_check: bool | None = None
    last_within_opening_hours: bool | None = None


class CallSession:
    def __init__(
        self,
        *,
        call_id: str,
        context: RestaurantContext,
        dialogue_engine: DialogueEngine,
        fallback_policy: FallbackPolicy,
        reservation_port: ReservationPort,
        call_log: CallLogPort,
        notification: NotificationPort,
        clock: Callable[[], datetime],
    ) -> None:
        self.call_id = call_id
        self.context = context
        self._engine = dialogue_engine
        self._fallback = fallback_policy
        self._reservation = reservation_port
        self._log = call_log
        self._notif = notification
        self._clock = clock
        self._state = _SessionState()

    async def begin(self) -> None:
        now = self._clock()
        self._state.started_at = now
        await self._log.start(self.call_id, self.context.id, now)

    async def process_user_turn(
        self, user_text: str, slot_updates: dict[str, Any]
    ) -> TurnResult:
        before = self._state.intent
        result = await self._engine.handle_turn(
            user_text=user_text,
            current_intent=before,
            slot_updates=slot_updates,
            context=self.context,
        )
        intent = result.updated_intent

        if is_correction(user_text):
            # Le client dément une information retenue : on l'efface au
            # lieu de la lui resservir. Sans ça, une erreur de
            # transcription devient définitive.
            contested = _contested_slot(user_text, intent)
            if contested is not None:
                intent = intent.without(contested)

        if intent != before:
            self._state.consecutive_no_progress_turns = 0
        elif is_clarification_request(user_text) or is_correction(user_text):
            # Demande de répétition ou contestation : c'est le bot qui a
            # échoué, pas le client qui piétine. Ne pas l'incriminer.
            pass
        else:
            self._state.consecutive_no_progress_turns += 1
        self._state.intent = intent
        self._state.last_bot_message = result.bot_says
        return result

    def check_fallback(
        self, *, user_requested_human: bool, group_size: int | None
    ) -> FallbackDecision | None:
        return self._fallback.decide(
            self.context,
            consecutive_no_progress_turns=self._state.consecutive_no_progress_turns,
            user_requested_human=user_requested_human,
            group_size=group_size,
        )

    @property
    def intent(self) -> ReservationIntent:
        """Intention en cours de construction.

        Exposée pour que `run_turn` puisse récapituler la réservation
        avant de la créer — la barrière de confirmation en dépend.
        """
        return self._state.intent

    @property
    def last_availability_check(self) -> bool | None:
        return self._state.last_availability_check

    @property
    def last_within_opening_hours(self) -> bool | None:
        return self._state.last_within_opening_hours

    async def request_callback(
        self, *, customer_phone: str, preferred_slot: str | None, note: str
    ) -> str:
        """Crée une demande de rappel via le ReservationPort. Renvoie l'id."""
        return await self._reservation.create_callback_request(
            restaurant_id=self.context.id,
            customer_phone=customer_phone,
            preferred_slot=preferred_slot,
            note=note,
        )

    async def finalize_if_complete(self, customer_phone: str | None) -> CallOutcome | None:
        intent = self._state.intent
        if not intent.is_complete():
            return None
        assert intent.date_time is not None
        assert intent.party_size is not None
        assert intent.customer_name is not None

        within_hours = self.context.is_open_at(intent.date_time)
        self._state.last_within_opening_hours = within_hours
        if not within_hours:
            return None

        available = await self._reservation.check_availability(
            restaurant_id=self.context.id,
            date_time=intent.date_time,
            party_size=intent.party_size,
        )
        self._state.last_availability_check = available
        if not available:
            return None

        reservation_id = await self._reservation.create(
            restaurant_id=self.context.id,
            date_time=intent.date_time,
            party_size=intent.party_size,
            customer_name=intent.customer_name,
            customer_phone=customer_phone,
        )
        await self._notif.send_confirmation(reservation_id)
        await self._end(CallOutcome.RESERVATION_CREATED)
        return CallOutcome.RESERVATION_CREATED

    # ── Interface du ConversationBrain ──────────────────────────────────
    #
    # Le modèle conduit la conversation ; le code garde la main sur ce qui
    # touche la base. `book` revérifie systématiquement la complétude, la
    # plage d'ouverture et la disponibilité, même quand le LLM affirme
    # qu'il faut réserver.

    def apply_slots(self, slots: dict[str, Any]) -> None:
        intent = self._state.intent
        if (day := slots.get("date")) is not None:
            intent = intent.with_date(day)
        if (hour := slots.get("time")) is not None:
            intent = intent.with_time(hour)
        if (size := slots.get("party_size")) is not None:
            intent = intent.with_party_size(int(size))
        if (name := slots.get("customer_name")) is not None:
            intent = intent.with_customer_name(str(name))
        self._state.intent = intent

    def clear_slots(self, names: list[str]) -> None:
        intent = self._state.intent
        for name in names:
            intent = intent.without(name)
        self._state.intent = intent

    async def check_availability(self, when: datetime, party_size: int) -> bool:
        available = await self._reservation.check_availability(
            restaurant_id=self.context.id,
            date_time=when,
            party_size=party_size,
        )
        self._state.last_availability_check = available
        return available

    async def book(self, customer_phone: str | None) -> CallOutcome | None:
        return await self.finalize_if_complete(customer_phone)

    async def end_with(self, outcome: CallOutcome) -> None:
        await self._end(outcome)

    async def _end(self, outcome: CallOutcome) -> None:
        if self._state.ended:
            return
        now = self._clock()
        duration = (now - (self._state.started_at or now)).total_seconds()
        await self._log.end(self.call_id, outcome, duration)
        self._state.ended = True


_SLOT_KEYWORDS = (
    ("customer_name", ("nom", "appelle", "appelais", "monsieur", "madame")),
    ("time", ("heure", "midi", "soir")),
    ("date", ("jour", "date", "demain", "lundi", "mardi", "mercredi",
              "jeudi", "vendredi", "samedi", "dimanche")),
    ("party_size", ("personne", "convive", "combien")),
)


def _contested_slot(text: str, intent: ReservationIntent) -> str | None:
    """Devine quel slot le client conteste.

    Heuristique volontairement simple et testable. À défaut d'indice
    lexical, on efface le nom : c'est de loin le slot le plus exposé aux
    erreurs de transcription.
    """
    lowered = text.lower()
    for slot, keywords in _SLOT_KEYWORDS:
        if any(k in lowered for k in keywords) and getattr(intent, slot) is not None:
            return slot
    if intent.customer_name is not None:
        return "customer_name"
    return None
