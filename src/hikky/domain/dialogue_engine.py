from dataclasses import dataclass
from datetime import datetime
from typing import Any

from hikky.domain.reservation_intent import ReservationIntent
from hikky.ports.language_model import LanguageModelPort


@dataclass(frozen=True, slots=True)
class TurnResult:
    bot_says: str
    updated_intent: ReservationIntent
    intent_complete: bool
    requires_availability_check: bool


class DialogueEngine:
    def __init__(self, llm: LanguageModelPort) -> None:
        self._llm = llm

    async def handle_turn(
        self,
        *,
        user_text: str,
        current_intent: ReservationIntent,
        slot_updates: dict[str, Any],
    ) -> TurnResult:
        intent = current_intent
        if (dt := slot_updates.get("date_time")) is not None:
            assert isinstance(dt, datetime)
            intent = intent.with_date_time(dt)
        if (n := slot_updates.get("party_size")) is not None:
            intent = intent.with_party_size(int(n))
        if (name := slot_updates.get("customer_name")) is not None:
            intent = intent.with_customer_name(str(name))

        check_needed = (
            intent.date_time is not None
            and intent.party_size is not None
            and (
                current_intent.date_time != intent.date_time
                or current_intent.party_size != intent.party_size
            )
        )

        prompt = self._build_prompt(user_text, intent)
        reply = await self._llm.complete(prompt)

        return TurnResult(
            bot_says=reply,
            updated_intent=intent,
            intent_complete=intent.is_complete(),
            requires_availability_check=check_needed,
        )

    def _build_prompt(
        self, user_text: str, intent: ReservationIntent
    ) -> list[dict[str, str]]:
        missing = intent.missing_slots()
        return [
            {
                "role": "system",
                "content": (
                    "Tu es l'assistant vocal d'un restaurant. Aide à prendre une réservation. "
                    f"Slots manquants: {sorted(missing) if missing else 'aucun'}."
                ),
            },
            {"role": "user", "content": user_text},
        ]
