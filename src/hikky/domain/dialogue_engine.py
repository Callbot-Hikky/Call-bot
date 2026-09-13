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


MAX_HISTORY_MESSAGES = 12

SYSTEM_PROMPT = """Tu es l'assistant vocal du restaurant {name}. Tu prends \
des réservations par téléphone.

RÈGLES IMPÉRATIVES :
- Réponds en UNE phrase courte, deux au maximum. Tu es à l'oral : une \
réponse longue est insupportable au téléphone.
- Ne pose qu'UNE seule question à la fois.
- Ne redemande JAMAIS une information déjà donnée (voir ci-dessous).
- Reste naturel et poli, sans formules ampoulées.
- N'invente jamais de disponibilité : tu ne connais pas le planning.
- Pas de listes à puces, pas de mise en forme : ce texte sera lu à voix haute.

Informations déjà recueillies :
{known}

Il te reste à obtenir : {missing}

Demande UNE seule de ces informations à la fois, formulée naturellement \
en français. N'emploie jamais de terme technique ni de mot anglais.
"""


class DialogueEngine:
    """Moteur de dialogue d'UN appel.

    Conserve l'historique de la conversation : sans lui, le LLM redémarre
    à zéro à chaque tour, repose les questions déjà posées et paraît
    stupide. L'instance est créée par appel (cf. `session_factory`), donc
    l'état ne fuit pas d'un appel à l'autre.
    """

    def __init__(self, llm: LanguageModelPort) -> None:
        self._llm = llm
        self._history: list[dict[str, str]] = []

    async def handle_turn(
        self,
        *,
        user_text: str,
        current_intent: ReservationIntent,
        slot_updates: dict[str, Any],
        context: Any | None = None,
    ) -> TurnResult:
        intent = current_intent
        if (dt := slot_updates.get("date_time")) is not None:
            assert isinstance(dt, datetime)
            intent = intent.with_date_time(dt)
        # `date` et `time` arrivent séparément : le client donne souvent
        # le jour avant l'heure ("demain matin"), et perdre la moitié
        # connue conduit le bot à redemander ce qu'il a déjà.
        if (day := slot_updates.get("date")) is not None:
            intent = intent.with_date(day)
        if (hour := slot_updates.get("time")) is not None:
            intent = intent.with_time(hour)
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

        prompt = self._build_prompt(user_text, intent, context)
        reply = await self._llm.complete(prompt)

        self._history.append({"role": "user", "content": user_text})
        self._history.append({"role": "assistant", "content": reply})
        if len(self._history) > MAX_HISTORY_MESSAGES:
            self._history = self._history[-MAX_HISTORY_MESSAGES:]

        return TurnResult(
            bot_says=reply,
            updated_intent=intent,
            intent_complete=intent.is_complete(),
            requires_availability_check=check_needed,
        )

    def _build_prompt(
        self,
        user_text: str,
        intent: ReservationIntent,
        context: Any | None = None,
    ) -> list[dict[str, str]]:
        # Libellés humains : le prompt exposait les identifiants techniques
        # et le modèle les récitait tels quels — en appel réel il a dit
        # « Customer name, s'il vous plaît ».
        labels = {
            "date": "le jour de la réservation",
            "time": "l'heure",
            "party_size": "le nombre de convives",
            "customer_name": "le nom du client",
        }
        missing = sorted(labels.get(s, s) for s in intent.missing_slots())
        known = []
        if intent.date is not None:
            known.append(f"- jour : {intent.date:%A %d %B}")
        if intent.time is not None:
            known.append(f"- heure : {intent.time:%Hh%M}")
        if intent.party_size is not None:
            known.append(f"- nombre de personnes : {intent.party_size}")
        if intent.customer_name is not None:
            known.append(f"- nom du client : {intent.customer_name}")

        system = SYSTEM_PROMPT.format(
            name=getattr(context, "name", "") or "",
            known="\n".join(known) if known else "- (rien pour l'instant)",
            missing=", ".join(missing) if missing else "rien, tout est complet",
        )

        return [
            {"role": "system", "content": system},
            *self._history,
            {"role": "user", "content": user_text},
        ]
