from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

from hikky.domain.outcomes import CallOutcome
from hikky.observability.latency import measure_latency

logger = logging.getLogger("hikky.turn_runner")


class SupportsSlotExtraction(Protocol):
    async def extract(self, user_text: str) -> dict[str, Any]: ...


class SlotExtractor:
    """Interface async : extrait des slots structurés d'un tour de parole.

    Vit dans le domaine — et non dans `pipeline/`, qui dépend de Pipecat —
    pour que le chemin AudioSocket puisse l'utiliser sans tirer Pipecat.
    """

    async def extract(self, user_text: str) -> dict[str, Any]:  # pragma: no cover
        raise NotImplementedError


class NoOpSlotExtractor(SlotExtractor):
    async def extract(self, user_text: str) -> dict[str, Any]:
        return {}


@dataclass(frozen=True, slots=True)
class TurnDecision:
    should_end: bool
    outcome: CallOutcome | None = None
    slot_updates: dict[str, Any] = field(default_factory=dict)


async def run_turn(
    *,
    session: Any,
    slot_extractor: SupportsSlotExtraction | None,
    user_text: str,
    customer_phone: str | None,
    speak: Callable[[str], Awaitable[None]],
) -> TurnDecision:
    slot_updates: dict[str, Any] = {}
    if slot_extractor is not None:
        async with measure_latency("slot_extraction"):
            slot_updates = await slot_extractor.extract(user_text)
        if slot_updates:
            logging.getLogger("hikky.conversation").info(
                "  slots : %s", slot_updates
            )

    async with measure_latency("dialogue_engine"):
        result = await session.process_user_turn(user_text, slot_updates)

    fallback = session.check_fallback(
        user_requested_human=False,
        group_size=slot_updates.get("party_size"),
    )
    if fallback is not None:
        logger.info(
            "fallback triggered",
            extra={"outcome": str(fallback.outcome), "reason": fallback.reason},
        )
        if (
            fallback.outcome == CallOutcome.CALLBACK_REQUESTED
            and customer_phone is not None
        ):
            await session.request_callback(
                customer_phone=customer_phone,
                preferred_slot=None,
                note=fallback.reason,
            )
        await speak(session.context.fallback_message)
        await session.end_with(fallback.outcome)
        return TurnDecision(
            should_end=True, outcome=fallback.outcome, slot_updates=slot_updates
        )

    # ── Barrière de confirmation ────────────────────────────────────────
    # On ne crée JAMAIS la réservation sur la seule complétude des slots :
    # le LLM peut les avoir remplis par déduction hasardeuse. On récapitule
    # et on attend un accord explicite. Sans ça, un « demain matin » suffit
    # à réserver à une heure inventée puis à raccrocher au nez du client.
    intent = getattr(session, "intent", None)
    complete = bool(intent is not None and intent.is_complete())

    if _awaiting_confirmation(session):
        if _is_agreement(user_text):
            outcome = await session.finalize_if_complete(customer_phone)
            _set_awaiting_confirmation(session, False)
            logger.info("réservation confirmée", extra={"outcome": str(outcome)})
            await speak(result.bot_says)
            return TurnDecision(
                should_end=outcome is not None,
                outcome=outcome,
                slot_updates=slot_updates,
            )
        if _is_refusal(user_text):
            _set_awaiting_confirmation(session, False)
            logger.info("réservation refusée par le client")
            await speak("D'accord, qu'est-ce que je corrige ?")
            return TurnDecision(should_end=False, slot_updates=slot_updates)
        await speak(_recap(session))
        return TurnDecision(should_end=False, slot_updates=slot_updates)

    if complete:
        _set_awaiting_confirmation(session, True)
        logger.info("slots complets — demande de confirmation")
        await speak(_recap(session))
        return TurnDecision(should_end=False, slot_updates=slot_updates)

    await speak(result.bot_says)
    return TurnDecision(should_end=False, slot_updates=slot_updates)


_AGREEMENT = ("oui", "c'est ça", "c'est ca", "exact", "tout à fait", "tout a fait",
              "parfait", "d'accord", "d accord", "confirme", "ok", "voilà", "voila")
_REFUSAL = ("non", "pas du tout", "erreur", "faux", "incorrect", "annule", "changer")


def _normalize(text: str) -> str:
    return " " + text.lower().strip() + " "


def _is_agreement(text: str) -> bool:
    t = _normalize(text)
    if any(f" {r} " in t or t.strip().startswith(r) for r in _REFUSAL):
        return False
    return any(a in t for a in _AGREEMENT)


def _is_refusal(text: str) -> bool:
    t = _normalize(text)
    return any(f" {r} " in t or t.strip().startswith(r) for r in _REFUSAL)


def _awaiting_confirmation(session: Any) -> bool:
    return bool(getattr(session, "_hikky_awaiting_confirmation", False))


def _set_awaiting_confirmation(session: Any, value: bool) -> None:
    try:
        session._hikky_awaiting_confirmation = value
    except AttributeError:  # dataclass slots — dégradation propre
        logger.warning("impossible de mémoriser l'état de confirmation")


_JOURS_FR = ("lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche")
_MOIS_FR = (
    "janvier", "février", "mars", "avril", "mai", "juin",
    "juillet", "août", "septembre", "octobre", "novembre", "décembre",
)


def _date_en_toutes_lettres(when: Any) -> str:
    """Formule prononçable par la synthèse vocale.

    Un `%d/%m` est lu n'importe comment : en appel réel le client a
    entendu « 27 » là où le bot affichait « 20/07 », et a corrigé une
    date pourtant juste.
    """
    return f"{_JOURS_FR[when.weekday()]} {when.day} {_MOIS_FR[when.month - 1]}"


def _heure_en_toutes_lettres(when: Any) -> str:
    if when.minute == 0:
        return f"{when.hour} heures"
    return f"{when.hour} heures {when.minute}"


def _recap(session: Any) -> str:
    intent = getattr(session, "intent", None)
    if intent is None:
        return "Je récapitule votre réservation. C'est bien cela ?"

    size = intent.party_size
    couverts = "1 personne" if size == 1 else f"{size} personnes"

    when = intent.date_time
    quand = (
        f"le {_date_en_toutes_lettres(when)} à {_heure_en_toutes_lettres(when)}"
        if when is not None
        else ""
    )
    return (
        f"Je récapitule : une table pour {couverts}, "
        f"{quand}, au nom de {intent.customer_name}. C'est bien cela ?"
    )
