"""Orchestration d'un tour piloté par le `ConversationBrain`.

Le code ne décide plus quoi dire — c'est le modèle. Le code applique ce
qu'il a compris, efface ce que le client conteste, et **exécute les
actions sous condition** :

- une réservation n'est créée que si l'intention est réellement complète,
  quoi qu'en dise le modèle ;
- une disponibilité n'est vérifiée que si un créneau est connu.

Cette asymétrie est volontaire : on accorde au LLM la conduite de la
conversation, jamais la décision d'écrire en base.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from hikky.domain.conversation_brain import BrainDecision
from hikky.domain.outcomes import CallOutcome

logger = logging.getLogger("hikky.brain_turn")
conversation = logging.getLogger("hikky.conversation")

MAX_HISTORY_MESSAGES = 16


@dataclass(frozen=True, slots=True)
class TurnDecision:
    should_end: bool
    outcome: CallOutcome | None = None
    slots: dict[str, Any] = field(default_factory=dict)


async def run_brain_turn(
    *,
    session: Any,
    brain: Any,
    user_text: str,
    customer_phone: str | None,
    history: list[dict[str, str]],
    speak: Callable[[str], Awaitable[None]],
    extractor: Any | None = None,
) -> TurnDecision:
    # L'extraction passe par un appel dédié, sans historique et sur un
    # prompt focalisé. Le conversationnel, lui, dérive à mesure que
    # l'historique s'allonge : en appel réel il annonçait « demain à
    # 12h45 » sans jamais émettre le moindre slot, et rien n'était
    # enregistré. Un 7B quantisé tient une tâche à la fois.
    if extractor is not None:
        extracted = await extractor.extract(user_text)
        if extracted:
            conversation.info("  extrait : %s", extracted)
            session.apply_slots(extracted)

    decision: BrainDecision = await brain.decide(
        user_text=user_text,
        intent=session.intent,
        context=session.context,
        history=history,
    )

    if decision.slots:
        if extractor is not None:
            # L'état appartient à l'extracteur. Le conversationnel a
            # inventé « party_size: 2 » sur une phrase sans le moindre
            # nombre ; ses slots sont donc ignorés dès qu'une source
            # fiable existe.
            conversation.info("  ignoré  : %s (source non fiable)", decision.slots)
        else:
            conversation.info("  compris : %s", decision.slots)
            session.apply_slots(decision.slots)
    if decision.clear:
        conversation.info("  efface  : %s", decision.clear)
        session.clear_slots(decision.clear)
    if decision.action != "none":
        conversation.info("  action  : %s", decision.action)

    reply = _shape_reply(decision, session.intent, history)

    history.append({"role": "user", "content": user_text})
    history.append({"role": "assistant", "content": reply})
    if len(history) > MAX_HISTORY_MESSAGES:
        del history[: len(history) - MAX_HISTORY_MESSAGES]

    if decision.action == "check_availability":
        await _check_availability(session)

    if decision.action == "transfer":
        await speak(reply)
        await session.end_with(CallOutcome.TRANSFERRED)
        return TurnDecision(should_end=True, outcome=CallOutcome.TRANSFERRED)

    # Le modèle demande "confirm" mais n'émet jamais "book" : en appel
    # réel il a annoncé « votre réservation est confirmée » alors que
    # rien n'avait été écrit. On ne lui confie donc pas ce déclenchement :
    # un accord explicite du client, sur une intention complète et après
    # une demande de confirmation, suffit.
    wants_booking = decision.action == "book" or (
        _confirmation_pending(session) and _is_agreement(user_text)
    )
    if _confirmation_pending(session) and _is_refusal(user_text):
        _set_confirmation_pending(session, False)

    if decision.action == "confirm" and session.intent.is_complete():
        # Le modèle réclame parfois confirmation alors qu'il manque le
        # nombre de convives ou le nom : demander « c'est bien cela ? »
        # sur une réservation incomplète désoriente le client.
        _set_confirmation_pending(session, True)

    if wants_booking:
        intent = session.intent
        if not intent.is_complete():
            # Le modèle a voulu réserver trop tôt. On refuse d'écrire en
            # base — mais surtout, on NE PRONONCE PAS sa phrase : elle
            # annonce un succès. En appel réel le bot a dit « votre
            # réservation est confirmée » alors que le nom manquait et
            # que rien n'avait été enregistré. On réclame ce qui manque.
            missing = intent.missing_slots()
            logger.warning(
                "action 'book' refusée — informations manquantes: %s",
                sorted(missing),
            )
            question = next(
                (q for slot, q in _QUESTIONS if slot in missing),
                "Il me manque une information pour finaliser.",
            )
            await speak(question)
            return TurnDecision(should_end=False, slots=decision.slots)

        _set_confirmation_pending(session, False)
        outcome = await session.book(customer_phone)
        # L'écriture en base ne laissait aucune trace : impossible de
        # vérifier après coup si le bot disait vrai en annonçant une
        # confirmation. On journalise systématiquement, succès comme échec.
        if outcome is None:
            logger.error(
                "RESERVATION NON CRÉÉE malgré une intention complète — "
                "créneau hors horaires ou indisponible"
            )
        else:
            logger.info(
                "RESERVATION CRÉÉE : %s, %s personnes, %s (%s)",
                intent.date_time,
                intent.party_size,
                intent.customer_name,
                outcome,
            )
        await speak(reply)
        return TurnDecision(
            should_end=outcome is not None, outcome=outcome, slots=decision.slots
        )

    await speak(reply)
    return TurnDecision(should_end=False, slots=decision.slots)


async def _check_availability(session: Any) -> None:
    intent = session.intent
    if intent.date is None or intent.time is None or intent.party_size is None:
        logger.info("vérification de disponibilité ignorée — créneau incomplet")
        return
    when = datetime.combine(intent.date, intent.time)
    available = await session.check_availability(when, intent.party_size)
    conversation.info("  dispo   : %s", "oui" if available else "non")


_QUESTIONS = (
    ("date", "Pour quel jour souhaitez-vous réserver ?"),
    ("time", "À quelle heure souhaitez-vous venir ?"),
    ("party_size", "Vous serez combien de personnes ?"),
    ("customer_name", "À quel nom dois-je noter la réservation ?"),
)


_JOURS = ("lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche")
_MOIS = ("janvier", "février", "mars", "avril", "mai", "juin", "juillet",
         "août", "septembre", "octobre", "novembre", "décembre")


def _heure_fr(hour: Any) -> str:
    if hour.hour == 12 and hour.minute == 0:
        return "midi"
    if hour.minute == 0:
        return f"{hour.hour} heures"
    return f"{hour.hour} heures {hour.minute}"


def build_recap(intent: Any) -> str:
    """Récapitulatif de confirmation, généré par le code.

    Le modèle produisait « Puis-je confirmer votre réservation ? » sans
    dire quoi, puis un récapitulatif qui oubliait le nom. Le client a dû
    réclamer trois fois avant de raccrocher. Ce message doit être exact
    et complet : il n'est pas confié au LLM, comme l'écriture en base.
    """
    size = intent.party_size
    couverts = "une personne" if size == 1 else f"{size} personnes"
    jour = intent.date
    quand = (
        f"le {_JOURS[jour.weekday()]} {jour.day} {_MOIS[jour.month - 1]}"
        if jour is not None
        else ""
    )
    heure = f" à {_heure_fr(intent.time)}" if intent.time is not None else ""
    nom = f", au nom de {intent.customer_name}" if intent.customer_name else ""
    return f"Je récapitule : {couverts} {quand}{heure}{nom}. C'est bien cela ?"


def _shape_reply(decision: Any, intent: Any, history: list[dict[str, str]]) -> str:
    """Choisit ce que le bot dit réellement.

    Deux messages échappent au modèle : le récapitulatif de confirmation,
    qui doit être exact, et les acquiescements creux du type « Parfait. »,
    qui laissent le client sans savoir quoi répondre.
    """
    if decision.action == "confirm" and intent.is_complete():
        return build_recap(intent)

    reply = decision.reply.strip()
    if decision.action in ("book", "transfer"):
        # Message de clôture : on n'y ajoute pas de question, l'appel se
        # termine. « C'est noté ! Puis-je enregistrer la réservation ? »
        # serait absurde après une écriture déjà effectuée.
        return reply
    if "?" not in reply and len(reply.split()) <= 4:
        missing = intent.missing_slots()
        for slot, question in _QUESTIONS:
            if slot in missing:
                logger.info("acquiescement creux remplacé — %s manquant", slot)
                return f"{reply} {question}".strip()
        return f"{reply} Puis-je enregistrer la réservation ?".strip()

    return _deduplicate(reply, history, intent)


def _deduplicate(reply: str, history: list[dict[str, str]], intent: Any) -> str:
    """Évite que le bot répète mot pour mot sa phrase précédente.

    Constaté en appel réel : « demain mercredi 22 juillet à midi ? »
    prononcé trois fois d'affilée pendant que le client s'agaçait. La
    consigne de reformulation dans le prompt ne suffit pas ; on reprend
    la main avec une question ciblée sur ce qui manque vraiment.
    """
    previous = next(
        (m["content"] for m in reversed(history) if m.get("role") == "assistant"),
        None,
    )
    if previous is None or reply.strip().lower() != previous.strip().lower():
        return reply

    missing = intent.missing_slots()
    for slot, question in _QUESTIONS:
        if slot in missing:
            logger.info("réponse répétée — question ciblée sur %s", slot)
            return question
    return "Puis-je confirmer votre réservation ?"


_AGREEMENT = (
    "oui", "c'est ça", "c'est ca", "exact", "tout à fait", "tout a fait",
    "parfait", "d'accord", "d accord", "confirme", "ok", "voilà", "voila",
)
_REFUSAL = ("non", "pas du tout", "erreur", "faux", "incorrect", "annule", "changer")


def _is_refusal(text: str) -> bool:
    lowered = f" {text.lower().strip()} "
    return any(f" {r} " in lowered or lowered.strip().startswith(r) for r in _REFUSAL)


def _is_agreement(text: str) -> bool:
    if _is_refusal(text):
        return False
    lowered = f" {text.lower().strip()} "
    return any(a in lowered for a in _AGREEMENT)


def _confirmation_pending(session: Any) -> bool:
    return bool(getattr(session, "_hikky_confirm_pending", False))


def _set_confirmation_pending(session: Any, value: bool) -> None:
    try:
        session._hikky_confirm_pending = value
    except AttributeError:
        logger.warning("session sans attribut libre — confirmation non mémorisée")
