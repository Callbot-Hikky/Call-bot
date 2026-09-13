"""Aiguilleur de tour : le code décide, le modèle reste pour l'intelligence.

Mesure qui motive ce module — même conversation, deux quantisations :

    Q4, prompt de production : « À quelle heure ? » × 3 après avoir reçu l'heure
    Q6, prompt de production : « À quelle heure ? » × 3 — identique

Les deux modèles échouent de la même façon face à une dizaine de
contraintes simultanées. Ce n'est donc pas leur capacité qui est en
cause, mais la charge : on ne peut pas leur confier le suivi de l'état.

Le code connaît exactement ce qui manque et choisit la question suivante.
Le modèle garde ce qu'il réussit :

- **comprendre** les phrases du client (l'extracteur ne s'est jamais trompé)
- **répondre** aux questions hors script, avec horaires et capacité

Sur un tour ordinaire, aucun appel conversationnel n'est nécessaire —
d'où la latence divisée par deux sur le chemin nominal.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any

# Ordre de collecte : le jour d'abord, le nom en dernier. C'est la
# progression naturelle d'une prise de réservation au téléphone.
SLOT_ORDER = ("date", "time", "party_size", "customer_name")


class Action(Enum):
    ASK_SLOT = "ask_slot"
    ANSWER_QUESTION = "answer_question"
    CONFIRM = "confirm"
    BOOK = "book"
    CORRECT = "correct"


@dataclass(frozen=True, slots=True)
class RouteDecision:
    action: Action
    slot: str | None = None


# Le client pose une question plutôt que d'y répondre. Sans cette
# détection, « À quel nom ? » recevait « C'est à quel nom ? » — le bot
# renvoyait la question au lieu d'y répondre.
_QUESTION_MARKERS = (
    "est-ce que", "êtes-vous", "etes-vous", "avez-vous", "vous êtes",
    "vous etes", "c'est quoi", "c'est quand", "quel est", "quelle est",
    "à quel", "a quel", "à quelle", "a quelle", "combien de temps",
    "pouvez-vous me dire", "peux-tu me dire", "répéter", "repeter",
    "pas compris", "pas entendu", "pardon", "comment ça", "qu'est-ce",
    "vous faites", "y a-t-il", "il y a", "je peux", "on peut",
)

# Sur un appel réel, le client a confirmé par « Ouais, à peu près » puis
# « Affirmatif » — aucun des deux n'était reconnu, et le bot a redemandé
# confirmation trois fois. La liste couvre donc le parler courant, pas
# seulement le français d'école.
_ACCORD = (
    "oui", "ouais", "ouaip", "yes", "yep", "si",
    "c'est ça", "c'est ca", "c'est bon", "c'est cela", "c'est exact",
    "exact", "exactement", "tout à fait", "tout a fait", "absolument",
    "affirmatif", "parfait", "impeccable", "nickel", "très bien",
    "tres bien", "d'accord", "d accord", "ça marche", "ca marche",
    "ça me va", "ca me va", "allez-y", "allez y", "confirme",
    "je confirme", "ok", "okay", "voilà", "voila", "bien sûr", "bien sur",
)
_REFUS = (
    "non", "nan", "négatif", "negatif", "pas du tout", "pas vraiment",
    "plutôt pas", "plutot pas", "erreur", "faux", "incorrect",
    "annule", "annulez", "annuler", "changer", "modifier",
)


def _normalise(text: str) -> str:
    return (text or "").lower().strip()


def _contient_expression(texte: str, expression: str) -> bool:
    """Cherche l'expression en respectant les frontières de mots.

    Une simple sous-chaîne prenait « Ouissem » pour un « oui » et
    « Louison » aussi : un nom propre valait confirmation de réservation.
    """
    return re.search(rf"(?<!\w){re.escape(expression)}(?!\w)", texte) is not None


def is_refusal(text: str) -> bool:
    t = _normalise(text)
    return any(_contient_expression(t, r) for r in _REFUS)


def is_agreement(text: str) -> bool:
    # Un refus prime : « non, c'est pas ça » contient « c'est ça ».
    if is_refusal(text):
        return False
    t = _normalise(text)
    return any(_contient_expression(t, a) for a in _ACCORD)


def is_client_question(text: str) -> bool:
    """Le client interroge-t-il, plutôt que de répondre ?

    Un point d'interrogation seul ne suffit pas : la transcription en
    ajoute ou en retire. On s'appuie aussi sur des tournures.
    """
    if not text:
        return False
    lowered = text.lower()
    if is_agreement(text) or is_refusal(text):
        return False
    if any(m in lowered for m in _QUESTION_MARKERS):
        return True
    return "?" in text and len(text.split()) > 2


def route_turn(
    intent: Any, user_text: str, awaiting_confirmation: bool
) -> RouteDecision:
    """Décide ce qui doit se passer, à partir de l'état — pas d'un LLM."""
    if is_client_question(user_text):
        return RouteDecision(Action.ANSWER_QUESTION)

    if awaiting_confirmation:
        if is_refusal(user_text):
            return RouteDecision(Action.CORRECT)
        if is_agreement(user_text) and intent.is_complete():
            return RouteDecision(Action.BOOK)

    missing = intent.missing_slots()
    for slot in SLOT_ORDER:
        if slot in missing:
            return RouteDecision(Action.ASK_SLOT, slot=slot)

    return RouteDecision(Action.CONFIRM)


# Plusieurs formulations par information : une phrase figée sonne
# robotique dès le deuxième appel.
_FORMULATIONS = {
    "date": (
        "Pour quel jour souhaitez-vous réserver ?",
        "Quel jour vous conviendrait ?",
        "Vous souhaitez venir quel jour ?",
    ),
    "time": (
        "À quelle heure souhaitez-vous venir ?",
        "Vous préférez quelle heure ?",
        "À quelle heure vous attendons-nous ?",
    ),
    "party_size": (
        "Vous serez combien de personnes ?",
        "Pour combien de personnes ?",
        "Combien serez-vous à table ?",
    ),
    "customer_name": (
        "C'est à quel nom ?",
        "À quel nom dois-je noter la réservation ?",
        "Puis-je avoir votre nom ?",
    ),
}


def phrase_for_slot(slot: str, deja_dites: set[str]) -> str:
    """Formulation naturelle, en évitant celles déjà employées."""
    choix = _FORMULATIONS.get(slot)
    if not choix:
        return "Pouvez-vous préciser ?"
    fraiches = [f for f in choix if f not in deja_dites]
    return random.choice(fraiches or list(choix))
