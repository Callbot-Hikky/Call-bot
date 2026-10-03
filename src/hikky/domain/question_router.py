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
#
# Observé en appel réel (transcriptions brutes) : « Oui, bonjour, je voulais
# savoir est ce que … halal ? » et « Non, non, je voulais vous poser la
# question, c'est quoi l'ambiance ? » n'étaient PAS des questions pour le
# bot — le « oui »/« non » de politesse en tête passait pour un accord ou un
# refus et court-circuitait la détection. « est ce que » sans tiret (sortie
# STT courante) n'était pas reconnu non plus, et « vous avez une terrasse »
# sans « ? » ne contenait aucun marqueur. D'où trois étages :
#
# 1. on retire les interjections de tête (« oui, bonjour, euh… ») ;
# 2. une TOURNURE FORTE suffit (« est-ce que », « c'est quoi », « avez-vous ») ;
# 3. une tournure FAIBLE (« vous avez », « c'est », « il y a ») ne suffit que
#    si la phrase parle d'un THÈME du restaurant (terrasse, halal, parking…)
#    — « il y a quatre personnes » reste une réponse, pas une question.
_QUESTION_FORTE = (
    "est-ce que", "est-ce qu'", "qu'est-ce",
    "c'est quoi", "c'est quand", "c'est où", "c'est ou", "c'est combien",
    "c'est comment", "quel est", "quelle est", "quels sont", "quelles sont",
    "à quel", "a quel", "à quelle", "a quelle", "combien de temps",
    "combien ça", "combien ca", "combien coûte", "combien coute",
    "combien c'est", "quel prix", "quels prix", "ça coûte", "ca coute",
    "pouvez-vous me", "pourriez-vous me", "peux-tu me", "vous pouvez me",
    "répéter", "repeter", "pas compris", "pas entendu", "pardon",
    "comment ça", "comment ca", "comment on", "comment vous", "comment fait",
    "où est", "ou est", "où se trouve", "où êtes", "où etes", "pourquoi",
    "y a-t-il", "avez-vous", "êtes-vous", "etes-vous", "faites-vous",
    "acceptez-vous", "proposez-vous", "servez-vous", "prenez-vous",
    "peut-on", "puis-je", "pourrais-je",
    "je voulais savoir", "je voudrais savoir", "j'aimerais savoir",
    "je veux savoir", "savoir si", "une question", "je me demandais",
    "dites-moi", "c'est possible", "possible de",
    # Question indirecte, observée en appel réel et routée à tort en réponse :
    # « Je voulais demander si le restaurant propose des plats végétariens ».
    "demander si", "je voulais demander", "je voudrais demander",
    "j'aimerais demander", "je peux demander", "puis-je demander",
)

# Tournures qui n'interrogent que si le sujet est un fait du restaurant.
# « le restaurant » à la troisième personne en fait partie : personne n'appelle
# pour INFORMER le bot sur son propre restaurant — « Le restaurant propose des
# plats halal. » est une question dont le STT a perdu l'intonation (ou a
# transcrit « Parce que » pour « Est-ce que », observé en appel réel).
_QUESTION_FAIBLE = re.compile(
    r"(?<!\w)(c'est|est|sont|ont|a|il y a|y a|vous avez|vous êtes|vous etes|"
    r"vous faites|vous acceptez|vous proposez|vous servez|vous prenez|"
    r"vous livrez|on peut|je peux|possible|ouverts?|ouvertes?|"
    r"le restaurant|l'établissement|l'etablissement|chez vous|"
    r"propose\w*|accepte\w*|dispose\w*|poss[èe]de\w*|sert|servent)(?!\w)"
)

# Ce dont un client s'enquiert au téléphone, hors créneau de réservation.
_THEMES_RESTAURANT = re.compile(
    r"(?<!\w)(halal|hallal|casher|kasher|v[ée]g[ée]\w*|vegan\w*|bio|gluten|"
    r"allerg\w*|terrasse|rooftop|toit|parking|gar[ée]r|stationn\w*|wifi|"
    r"climatis\w*|clim|chaises? hautes?|b[ée]b[ée]s?|enfants?|poussettes?|"
    r"animau\w*|chiens?|chats?|accessib\w*|fauteuils?|handicap\w*|pmr|"
    r"carte|cb|esp[èe]ces|tickets?|ch[èe]ques?|paiement|payer|prix|tarifs?|"
    r"cher|ch[èe]re|co[ûu]te|budget|menus?|plats?|formules?|desserts?|"
    r"sp[ée]cialit\w*|cuisine|pizzas?|poissons?|viandes?|emporter|livr\w*|"
    r"ambiance|adresse|situ[ée]s?|m[ée]tro|ouverts?|ferm[ée]s?|horaires?|"
    r"tenue|privatis\w*|anniversaires?|f[êe]tes?|boissons?|vins?|alcool|"
    r"bar|musique|fumeurs?|toilettes|wc|groupes?)(?!\w)"
)

# Politesses et hésitations en tête de phrase : elles ne portent aucun sens
# et masquaient la question qui suivait.
_INTERJECTIONS = re.compile(
    r"^(?:(?:oui|ouais|non|nan|bonjour|bonsoir|allô|allo|alors|euh|bah|ben|"
    r"hein|merci|et|donc|dites|voilà|voila|en fait|du coup|d'accord|ok|okay|"
    r"excusez-moi|excusez moi|s'il vous plaît|s'il vous plait|svp)"
    r"[\s,.!;:]+)+"
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


def _normaliser_question(text: str) -> str:
    """Aplanit les variantes de transcription avant de chercher les tournures.

    Le STT écrit « est ce que », « est-ce-que », des apostrophes typographiques
    et des espaces en trop : la détection ne doit dépendre d'aucun de ces
    détails.
    """
    t = _normalise(text).replace("’", "'").replace("‘", "'")
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"\best[ -]ce[ -]qu", "est-ce qu", t)
    t = re.sub(r"\bqu[' ]est[ -]ce\b", "qu'est-ce", t)
    t = re.sub(r"\by[' ]?a[ -]t[ -]?'?il\b", "y a-t-il", t)
    return t


def _sans_interjections(texte: str) -> str:
    return _INTERJECTIONS.sub("", texte).strip()


def is_client_question(text: str) -> bool:
    """Le client interroge-t-il, plutôt que de répondre ?

    Un point d'interrogation seul ne suffit pas : la transcription en
    ajoute ou en retire. On s'appuie sur des tournures, après avoir ôté les
    « oui », « non », « bonjour » de politesse qui précèdent souvent la
    vraie question au téléphone.
    """
    if not text:
        return False
    coeur = _sans_interjections(_normaliser_question(text))
    if not coeur:
        return False
    if any(_contient_expression(coeur, m) for m in _QUESTION_FORTE):
        return True
    # Un accord ou un refus net n'est pas une question — mais on le juge sur
    # le cœur de la phrase, pas sur le « oui » de politesse qui l'ouvre.
    if is_agreement(coeur) or is_refusal(coeur):
        return False
    if "?" in text and len(text.split()) > 2:
        return True
    return bool(_QUESTION_FAIBLE.search(coeur) and _THEMES_RESTAURANT.search(coeur))


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
