"""Que dire quand on n'a rien compris.

Mesuré sur 8 278 tours de dialogue (Bohus & Rudnicky, 2005) : « pouvez-vous
répéter ? » récupère 33,7 % des situations, AVANCER sur une autre question 64,4 %.
Et après deux échecs, un troisième reprompt identique ne sert à rien : on clôt
poliment plutôt que de noter une réservation fausse (règle « 3 no-match -> humain »).

Fonction pure : des informations entrent, une phrase et une décision sortent.
"""

from __future__ import annotations

MAX_ATTEMPTS = 3

GOODBYE_AFTER_FAILURES = (
    "Je suis désolée, je vous entends très mal et je ne voudrais pas noter une "
    "réservation erronée. N'hésitez pas à rappeler, l'équipe du restaurant se fera "
    "un plaisir de vous répondre. Bonne journée !"
)
CONFIRM_AGAIN = "Vous confirmez la réservation ? Dites oui, ou non."
PREFIX_FIRST = "Je vous entends mal. "
PREFIX_SECOND = "La ligne est mauvaise, je reprends. "


def repair_reply(
    attempt: int,
    *,
    awaiting_confirmation: bool,
    next_question: str | None,
    recap: str | None,
) -> tuple[str, bool]:
    """La phrase à dire au `attempt`-ième échec consécutif, et s'il faut raccrocher.

    `next_question` : la question du créneau encore manquant (on avance dessus).
    `recap` : le récapitulatif, si plus rien ne manque.
    """
    if attempt >= MAX_ATTEMPTS:
        return GOODBYE_AFTER_FAILURES, True
    prefix = PREFIX_FIRST if attempt == 1 else PREFIX_SECOND
    if awaiting_confirmation:
        return prefix + CONFIRM_AGAIN, False
    if next_question:
        return prefix + next_question, False
    return prefix + (recap or CONFIRM_AGAIN), False
