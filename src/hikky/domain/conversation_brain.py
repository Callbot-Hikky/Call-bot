"""Réponse aux questions hors script — le seul rôle conversationnel du LLM.

Le parcours de réservation est piloté par le code (`routed_turn` +
`question_router`) ; le modèle n'intervient que lorsque le client pose une
vraie question (horaires, options…). Une tâche unique et un prompt court se
sont montrés bien plus fiables, sous forte charge, qu'un moteur généraliste
chargé de dizaines de contraintes.
"""

from __future__ import annotations

import logging
from datetime import date, time

from hikky.ports.language_model import LanguageModelPort

logger = logging.getLogger("hikky.answerer")

_JOURS = ("lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche")
_MOIS = ("janvier", "février", "mars", "avril", "mai", "juin", "juillet",
         "août", "septembre", "octobre", "novembre", "décembre")


ANSWER_PROMPT = """Tu es l'assistant vocal du restaurant {name}, au téléphone.

HORAIRES :
{hours}

CAPACITÉ : {capacity} couverts, groupes de {max_group} personnes maximum.

RÉSERVATION EN COURS :
{known}

Le client vient de te poser une question. Réponds-y en UNE phrase courte, \
naturelle, à partir des informations ci-dessus. Si tu ne sais pas, dis-le \
simplement. Dis les dates et heures en toutes lettres, jamais en chiffres.

Réponds uniquement par la phrase à dire, sans JSON ni commentaire."""


class QuestionAnswerer:
    """Répond aux questions hors script — le seul rôle conversationnel.

    Une tâche unique, contre la dizaine de contraintes d'un prompt
    généraliste. La mesure a montré que les quantisations testées échouaient
    identiquement sous forte charge, et réussissaient dès qu'on l'allégeait.
    """

    def __init__(self, llm: LanguageModelPort) -> None:
        self._llm = llm

    async def answer(self, *, user_text, intent, context, history) -> str:
        system = ANSWER_PROMPT.format(
            name=context.name,
            hours=_format_hours(context),
            capacity=context.total_capacity,
            max_group=context.rules.max_group_size,
            known=_format_known(intent),
        )
        messages = [
            {"role": "system", "content": system},
            *history[-6:],
            {"role": "user", "content": user_text},
        ]
        try:
            raw = await self._llm.complete(messages)
        except Exception:  # noqa: BLE001
            logger.warning("réponse hors script échouée", exc_info=True)
            return "Pardon, pouvez-vous répéter ?"
        texte = (raw or "").strip()
        return texte if texte and len(texte) < 400 else "Pardon, pouvez-vous répéter ?"


def _format_hours(context) -> str:
    lines = []
    for slot in sorted(context.opening_hours, key=lambda s: s.weekday):
        lines.append(
            f"- {_JOURS[slot.weekday]} : {slot.opens:%H:%M} à {slot.closes:%H:%M}"
        )
    closed = set(range(7)) - {s.weekday for s in context.opening_hours}
    for day in sorted(closed):
        lines.append(f"- {_JOURS[day]} : FERMÉ")
    return "\n".join(lines)


def _date_fr(day: date) -> str:
    """Date prononçable. Le format ISO ressortait tel quel à l'oral."""
    return f"{_JOURS[day.weekday()]} {day.day} {_MOIS[day.month - 1]}"


def _heure_fr(hour: time) -> str:
    """Heure prononçable. « 12:00 » se lisait « douze deux points zéro zéro »."""
    if hour.hour == 12 and hour.minute == 0:
        return "midi"
    if hour.hour == 0 and hour.minute == 0:
        return "minuit"
    if hour.minute == 0:
        return f"{hour.hour} heures"
    return f"{hour.hour} heures {hour.minute}"


def _format_known(intent) -> str:
    lines = []
    if intent.date is not None:
        lines.append(f"- jour : {_date_fr(intent.date)}")
    if intent.time is not None:
        lines.append(f"- heure : {_heure_fr(intent.time)}")
    if intent.party_size is not None:
        lines.append(f"- nombre de convives : {intent.party_size}")
    if intent.customer_name is not None:
        lines.append(f"- nom : {intent.customer_name}")
    return "\n".join(lines) if lines else "- (rien encore)"
