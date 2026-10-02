"""Réponse aux questions hors script — le seul rôle conversationnel du LLM.

Le parcours de réservation est piloté par le code (`routed_turn` +
`question_router`) ; le modèle n'intervient que lorsque le client pose une
vraie question (horaires, options…). Une tâche unique et un prompt court se
sont montrés bien plus fiables, sous forte charge, qu'un moteur généraliste
chargé de dizaines de contraintes.
"""

from __future__ import annotations

import logging
import re
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

IMPORTANT : la réservation ci-dessus n'est PAS encore enregistrée. Ne dis \
JAMAIS qu'elle est confirmée, réservée, notée, validée ou enregistrée, et \
n'annonce aucune réservation. Tu réponds seulement à la question.

Réponds uniquement par la phrase à dire, sans JSON ni commentaire."""


# `answer()` n'est appelé que hors parcours de réservation : aucune réservation
# n'est créée ici. Toute affirmation qu'elle est faite est donc FAUSSE par
# construction. Observé en appel réel : « votre réservation est confirmée pour
# ce soir à vingt et une heures sous le nom Général » — rien n'était réservé.
_FAUSSE_CONFIRMATION = re.compile(
    r"\b(r[ée]servation (est |a [ée]t[ée] )?(confirm|enregistr|not[ée]|valid|prise|faite)\w*|"
    r"est confirm\w*|c'est (not[ée]|r[ée]serv[ée]|enregistr[ée]|confirm[ée])|"
    r"bien not[ée]|j'ai (not[ée]|r[ée]serv[ée]|enregistr[ée]|confirm[ée])|"
    r"je (vous )?confirme|est r[ée]serv[ée]e?)\b",
    re.IGNORECASE,
)
_REPONSE_SURE = "Je comprends."


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
        if not texte or len(texte) >= 400:
            return "Pardon, pouvez-vous répéter ?"
        if _FAUSSE_CONFIRMATION.search(texte):
            # Fausse confirmation : on ne la prononce jamais. Réponse neutre ;
            # l'appelant (_ensure_progress) enchaîne sur le slot manquant.
            logger.warning("answerer a annoncé une réservation inexistante — neutralisé: %r", texte)
            return _REPONSE_SURE
        return texte


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
