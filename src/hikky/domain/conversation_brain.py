"""Moteur conversationnel — un seul appel LLM, tout le contexte, des actions.

Remplace le duo `LLMSlotExtractor` + `DialogueEngine`, qui faisait deux
appels dont aucun ne voyait le restaurant. Le modèle qui parlait ne
connaissait que le nom de l'établissement et la liste des cases vides :
il ne pouvait donc que réclamer ces cases, et répétait la même phrase dès
que le client sortait du script.

Ce moteur lui donne :

- les horaires d'ouverture, la capacité, la taille de groupe maximale
- la date et l'heure courantes (fuseau du restaurant)
- ce qui est déjà connu de la réservation
- l'historique complet de la conversation
- la possibilité de **demander une action** : vérifier une disponibilité,
  effacer une information contestée, confirmer, réserver, transférer

La sortie est un JSON strict plutôt qu'un appel de fonction natif : avec
un modèle 7B quantisé, le tool-calling devient vite erratique là où un
schéma JSON simple reste fiable.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable
from datetime import date, datetime, time
from typing import Any

from hikky.domain.reservation_intent import ReservationIntent
from hikky.domain.restaurant_context import RestaurantContext
from hikky.ports.language_model import LanguageModelPort

logger = logging.getLogger("hikky.brain")

KNOWN_ACTIONS = ("none", "check_availability", "confirm", "book", "transfer")

SAFE_REPLY = "Pardon, je n'ai pas bien saisi. Pouvez-vous répéter ?"

_JOURS = ("lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche")
_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)

SYSTEM_PROMPT = """Tu es l'assistant vocal du restaurant {name}. Tu réponds \
au téléphone et tu prends des réservations.

NOUS SOMMES LE {today}, IL EST {now}.

HORAIRES D'OUVERTURE :
{hours}

CAPACITÉ : {capacity} couverts. Groupes de {max_group} personnes maximum \
au-delà, il faut passer par un humain.

CE QUE TU SAIS DÉJÀ DE CETTE RÉSERVATION :
{known}

SOIS BREF — C'EST LA RÈGLE LA PLUS IMPORTANTE :
- **15 mots maximum.** Chaque mot prononcé fait attendre le client, car \
ta réponse est lue à voix haute en temps réel.
- Ne répète PAS ce que le client vient de dire, ni les informations que \
tu connais déjà. Il les a en tête.
- Pose seulement ta question. Un récapitulatif ne se fait QU'UNE fois, \
au moment de la confirmation finale.

Exemples de bonnes réponses :
- "Vous serez combien de personnes ?"
- "À quelle heure souhaitez-vous venir ?"
- "C'est à quel nom ?"
- "Très bien. À quelle heure ?"

Exemple de MAUVAISE réponse, bien trop longue :
- "Parfait, vous souhaitez réserver pour mercredi 22 juillet à midi pour \
deux personnes. Quel est votre nom s'il vous plaît ?"

TU PARLES AU TÉLÉPHONE :
- Dis les dates en toutes lettres : "mercredi 22 juillet", JAMAIS en \
chiffres ni au format ISO. Ton texte est lu par une synthèse vocale, qui \
prononce une date en chiffres n'importe comment.
- Dis les heures ainsi : "20 heures", "12 heures 45".
- UNE phrase courte, deux au maximum. Jamais de liste, jamais de mise en \
forme : ton texte est lu à voix haute.
- UNE seule question à la fois.
- Ne redemande jamais une information de la liste ci-dessus.
- Si tu répètes une question, reformule-la autrement.

N'INVENTE RIEN :
- Une heure vague ("matin", "midi", "dans la soirée") n'est PAS une heure : \
demande l'heure exacte au lieu de la choisir toi-même.
- Le jour et l'heure sont indépendants : tu peux connaître l'un sans l'autre.
- Tu ne connais pas les disponibilités : demande une vérification.

NE CONFIRME JAMAIS TROP TÔT :
- Tant qu'il manque une information de la liste ci-dessus, demande-la. \
Ne propose pas de récapituler, ne dis pas "je vous réserve", n'annonce \
aucune confirmation.
- N'annonce une réservation comme enregistrée QUE si tu as demandé \
"book" ET que rien ne manque.
- Il te faut TOUJOURS le nom du client avant de finaliser.

SI LE CLIENT TE CORRIGE ou dit que tu te trompes, excuse-toi brièvement et \
signale l'information à effacer. Ne répète jamais une donnée qu'il conteste.

RÉPONDS UNIQUEMENT PAR CE JSON, sans texte autour :

{{"reply": "ce que tu dis à voix haute",
  "slots": {{"date": "AAAA-MM-JJ", "time": "HH:MM", "party_size": 2, \
"customer_name": "Nom"}},
  "clear": ["champ à effacer"],
  "action": "none"}}

- "slots" : uniquement ce que le client vient d'énoncer clairement. Omets \
tout le reste.
- "clear" : les champs que le client conteste, parmi "date", "time", \
"party_size", "customer_name". Liste vide sinon.
Ces noms de champs sont techniques : ne les prononce JAMAIS à voix haute. \
Dis "votre nom", jamais "customer name".

- "action" : "check_availability" pour vérifier un créneau, "confirm" pour \
faire valider le récapitulatif, "book" quand le client a explicitement \
accepté, "transfer" pour passer un humain, "none" sinon.
"""


class BrainDecision:
    """Ce que le modèle a décidé pour ce tour."""

    __slots__ = ("reply", "slots", "clear", "action")

    def __init__(
        self,
        reply: str,
        slots: dict[str, Any] | None = None,
        clear: list[str] | None = None,
        action: str = "none",
    ) -> None:
        self.reply = reply
        self.slots = slots or {}
        self.clear = clear or []
        self.action = action if action in KNOWN_ACTIONS else "none"

    def __repr__(self) -> str:  # pragma: no cover — confort de debug
        return (
            f"BrainDecision(reply={self.reply!r}, slots={self.slots!r}, "
            f"clear={self.clear!r}, action={self.action!r})"
        )


class ConversationBrain:
    def __init__(
        self,
        llm: LanguageModelPort,
        *,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._llm = llm
        self._clock = clock or datetime.now

    async def decide(
        self,
        *,
        user_text: str,
        intent: ReservationIntent,
        context: RestaurantContext,
        history: list[dict[str, str]],
    ) -> BrainDecision:
        messages = [
            {"role": "system", "content": self._system_prompt(intent, context)},
            *history,
            {"role": "user", "content": user_text},
        ]
        try:
            raw = await self._llm.complete(messages)
        except Exception:  # noqa: BLE001 — un appel doit survivre à un LLM KO
            logger.warning("appel LLM échoué", exc_info=True)
            return BrainDecision(SAFE_REPLY)
        return self._parse(raw)

    def _system_prompt(
        self, intent: ReservationIntent, context: RestaurantContext
    ) -> str:
        now = self._clock()
        return SYSTEM_PROMPT.format(
            name=context.name,
            today=_date_fr(now.date()),
            now=now.strftime("%H:%M"),
            hours=_format_hours(context),
            capacity=context.total_capacity,
            max_group=context.rules.max_group_size,
            known=_format_known(intent),
        )

    def _parse(self, raw: str) -> BrainDecision:
        match = _JSON_BLOCK.search(raw or "")
        if not match:
            logger.warning("aucun JSON dans la réponse: %r", raw)
            return BrainDecision(_plain_text_fallback(raw))
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError:
            logger.warning("JSON illisible: %r", raw)
            return BrainDecision(_plain_text_fallback(raw))
        if not isinstance(payload, dict):
            return BrainDecision(SAFE_REPLY)

        reply = payload.get("reply")
        return BrainDecision(
            reply=reply.strip() if isinstance(reply, str) and reply.strip() else SAFE_REPLY,
            slots=_parse_slots(payload.get("slots")),
            clear=_parse_clear(payload.get("clear")),
            action=payload.get("action", "none"),
        )


def _plain_text_fallback(raw: str) -> str:
    """Le modèle a parlé sans JSON : on récupère sa phrase plutôt que rien."""
    text = (raw or "").strip()
    return text if text and len(text) < 400 else SAFE_REPLY


def _format_hours(context: RestaurantContext) -> str:
    lines = []
    for slot in sorted(context.opening_hours, key=lambda s: s.weekday):
        lines.append(
            f"- {_JOURS[slot.weekday]} : {slot.opens:%H:%M} à {slot.closes:%H:%M}"
        )
    closed = set(range(7)) - {s.weekday for s in context.opening_hours}
    for day in sorted(closed):
        lines.append(f"- {_JOURS[day]} : FERMÉ")
    return "\n".join(lines)


_MOIS = ("janvier", "février", "mars", "avril", "mai", "juin", "juillet",
         "août", "septembre", "octobre", "novembre", "décembre")


def _date_fr(day: date) -> str:
    """Date prononçable. Le format ISO ressortait tel quel à l'oral."""
    return f"{_JOURS[day.weekday()]} {day.day} {_MOIS[day.month - 1]}"


def _heure_fr(hour: time) -> str:
    """Heure prononçable. Le modèle recopiait « 12:00 » depuis l'état, que
    la synthèse lit « douze deux points zéro zéro »."""
    if hour.hour == 12 and hour.minute == 0:
        return "midi"
    if hour.hour == 0 and hour.minute == 0:
        return "minuit"
    if hour.minute == 0:
        return f"{hour.hour} heures"
    return f"{hour.hour} heures {hour.minute}"


def _format_known(intent: ReservationIntent) -> str:
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


def _parse_slots(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        return {}
    out: dict[str, Any] = {}

    day = raw.get("date")
    if isinstance(day, str):
        try:
            out["date"] = date.fromisoformat(day)
        except ValueError:
            logger.debug("date illisible: %r", day)

    hour = raw.get("time")
    if isinstance(hour, str):
        try:
            out["time"] = time.fromisoformat(hour)
        except ValueError:
            logger.debug("heure illisible: %r", hour)

    size = raw.get("party_size")
    if isinstance(size, int) and size > 0:
        out["party_size"] = size

    name = raw.get("customer_name")
    if isinstance(name, str) and name.strip():
        out["customer_name"] = name.strip()

    return out


def _parse_clear(raw: Any) -> list[str]:
    if not isinstance(raw, list):
        return []
    allowed = {"date", "time", "party_size", "customer_name"}
    return [s for s in raw if isinstance(s, str) and s in allowed]


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

    Une tâche unique, contre la dizaine de contraintes du prompt
    précédent. La mesure a montré que les deux quantisations testées
    échouaient identiquement sous forte charge, et réussissaient dès
    qu'on l'allégeait.
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
