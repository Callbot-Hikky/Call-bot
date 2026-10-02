"""Réponse aux questions hors script — le seul rôle conversationnel du LLM.

Le parcours de réservation est piloté par le code (`routed_turn` +
`question_router`) ; le modèle n'intervient que lorsque le client pose une
vraie question (horaires, options…). Une tâche unique et un prompt court se
sont montrés bien plus fiables, sous forte charge, qu'un moteur généraliste
chargé de dizaines de contraintes.

Jusqu'ici le modèle ne connaissait du restaurant que ses horaires et sa
capacité : à « vous avez une terrasse ? », il inventait. Il reçoit
maintenant les attributs du restaurant (backend) et les passages de sa base
de connaissances qui ressemblent à la question. Quand rien ne répond, le bot
le dit et remonte la question : c'est le restaurateur qui lui apprendra la
réponse, jamais le client.
"""

from __future__ import annotations

import asyncio
import logging
import re
from datetime import date, time
from typing import Any

from hikky.domain.knowledge import KnowledgePassage
from hikky.ports.knowledge import KnowledgePort
from hikky.ports.language_model import LanguageModelPort

logger = logging.getLogger("hikky.answerer")
# Les questions auxquelles le bot n'a pas pu répondre : à relire pour enrichir
# les attributs du restaurant côté backend. Avec une base de connaissances
# branchée, elles y sont aussi remontées (`report_unanswered`).
sans_reponse = logging.getLogger("hikky.questions_sans_reponse")

_JOURS = ("lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche")
_MOIS = ("janvier", "février", "mars", "avril", "mai", "juin", "juillet",
         "août", "septembre", "octobre", "novembre", "décembre")


ANSWER_PROMPT = """Tu es l'assistant vocal du restaurant {name}, au téléphone.

HORAIRES :
{hours}

CAPACITÉ : {capacity} couverts, groupes de {max_group} personnes maximum.
{address}
INFORMATIONS PRATIQUES :
{facts}
{passages}
RÉSERVATION EN COURS :
{known}

Le client vient de te poser une question. Réponds-y en UNE phrase courte \
(quinze mots maximum), naturelle, uniquement à partir des informations \
ci-dessus. N'invente rien. Si la réponse n'y figure pas, réponds uniquement \
par le mot {unknown}. Dis les dates et heures en toutes lettres, jamais en \
chiffres.

IMPORTANT : la réservation ci-dessus n'est PAS encore enregistrée. Ne dis \
JAMAIS qu'elle est confirmée, réservée, notée, validée ou enregistrée, et \
n'annonce aucune réservation. Tu réponds seulement à la question.

Réponds uniquement par la phrase à dire, sans JSON ni commentaire."""

# Mot que le modèle renvoie quand il n'a pas l'information. Le code le
# remplace par une phrase écrite : sans ça, un petit modèle préfère
# inventer une terrasse plutôt qu'avouer qu'il n'en sait rien.
UNKNOWN = "INCONNU"

REPLY_UNKNOWN_REPORTED = (
    "Je n'ai pas cette information, je transmets votre question au restaurant."
)
# Sans base de connaissances branchée, on ne promet pas de transmettre. Court
# et honnête : au téléphone, « je n'ai pas d'information concernant l'halal
# de notre restaurant, je vous conseille de contacter notre gérant… » (observé)
# est trop long et laisse l'appelant sans issue.
REPLY_UNKNOWN = "Je n'ai pas cette information, l'équipe pourra vous renseigner sur place."
_REPONSE_INCONNUE = REPLY_UNKNOWN

# En dessous, le passage ne parle sans doute pas de la question : on ne le
# montre pas au modèle, qui s'en servirait pour répondre à côté. Valeur de
# départ, à ajuster sur de vrais appels.
DEFAULT_MIN_SCORE = 0.35

# Le client attend au téléphone : passé ce délai, on répond sans la base
# plutôt que de laisser un blanc.
KNOWLEDGE_TIMEOUT_SECONDS = 1.5

# Le modèle a un contexte court : un passage long évincerait les horaires.
MAX_PASSAGE_CHARS = 400


# `answer()` n'est appelé que hors parcours de réservation : aucune réservation
# n'est créée ici. Toute affirmation qu'elle est faite est donc FAUSSE par
# construction. Observé en appel réel : « votre réservation est confirmée pour
# ce soir à vingt et une heures sous le nom Général » — rien n'était réservé.
#
# La garde vise la RÉSERVATION (ou la table, le créneau), pas le verbe
# « confirmer » en soi : « Oui, je vous confirme que nous avons une terrasse »
# est une réponse légitime et ne doit pas finir en « Je comprends. ».
_FAUSSE_CONFIRMATION = re.compile(
    r"\b((r[ée]servation|table|cr[ée]neau|demande)\w* (est |a [ée]t[ée] |sera )?(bien )?"
    r"(confirm|enregistr|not[ée]|valid|pris|fait|r[ée]serv)\w*|"
    r"c'est (not[ée]|r[ée]serv[ée]|enregistr[ée])\w*|"
    r"bien not[ée]|j'ai (not[ée]|r[ée]serv[ée]|enregistr[ée]|confirm[ée])\w*|"
    r"je (vous )?confirme(?! que)|"
    r"vous (êtes|etes) (attendus?|inscrits?)\b)",
    re.IGNORECASE,
)
_REPONSE_SURE = "Je comprends."

# Le modèle dit qu'il ne sait pas, mais avec ses mots au lieu du mot convenu :
# on le reconnaît quand même, pour ne jamais prononcer une tirade d'excuses.
_SANS_INFORMATION = re.compile(
    r"je n'ai pas (cette |d'|l'|de |d'autres? )?(information|info|détail|precision|précision)|"
    r"je ne (sais|dispose|connais|peux pas (vous )?(dire|répondre|renseigner))|"
    r"je n'en sais|pas (en mesure|capable) de|aucune information|je l'ignore",
    re.IGNORECASE,
)


class QuestionAnswerer:
    """Répond aux questions hors script — le seul rôle conversationnel.

    Une tâche unique, contre la dizaine de contraintes d'un prompt
    généraliste. La mesure a montré que les quantisations testées échouaient
    identiquement sous forte charge, et réussissaient dès qu'on l'allégeait.

    Avec une base de connaissances, la réponse s'appuie sur ce que le
    restaurateur a écrit. Quand rien ne répond, le bot le dit et remonte la
    question : c'est le restaurateur qui lui apprendra la réponse, jamais
    le client.
    """

    def __init__(
        self,
        llm: LanguageModelPort,
        *,
        knowledge: KnowledgePort | None = None,
        min_score: float = DEFAULT_MIN_SCORE,
        knowledge_timeout_seconds: float = KNOWLEDGE_TIMEOUT_SECONDS,
    ) -> None:
        self._llm = llm
        self._knowledge = knowledge
        self._min_score = min_score
        self._knowledge_timeout = knowledge_timeout_seconds
        # Références gardées : une tâche sans référence peut être ramassée
        # avant d'avoir tourné.
        self._pending: set[asyncio.Task[None]] = set()

    async def answer(self, *, user_text, intent, context, history) -> str:
        passages = await self._relevant_passages(user_text)
        adresse = getattr(context, "address", None)
        system = ANSWER_PROMPT.format(
            name=context.name,
            hours=_format_hours(context),
            capacity=context.total_capacity,
            max_group=context.rules.max_group_size,
            address=f"\nADRESSE : {adresse}\n" if adresse else "",
            facts=_format_facts(getattr(context, "attributes", None)),
            passages=_format_passages(passages),
            known=_format_known(intent),
            unknown=UNKNOWN,
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
        if _is_unknown(texte) or _SANS_INFORMATION.search(texte):
            sans_reponse.warning("question sans réponse : %r (modèle : %r)", user_text, texte)
            return self._admit_unknown(user_text)
        return texte

    async def wait_pending(self) -> None:
        """Attend les signalements en cours. Utile en fin d'appel et en test."""
        if self._pending:
            await asyncio.gather(*self._pending, return_exceptions=True)

    async def _relevant_passages(self, question: str) -> list[KnowledgePassage]:
        if self._knowledge is None:
            return []
        try:
            passages = await asyncio.wait_for(
                self._knowledge.search(question), timeout=self._knowledge_timeout
            )
        except TimeoutError:
            logger.info("base de connaissances trop lente — réponse sans elle")
            return []
        except Exception:  # noqa: BLE001 — le port promet de ne pas lever ; ceinture et bretelles
            logger.warning("base de connaissances en échec", exc_info=True)
            return []
        return [p for p in passages if p.score >= self._min_score]

    def _admit_unknown(self, question: str) -> str:
        if self._knowledge is None:
            return REPLY_UNKNOWN
        # En tâche de fond : le client n'attend pas un aller-retour réseau
        # pour entendre « je ne sais pas ».
        task = asyncio.create_task(self._knowledge.report_unanswered(question))
        self._pending.add(task)
        task.add_done_callback(self._pending.discard)
        return REPLY_UNKNOWN_REPORTED


def _is_unknown(texte: str) -> bool:
    return texte.strip(" .!«»\"'").upper().startswith(UNKNOWN)


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


# Libellés parlés des clés déclarées par le backend (`restaurants.attributes`).
# Une clé inconnue est lue telle quelle, soulignés remplacés par des espaces :
# le restaurateur peut déclarer n'importe quoi sans changement de schéma.
_LIBELLES: dict[str, str] = {
    "halal": "halal",
    "casher": "casher",
    "kosher": "casher",
    "vegan": "options véganes",
    "vegetarian": "options végétariennes",
    "bio": "produits bio",
    "gluten_free": "options sans gluten",
    "terrace": "terrasse",
    "rooftop": "rooftop",
    "wifi": "wifi",
    "high_chairs": "chaises hautes pour enfants",
    "pets_allowed": "animaux acceptés",
    "private_parking": "parking privé",
    "air_conditioning": "climatisation",
    "wheelchair_accessible": "accès fauteuil roulant",
    "delivery": "livraison",
    "takeaway": "vente à emporter",
    "cash_only": "paiement uniquement en espèces",
    "card_payment": "paiement par carte bancaire",
    "meal_vouchers": "tickets restaurant acceptés",
    "reservation_recommended": "réservation recommandée",
    "ambiance": "ambiance",
    "price_range": "gamme de prix",
    "cuisine_type": "type de cuisine",
}
_GAMMES_DE_PRIX = {
    "€": "économique",
    "€€": "prix modérés",
    "€€€": "haut de gamme",
    "€€€€": "très haut de gamme",
}


def _format_facts(attributes: Any) -> str:
    """Les attributs du restaurant, à plat et en français prononçable.

    Le JSON du backend est groupé (`dietary`, `equipments`…) ; le modèle n'a
    pas besoin des groupes, seulement des faits : « - terrasse : oui ».
    """
    lignes: list[str] = []
    for cle, valeur in _aplatir(attributes):
        libelle = _LIBELLES.get(cle, cle.replace("_", " "))
        if isinstance(valeur, bool):
            texte = "oui" if valeur else "non"
        elif isinstance(valeur, (list, tuple, set)):
            texte = ", ".join(str(v) for v in valeur if v is not None and str(v).strip())
        elif valeur is None:
            continue
        else:
            texte = str(valeur).strip()
        if cle == "price_range":
            texte = _GAMMES_DE_PRIX.get(texte, texte)
        if texte:
            lignes.append(f"- {libelle} : {texte}")
    return "\n".join(lignes) if lignes else "- (aucune information supplémentaire)"


def _format_passages(passages: list[KnowledgePassage]) -> str:
    """Ce que le restaurateur a écrit dans sa base, ou rien du tout."""
    if not passages:
        return ""
    lignes = [f"- {p.title} : {p.content[:MAX_PASSAGE_CHARS]}" for p in passages]
    return "\nCE QUE LE RESTAURANT A ÉCRIT :\n" + "\n".join(lignes) + "\n"


def _aplatir(valeur: Any) -> list[tuple[str, Any]]:
    if not isinstance(valeur, dict):
        return []
    plat: list[tuple[str, Any]] = []
    for cle, v in valeur.items():
        if isinstance(v, dict):
            plat.extend(_aplatir(v))
        else:
            plat.append((str(cle), v))
    return plat


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
