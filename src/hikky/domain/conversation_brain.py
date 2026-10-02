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
from typing import Any

from hikky.ports.language_model import LanguageModelPort

logger = logging.getLogger("hikky.answerer")
# Les questions auxquelles le bot n'a pas pu répondre : à relire pour enrichir
# les attributs du restaurant côté backend. Aucune route de rappel n'existe
# aujourd'hui (voir `call_ingest_adapter.create_callback_request`), on ne
# promet donc rien au client — on trace.
sans_reponse = logging.getLogger("hikky.questions_sans_reponse")

_JOURS = ("lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche")
_MOIS = ("janvier", "février", "mars", "avril", "mai", "juin", "juillet",
         "août", "septembre", "octobre", "novembre", "décembre")


ANSWER_PROMPT = """Tu es l'assistant vocal du restaurant {name}, au téléphone.

HORAIRES :
{hours}

CAPACITÉ : {capacity} couverts, groupes de {max_group} personnes maximum.
{address}
CE QUE PROPOSE LE RESTAURANT :
{facts}

RÉSERVATION EN COURS :
{known}

Le client vient de te poser une question. Réponds-y en UNE phrase courte \
(quinze mots maximum), naturelle, uniquement à partir des informations \
ci-dessus. Si la réponse n'y figure pas, dis exactement : « Je n'ai pas \
cette information. » Dis les dates et heures en toutes lettres, jamais en \
chiffres.

IMPORTANT : la réservation ci-dessus n'est PAS encore enregistrée. Ne dis \
JAMAIS qu'elle est confirmée, réservée, notée, validée ou enregistrée, et \
n'annonce aucune réservation. Tu réponds seulement à la question.

Réponds uniquement par la phrase à dire, sans JSON ni commentaire."""


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

# Le modèle ne sait pas : au téléphone, « je n'ai pas d'information concernant
# l'halal de notre restaurant, je vous conseille de contacter notre gérant pour
# plus de détails » (observé) est trop long et laisse l'appelant sans issue.
# On remplace par une phrase courte, honnête, qui indique où obtenir la
# réponse ; `_ensure_progress` enchaîne ensuite sur la réservation.
_SANS_INFORMATION = re.compile(
    r"je n'ai pas (cette |d'|l'|de |d'autres? )?(information|info|détail|precision|précision)|"
    r"je ne (sais|dispose|connais|peux pas (vous )?(dire|répondre|renseigner))|"
    r"je n'en sais|pas (en mesure|capable) de|aucune information|je l'ignore",
    re.IGNORECASE,
)
_REPONSE_INCONNUE = (
    "Je n'ai pas cette information, l'équipe pourra vous renseigner sur place."
)


class QuestionAnswerer:
    """Répond aux questions hors script — le seul rôle conversationnel.

    Une tâche unique, contre la dizaine de contraintes d'un prompt
    généraliste. La mesure a montré que les quantisations testées échouaient
    identiquement sous forte charge, et réussissaient dès qu'on l'allégeait.
    """

    def __init__(self, llm: LanguageModelPort) -> None:
        self._llm = llm

    async def answer(self, *, user_text, intent, context, history) -> str:
        adresse = getattr(context, "address", None)
        system = ANSWER_PROMPT.format(
            name=context.name,
            hours=_format_hours(context),
            capacity=context.total_capacity,
            max_group=context.rules.max_group_size,
            address=f"\nADRESSE : {adresse}\n" if adresse else "",
            facts=_format_facts(getattr(context, "attributes", None)),
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
        if _SANS_INFORMATION.search(texte):
            sans_reponse.warning("question sans réponse : %r (modèle : %r)", user_text, texte)
            return _REPONSE_INCONNUE
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
