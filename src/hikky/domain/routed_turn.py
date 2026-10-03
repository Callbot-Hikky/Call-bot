"""Tour de dialogue piloté par le code, intelligence confiée au modèle.

Répartition des rôles, établie par la mesure et non par principe :

- **le code** tient l'état et choisit la question suivante — les deux
  quantisations testées reposaient trois fois la même question après
  avoir reçu la réponse ;
- **le modèle** comprend les phrases (l'extracteur ne s'est jamais
  trompé) et répond aux questions hors script, où il excelle.

Conséquence sur la latence : sur un tour ordinaire, un seul appel LLM au
lieu de deux. L'appel conversationnel n'a lieu que si le client pose
réellement une question.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from hikky.domain.outcomes import CallOutcome
from hikky.domain.phraseur import Intention
from hikky.domain.question_router import Action, phrase_for_slot, route_turn

logger = logging.getLogger("hikky.routed_turn")
conversation = logging.getLogger("hikky.conversation")

MAX_HISTORY_MESSAGES = 12

_JOURS = ("lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche")
_MOIS = ("janvier", "février", "mars", "avril", "mai", "juin", "juillet",
         "août", "septembre", "octobre", "novembre", "décembre")


@dataclass(frozen=True, slots=True)
class TurnOutcome:
    should_end: bool
    outcome: CallOutcome | None = None
    awaiting_confirmation: bool = False
    slots: dict[str, Any] = field(default_factory=dict)


def _heure_fr(hour: Any) -> str:
    if hour.hour == 12 and hour.minute == 0:
        return "midi"
    if hour.hour == 0 and hour.minute == 0:
        return "minuit"
    if hour.minute == 0:
        return f"{hour.hour} heures"
    return f"{hour.hour} heures {hour.minute}"


def build_recap(intent: Any) -> str:
    """Récapitulatif complet, généré par le code.

    Le modèle produisait « Puis-je confirmer votre réservation ? » sans
    dire quoi, puis oubliait le nom. Ce message doit être exact.
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


def _refus_si_groupe_trop_grand(session: Any) -> str | None:
    """Message de refus si la taille du groupe dépasse le plafond, sinon None.

    Le plafond vient du contexte du restaurant (`rules.max_group_size`, alimenté
    par la politique du backend). Au-delà, un très grand groupe relève d'un
    échange humain — le bot le dit clairement plutôt que d'enchaîner sur
    « souhaitez-vous une autre heure ? », qui n'a jamais de sens ici.
    """
    rules = getattr(getattr(session, "context", None), "rules", None)
    max_group = getattr(rules, "max_group_size", None)
    party = getattr(getattr(session, "intent", None), "party_size", None)
    if max_group is None or party is None or party <= max_group:
        return None
    return (
        f"Je suis désolé, pour un groupe de plus de {max_group} personnes je ne "
        "peux pas prendre la réservation par téléphone. Souhaitez-vous réserver "
        "pour un plus petit groupe ?"
    )


_CRENEAUX_MARKERS = (
    "disponib",
    "créneau",
    "creneau",
    "quelles heures",
    "quels horaires",
    "heures libres",
    "heure libre",
    "de la place",
    "des places",
    "vous avez de la place",
)


def _demande_de_creneaux(user_text: str) -> bool:
    """Vrai si le client demande nos créneaux / heures disponibles."""
    texte = (user_text or "").lower()
    return any(marker in texte for marker in _CRENEAUX_MARKERS)


def _jour_fr(jour: Any) -> str:
    return f"le {_JOURS[jour.weekday()]} {jour.day} {_MOIS[jour.month - 1]}"


def _lister_creneaux(session: Any) -> str:
    """Énumère les plages d'ouverture pour le jour demandé (déterministe)."""
    context = getattr(session, "context", None)
    intent = getattr(session, "intent", None)
    jour = getattr(intent, "date", None) if intent is not None else None
    horaires = list(getattr(context, "opening_hours", None) or [])
    if jour is None:
        return "Bien sûr. Pour quel jour souhaitez-vous connaître nos disponibilités ?"
    plages = sorted(
        (oh for oh in horaires if oh.weekday == jour.weekday()),
        key=lambda oh: oh.opens,
    )
    if not plages:
        return (
            f"Je suis désolé, nous sommes fermés {_jour_fr(jour)}. "
            "Souhaitez-vous choisir un autre jour ?"
        )
    texte = " et ".join(
        f"de {_heure_fr(oh.opens)} à {_heure_fr(oh.closes)}" for oh in plages
    )
    return (
        f"{_jour_fr(jour).capitalize()}, nous vous accueillons {texte}. "
        "À quelle heure souhaitez-vous venir ?"
    )


async def _verifier_dispo_avant_nom(
    session: Any,
    history: list[dict[str, str]],
    speak: Callable[[str], Awaitable[None]],
    user_text: str,
) -> TurnOutcome | None:
    """Vérifie la dispo du créneau avant de demander le nom.

    Renvoie un TurnOutcome (message d'indisponibilité + heure effacée) si le
    créneau n'est pas réservable, sinon None pour laisser le flux demander le
    nom. Sur une session basique (sans support de dispo détaillée), on
    n'intervient pas.
    """
    detail_fn = getattr(session, "availability_detail", None)
    confirmed_fn = getattr(session, "availability_confirmed_for", None)
    mark_fn = getattr(session, "mark_availability_ok", None)
    if detail_fn is None or confirmed_fn is None or mark_fn is None:
        return None

    intent = session.intent
    quand = getattr(intent, "date_time", None)
    couverts = getattr(intent, "party_size", None)
    if quand is None or couverts is None:
        return None
    if confirmed_fn(quand, couverts):
        return None

    detail = await detail_fn(quand, couverts)
    if getattr(detail, "available", False):
        mark_fn(quand, couverts)
        return None

    raison = getattr(detail, "reason", None)
    alternatives = getattr(detail, "alternatives", None) or []
    if raison == "closed":
        reply = (
            "Je suis désolé, nous sommes fermés à cette heure-là. "
            "Souhaitez-vous choisir une autre heure ?"
        )
    else:
        reply = "Je suis désolé, nous sommes complets à cette heure-là."
        if alternatives:
            heures = ", ".join(_heure_fr(a) for a in alternatives[:3])
            reply += f" Nous aurions de la place à {heures}."
        reply += " Souhaitez-vous une autre heure ?"

    session.clear_slots(["time"])
    logger.info("créneau indisponible avant collecte du nom — nouvelle heure demandée")
    await _remember(history, user_text, reply)
    await speak(reply)
    return TurnOutcome(should_end=False, slots={})


async def run_routed_turn(
    *,
    session: Any,
    user_text: str,
    customer_phone: str | None,
    history: list[dict[str, str]],
    extractor: Any,
    answerer: Any,
    awaiting_confirmation: bool,
    speak: Callable[[str], Awaitable[None]],
    recent_phrasings: set[str] | None = None,
    phraseur: Any = None,
) -> TurnOutcome:
    # 1. Comprendre — seul rôle du modèle sur un tour ordinaire.
    slots: dict[str, Any] = {}
    seul_le_nom_manque = session.intent.missing_slots() == {"customer_name"}
    if extractor is not None:
        slots = await extractor.extract(user_text)
    if seul_le_nom_manque and "customer_name" not in slots:
        # On vient de demander le nom : une réponse d'un mot (« Royal. ») EST le nom,
        # même si l'extracteur LLM ne l'a pas reconnu comme tel (observé en appel réel :
        # le bot a redemandé le nom au client qui venait de le donner).
        nom = _nom_depuis_reponse_courte(user_text)
        if nom:
            slots = {**slots, "customer_name": nom}
    if slots:
        conversation.info("  extrait : %s", slots)
        session.apply_slots(slots)

    # Plafond de groupe : au-delà, l'assistant ne prend pas la réservation par
    # téléphone. On refuse tôt et clairement, plutôt que de proposer d'autres
    # horaires — ce qui n'aiderait jamais pour un groupe trop grand et laissait
    # croire à tort que le problème venait de l'heure.
    refus_taille = _refus_si_groupe_trop_grand(session)
    if refus_taille is not None:
        logger.info("groupe au-delà du plafond — réservation refusée")
        await _remember(history, user_text, refus_taille)
        await speak(refus_taille)
        return TurnOutcome(should_end=False, slots=slots)

    # 2. Décider — sans LLM : l'état suffit.
    decision = route_turn(session.intent, user_text, awaiting_confirmation)
    conversation.info("  route   : %s", decision.action.value)

    # Le client demande nos créneaux / heures disponibles → réponse
    # déterministe (au lieu de laisser le modèle broder et tourner en rond).
    if decision.action is Action.ANSWER_QUESTION and _demande_de_creneaux(user_text):
        reply = _lister_creneaux(session)
        await _remember(history, user_text, reply)
        await speak(reply)
        return TurnOutcome(
            should_end=False, awaiting_confirmation=awaiting_confirmation, slots=slots
        )

    # Dès que jour + heure + nombre sont connus, vérifier la disponibilité —
    # AVANT de demander le nom ET avant de récapituler. Sinon, si le client a
    # donné son nom tôt, l'indisponibilité n'était annoncée qu'au tout dernier
    # moment (au moment de réserver). On propose une autre heure tout de suite.
    if decision.action is Action.CONFIRM or (
        decision.action is Action.ASK_SLOT and decision.slot == "customer_name"
    ):
        indispo = await _verifier_dispo_avant_nom(session, history, speak, user_text)
        if indispo is not None:
            return indispo

    # 3. Agir.
    if decision.action is Action.ANSWER_QUESTION:
        # Le client interroge : c'est ici que l'intelligence sert.
        reply = await answerer.answer(
            user_text=user_text,
            intent=session.intent,
            context=session.context,
            history=history,
        )
        reply = _ensure_progress(reply, session.intent, recent_phrasings, history)
        await _remember(history, user_text, reply)
        await speak(reply)
        return TurnOutcome(
            should_end=False,
            awaiting_confirmation=awaiting_confirmation,
            slots=slots,
        )

    if decision.action is Action.BOOK:
        outcome = await session.book(customer_phone)
        if outcome is None:
            logger.error("RESERVATION NON CRÉÉE — créneau indisponible ou hors horaires")
            reply = "Ce créneau n'est pas disponible. Souhaitez-vous une autre heure ?"
            await _remember(history, user_text, reply)
            await speak(reply)
            return TurnOutcome(should_end=False, slots=slots)

        intent = session.intent
        logger.info(
            "RESERVATION CRÉÉE : %s, %s personnes, %s (%s)",
            intent.date_time, intent.party_size, intent.customer_name, outcome,
        )
        reply = (
            f"C'est noté, {intent.customer_name}. "
            f"Nous vous attendons {_JOURS[intent.date.weekday()]} "
            f"{intent.date.day} {_MOIS[intent.date.month - 1]} "
            f"à {_heure_fr(intent.time)}. "
            "Vous allez recevoir un SMS de confirmation. Bonne journée !"
        )
        await _remember(history, user_text, reply)
        await speak(reply)
        return TurnOutcome(should_end=True, outcome=outcome, slots=slots)

    if decision.action is Action.CORRECT:
        reply = "D'accord. Qu'est-ce que je corrige ?"
        await _remember(history, user_text, reply)
        await speak(reply)
        return TurnOutcome(should_end=False, slots=slots)

    if decision.action is Action.CONFIRM:
        reply = build_recap(session.intent)
        await _remember(history, user_text, reply)
        await speak(reply)
        return TurnOutcome(should_end=False, awaiting_confirmation=True, slots=slots)

    # ASK_SLOT — le cas nominal. Le code choisit l'information à demander,
    # le modèle la met en mots en tenant compte de ce que le client vient
    # de dire. Sans ce dernier point, le bot récitait la même phrase quel
    # que soit l'interlocuteur : c'est ce qui le faisait sonner robotique.
    figee = phrase_for_slot(decision.slot or "", recent_phrasings or set())
    # Le modèle n'est consulté que s'il a quelque chose à accuser réception
    # (un slot vient d'être compris). Phrase incomprise → rien à reformuler ; et
    # la demande du nom est déjà naturelle en figé. Sinon on attendait 1,2 s le
    # modèle pour retomber sur la phrase figée — observé 2× sur un appel.
    a_quelque_chose_a_dire = bool(slots) and decision.slot != "customer_name"
    if phraseur is not None and a_quelque_chose_a_dire:
        reply = await phraseur.formuler(
            Intention(
                besoin="demander_slot",
                slot=decision.slot,
                repli=figee,
                derniere_phrase_client=user_text,
            )
        )
    else:
        reply = figee
    if recent_phrasings is not None:
        recent_phrasings.add(reply)
    await _remember(history, user_text, reply)
    await speak(reply)
    return TurnOutcome(
        should_end=False, awaiting_confirmation=awaiting_confirmation, slots=slots
    )


# Formules qui entourent un nom au téléphone ; on ne garde que le nom.
_FORMULES_NOM = re.compile(
    r"^(c'est |c’est |je m'appelle |je m’appelle |au nom de |pour |"
    r"monsieur |madame |mademoiselle |m\. |mme |mr |mlle )+",
    re.IGNORECASE,
)
# Réponses courtes qui ne sont PAS un nom : acquiescements, politesse, nombres,
# moments. Un nom ne contient ni chiffre ni ces mots.
_PAS_UN_NOM = re.compile(
    r"\d|\b(oui|non|ok|d'accord|pardon|quoi|comment|heures?|midi|minuit|soir|matin|"
    r"demain|aujourd'hui|lundi|mardi|mercredi|jeudi|vendredi|samedi|dimanche|"
    r"personnes?|couverts?|table|réserv\w*|merci|bonjour|allo|allô|attendez|"
    r"un|deux|trois|quatre|cinq|six|sept|huit|neuf|dix|vingt|trente)\b",
    re.IGNORECASE,
)


def _nom_depuis_reponse_courte(user_text: str) -> str | None:
    """Le nom donné en réponse directe à « À quel nom ? », ou None.

    Trois mots au plus, lettres seulement (tirets et apostrophes admis), sans
    vocabulaire d'heure, de nombre ou d'acquiescement. Volontairement strict :
    un faux nom retenu réserverait sous une mauvaise identité, alors qu'un nom
    manqué coûte juste une redemande.
    """
    texte = (user_text or "").strip().strip(".!?,;: ").strip()
    texte = _FORMULES_NOM.sub("", texte).strip()
    if not texte or len(texte) > 40 or _PAS_UN_NOM.search(texte):
        return None
    mots = texte.split()
    if not 1 <= len(mots) <= 3:
        return None
    if not all(re.fullmatch(r"[A-Za-zÀ-ÖØ-öø-ÿ][A-Za-zÀ-ÖØ-öø-ÿ'’-]*", m) for m in mots):
        return None
    return " ".join(m[:1].upper() + m[1:] for m in mots)


def _reservation_engagee(intent: Any) -> bool:
    """Le client a-t-il commencé une réservation (au moins un créneau connu) ?"""
    from hikky.domain.question_router import SLOT_ORDER

    return len(intent.missing_slots()) < len(SLOT_ORDER)


def _ensure_progress(
    reply: str,
    intent: Any,
    recent: set[str] | None,
    history: list[dict[str, str]] | None = None,
) -> str:
    """Après avoir répondu au client, reprendre la réservation — seulement si
    elle est engagée, et sans insister.

    Observé en appel réel : « à chaque fin de réponse l'IA me casse la tête
    pour réserver une table ». Un appelant qui ne fait que se renseigner
    recevait une relance commerciale après CHAQUE réponse. Règles :
    - aucun créneau connu → on répond, point ; c'est au client de dire s'il
      veut réserver (la salutation l'a déjà invité à parler) ;
    - réservation en cours → on reprend sur ce qui manque, mais jamais deux
      fois de suite : si notre phrase précédente était déjà une question et
      que le client en pose une autre, on répond seulement.
    """
    reply = (reply or "").strip()
    if not reply:
        reply = "Pardon, pouvez-vous répéter ?"
    if reply.rstrip().endswith("?"):
        return reply
    if not _reservation_engagee(intent):
        return reply
    if history:
        derniere = history[-1]
        if derniere.get("role") == "assistant" and derniere.get("content", "").rstrip().endswith("?"):
            return reply
    missing = intent.missing_slots()
    from hikky.domain.question_router import SLOT_ORDER

    for slot in SLOT_ORDER:
        if slot in missing:
            return f"{reply} {phrase_for_slot(slot, recent or set())}"
    return f"{reply} Puis-je confirmer votre réservation ?"


async def _remember(
    history: list[dict[str, str]], user_text: str, reply: str
) -> None:
    history.append({"role": "user", "content": user_text})
    history.append({"role": "assistant", "content": reply})
    if len(history) > MAX_HISTORY_MESSAGES:
        del history[: len(history) - MAX_HISTORY_MESSAGES]
