"""Extracteur de slots basé sur le LLM.

À chaque tour utilisateur, on appelle le LLM en mode JSON strict avec
un prompt système qui lui demande d'extraire — quand ils sont
clairement présents — `date_time`, `party_size` et `customer_name`.

C'est ce qui débloque le critère §9.3 du spec : sans cet extracteur,
`ReservationIntent` ne se remplit jamais et la réservation ne peut
pas être finalisée.

L'extracteur fait UN appel LLM supplémentaire par tour. C'est un coût
assumé pour le POC ; une future itération pourra fusionner extraction
et génération de réponse via du tool-calling structuré dans un seul
appel.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable
from datetime import date, datetime, time
from typing import Any

from hikky.domain.turn_runner import SlotExtractor
from hikky.ports.language_model import LanguageModelPort

logger = logging.getLogger("hikky.slot_extractor")

SYSTEM_PROMPT = """Tu es un extracteur de slots pour un assistant vocal de \
réservation de restaurant.

Étant donnée la dernière phrase de l'utilisateur (en français), extrais \
UNIQUEMENT les informations clairement présentes parmi :

- "date" : jour de la réservation, format ISO "YYYY-MM-DD". Résous les \
expressions relatives ("demain", "vendredi", "ce soir") par rapport à \
la date du jour.
- "time" : heure de la réservation, format "HH:MM" (24h). UNIQUEMENT si \
une heure précise est énoncée.
- "party_size" : nombre de personnes, entier strictement positif.
- "customer_name" : nom du client (chaîne).

Date du jour : {today_iso} ({weekday_name}).

RÈGLE ABSOLUE — N'INVENTE JAMAIS.

"date" et "time" sont INDÉPENDANTS : émets celui que tu connais, omets \
l'autre. Un moment de journée vague ("matin", "midi", "soirée", \
"dans l'après-midi") n'est PAS une heure — n'émets alors pas "time", \
mais émets bien "date" si le jour est connu. C'est à l'assistant de \
demander l'heure exacte, pas à toi de la choisir.

De même, n'émets "party_size" que si un nombre de personnes est \
réellement énoncé, et "customer_name" que si un nom est réellement \
donné. Un slot absent doit rester absent : il vaut mieux redemander \
que réserver sur une supposition.

Réponds par UN SEUL objet JSON minifié contenant uniquement les clés \
des slots détectés (omets les autres). Si rien n'est extrait, réponds \
exactement `{{}}`. Aucune explication, aucun markdown, aucun texte hors \
du JSON.

Exemples :
- "Je voudrais réserver" → {{}}
- "Demain à 20 heures pour 4" → {{"date": "{tomorrow_iso}", \
"time": "20:00", "party_size": 4}}
- "Au nom de Dupont" → {{"customer_name": "Dupont"}}
- "On serait 6" → {{"party_size": 6}}
- "demain matin" → {{"date": "{tomorrow_iso}"}}   (jour connu, heure inconnue)
- "dans la soirée" → {{}}   (ni jour ni heure precise)
- "plutôt vers midi" → {{}}   (moment vague, pas une heure)
- "demain à midi" → {{"date": "{tomorrow_iso}", "time": "12:00"}}
- "à vingt heures" → {{"time": "20:00"}}   (heure seule, jour inconnu)
"""

_WEEKDAYS_FR = [
    "lundi",
    "mardi",
    "mercredi",
    "jeudi",
    "vendredi",
    "samedi",
    "dimanche",
]

# Récupère le 1er bloc JSON dans la réponse (au cas où le LLM ajoute du texte)
_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


class LLMSlotExtractor(SlotExtractor):
    def __init__(
        self,
        llm: LanguageModelPort,
        *,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._llm = llm
        self._clock = clock or datetime.now

    async def extract(self, user_text: str) -> dict[str, Any]:
        if not user_text or not user_text.strip():
            return {}

        now = self._clock()
        from datetime import timedelta

        tomorrow = now + timedelta(days=1)
        system = SYSTEM_PROMPT.format(
            today_iso=now.date().isoformat(),
            weekday_name=_WEEKDAYS_FR[now.weekday()],
            tomorrow_iso=tomorrow.date().isoformat(),
        )

        try:
            raw = await self._llm.complete(
                [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_text},
                ]
            )
        except Exception:  # noqa: BLE001 — extraction est best-effort
            logger.warning("LLM slot extraction call failed", exc_info=True)
            return {}

        return _drop_invented_time(self._parse(raw, now), user_text)

    def _parse(self, raw: str, now: datetime | None = None) -> dict[str, Any]:
        today = (now or self._clock()).date()
        match = _JSON_BLOCK.search(raw or "")
        if not match:
            logger.debug("No JSON object in LLM reply: %r", raw)
            return {}
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            logger.warning("Could not parse JSON from LLM: %r", raw)
            return {}
        if not isinstance(parsed, dict):
            return {}

        out: dict[str, Any] = {}

        dt_raw = parsed.get("date_time")  # retro-compat

        day_raw = parsed.get("date")
        if isinstance(day_raw, str):
            try:
                parsed_day = _sane_date(date.fromisoformat(day_raw), today)
            except ValueError:
                logger.debug("Bad ISO date from LLM: %r", day_raw)
            else:
                if parsed_day is not None:
                    out["date"] = parsed_day

        time_raw = parsed.get("time")
        if isinstance(time_raw, str):
            try:
                out["time"] = time.fromisoformat(time_raw)
            except ValueError:
                logger.debug("Bad ISO time from LLM: %r", time_raw)
        if isinstance(dt_raw, str):
            try:
                out["date_time"] = datetime.fromisoformat(dt_raw)
            except ValueError:
                logger.debug("Bad ISO date_time from LLM: %r", dt_raw)

        ps_raw = parsed.get("party_size")
        if isinstance(ps_raw, int) and ps_raw > 0:
            out["party_size"] = ps_raw

        name_raw = parsed.get("customer_name")
        if isinstance(name_raw, str) and name_raw.strip():
            out["customer_name"] = name_raw.strip()

        return out


# Moments de journée VAGUES : ils indiquent une période, pas une heure. Le
# modèle infère parfois une heure précise (« ce soir » → 18:00) malgré la
# consigne ; on retire alors l'heure de façon déterministe.
_VAGUE_PERIOD = re.compile(
    r"\b(ce soir|cette soir[ée]+|en soir[ée]+|dans la soir[ée]+|"
    r"ce matin|dans la matin[ée]+|cet? apr[eè]s[- ]?midi|"
    r"dans l'apr[eè]s[- ]?midi|en journ[ée]+|dans la journ[ée]+|tant[ôo]t)\b",
    re.IGNORECASE,
)
# Une heure réellement énoncée : un chiffre, ou « midi »/« minuit » (qui SONT
# des heures valides et doivent, eux, être conservés).
_EXPLICIT_HOUR = re.compile(r"\d|\bmidi\b|\bminuit\b", re.IGNORECASE)


def _drop_invented_time(slots: dict[str, Any], user_text: str) -> dict[str, Any]:
    """Retire l'heure quand la phrase ne contient qu'un moment vague.

    « ce soir », « dans l'après-midi »… ne sont pas des heures : c'est à
    l'assistant de demander l'heure exacte, pas au modèle de la deviner. Si
    la phrase donne une vraie heure (chiffre, « midi », « minuit »), on garde.
    """
    texte = user_text or ""
    if (
        "time" in slots
        and _VAGUE_PERIOD.search(texte)
        and not _EXPLICIT_HOUR.search(texte)
    ):
        return {clef: valeur for clef, valeur in slots.items() if clef != "time"}
    return slots


# Fenêtre de réservation plausible. Au-delà, la valeur est jetée plutôt
# que corrigée : mieux vaut redemander que réserver n'importe quand.
_MAX_DAYS_AHEAD = 366


def _sane_date(day: date, today: date) -> date | None:
    """Corrige l'année hallucinée par le modèle.

    Constaté en conditions réelles : sur « demain » le LLM a renvoyé
    2023-07-22 — son année d'entraînement — alors qu'on était en 2026.
    Une date passée est réinterprétée comme sa prochaine occurrence ;
    une date absurdement lointaine est abandonnée.
    """
    if day >= today:
        return day if (day - today).days <= _MAX_DAYS_AHEAD else None

    for year in (today.year, today.year + 1):
        try:
            candidate = day.replace(year=year)
        except ValueError:  # 29 février d'une année non bissextile
            continue
        if candidate >= today:
            logger.info("année corrigée: %s -> %s", day, candidate)
            return candidate
    return None
