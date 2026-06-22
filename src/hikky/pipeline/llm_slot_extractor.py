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
from datetime import datetime
from typing import Any

from hikky.pipeline.dialogue_processor import SlotExtractor
from hikky.ports.language_model import LanguageModelPort

logger = logging.getLogger("hikky.slot_extractor")

SYSTEM_PROMPT = """Tu es un extracteur de slots pour un assistant vocal de \
réservation de restaurant.

Étant donnée la dernière phrase de l'utilisateur (en français), extrais \
UNIQUEMENT les informations clairement présentes parmi :

- "date_time" : date et heure de la réservation, au format ISO 8601 \
"YYYY-MM-DDTHH:MM" (24h). Résous les expressions relatives ("demain", \
"ce soir", "vendredi") par rapport à la date du jour.
- "party_size" : nombre de personnes, entier strictement positif.
- "customer_name" : nom du client (chaîne).

Date du jour : {today_iso} ({weekday_name}).

Réponds par UN SEUL objet JSON minifié contenant uniquement les clés \
des slots détectés (omets les autres). Si rien n'est extrait, réponds \
exactement `{{}}`. Aucune explication, aucun markdown, aucun texte hors \
du JSON.

Exemples :
- "Je voudrais réserver" → {{}}
- "Demain à 20 heures pour 4" → {{"date_time": "{tomorrow_iso}T20:00", \
"party_size": 4}}
- "Au nom de Dupont" → {{"customer_name": "Dupont"}}
- "On serait 6" → {{"party_size": 6}}
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

        return self._parse(raw)

    def _parse(self, raw: str) -> dict[str, Any]:
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

        dt_raw = parsed.get("date_time")
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
