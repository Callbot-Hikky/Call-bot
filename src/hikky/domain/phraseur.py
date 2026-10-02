"""Rend la parole au modèle, sans lui rendre l'état.

L'aiguilleur (`question_router`) a corrigé un vrai défaut : le modèle
perdait le fil de la réservation et redemandait trois fois la même
chose. Mais le remède lui a retiré la parole entièrement — chaque phrase
du bot est devenue une chaîne figée piochée dans trois variantes. Sur un
appel réel, le modèle n'était sollicité pour parler aucune fois. D'où
l'impression de parler à un serveur vocal.

Le partage retenu :

    le code   : quel slot manque, quoi confirmer, quoi enregistrer
                — l'état, qu'il ne doit jamais lâcher
    le modèle : la formulation, qui tient compte de ce que le client
                vient de dire

Deux garde-fous, tirés d'échecs observés :

- **le repli** : au téléphone, une phrase un peu raide vaut mieux que
  trois secondes de silence. Toute défaillance retombe sur la phrase
  figée, qui reste correcte.
- **les intentions engageantes** (récapitulatif, confirmation de
  réservation) ne passent pas par le modèle. Il a déjà annoncé des
  réservations qui n'existaient pas ; ces phrases-là engagent le
  restaurant et restent écrites par le code.
"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger("hikky.phraseur")

DEFAULT_TIMEOUT_SECONDS = 1.2

# Au-delà, le modèle est parti en tirade — insupportable au téléphone.
MAX_CARACTERES = 180

# Ces intentions engagent le restaurant : leur formulation reste au code.
INTENTIONS_ENGAGEANTES = frozenset({"recapituler", "confirmer_reservation"})

PROMPT = """Tu es l'hôte du restaurant {restaurant}, au téléphone.

Le client vient de dire : « {client} »

Tu dois maintenant obtenir cette information : {besoin}

Réponds en UNE phrase courte et naturelle, en français, à l'oral.
Accuse brièvement réception de ce que le client a dit, puis pose la
question. N'invente aucune information. Ne récapitule pas.
Ta phrase DOIT être une question qui demande cette information. Ne
répète pas et ne confirme JAMAIS une valeur pour cette information
(pas de « je confirme », « c'est bien », « c'est noté ») : si tu la
demandes, c'est que tu ne l'as pas.

Ta phrase :"""


# Vocabulaire d'affirmation : une phrase censée DEMANDER un slot ne peut
# pas affirmer ou confirmer une valeur. Observé en appel réel : le modèle a
# « accusé réception » d'une heure que le système n'avait pas retenue, en
# sortant « Je confirme vingt et une heures ? » — une fausse confirmation.
_AFFIRMATION = re.compile(
    r"\b(je confirme|c'est bien|est confirm\w*|confirm[ée]e?s?\b|"
    r"c'est not[ée]|bien not[ée]|est not[ée]|j'ai not[ée]|"
    r"est r[ée]serv[ée]|c'est r[ée]serv[ée]|enregistr[ée]e?s?)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class Intention:
    """Ce que le code a décidé de dire, avant sa mise en mots.

    `repli` est la phrase figée : elle est toujours correcte, et sert dès
    que le modèle échoue, tarde ou dérape.
    """

    besoin: str
    slot: str | None
    repli: str
    derniere_phrase_client: str = ""


class Phraseur:
    def __init__(
        self,
        model: Any,
        *,
        restaurant: str = "Le Bistrot du Coin",
        timeout_secondes: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        self._model = model
        self._restaurant = restaurant
        self._timeout = timeout_secondes

    async def formuler(self, intention: Intention) -> str:
        if intention.besoin in INTENTIONS_ENGAGEANTES:
            # Le modèle n'est même pas consulté : ces phrases engagent.
            return intention.repli

        prompt = PROMPT.format(
            restaurant=self._restaurant,
            client=intention.derniere_phrase_client or "(rien)",
            besoin=intention.slot or intention.besoin,
        )

        try:
            brut = await asyncio.wait_for(
                self._model.complete([{"role": "user", "content": prompt}]),
                timeout=self._timeout,
            )
        except TimeoutError:
            logger.info("modèle trop lent — repli sur la phrase figée")
            return intention.repli
        except Exception:  # noqa: BLE001 — aucune panne du modèle ne coupe l'appel
            logger.warning("modèle indisponible — repli", exc_info=True)
            return intention.repli

        return self._retenir_ou_replier(brut, intention.repli)

    def _retenir_ou_replier(self, brut: Any, repli: str) -> str:
        if not isinstance(brut, str):
            return repli
        phrase = brut.strip().strip('"').strip()
        if not phrase:
            return repli
        if len(phrase) > MAX_CARACTERES:
            logger.info("formulation trop longue (%d car.) — repli", len(phrase))
            return repli
        # Une demande de slot est une question : sans « ? » ou avec un
        # vocabulaire d'affirmation, le modèle a dérapé (fausse confirmation).
        # La phrase figée, elle, est toujours une vraie question correcte.
        if not phrase.rstrip().endswith("?") or _AFFIRMATION.search(phrase):
            logger.info("formulation affirme/confirme au lieu de demander — repli: %r", phrase)
            return repli
        return phrase
