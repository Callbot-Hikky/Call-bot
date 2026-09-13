"""Détecte que le client reprend la parole pendant que le bot parle.

Le problème central n'est pas de détecter la voix — c'est de ne pas
détecter **celle du bot**. La ligne téléphonique renvoie un écho de ce
qui est diffusé ; un détecteur naïf s'interrompt tout seul dès la
première syllabe et le bot ne finit plus aucune phrase.

Un VAD entraîné (Silero) ne résout pas ce point : l'écho *est* de la
parole humaine, correctement synthétisée. Il la classerait comme telle.
Ce qui distingue l'écho de la voix du client, c'est son niveau — il
revient atténué par la ligne. D'où une détection fondée sur l'énergie,
avec trois garde-fous :

1. un seuil nettement plus haut que celui de la fin de tour ;
2. plusieurs trames consécutives — un claquement ne dure pas ;
3. un délai de garde au début, le temps que le bot cesse de saturer
   la ligne.

L'audio capté pendant l'interruption appartient déjà au client : il est
conservé, sinon la transcription commencerait au milieu d'un mot.
"""

from __future__ import annotations

import logging

from hikky.adapters.telephony.audio_resample import frame_rms

logger = logging.getLogger("hikky.barge_in")

# Le seuil de fin de tour est à 500. L'écho du bot le dépasse largement :
# le réutiliser ici ferait couper le bot par sa propre voix.
SEUIL_INTERRUPTION_PAR_DEFAUT = 3000.0

# 3 trames = 60 ms de parole soutenue. En dessous, on attrape les bruits.
TRAMES_CONSECUTIVES_PAR_DEFAUT = 3

# 15 trames = 300 ms pendant lesquelles on ne cherche pas à détecter :
# la ligne est encore saturée par le début de la phrase du bot.
TRAMES_DE_GARDE_PAR_DEFAUT = 15


class DetecteurInterruption:
    def __init__(
        self,
        *,
        seuil_rms: float = SEUIL_INTERRUPTION_PAR_DEFAUT,
        trames_consecutives: int = TRAMES_CONSECUTIVES_PAR_DEFAUT,
        trames_de_garde: int = TRAMES_DE_GARDE_PAR_DEFAUT,
    ) -> None:
        self._seuil = seuil_rms
        self._requises = max(1, trames_consecutives)
        self._garde = max(0, trames_de_garde)
        self._vues = 0
        self._consecutives = 0
        self._capte = bytearray()

    def observer(self, trame: bytes) -> bool:
        """Rend True au moment précis où le client prend la parole."""
        self._vues += 1
        if self._vues <= self._garde:
            return False

        if frame_rms(trame) < self._seuil:
            # La série est rompue : ce n'était pas de la parole soutenue.
            self._consecutives = 0
            self._capte.clear()
            return False

        self._consecutives += 1
        self._capte.extend(trame)

        if self._consecutives < self._requises:
            return False

        logger.info(
            "le client reprend la parole (%d trames au-dessus de %.0f)",
            self._consecutives, self._seuil,
        )
        return True

    def audio_capte(self) -> bytes:
        """La parole du client déjà entendue, à joindre au tour suivant."""
        return bytes(self._capte)

    def reinitialiser(self) -> None:
        self._vues = 0
        self._consecutives = 0
        self._capte.clear()
