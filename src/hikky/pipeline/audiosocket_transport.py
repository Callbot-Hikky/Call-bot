"""Transport Pipecat pour Asterisk AudioSocket.

Pipecat ne fournit pas ce transport : il connaît Twilio, LiveKit, Daily,
WebRTC, mais pas le protocole AudioSocket. On l'ajoute pour bénéficier de
tout ce que le framework apporte et que la boucle maison n'a jamais eu :

- **le barge-in** — écouter pendant que le bot parle et l'interrompre
  net dès que le client reprend la parole ;
- **un vrai détecteur de parole** (Silero, qui accepte le 8 kHz
  téléphonique) au lieu d'un seuil d'énergie ;
- **une détection de fin de tour** qui ne tranche plus au premier
  silence de 500 ms, au milieu d'une phrase.

Contraintes du protocole, apprises en production :

- Asterisk n'accepte **que des trames de 20 ms** (320 octets à 8 kHz).
  Un paquet plus gros est joué n'importe comment — la voix devient
  inintelligible.
- Le premier paquet reçu porte l'UUID de l'appel, qui sert de clé
  d'idempotence côté réservation.
"""

from __future__ import annotations

import asyncio
import logging
import struct
import time
from typing import Any
from uuid import UUID

from pipecat.frames.frames import InputAudioRawFrame, OutputAudioRawFrame
from pipecat.transports.base_input import BaseInputTransport
from pipecat.transports.base_output import BaseOutputTransport
from pipecat.transports.base_transport import BaseTransport, TransportParams

logger = logging.getLogger("hikky.audiosocket_transport")

# Types de paquets AudioSocket (cf. audiosocket_protocol.py).
_HANGUP, _UUID, _DTMF, _AUDIO, _ERREUR = 0x00, 0x01, 0x03, 0x10, 0xFF

# 20 ms à 8 kHz en 16 bits mono. Asterisk n'en accepte pas d'autre.
TRAME_20MS_OCTETS = 320
TRAME_20MS_SECONDES = 0.020

# Au-dela de ce retard, on considere qu'il y a eu un silence (ou une
# interruption) et on repart de l'instant present. Sans ce garde-fou, le
# transport « rattraperait » son retard en rafale a la reprise de parole,
# ce qui est precisement ce que le cadencement cherche a eviter.
DERIVE_MAX_SECONDES = 0.200


class AudioSocketTransportParams(TransportParams):
    """Paramètres du transport. Le 8 kHz est imposé par le protocole."""


class AudioSocketInputTransport(BaseInputTransport):
    """Lit les paquets Asterisk et les pousse dans la pipeline."""

    def __init__(
        self, reader: Any, params: TransportParams, parent: AudioSocketTransport, **kwargs
    ) -> None:
        super().__init__(params, **kwargs)
        self._reader = reader
        self._parent = parent
        self._tache: asyncio.Task | None = None
        self._actif = False

    async def start(self, frame: Any) -> None:
        await super().start(frame)
        self._actif = True
        self._tache = asyncio.create_task(self._boucle_lecture())

    async def stop(self, frame: Any) -> None:
        self._actif = False
        if self._tache is not None:
            self._tache.cancel()
            try:
                await self._tache
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
        await super().stop(frame)

    async def _boucle_lecture(self) -> None:
        try:
            await self.lire_paquets_pour_test()
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 — frontière réseau
            logger.exception("lecture AudioSocket interrompue")

    async def lire_paquets_pour_test(self) -> None:
        """Boucle de lecture, exposée pour être pilotée en test.

        Sans ce point d'entrée, tester la conversion paquet -> frame
        exigerait de démarrer toute la machinerie Pipecat.
        """
        while True:
            try:
                entete = await self._reader.readexactly(3)
            except (asyncio.IncompleteReadError, ConnectionError, OSError):
                logger.info("le pair a fermé la connexion")
                return

            type_paquet = entete[0]
            (longueur,) = struct.unpack(">H", entete[1:3])
            charge = await self._reader.readexactly(longueur) if longueur else b""

            if type_paquet == _HANGUP:
                logger.info("raccrochage demandé par Asterisk")
                return
            if type_paquet == _UUID and longueur == 16:
                self._parent.call_id = str(UUID(bytes=charge))
                logger.info("appel lié : %s", self._parent.call_id)
                continue
            if type_paquet == _DTMF:
                continue
            if type_paquet == _ERREUR:
                logger.warning("paquet d'erreur Asterisk")
                continue
            if type_paquet == _AUDIO and charge:
                frame = InputAudioRawFrame(
                    audio=charge,
                    sample_rate=self._params.audio_in_sample_rate or 8000,
                    num_channels=1,
                )
                resultat = self.push_audio_frame(frame)
                if asyncio.iscoroutine(resultat):
                    await resultat


class AudioSocketOutputTransport(BaseOutputTransport):
    """Découpe l'audio de la pipeline en trames de 20 ms pour Asterisk."""

    def __init__(self, writer: Any, params: TransportParams, **kwargs) -> None:
        super().__init__(params, **kwargs)
        self._writer = writer
        self._verrou = asyncio.Lock()
        self._reste = bytearray()
        # Instant auquel la prochaine trame doit partir. 0 = flux a l'arret.
        self._prochaine_echeance = 0.0

    async def write_audio_frame(self, frame: OutputAudioRawFrame) -> bool:
        """Émet l'audio en trames de 20 ms strictes.

        Un paquet plus gros est accepté par le protocole mais joué de
        travers par Asterisk : la voix devient inintelligible. Le reliquat
        est conservé entre deux appels pour ne pas insérer de silence
        artificiel au milieu d'une phrase.
        """
        async with self._verrou:
            self._reste.extend(frame.audio)
            paquets = bytearray()
            trames: list[bytes] = []
            while len(self._reste) >= TRAME_20MS_OCTETS:
                trame = bytes(self._reste[:TRAME_20MS_OCTETS])
                del self._reste[:TRAME_20MS_OCTETS]
                trames.append(struct.pack(">BH", _AUDIO, len(trame)) + trame)

            if not trames and self._reste:
                # Fin de parole : on complète la dernière trame plutôt que
                # de la garder indéfiniment en attente.
                trame = bytes(self._reste).ljust(TRAME_20MS_OCTETS, b"\x00")
                self._reste.clear()
                trames.append(struct.pack(">BH", _AUDIO, len(trame)) + trame)

            if not trames:
                return True

            # Cadencement temps reel : une trame toutes les 20 ms.
            #
            # Le TTS produit la parole bien plus vite que le temps reel. Tout
            # ecrire d'un bloc faisait expedier a Asterisk des centaines de
            # paquets RTP en quelques millisecondes ; le reseau ecretait ces
            # rafales et 10 % des paquets se perdaient, hachant la voix.
            #
            # L'echeance est absolue et monotone, jamais « maintenant + 20 ms » :
            # cette derniere forme ajoute le delai au temps DEJA ecoule et
            # accumule le retard de chaque iteration jusqu'a desynchroniser
            # l'audio de plusieurs centaines de millisecondes sur une phrase.
            try:
                for paquet in trames:
                    maintenant = time.monotonic()
                    if (
                        self._prochaine_echeance == 0.0
                        or maintenant - self._prochaine_echeance > DERIVE_MAX_SECONDES
                    ):
                        self._prochaine_echeance = maintenant
                    delai = self._prochaine_echeance - maintenant
                    if delai > 0:
                        await asyncio.sleep(delai)
                    self._writer.write(paquet)
                    self._prochaine_echeance += TRAME_20MS_SECONDES
                await self._writer.drain()
            except (ConnectionError, OSError):
                logger.info("flux coupé pendant la parole")
                return False
            return True

    async def vider_pour_interruption(self) -> None:
        """Jette l'audio en attente quand le client interrompt le bot.

        Sans ça le bot continuerait de prononcer la fin de sa phrase après
        avoir été coupé — le barge-in n'aurait aucun effet audible.
        """
        async with self._verrou:
            if self._reste:
                logger.info("interruption : %d octets abandonnés", len(self._reste))
            self._reste.clear()
            # Le flux reprendra a l'instant present : sans cette remise a zero,
            # la reprise de parole rattraperait en rafale le temps de silence.
            self._prochaine_echeance = 0.0


class AudioSocketTransport(BaseTransport):
    """Assemble l'entrée et la sortie autour d'une connexion Asterisk."""

    def __init__(
        self,
        *,
        reader: Any,
        writer: Any,
        params: TransportParams,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.params = params
        self.call_id: str | None = None
        self._input = AudioSocketInputTransport(reader, params, self, name=self._input_name)
        self._output = AudioSocketOutputTransport(writer, params, name=self._output_name)

    def input(self) -> AudioSocketInputTransport:
        return self._input

    def output(self) -> AudioSocketOutputTransport:
        return self._output
