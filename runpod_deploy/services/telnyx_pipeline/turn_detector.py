"""Savoir quand le client parle, quand il a fini, et quand il coupe le bot.

Un détecteur volontairement simple : le niveau sonore de chaque paquet de 20 ms.
Au-dessus d'un seuil, ça parle ; après assez de silence, la phrase est finie.
Il coûte zéro milliseconde et zéro mémoire GPU sur une carte déjà pleine, et il se
règle à vue dans les journaux (chaque énoncé est tracé avec son niveau).

Ce module ne connaît que des octets et des compteurs : pas de réseau, pas de
modèle, pas d'horloge. C'est ce qui le rend testable — et c'est la pièce où se
sont cachés les bugs les plus coûteux de l'appel (phrase coupée, premier mot rogné).
"""

from __future__ import annotations

from dataclasses import dataclass

from .audio import rms

# ── Constantes, toutes issues d'appels réels ──────────────────────────────────
SPEECH_THRESHOLD = 800  # niveau au-dessus duquel « ça parle » (bruit de ligne + marge)
# 35 paquets de silence = 700 ms de silence = fin de phrase.
# 500 ms coupait 2 phrases sur 10 en plein mot (« si le restaurant est à l'al… »).
END_SILENCE_FRAMES = 35
MIN_SPEECH_FRAMES = 12  # 12 paquets de parole minimum = 240 ms, sinon c'est un bruit
BARGEIN_THRESHOLD = 2500  # parole soutenue, nettement au-dessus du bruit, pour couper le bot
BARGEIN_MIN_FRAMES = 12  # … pendant 240 ms d'affilée (un « mmh » ne coupe pas)
# 400 ms d'audio RÉEL gardées avant l'attaque : le seuil d'énergie rogne la première syllabe, sinon.
PREROLL_BYTES = 6400


@dataclass(frozen=True)
class Utterance:
    """Une phrase du client, prête pour la reconnaissance vocale."""

    pcm: bytes  # PCM16 8 kHz, pré-roll inclus
    speech_frames: int  # nombre de paquets jugés « parole »
    during_bot_speech: bool  # dite pendant que le bot parlait (à traiter après lui)


@dataclass(frozen=True)
class BargeIn:
    """Le client a parlé fort et longtemps pendant que le bot parlait : on le coupe."""

    rms: int


class TurnDetector:
    def __init__(
        self,
        *,
        speech_threshold: int = SPEECH_THRESHOLD,
        end_silence_frames: int = END_SILENCE_FRAMES,
        min_speech_frames: int = MIN_SPEECH_FRAMES,
        bargein_threshold: int = BARGEIN_THRESHOLD,
        bargein_min_frames: int = BARGEIN_MIN_FRAMES,
        preroll_bytes: int = PREROLL_BYTES,
    ) -> None:
        self._speech_threshold = speech_threshold
        self._end_silence_frames = end_silence_frames
        self._min_speech_frames = min_speech_frames
        self._bargein_threshold = bargein_threshold
        self._bargein_min_frames = bargein_min_frames
        self._preroll_bytes = preroll_bytes
        self._utterance = bytearray()
        self._preroll = bytearray()
        self._speech = 0
        self._silence = 0
        self._loud = 0

    def feed(self, pcm: bytes, *, bot_speaking: bool) -> list[Utterance | BargeIn]:
        """Un paquet de 20 ms entre ; zéro, un ou deux événements sortent."""
        events: list[Utterance | BargeIn] = []
        level = rms(pcm)

        # Barge-in : seulement pendant que le bot parle, et seulement une parole
        # soutenue et nette — pas un bruit de fond, pas un acquiescement.
        if bot_speaking and level > self._bargein_threshold:
            self._loud += 1
            if self._loud >= self._bargein_min_frames:
                events.append(BargeIn(rms=level))
                self._loud = 0
        else:
            self._loud = 0

        if level >= self._speech_threshold:
            if self._speech == 0:
                self._utterance += self._preroll  # attaque : coller le pré-roll
            self._utterance += pcm
            self._speech += 1
            self._silence = 0
        elif self._speech > 0:
            self._utterance += pcm
            self._silence += 1
            if self._silence >= self._end_silence_frames:
                events.extend(self._close(bot_speaking))

        self._preroll += pcm
        if len(self._preroll) > self._preroll_bytes:
            del self._preroll[: len(self._preroll) - self._preroll_bytes]
        return events

    def _close(self, bot_speaking: bool) -> list[Utterance]:
        pcm, frames = bytes(self._utterance), self._speech
        self._utterance = bytearray()
        self._speech = 0
        self._silence = 0
        if not bot_speaking and frames >= self._min_speech_frames:
            return [Utterance(pcm=pcm, speech_frames=frames, during_bot_speech=False)]
        if bot_speaking and frames >= 2 * self._min_speech_frames:
            # Le client a parlé par-dessus le bot sans le couper : avant, c'était jeté
            # (jusqu'à 4,4 s de parole perdues par appel). On le garde pour après.
            return [Utterance(pcm=pcm, speech_frames=frames, during_bot_speech=True)]
        return []
