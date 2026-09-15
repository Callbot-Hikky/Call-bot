"""Découpage des tours de parole par détection d'activité vocale (VAD).

Le serveur AudioSocket décidait de la fin d'un tour par un simple seuil
d'énergie (RMS) sur chaque trame. Sur une ligne téléphonique 8 kHz bruitée
c'est fragile : un mot prononcé doucement passe sous le seuil et la phrase
est coupée en plein milieu (« les, les, les »), tandis qu'un bruit de fond
régulier reste au-dessus du seuil et deux phrases se retrouvent collées.

Un VAD (`webrtcvad`, conçu pour la téléphonie) classe chaque trame de 20 ms
en parole / non-parole à partir de caractéristiques spectrales, bien plus
robustes au niveau de bruit qu'une énergie brute. Cet endpointer encapsule la
machine à états qui vivait, éparse, dans la boucle du serveur :

  - on ignore le silence de tête ;
  - on démarre l'énoncé à la première trame de parole, en incluant un court
    « préroll » (les dernières trames de silence) pour ne pas rogner le
    premier phonème ;
  - un silence bref (< hangover) ne coupe PAS l'énoncé ;
  - on clôt après `hangover_frames` trames non-parole consécutives, ou à
    `max_utterance_frames` (garde-fou anti-monologue) ;
  - un énoncé plus court que `min_speech_frames` est jeté (bruit isolé).

Le backend VAD est injectable : les tests tournent sans `webrtcvad`, et on
peut brancher un autre détecteur sans toucher à la machine à états.
"""

from __future__ import annotations

import logging
from collections import deque
from collections.abc import Callable

logger = logging.getLogger("hikky.vad")

# Type du backend : (trame_pcm16, sample_rate) -> parole ?
VadBackend = Callable[[bytes, int], bool]


def _webrtc_backend(aggressiveness: int) -> VadBackend:
    """Backend par défaut : webrtcvad. Import paresseux (comme faster-whisper)
    pour que la suite de tests n'exige pas la librairie."""
    import webrtcvad  # lazy

    vad = webrtcvad.Vad(aggressiveness)

    def _is_speech(frame: bytes, sample_rate: int) -> bool:
        try:
            return vad.is_speech(frame, sample_rate)
        except Exception:  # noqa: BLE001 — trame de taille inattendue, etc.
            return False

    return _is_speech


class VadEndpointer:
    def __init__(
        self,
        *,
        frame_bytes: int = 320,
        sample_rate: int = 8000,
        min_speech_frames: int = 8,
        hangover_frames: int = 20,
        start_padding_frames: int = 5,
        max_utterance_frames: int = 750,
        aggressiveness: int = 2,
        vad: VadBackend | None = None,
    ) -> None:
        self._frame_bytes = frame_bytes
        self._sample_rate = sample_rate
        self._min_speech = min_speech_frames
        self._hangover = hangover_frames
        self._padding = start_padding_frames
        self._max_frames = max_utterance_frames
        self._vad = vad or _webrtc_backend(aggressiveness)

        self._preroll: deque[bytes] = deque(maxlen=max(0, start_padding_frames))
        self._utterance = bytearray()
        self._frames = 0
        self._speech_frames = 0
        self._silence_run = 0
        self._started = False

    def _reset(self) -> None:
        self._preroll.clear()
        self._utterance.clear()
        self._frames = 0
        self._speech_frames = 0
        self._silence_run = 0
        self._started = False

    def seed(self, pcm: bytes) -> None:
        """Amorce l'énoncé avec de l'audio déjà entendu (reprise après une
        interruption). L'énoncé est considéré comme démarré."""
        if not pcm:
            return
        self._started = True
        self._utterance.extend(pcm)
        frames = max(1, len(pcm) // self._frame_bytes)
        self._frames += frames
        self._speech_frames += frames
        self._silence_run = 0

    def _is_speech(self, frame: bytes) -> bool:
        # webrtcvad exige une trame d'exactement 10/20/30 ms ; toute trame de
        # taille non standard (dernier paquet tronqué) est traitée en silence.
        if len(frame) != self._frame_bytes:
            return False
        return self._vad(frame, self._sample_rate)

    def feed(self, frame: bytes) -> bytes | None:
        """Consomme une trame de 20 ms. Renvoie l'énoncé complet (PCM) quand un
        point de coupure est détecté, sinon None."""
        speech = self._is_speech(frame)

        if not self._started:
            # On mémorise les dernières trames de silence pour les rejouer en
            # tête d'énoncé (préroll) et ne pas rogner l'attaque.
            self._preroll.append(frame)
            if not speech:
                return None
            # Attaque : on démarre en incluant le préroll.
            for pre in self._preroll:
                self._utterance.extend(pre)
            self._frames = len(self._preroll)
            self._preroll.clear()
            self._started = True
            self._speech_frames = 1
            self._silence_run = 0
            return self._maybe_end_on_max()

        # Énoncé en cours.
        self._utterance.extend(frame)
        self._frames += 1
        if speech:
            self._speech_frames += 1
            self._silence_run = 0
        else:
            self._silence_run += 1
            if self._silence_run >= self._hangover:
                return self._finish()
        return self._maybe_end_on_max()

    def _maybe_end_on_max(self) -> bytes | None:
        if self._frames >= self._max_frames:
            logger.info("énoncé coupé à la longueur max (%d trames)", self._frames)
            return self._finish()
        return None

    def _finish(self) -> bytes | None:
        # Un énoncé trop court, c'est du bruit isolé : on le jette plutôt que de
        # faire halluciner Whisper sur 100 ms de souffle.
        assez = self._speech_frames >= self._min_speech
        pcm = bytes(self._utterance) if assez else None
        self._reset()
        return pcm
