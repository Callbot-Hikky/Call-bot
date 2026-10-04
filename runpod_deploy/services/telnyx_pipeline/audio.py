"""Conversions audio, et rien d'autre.

Ce que la ligne envoie : des octets G.711 — A-law (Europe) ou µ-law (États-Unis) —,
8 000 par seconde, livrés par paquets de 20 ms. Ce que nos modèles veulent : du PCM
16 bits — 16 kHz pour la reconnaissance vocale, 24 kHz en sortie de la synthèse.
Ce qu'on renvoie à Telnyx : le même codec que la ligne, A-law 8 kHz (demandé dans le
TeXML, `bidirectionalCodec="PCMA"`), en trames de 160 octets (= 20 ms).

Toutes les fonctions sont pures : des octets entrent, des octets sortent.
"""

from __future__ import annotations

import audioop  # fourni par audioop-lts (backport officiel, Python >= 3.13) : codec G.711
import os
import wave
from collections.abc import Iterator

import numpy as np
import soxr  # rééchantillonnage de qualité (polyphase). Dépendance obligatoire, installée par

# pod_setup.sh : absente = erreur de déploiement, on échoue au démarrage, pas en appel.

TELEPHONY_RATE = 8000  # cadence de la ligne (échantillons par seconde)
STT_RATE = 16000  # ce que la reconnaissance vocale attend
TTS_RATE = 24000  # ce que la synthèse produit
FRAME_MS = 20  # durée d'un paquet Telnyx
G711_FRAME_BYTES = (
    TELEPHONY_RATE * FRAME_MS // 1000
)  # 160 octets G.711 (1 octet/échantillon) = 20 ms
PCM_FRAME_BYTES = G711_FRAME_BYTES * 2  # 320 octets PCM16 = 20 ms


def decode_telephony(raw: bytes, *, alaw: bool) -> bytes:
    """Octets de la ligne -> PCM16 8 kHz. Le codec est lu dans l'événement `start`
    de Telnyx : la ligne européenne envoie du PCMA (A-law), pas du PCMU."""
    return audioop.alaw2lin(raw, 2) if alaw else audioop.ulaw2lin(raw, 2)


def rms(pcm: bytes) -> int:
    """Niveau sonore moyen d'un bloc PCM16 (c'est tout notre détecteur de parole)."""
    return audioop.rms(pcm, 2) if pcm else 0


def peak(pcm: bytes) -> int:
    return audioop.max(pcm, 2) if pcm else 0


def to_stt_rate(pcm8k: bytes) -> bytes:
    """8 kHz téléphone -> 16 kHz pour la reconnaissance vocale."""
    x = np.frombuffer(pcm8k, dtype="<i2")
    return soxr.resample(x, TELEPHONY_RATE, STT_RATE, quality="HQ").astype("<i2").tobytes()


def split_into_frames(g711: bytes) -> list[bytes]:
    """Découpe une phrase A-law en paquets de 160 octets (= 20 ms), l'unité de Telnyx.

    Pour 1 s de voix : len(g711) = 8 000 -> 50 paquets. Le dernier peut être plus
    court si la phrase ne tombe pas sur un multiple de 20 ms ; Telnyx l'accepte.
    """
    frames = []
    for start in range(0, len(g711), G711_FRAME_BYTES):  # 0, 160, 320, ... : début de chaque paquet
        end = start + G711_FRAME_BYTES
        frames.append(g711[start:end])  # copie les octets de `start` (inclus) à `end` (exclu)
    return frames


def pcm_to_telephony_frames(pcm: bytes, rate: int) -> list[bytes]:
    """PCM16 (à `rate` Hz) -> trames A-law de 20 ms, pour une phrase complète.

    Une phrase entière est un flux à un seul morceau : même code que le flux,
    utilisé pour la salutation mise en cache et le repli quand le flux TTS échoue.
    """
    return StreamingDownsampler(rate).feed(pcm)


class StreamingDownsampler:
    """Même conversion que `pcm_to_telephony_frames`, mais morceau par morceau.

    Le rééchantillonneur garde son état entre deux morceaux : sans ça, chaque
    raccord fait un clic audible au téléphone.
    """

    def __init__(self, src_rate: int) -> None:
        self._src_rate = src_rate
        self._state = None

    def feed(self, pcm: bytes) -> list[bytes]:
        """Un morceau PCM entre, des paquets A-law de 20 ms sortent."""
        # 1. redimensionner (en reprenant où le morceau précédent s'est arrêté)
        pcm8, self._state = audioop.ratecv(pcm, 2, 1, self._src_rate, TELEPHONY_RATE, self._state)
        g711 = audioop.lin2alaw(pcm8, 2)  # 2. conversion en A-law
        return split_into_frames(g711)  # 3. découper


def pop_length_prefixed(buf: bytearray) -> Iterator[bytes]:
    """Lit les morceaux complets d'un flux « taille sur 4 octets, puis données ».

    C'est le format du flux TTS entre nos deux services : zéro décodage, un entier
    puis du PCM brut. Les morceaux rendus sont retirés de `buf` ; un morceau
    incomplet y reste en attente de la suite.
    """
    while len(buf) >= 4:
        size = int.from_bytes(buf[:4], "big")
        if len(buf) < 4 + size:
            return
        chunk = bytes(buf[4 : 4 + size])
        del buf[: 4 + size]
        yield chunk


def write_wav(path: str, pcm8k: bytes) -> str:
    """Enregistre un énoncé en WAV 8 kHz pour l'écoute hors ligne (débogage).
    C'est ce qui a permis de trancher « STT sourd » contre « VAD qui coupe »."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(TELEPHONY_RATE)
        w.writeframes(pcm8k)
    return path
