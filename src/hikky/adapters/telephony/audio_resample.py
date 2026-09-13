"""Rééchantillonnage PCM 16 bits pour la chaîne téléphonique.

Le trajet audio traverse trois taux : la voix synthétisée (16 à 44,1 kHz
selon le moteur), le 8 kHz imposé par AudioSocket, et le 16 kHz attendu
par Whisper. Chaque conversion doit être correcte, sous peine d'une voix
« robotique » qu'on attribuerait à tort au moteur de synthèse.

Méthode : interpolation par sinus cardinal fenêtré (Blackman). Le noyau
est comprimé du rapport de décimation, ce qui assure **en une seule
passe** le filtrage anti-repliement et l'interpolation. Une interpolation
linéaire, même précédée d'un filtre, laisse fuir des artefacts et atténue
le haut de la bande utile.

Le noyau est tabulé et mémorisé par couple de taux : sans ça, le coût des
appels à `sin()` rendrait la conversion trop lente pour du temps réel.
"""

from __future__ import annotations

import math
import struct
from functools import lru_cache

_INT16_MIN = -32768
_INT16_MAX = 32767

# Demi-largeur du noyau en périodes de sinc. 12 lobes suffisent pour une
# réjection très supérieure à ce que le 8 kHz téléphonique peut exiger.
_KERNEL_HALF_LOBES = 12

# Pas de tabulation du noyau. 512 points par lobe rendent l'erreur
# d'interpolation dans la table négligeable devant celle du signal.
_TABLE_STEPS_PER_LOBE = 512

# Marge sous la Nyquist cible : coupe à 92 % pour laisser une bande de
# transition au filtre sans rogner la voix (intelligibilité jusqu'à 3400 Hz).
_CUTOFF_MARGIN = 0.92


def _sinc(x: float) -> float:
    if x == 0.0:
        return 1.0
    pix = math.pi * x
    return math.sin(pix) / pix


@lru_cache(maxsize=32)
def _kernel_table() -> tuple[float, ...]:
    """Tabule sinc(x) * fenêtre de Blackman sur [0, _KERNEL_HALF_LOBES]."""
    size = _KERNEL_HALF_LOBES * _TABLE_STEPS_PER_LOBE + 1
    table = []
    for i in range(size):
        x = i / _TABLE_STEPS_PER_LOBE
        t = x / _KERNEL_HALF_LOBES
        window = 0.42 - 0.5 * math.cos(math.pi * (1.0 - t)) + 0.08 * math.cos(
            2.0 * math.pi * (1.0 - t)
        )
        table.append(_sinc(x) * window)
    return tuple(table)


def _kernel(x: float) -> float:
    ax = abs(x)
    if ax >= _KERNEL_HALF_LOBES:
        return 0.0
    table = _kernel_table()
    pos = ax * _TABLE_STEPS_PER_LOBE
    i = int(pos)
    frac = pos - i
    return table[i] + (table[i + 1] - table[i]) * frac


try:  # NumPy vit côté GPU (dépendance de faster-whisper), pas côté tests
    import numpy as _numpy
except ImportError:  # pragma: no cover
    _numpy = None


def resample_pcm16(data: bytes, src_rate: int, dst_rate: int) -> bytes:
    """Rééchantillonne, via NumPy si disponible.

    Le chemin Python pur coûtait 1,68 s par tour de parole (425 ms en
    entrée, 1253 ms en sortie) sur un budget total de 6,2 s avant que
    l'appelant n'entende quoi que ce soit. Le calcul est identique — même
    noyau, mêmes coefficients — seule la boucle est vectorisée.
    """
    if src_rate <= 0 or dst_rate <= 0:
        raise ValueError(f"sample rates must be positive, got {src_rate} -> {dst_rate}")
    if not data:
        return b""
    if src_rate == dst_rate:
        return data
    if _numpy is not None:
        return _resample_numpy(data, src_rate, dst_rate)
    return _resample_pure_python(data, src_rate, dst_rate)


def _resample_numpy(data: bytes, src_rate: int, dst_rate: int) -> bytes:
    np = _numpy
    samples = np.frombuffer(data, dtype="<i2").astype(np.float64)
    count = samples.size
    if count == 0:
        return b""

    ratio = dst_rate / src_rate
    out_count = max(1, round(count * ratio))
    scale = min(1.0, ratio) * _CUTOFF_MARGIN
    half_width = _KERNEL_HALF_LOBES / scale

    # Même pas que le chemin Python : `count / out_count`, et non `1/ratio`
    # — l'arrondi de `out_count` les rend légèrement différents.
    step = count / out_count
    centers = np.arange(out_count, dtype=np.float64) * step

    span = int(math.ceil(half_width)) + 1
    offsets = np.arange(-span, span + 1, dtype=np.int64)
    idx = np.floor(centers).astype(np.int64)[:, None] + offsets[None, :]

    # Les indices hors bornes sont ANNULÉS, pas rabattus : le chemin
    # Python les ignore, les rabattre dupliquerait les échantillons de
    # bord et fausserait la normalisation.
    valid = (idx >= 0) & (idx < count)
    weights = np.where(valid, _kernel_numpy((centers[:, None] - idx) * scale), 0.0)
    safe_idx = np.clip(idx, 0, count - 1)

    norm = weights.sum(axis=1)
    norm[norm == 0.0] = 1.0
    values = (samples[safe_idx] * weights).sum(axis=1) / norm

    out = np.clip(np.rint(values), _INT16_MIN, _INT16_MAX).astype("<i2")
    return out.tobytes()


def _kernel_numpy(x):
    np = _numpy
    ax = np.abs(x)
    table = np.asarray(_kernel_table(), dtype=np.float64)
    pos = ax * _TABLE_STEPS_PER_LOBE
    i = np.minimum(pos.astype(np.int64), table.size - 2)
    frac = pos - i
    out = table[i] + (table[i + 1] - table[i]) * frac
    return np.where(ax >= _KERNEL_HALF_LOBES, 0.0, out)


def _resample_pure_python(data: bytes, src_rate: int, dst_rate: int) -> bytes:
    if not data:
        return b""
    if src_rate == dst_rate:
        return data

    count = len(data) // 2
    if count == 0:
        return b""

    samples = struct.unpack(f"<{count}h", data[: count * 2])
    ratio = dst_rate / src_rate
    out_count = max(1, round(count * ratio))

    # En décimation, le noyau est comprimé pour couper sous la Nyquist
    # cible. En interpolation, il reste à l'échelle de la source.
    scale = min(1.0, ratio) * _CUTOFF_MARGIN
    half_width = _KERNEL_HALF_LOBES / scale
    step = 1.0 / ratio

    out = []
    for i in range(out_count):
        center = i * step
        first = int(math.ceil(center - half_width))
        last = int(math.floor(center + half_width))
        if first < 0:
            first = 0
        if last >= count:
            last = count - 1

        acc = 0.0
        norm = 0.0
        for j in range(first, last + 1):
            w = _kernel((center - j) * scale)
            if w != 0.0:
                acc += samples[j] * w
                norm += w

        value = acc / norm if norm != 0.0 else 0.0
        out.append(max(_INT16_MIN, min(_INT16_MAX, int(round(value)))))

    return struct.pack(f"<{out_count}h", *out)


def frame_rms(frame: bytes) -> float:
    count = len(frame) // 2
    if count == 0:
        return 0.0
    samples = struct.unpack(f"<{count}h", frame[: count * 2])
    return math.sqrt(sum(s * s for s in samples) / count)


class StreamingResampler:
    """Rééchantillonne au fil de l'eau, sans discontinuité aux raccords.

    XTTS livre son premier morceau en 364 ms mais met plus de 4 s pour une
    phrase entière. Attendre la totalité avant de rééchantillonner détruit
    tout l'intérêt du flux — c'est ce que faisait `_speak`.

    La difficulté : le noyau du filtre a besoin de contexte de part et
    d'autre de chaque échantillon. Rééchantillonner un morceau isolément
    lui retire ce contexte aux bords et produit un clic audible. On
    conserve donc une marge de recouvrement entre deux appels, et on ne
    livre que la partie dont le voisinage est complet.
    """

    def __init__(self, src_rate: int, dst_rate: int) -> None:
        if src_rate <= 0 or dst_rate <= 0:
            raise ValueError(f"taux invalides : {src_rate} -> {dst_rate}")
        self._src = src_rate
        self._dst = dst_rate
        self._passthrough = src_rate == dst_rate
        ratio = dst_rate / src_rate
        scale = min(1.0, ratio) * _CUTOFF_MARGIN
        # Contexte requis par le noyau, exprimé en échantillons source.
        self._margin = int(math.ceil(_KERNEL_HALF_LOBES / scale)) + 2
        self._tail = b""          # échantillons gardés pour le contexte
        self._emitted = 0         # échantillons de sortie déjà livrés
        self._consumed = 0        # échantillons source déjà retirés du flux

    def push(self, chunk: bytes) -> bytes:
        """Absorbe un morceau et rend ce qui peut être diffusé sans risque."""
        if self._passthrough:
            return chunk
        if not chunk:
            return b""

        buffer = self._tail + chunk
        available = len(buffer) // 2
        if available <= 2 * self._margin:
            self._tail = buffer
            return b""

        # On ne rééchantillonne que la portion dont le voisinage droit est
        # connu ; la marge finale attend le morceau suivant.
        usable = available - self._margin
        out = self._render(buffer, usable)
        keep_from = (usable - self._margin) * 2
        self._tail = buffer[keep_from:]
        self._consumed += usable - self._margin
        return out

    def flush(self) -> bytes:
        """Livre le reliquat une fois le flux terminé."""
        if self._passthrough:
            return b""
        if not self._tail:
            return b""
        out = self._render(self._tail, len(self._tail) // 2)
        self._tail = b""
        return out

    def _render(self, buffer: bytes, sample_count: int) -> bytes:
        """Rééchantillonne `buffer` et ne rend que la part encore due."""
        rendered = resample_pcm16(buffer[: sample_count * 2], self._src, self._dst)
        total = len(rendered) // 2
        # Position absolue du début de ce tampon dans le flux source.
        start_ratio = self._consumed * self._dst / self._src
        already = max(0, self._emitted - int(round(start_ratio)))
        if already >= total:
            return b""
        out = rendered[already * 2 :]
        self._emitted += total - already
        return out
