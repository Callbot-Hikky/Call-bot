import struct

import pytest

from hikky.adapters.telephony.audio_resample import frame_rms, resample_pcm16


def _pcm(samples: list[int]) -> bytes:
    return struct.pack(f"<{len(samples)}h", *samples)


def _unpack(data: bytes) -> list[int]:
    return list(struct.unpack(f"<{len(data) // 2}h", data))


def test_resample_same_rate_is_identity():
    data = _pcm([0, 100, -100, 32767, -32768])
    assert resample_pcm16(data, 8000, 8000) == data


def test_resample_empty_input_returns_empty():
    assert resample_pcm16(b"", 8000, 16000) == b""


def test_upsampling_doubles_sample_count():
    data = _pcm([0] * 160)
    out = resample_pcm16(data, 8000, 16000)
    assert len(_unpack(out)) == 320


def test_downsampling_from_piper_rate_to_telephony_rate():
    data = _pcm([0] * 22050)
    out = resample_pcm16(data, 22050, 8000)
    assert len(_unpack(out)) == 8000


def test_resample_preserves_constant_signal():
    data = _pcm([1000] * 100)
    out = _unpack(resample_pcm16(data, 8000, 16000))
    assert all(abs(s - 1000) <= 1 for s in out)


def test_resample_clamps_to_int16_range():
    data = _pcm([32767] * 50)
    out = _unpack(resample_pcm16(data, 8000, 16000))
    assert all(-32768 <= s <= 32767 for s in out)


def test_resample_rejects_non_positive_rate():
    with pytest.raises(ValueError):
        resample_pcm16(_pcm([0]), 0, 8000)


def test_frame_rms_of_silence_is_zero():
    assert frame_rms(_pcm([0] * 160)) == 0.0


def test_frame_rms_of_loud_signal_is_high():
    assert frame_rms(_pcm([10000] * 160)) == pytest.approx(10000, rel=0.01)


def test_frame_rms_of_empty_frame_is_zero():
    assert frame_rms(b"") == 0.0


def _tone(freq: float, rate: int, seconds: float, amp: int = 8000) -> bytes:
    import math

    n = int(rate * seconds)
    return struct.pack(
        f"<{n}h",
        *[int(amp * math.sin(2 * math.pi * freq * i / rate)) for i in range(n)],
    )


def test_downsampling_attenuates_frequencies_above_target_nyquist():
    """Un 6 kHz descendu en 8 kHz se replierait en 2 kHz sans filtre.

    C'est la cause classique d'une voix « métallique » : tout le contenu
    au-dessus de 4 kHz revient en distorsion audible dans les graves.
    """
    tone = _tone(6000, 22050, 0.25)
    out = resample_pcm16(tone, 22050, 8000)
    assert frame_rms(out) < frame_rms(tone) * 0.25


def test_downsampling_preserves_speech_band_frequencies():
    tone = _tone(500, 22050, 0.25)
    out = resample_pcm16(tone, 22050, 8000)
    assert frame_rms(out) > frame_rms(tone) * 0.7


def test_upsampling_does_not_attenuate_signal():
    tone = _tone(500, 8000, 0.25)
    out = resample_pcm16(tone, 8000, 16000)
    assert frame_rms(out) > frame_rms(tone) * 0.9


def _rms_of_tone_after_resample(freq, src, dst, seconds=0.3):
    tone = _tone(freq, src, seconds)
    return frame_rms(resample_pcm16(tone, src, dst)) / max(frame_rms(tone), 1e-9)


def test_passband_is_flat_across_the_speech_range():
    """L'interpolation lineaire attenue le haut de la bande utile.

    La voix humaine porte l'essentiel de son intelligibilite entre 300 Hz
    et 3400 Hz (bande telephonique). Un 3000 Hz ne doit pas ressortir
    nettement plus faible qu'un 300 Hz, sinon la voix parait etouffee.
    """
    low = _rms_of_tone_after_resample(300, 24000, 8000)
    high = _rms_of_tone_after_resample(3000, 24000, 8000)
    assert high > low * 0.7, f"droop excessif: 300Hz={low:.3f} 3000Hz={high:.3f}"


def test_stopband_is_strongly_attenuated():
    ratio = _rms_of_tone_after_resample(6000, 24000, 8000)
    assert ratio < 0.05, f"repliement residuel: {ratio:.3f}"


def test_integer_ratio_downsampling_is_clean():
    ratio = _rms_of_tone_after_resample(1000, 24000, 8000)
    assert ratio > 0.85


# ── Chemin rapide NumPy ─────────────────────────────────────────────────
#
# Mesure en production : 425 ms pour le 8k->16k d'entree, 1253 ms pour le
# 24k->8k de sortie. 1,68 s par tour de parole, sur un budget total de
# 6,2 s. Le noyau evalue ~78 coefficients par echantillon de sortie, en
# Python pur. NumPy est deja installe cote GPU.
#
# Exigence absolue : sortie IDENTIQUE au Python pur, dont la qualite
# sonore a ete validee a l'oreille.

def test_numpy_and_python_paths_agree_on_downsampling():
    from hikky.adapters.telephony import audio_resample as ar

    tone = _tone(440, 24000, 0.25) + _tone(1800, 24000, 0.25)
    rapide = ar.resample_pcm16(tone, 24000, 8000)
    lent = ar._resample_pure_python(tone, 24000, 8000)
    assert len(rapide) == len(lent)
    a = _unpack(rapide)
    b = _unpack(lent)
    ecart = max(abs(x - y) for x, y in zip(a, b, strict=True))
    assert ecart <= 1, f"les deux chemins divergent de {ecart}"


def test_numpy_and_python_paths_agree_on_upsampling():
    from hikky.adapters.telephony import audio_resample as ar

    tone = _tone(600, 8000, 0.25)
    a = _unpack(ar.resample_pcm16(tone, 8000, 16000))
    b = _unpack(ar._resample_pure_python(tone, 8000, 16000))
    assert max(abs(x - y) for x, y in zip(a, b, strict=True)) <= 1


def test_pure_python_fallback_when_numpy_is_absent(monkeypatch):
    """Le socle du projet ne depend pas de NumPy : la suite de tests doit
    tourner sans lui."""
    from hikky.adapters.telephony import audio_resample as ar

    monkeypatch.setattr(ar, "_numpy", None)
    out = ar.resample_pcm16(_tone(440, 24000, 0.1), 24000, 8000)
    assert len(out) > 0


def test_downsampling_quality_is_preserved_by_the_fast_path():
    """Les garanties acoustiques validees a l'oreille doivent tenir."""
    assert _rms_of_tone_after_resample(300, 24000, 8000) > 0.9
    assert _rms_of_tone_after_resample(3000, 24000, 8000) > 0.9
    assert _rms_of_tone_after_resample(6000, 24000, 8000) < 0.05


# ── Reechantillonnage en flux ───────────────────────────────────────────
#
# XTTS livre son premier morceau en 364 ms mais met 4152 ms pour une
# phrase entiere. `_speak` attendait TOUT avant de rechantillonner et
# d'emettre : le benefice du flux etait integralement detruit au dernier
# maillon. Pour diffuser au fil de l'eau il faut rechantillonner morceau
# par morceau — sans recreer de discontinuites aux raccords, ce que le
# filtre provoque s'il perd son contexte.

def test_streaming_resampler_introduces_no_audible_click():
    """Ce qui compte n'est pas l'egalite octet a octet mais l'absence de
    discontinuite : un decalage d'un echantillon (0,125 ms) est inaudible,
    un saut d'amplitude s'entend comme un clic.

    On compare donc le saut maximal entre echantillons consecutifs du flux
    a celui du meme signal traite en bloc.
    """
    from hikky.adapters.telephony.audio_resample import StreamingResampler

    source = _tone(440, 24000, 0.2) + _tone(1500, 24000, 0.2)
    bloc = _unpack(resample_pcm16(source, 24000, 8000))

    rs = StreamingResampler(24000, 8000)
    morceaux = [source[i : i + 4800] for i in range(0, len(source), 4800)]
    flux = _unpack(b"".join(rs.push(m) for m in morceaux) + rs.flush())

    assert abs(len(flux) - len(bloc)) <= 4, (
        f"{len(flux)} echantillons en flux contre {len(bloc)} en bloc"
    )

    saut_bloc = max(abs(b - a) for a, b in zip(bloc, bloc[1:], strict=False))
    saut_flux = max(abs(b - a) for a, b in zip(flux, flux[1:], strict=False))
    assert saut_flux <= saut_bloc * 1.2, (
        f"clic aux raccords : saut {saut_flux} en flux contre {saut_bloc} en bloc"
    )


def test_streaming_resampler_preserves_signal_energy():
    from hikky.adapters.telephony.audio_resample import StreamingResampler

    source = _tone(800, 24000, 0.3)
    bloc = frame_rms(resample_pcm16(source, 24000, 8000))
    rs = StreamingResampler(24000, 8000)
    flux = frame_rms(
        b"".join(rs.push(source[i : i + 4800]) for i in range(0, len(source), 4800))
        + rs.flush()
    )
    assert abs(flux - bloc) < bloc * 0.05, f"energie {flux:.0f} contre {bloc:.0f}"


def test_streaming_resampler_emits_before_the_end():
    from hikky.adapters.telephony.audio_resample import StreamingResampler

    rs = StreamingResampler(24000, 8000)
    premier = rs.push(_tone(440, 24000, 0.2))
    assert len(premier) > 0, "rien emis avant la fin : le flux ne sert a rien"


def test_streaming_resampler_handles_tiny_chunks():
    from hikky.adapters.telephony.audio_resample import StreamingResampler

    rs = StreamingResampler(24000, 8000)
    source = _tone(440, 24000, 0.1)
    sortie = b"".join(rs.push(source[i : i + 96]) for i in range(0, len(source), 96))
    sortie += rs.flush()
    assert len(sortie) > 0


def test_streaming_resampler_is_a_passthrough_at_equal_rates():
    from hikky.adapters.telephony.audio_resample import StreamingResampler

    rs = StreamingResampler(8000, 8000)
    data = _pcm([100, 200, 300])
    assert rs.push(data) + rs.flush() == data
