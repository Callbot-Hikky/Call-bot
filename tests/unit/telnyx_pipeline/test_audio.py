"""Conversions audio : codecs téléphoniques, cadences, découpe en trames."""

import struct

from telnyx_pipeline import audio


def _pcm(n_samples: int, level: int = 1000) -> bytes:
    return struct.pack("<" + "h" * n_samples, *([level, -level] * (n_samples // 2)))


def test_decodage_a_law_et_mu_law_rendent_du_pcm16():
    raw = bytes(range(160))
    assert len(audio.decode_telephony(raw, alaw=True)) == 320
    assert len(audio.decode_telephony(raw, alaw=False)) == 320
    assert audio.decode_telephony(raw, alaw=True) != audio.decode_telephony(raw, alaw=False)


def test_vers_16k_double_la_duree():
    pcm8k = _pcm(8000)  # 1 s
    out = audio.to_stt_rate(pcm8k)
    assert abs(len(out) - 2 * len(pcm8k)) <= 64


def test_pcm_24k_vers_trames_mu_law_de_20ms():
    frames = audio.pcm_to_ulaw_frames(_pcm(24000), audio.TTS_RATE)  # 1 s
    assert all(len(f) == audio.ULAW_FRAME_BYTES for f in frames[:-1])
    assert 49 <= len(frames) <= 51


def test_le_sous_echantillonneur_en_flux_garde_son_etat_entre_les_morceaux():
    """Deux morceaux consécutifs doivent donner la même chose qu'un seul bloc : sans
    état continu, un clic s'entend à chaque raccord."""
    pcm = _pcm(24000)
    d = audio.StreamingDownsampler(audio.TTS_RATE)
    a = b"".join(d.feed(pcm[:24000])) + b"".join(d.feed(pcm[24000:]))
    one_shot = b"".join(audio.pcm_to_ulaw_frames(pcm, audio.TTS_RATE))
    assert abs(len(a) - len(one_shot)) <= audio.ULAW_FRAME_BYTES


def test_lecture_des_morceaux_prefixes_par_leur_taille():
    buf = bytearray()
    buf += (3).to_bytes(4, "big") + b"abc" + (2).to_bytes(4, "big") + b"d"  # 2e incomplet
    assert list(audio.pop_length_prefixed(buf)) == [b"abc"]
    assert bytes(buf) == (2).to_bytes(4, "big") + b"d"
    buf += b"e"
    assert list(audio.pop_length_prefixed(buf)) == [b"de"]
    assert bytes(buf) == b""


def test_niveau_sonore():
    assert audio.rms(_pcm(160, 1000)) == 1000
    assert audio.peak(_pcm(160, 1000)) == 1000
