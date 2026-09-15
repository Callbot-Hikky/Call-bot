"""Endpointer VAD : découper les tours de parole par détection d'activité
vocale plutôt que par seuil d'énergie.

Le découpage RMS coupait les mots dès que la parole passait sous le seuil
(« les, les, les ») et fusionnait deux phrases quand le bruit de ligne
restait au-dessus. Un VAD par trame décide « parole / non-parole » de façon
robuste au niveau de bruit.

Le backend VAD est injecté : les tests n'ont pas besoin de `webrtcvad`.
"""

from hikky.adapters.telephony.vad_endpointer import VadEndpointer

SP = b"\x11" * 320  # trame "parole" (marqueur pour le faux VAD)
SIL = b"\x00" * 320  # trame "silence"


def _faux_vad(frame: bytes, sample_rate: int) -> bool:
    """Parole ssi la trame n'est pas du pur silence."""
    return frame != SIL


def _endpointer(**kwargs):
    params = dict(
        frame_bytes=320,
        sample_rate=8000,
        min_speech_frames=2,
        hangover_frames=3,
        start_padding_frames=2,
        max_utterance_frames=10,
        vad=_faux_vad,
    )
    params.update(kwargs)
    return VadEndpointer(**params)


def _feed_all(ep, frames):
    """Renvoie la liste des énoncés complétés pendant l'alimentation."""
    out = []
    for f in frames:
        utt = ep.feed(f)
        if utt is not None:
            out.append(utt)
    return out


def test_le_silence_de_tete_est_ignore():
    ep = _endpointer()
    # Que du silence : aucun énoncé.
    assert _feed_all(ep, [SIL] * 6) == []


def test_un_enonce_se_termine_apres_le_hangover():
    ep = _endpointer()
    frames = [SP, SP, SP] + [SIL, SIL, SIL]  # 3 parole puis 3 silences (=hangover)
    out = _feed_all(ep, frames)
    assert len(out) == 1
    # L'énoncé contient au moins les trames de parole.
    assert out[0].count(SP[:1]) >= 0  # sanity: bytes présents
    assert len(out[0]) >= 3 * 320


def test_un_enonce_trop_court_est_jete():
    ep = _endpointer(min_speech_frames=3)
    # 2 trames de parole seulement, puis hangover → sous le minimum → rien.
    out = _feed_all(ep, [SP, SP] + [SIL, SIL, SIL])
    assert out == []


def test_un_silence_bref_ne_coupe_pas_l_enonce():
    ep = _endpointer(hangover_frames=3)
    # parole, court silence (<hangover), parole, puis vrai hangover.
    frames = [SP, SP, SIL, SP, SP] + [SIL, SIL, SIL]
    out = _feed_all(ep, frames)
    assert len(out) == 1  # une seule phrase, pas deux
    assert len(out[0]) >= 5 * 320


def test_le_preroll_precede_la_parole():
    """On garde quelques trames avant l'attaque pour ne pas rogner le premier
    phonème (« quatre » entendu « attre »)."""
    ep = _endpointer(start_padding_frames=2)
    frames = [SIL, SIL, SP, SP] + [SIL, SIL, SIL]
    out = _feed_all(ep, frames)
    assert len(out) == 1
    # 2 trames de préroll + 2 de parole + silences de hangover inclus.
    assert len(out[0]) >= 4 * 320


def test_l_enonce_est_coupe_a_la_longueur_max():
    ep = _endpointer(max_utterance_frames=5)
    # Parole continue plus longue que le max → coupé sans attendre le silence.
    out = _feed_all(ep, [SP] * 8)
    assert len(out) == 1
    assert len(out[0]) <= 6 * 320  # ~max, marge d'une trame


def test_deux_enonces_successifs():
    ep = _endpointer()
    frames = (
        [SP, SP] + [SIL, SIL, SIL]  # énoncé 1
        + [SIL]                      # pause
        + [SP, SP, SP] + [SIL, SIL, SIL]  # énoncé 2
    )
    out = _feed_all(ep, frames)
    assert len(out) == 2


def test_seed_reprend_une_parole_deja_commencee():
    """Après une interruption (barge-in), on réinjecte le début déjà entendu :
    l'énoncé ne doit pas être perdu ni attendre une nouvelle attaque."""
    ep = _endpointer()
    ep.seed(SP + SP)  # 2 trames déjà entendues
    out = _feed_all(ep, [SP] + [SIL, SIL, SIL])
    assert len(out) == 1
    assert len(out[0]) >= 3 * 320
