"""Détection de parole : la seule pièce qui décide « le client a fini sa phrase ».

Les constantes viennent d'appels réels (voir les commentaires du module) ; les
scénarios ci-dessous rejouent les bugs de la semaine du 2026-10-03.
"""

import struct

from telnyx_pipeline.turn_detector import BargeIn, TurnDetector, Utterance

FRAME = 160  # échantillons PCM16 par trame de 20 ms à 8 kHz


def _frame(level: int) -> bytes:
    """Une trame de 20 ms d'un signal carré d'amplitude `level` (RMS = level)."""
    return struct.pack("<" + "h" * FRAME, *([level, -level] * (FRAME // 2)))


SILENCE, VOICE, LOUD = _frame(100), _frame(2000), _frame(4000)


def _speak(det, frames, bot_speaking=False):
    events = []
    for f in frames:
        events.extend(det.feed(f, bot_speaking=bot_speaking))
    return events


def test_une_phrase_puis_un_silence_de_700ms_donne_un_enonce():
    det = TurnDetector()
    events = _speak(det, [SILENCE] * 5 + [VOICE] * 20 + [SILENCE] * 34)
    assert events == []  # 34 trames de silence = 680 ms : pas encore
    events = _speak(det, [SILENCE])  # 35e trame : 700 ms
    assert len(events) == 1 and isinstance(events[0], Utterance)
    assert events[0].speech_frames == 20 and not events[0].during_bot_speech


def test_le_pre_roll_est_colle_devant_la_phrase():
    """Le VAD à énergie rogne la première syllabe : on garde 400 ms d'audio réel
    AVANT l'attaque (appel réel : « a l'al » au lieu de « halal »)."""
    det = TurnDetector()
    (utt,) = _speak(det, [SILENCE] * 30 + [VOICE] * 20 + [SILENCE] * 35)
    # 20 trames de voix + 35 de silence de fin + pré-roll (6 400 octets = 20 trames)
    assert len(utt.pcm) == (20 + 35) * FRAME * 2 + 6400


def test_un_blip_plus_court_que_240ms_est_ignore():
    det = TurnDetector()
    assert _speak(det, [VOICE] * 11 + [SILENCE] * 35) == []


def test_oui_de_240ms_passe():
    det = TurnDetector()
    events = _speak(det, [VOICE] * 12 + [SILENCE] * 35)
    assert len(events) == 1


def test_pendant_que_le_bot_parle_une_phrase_courte_est_jetee_une_longue_gardee():
    det = TurnDetector()
    assert _speak(det, [VOICE] * 15 + [SILENCE] * 35, bot_speaking=True) == []
    (utt,) = _speak(det, [VOICE] * 30 + [SILENCE] * 35, bot_speaking=True)
    assert utt.during_bot_speech is True


def test_le_client_qui_parle_fort_240ms_coupe_le_bot():
    det = TurnDetector()
    events = _speak(det, [LOUD] * 11, bot_speaking=True)
    assert not any(isinstance(e, BargeIn) for e in events)
    events = _speak(det, [LOUD], bot_speaking=True)
    assert any(isinstance(e, BargeIn) for e in events)


def test_une_voix_normale_ne_coupe_pas_le_bot():
    det = TurnDetector()
    events = _speak(det, [VOICE] * 50, bot_speaking=True)
    assert not any(isinstance(e, BargeIn) for e in events)


def test_le_barge_in_n_est_pas_declenche_quand_le_bot_ne_parle_pas():
    det = TurnDetector()
    events = _speak(det, [LOUD] * 30, bot_speaking=False)
    assert not any(isinstance(e, BargeIn) for e in events)
