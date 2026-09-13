"""Detecter que le client reprend la parole pendant que le bot parle.

Le probleme central n'est pas de detecter la voix — c'est de ne pas
detecter *celle du bot*. La ligne telephonique renvoie un echo de ce que
le bot diffuse ; un detecteur naif s'interrompt tout seul des la
premiere syllabe, et le bot ne finit plus aucune phrase.

Trois garde-fous, dans l'ordre d'importance :

1. un seuil nettement plus haut que celui de la detection de fin de tour
   — l'echo revient attenue, la voix du client non ;
2. plusieurs trames consecutives — un claquement, un souffle ou un pic
   d'echo ne durent pas ;
3. un delai de garde apres le debut de la parole du bot, le temps que
   son propre son cesse de saturer la ligne.

Et une exigence de correction : l'audio capte pendant l'interruption
appartient deja au client. Le jeter ferait perdre le debut de sa phrase,
et la transcription commencerait au milieu d'un mot.
"""

import struct

from hikky.adapters.telephony.barge_in import DetecteurInterruption

TRAME = 160  # echantillons = 20 ms a 8 kHz


def _trame(amplitude: int) -> bytes:
    return struct.pack(f"<{TRAME}h", *([amplitude] * TRAME))


SILENCE = _trame(0)
ECHO = _trame(700)      # la voix du bot qui revient, attenuee
CLIENT = _trame(6000)   # quelqu'un parle vraiment


def _detecteur(**kwargs) -> DetecteurInterruption:
    params = {
        "seuil_rms": 3000.0,
        "trames_consecutives": 3,
        "trames_de_garde": 0,
    }
    params.update(kwargs)
    return DetecteurInterruption(**params)


# ── Ne pas s'interrompre soi-meme ───────────────────────────────────────


def test_le_silence_n_interrompt_pas():
    d = _detecteur()
    for _ in range(20):
        assert d.observer(SILENCE) is False


def test_l_echo_du_bot_n_interrompt_pas():
    """Le defaut qui rend le barge-in inutilisable : le bot se coupe lui-meme."""
    d = _detecteur()
    for _ in range(30):
        assert d.observer(ECHO) is False


def test_un_pic_bref_n_interrompt_pas():
    """Un claquement de porte dure une trame, pas trois."""
    d = _detecteur()
    assert d.observer(CLIENT) is False
    assert d.observer(CLIENT) is False
    assert d.observer(SILENCE) is False
    # Le compteur est reparti de zero : deux trames ne suffisent plus.
    assert d.observer(CLIENT) is False
    assert d.observer(CLIENT) is False


# ── Interrompre quand c'est vraiment le client ──────────────────────────


def test_une_parole_soutenue_interrompt():
    d = _detecteur()
    assert d.observer(CLIENT) is False
    assert d.observer(CLIENT) is False
    assert d.observer(CLIENT) is True


def test_le_delai_de_garde_protege_le_debut_de_phrase():
    """Au demarrage, le bot sature la ligne : on n'ecoute pas encore."""
    d = _detecteur(trames_de_garde=5)
    for _ in range(5):
        assert d.observer(CLIENT) is False
    # Garde ecoulee : la detection reprend ses droits.
    assert d.observer(CLIENT) is False
    assert d.observer(CLIENT) is False
    assert d.observer(CLIENT) is True


# ── Ne pas perdre le debut de la phrase du client ───────────────────────


def test_l_audio_capte_pendant_l_interruption_est_conserve():
    """Sans cela, la transcription commence au milieu d'un mot."""
    d = _detecteur()
    d.observer(CLIENT)
    d.observer(CLIENT)
    d.observer(CLIENT)

    garde = d.audio_capte()

    assert len(garde) == 3 * TRAME * 2
    assert garde.startswith(CLIENT)


def test_le_silence_precedent_n_est_pas_conserve():
    """Seule la parole compte : le silence d'avant alourdirait la transcription."""
    d = _detecteur()
    for _ in range(10):
        d.observer(SILENCE)
    d.observer(CLIENT)
    d.observer(CLIENT)
    d.observer(CLIENT)

    assert len(d.audio_capte()) == 3 * TRAME * 2


def test_une_reinitialisation_vide_la_garde():
    d = _detecteur()
    d.observer(CLIENT)
    d.observer(CLIENT)
    d.observer(CLIENT)
    d.reinitialiser()

    assert d.audio_capte() == b""
    assert d.observer(CLIENT) is False


# ── Le seuil doit rester au-dessus de celui de fin de tour ──────────────


def test_le_seuil_d_interruption_est_plus_exigeant():
    """500 est le seuil de fin de tour ; l'echo le depasse largement.

    Reutiliser ce seuil pour l'interruption ferait couper le bot par son
    propre echo a chaque phrase.
    """
    from hikky.adapters.telephony.barge_in import SEUIL_INTERRUPTION_PAR_DEFAUT

    assert SEUIL_INTERRUPTION_PAR_DEFAUT > 500.0
