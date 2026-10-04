"""L'état d'un appel : énoncé gardé pendant la parole du bot, répétitions, flux Telnyx."""

from telnyx_pipeline.call_state import PENDING_MAX_AGE_S, CallState
from telnyx_pipeline.inbound import StreamRegistry, texml_connect


def test_l_enonce_garde_est_rejoue_s_il_est_recent():
    s = CallState()
    s.keep_pending(b"a" * 100, now=10.0)
    assert s.pop_pending(now=12.0) == b"a" * 100
    assert s.pop_pending(now=12.0) is None


def test_un_enonce_garde_trop_vieux_est_perime():
    """Appel réel (Paul) : la répétition rejouée après la réponse faisait répondre deux
    fois à la même question. Au-delà de 6 s, le client a déjà avancé."""
    s = CallState()
    s.keep_pending(b"a", now=10.0)
    assert s.pop_pending(now=10.0 + PENDING_MAX_AGE_S + 0.1) is None


def test_seul_le_plus_recent_est_garde():
    s = CallState()
    s.keep_pending(b"premier", now=1.0)
    s.keep_pending(b"second", now=2.0)
    assert s.pop_pending(now=3.0) == b"second"


def test_une_repetition_du_dernier_texte_est_reconnue():
    s = CallState()
    assert not s.is_repeat("Est-ce que vous avez une terrasse ?")
    assert s.is_repeat("est ce que vous avez une terrasse")
    assert not s.is_repeat("Et un parking ?")


def test_un_seul_stream_par_appel_telnyx():
    """Appel réel : deux POST inbound à 7 s d'écart pour un appel -> deux bots."""
    reg = StreamRegistry(ttl_s=3600)
    assert reg.accept("call-1", now=0.0)
    assert not reg.accept("call-1", now=7.0)
    assert reg.accept("call-2", now=8.0)
    assert reg.accept("call-1", now=3700.0)  # oublié après une heure


def test_un_inbound_sans_identifiant_est_toujours_accepte():
    reg = StreamRegistry()
    assert reg.accept("", now=0.0) and reg.accept("", now=1.0)


def test_la_consigne_texml_pointe_sur_notre_websocket():
    xml = texml_connect("pod-19123.proxy.runpod.net")
    assert "wss://pod-19123.proxy.runpod.net/telnyx/stream" in xml
    assert 'bidirectionalCodec="PCMA"' in xml  # A-law : le même codec que la ligne envoie
