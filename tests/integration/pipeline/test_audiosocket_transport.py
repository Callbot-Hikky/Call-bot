"""Transport Pipecat pour AudioSocket — c'est lui qui apporte le barge-in.

Le chemin AudioSocket actuel n'ecoute pas pendant que le bot parle : le
client et le bot se coupent mutuellement, et la fin de parole est
detectee par un simple seuil d'energie a 500 ms, qui tranche au milieu
des phrases.

Pipecat resout cela nativement — VAD Silero (qui accepte le 8 kHz
telephonique), detection de tour, et interruptions activees par
`VADUserTurnStartStrategy(enable_interruptions=True)`. Encore faut-il
lui fournir un transport qui parle AudioSocket, ce qu'il ne propose pas.
"""

import struct
import uuid

from pipecat.frames.frames import InputAudioRawFrame, OutputAudioRawFrame

from hikky.pipeline.audiosocket_transport import (
    AudioSocketTransport,
    AudioSocketTransportParams,
)

TRAME_20MS = 320  # 8 kHz, 16 bits, mono


def _trame_audio(valeur: int = 1000) -> bytes:
    return struct.pack("<160h", *([valeur] * 160))


class _LecteurScripte:
    """Simule Asterisk : livre des paquets AudioSocket puis ferme."""

    def __init__(self, paquets: bytes) -> None:
        self._data = paquets
        self._pos = 0

    async def readexactly(self, n: int) -> bytes:
        import asyncio

        if self._pos + n > len(self._data):
            raise asyncio.IncompleteReadError(self._data[self._pos :], n)
        bloc = self._data[self._pos : self._pos + n]
        self._pos += n
        return bloc


class _EcrivainCapturant:
    def __init__(self) -> None:
        self.ecrit = bytearray()
        self.drains = 0

    def write(self, data: bytes) -> None:
        self.ecrit.extend(data)

    async def drain(self) -> None:
        self.drains += 1

    def close(self) -> None:
        return None

    async def wait_closed(self) -> None:
        return None


def _paquet_audio(charge: bytes) -> bytes:
    return struct.pack(">BH", 0x10, len(charge)) + charge


def _paquet_hangup() -> bytes:
    return struct.pack(">BH", 0x00, 0)


def _params() -> AudioSocketTransportParams:
    return AudioSocketTransportParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        audio_in_sample_rate=8000,
        audio_out_sample_rate=8000,
    )


# ── Construction ────────────────────────────────────────────────────────


def test_transport_expose_une_entree_et_une_sortie():
    t = AudioSocketTransport(
        reader=_LecteurScripte(b""), writer=_EcrivainCapturant(), params=_params()
    )
    assert t.input() is not None
    assert t.output() is not None


def test_le_transport_travaille_en_8_khz():
    """AudioSocket impose le 8 kHz : toute autre valeur casserait la lecture."""
    t = AudioSocketTransport(
        reader=_LecteurScripte(b""), writer=_EcrivainCapturant(), params=_params()
    )
    assert t.params.audio_in_sample_rate == 8000
    assert t.params.audio_out_sample_rate == 8000


# ── Entree : Asterisk -> Pipecat ────────────────────────────────────────


async def test_les_paquets_audio_deviennent_des_frames_pipecat():
    paquets = _paquet_audio(_trame_audio()) * 3 + _paquet_hangup()
    entree = AudioSocketTransport(
        reader=_LecteurScripte(paquets), writer=_EcrivainCapturant(), params=_params()
    ).input()

    recues = []
    entree.push_audio_frame = lambda f: recues.append(f)  # type: ignore[assignment]
    await entree.lire_paquets_pour_test()

    assert len(recues) == 3
    assert all(isinstance(f, InputAudioRawFrame) for f in recues)
    assert recues[0].sample_rate == 8000
    assert len(recues[0].audio) == TRAME_20MS


async def test_le_paquet_uuid_fournit_l_identifiant_d_appel():
    call_uuid = uuid.uuid4()
    paquets = struct.pack(">BH", 0x01, 16) + call_uuid.bytes + _paquet_hangup()
    transport = AudioSocketTransport(
        reader=_LecteurScripte(paquets), writer=_EcrivainCapturant(), params=_params()
    )
    entree = transport.input()
    entree.push_audio_frame = lambda f: None  # type: ignore[assignment]
    await entree.lire_paquets_pour_test()
    assert transport.call_id == str(call_uuid)


async def test_le_raccrochage_arrete_la_lecture():
    paquets = _paquet_hangup() + _paquet_audio(_trame_audio()) * 5
    entree = AudioSocketTransport(
        reader=_LecteurScripte(paquets), writer=_EcrivainCapturant(), params=_params()
    ).input()
    recues = []
    entree.push_audio_frame = lambda f: recues.append(f)  # type: ignore[assignment]
    await entree.lire_paquets_pour_test()
    assert recues == []


# ── Sortie : Pipecat -> Asterisk ────────────────────────────────────────


async def test_l_audio_sortant_respecte_les_trames_de_20_ms():
    """Asterisk n'accepte que des trames de 20 ms — un paquet plus gros
    casse la lecture, ce qui est arrive en production."""
    ecrivain = _EcrivainCapturant()
    sortie = AudioSocketTransport(
        reader=_LecteurScripte(b""), writer=ecrivain, params=_params()
    ).output()

    await sortie.write_audio_frame(
        OutputAudioRawFrame(audio=_trame_audio() * 5, sample_rate=8000, num_channels=1)
    )

    tailles = _tailles_des_paquets(bytes(ecrivain.ecrit))
    assert tailles, "aucun paquet emis"
    assert all(t == TRAME_20MS for t in tailles), tailles


async def test_une_trame_incomplete_est_completee_par_du_silence():
    ecrivain = _EcrivainCapturant()
    sortie = AudioSocketTransport(
        reader=_LecteurScripte(b""), writer=ecrivain, params=_params()
    ).output()
    await sortie.write_audio_frame(
        OutputAudioRawFrame(audio=b"\x01\x02" * 50, sample_rate=8000, num_channels=1)
    )
    assert _tailles_des_paquets(bytes(ecrivain.ecrit)) == [TRAME_20MS]


async def test_l_interruption_vide_l_audio_en_attente():
    """Le barge-in n'a de sens que si l'audio deja mis en file est jete :
    sinon le bot continue de parler apres avoir ete interrompu."""
    ecrivain = _EcrivainCapturant()
    sortie = AudioSocketTransport(
        reader=_LecteurScripte(b""), writer=ecrivain, params=_params()
    ).output()
    await sortie.write_audio_frame(
        OutputAudioRawFrame(audio=_trame_audio() * 20, sample_rate=8000, num_channels=1)
    )
    avant = len(ecrivain.ecrit)
    await sortie.vider_pour_interruption()
    await sortie.write_audio_frame(
        OutputAudioRawFrame(audio=_trame_audio(), sample_rate=8000, num_channels=1)
    )
    assert len(ecrivain.ecrit) > avant


def _tailles_des_paquets(donnees: bytes) -> list[int]:
    tailles, pos = [], 0
    while pos + 3 <= len(donnees):
        (longueur,) = struct.unpack(">H", donnees[pos + 1 : pos + 3])
        if donnees[pos] == 0x10:
            tailles.append(longueur)
        pos += 3 + longueur
    return tailles
