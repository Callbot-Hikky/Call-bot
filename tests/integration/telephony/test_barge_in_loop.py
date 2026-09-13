"""Le client doit pouvoir couper le bot.

Aujourd'hui la boucle ne lit rien pendant qu'elle parle : le client et le
bot se coupent mutuellement, et tout ce que le client dit pendant une
phrase du bot est perdu.

Ces tests portent sur l'integration, pas sur la detection elle-meme
(couverte par `tests/unit/telephony/test_barge_in.py`) :

- la diffusion s'arrete vraiment quand le client parle ;
- ce que le client a dit pendant l'interruption n'est pas perdu ;
- l'echo du bot ne declenche rien — sinon il ne finit aucune phrase.
"""

from __future__ import annotations

import asyncio
import struct

from hikky.adapters.telephony.asterisk_audiosocket_adapter import (
    AsteriskAudioSocketAdapter,
)
from hikky.adapters.telephony.asterisk_audiosocket_server import (
    AudioSocketServerConfig,
    _speak,
)
from hikky.adapters.telephony.audiosocket_protocol import encode_audio

from .test_audiosocket_ai_loop import (
    CALL_ID,
    _CapturingWriter,
    _FakeTTS,
    _ScriptedReader,
)


def _trame(amplitude: int) -> bytes:
    return struct.pack("<160h", *([amplitude] * 160))


CLIENT = _trame(9000)
ECHO = _trame(700)
SILENCE = _trame(0)


class _TTSLong:
    """Voix longue : sans interruption, la diffusion dure plusieurs secondes."""

    sample_rate = 8000

    def __init__(self) -> None:
        self.spoken: list[str] = []

    async def synthesize(self, text: str):
        self.spoken.append(text)
        # 4 secondes a 8 kHz.
        payload = struct.pack("<32000h", *([1500] * 32000))

        async def _stream():
            yield payload

        return _stream()


def _config(**kwargs) -> AudioSocketServerConfig:
    params = {
        "sample_rate": 8000,
        "tts_sample_rate": 8000,
        "realtime_playback": True,
        "barge_in_enabled": True,
        "barge_in_guard_frames": 0,
    }
    params.update(kwargs)
    return AudioSocketServerConfig(**params)


class _Deps:
    def __init__(self, tts) -> None:
        self.tts_adapter = tts


def _flux(trames: list[bytes]) -> bytes:
    return b"".join(encode_audio(t) for t in trames)


class _LecteurSansFin:
    """Ligne ouverte : elle transmet tant que l'appel dure.

    Un flux fini ferait croire a un raccrochage, ce qui arrete la
    diffusion pour une raison etrangere au barge-in.
    """

    def __init__(self, trame: bytes) -> None:
        self._paquet = encode_audio(trame)
        self._tampon = bytearray()

    async def readexactly(self, n: int) -> bytes:
        while len(self._tampon) < n:
            self._tampon.extend(self._paquet)
            await asyncio.sleep(0)
        bloc = bytes(self._tampon[:n])
        del self._tampon[:n]
        return bloc


async def test_la_parole_du_client_arrete_la_diffusion():
    tts = _TTSLong()
    writer = _CapturingWriter()
    adapter = AsteriskAudioSocketAdapter()
    adapter.bind_call(CALL_ID, writer)
    reader = _LecteurSansFin(CLIENT)

    resultat = await _speak(
        adapter, CALL_ID, "Une phrase assez longue.", _Deps(tts), _config(),
        reader=reader,
    )

    assert resultat.interrompu is True
    # 4 s de voix a 8 kHz = 64000 octets. Coupe tot, on en emet bien moins.
    assert len(writer.written) < 32000, len(writer.written)


async def test_ce_que_le_client_dit_pendant_l_interruption_est_conserve():
    """Sinon la transcription commence au milieu d'un mot."""
    tts = _TTSLong()
    writer = _CapturingWriter()
    adapter = AsteriskAudioSocketAdapter()
    adapter.bind_call(CALL_ID, writer)
    reader = _LecteurSansFin(CLIENT)

    resultat = await _speak(
        adapter, CALL_ID, "Une phrase assez longue.", _Deps(tts), _config(),
        reader=reader,
    )

    assert resultat.audio_client, "le debut de la phrase du client est perdu"
    assert resultat.audio_client.startswith(CLIENT)


async def test_l_echo_du_bot_ne_l_interrompt_pas():
    """Le defaut qui rendrait le barge-in inutilisable."""
    tts = _TTSLong()
    writer = _CapturingWriter()
    adapter = AsteriskAudioSocketAdapter()
    adapter.bind_call(CALL_ID, writer)
    reader = _LecteurSansFin(ECHO)

    resultat = await _speak(
        adapter, CALL_ID, "Une phrase assez longue.", _Deps(tts), _config(),
        reader=reader,
    )

    assert resultat.interrompu is False
    assert len(writer.written) > 32000, "la phrase a ete tronquee par l'echo"


async def test_sans_reader_le_comportement_est_inchange():
    """Le chemin existant doit continuer de fonctionner tel quel."""
    tts = _FakeTTS(sample_rate=8000)
    writer = _CapturingWriter()
    adapter = AsteriskAudioSocketAdapter()
    adapter.bind_call(CALL_ID, writer)

    resultat = await _speak(
        adapter, CALL_ID, "Bonjour.", _Deps(tts), _config(),
    )

    assert resultat.interrompu is False
    assert tts.spoken == ["Bonjour."]


async def test_le_barge_in_peut_etre_desactive():
    """Filet de securite : si l'echo pose probleme en production."""
    tts = _TTSLong()
    writer = _CapturingWriter()
    adapter = AsteriskAudioSocketAdapter()
    adapter.bind_call(CALL_ID, writer)
    reader = _LecteurSansFin(CLIENT)

    resultat = await _speak(
        adapter, CALL_ID, "Une phrase assez longue.", _Deps(tts),
        _config(barge_in_enabled=False), reader=reader,
    )

    assert resultat.interrompu is False
    assert len(writer.written) > 32000


async def test_le_raccrochage_pendant_la_parole_est_signale():
    """Si le client raccroche, inutile de finir la phrase."""
    tts = _TTSLong()
    writer = _CapturingWriter()
    adapter = AsteriskAudioSocketAdapter()
    adapter.bind_call(CALL_ID, writer)
    from hikky.adapters.telephony.audiosocket_protocol import encode_hangup

    reader = _ScriptedReader(encode_hangup())

    resultat = await _speak(
        adapter, CALL_ID, "Une phrase assez longue.", _Deps(tts), _config(),
        reader=reader,
    )

    assert resultat.raccroche is True
    assert resultat.interrompu is False


async def test_l_interruption_ne_remonte_pas_d_audio_vide():
    """Une interruption sans audio ferait repartir la collecte sur du vide."""
    tts = _TTSLong()
    writer = _CapturingWriter()
    adapter = AsteriskAudioSocketAdapter()
    adapter.bind_call(CALL_ID, writer)
    reader = _LecteurSansFin(CLIENT)

    resultat = await _speak(
        adapter, CALL_ID, "Une phrase assez longue.", _Deps(tts), _config(),
        reader=reader,
    )

    assert resultat.interrompu is True
    # Multiple entier de trames : la boucle en deduit un nombre de trames.
    assert len(resultat.audio_client) % 320 == 0
    assert len(resultat.audio_client) > 0
