"""Faire parler le bot : envoi à Telnyx, interruption, historique tronqué.

Le WebSocket et le service TTS sont remplacés par des doublures : on vérifie ce qui
part sur la ligne et ce qui reste dans l'historique, sans réseau ni GPU.
"""

import base64
import json
import struct

import pytest
from telnyx_pipeline import audio
from telnyx_pipeline.call_state import CallState
from telnyx_pipeline.speaker import TelnyxSpeaker


class FakeWs:
    def __init__(self) -> None:
        self.sent: list[dict] = []

    async def send_text(self, text: str) -> None:
        self.sent.append(json.loads(text))


class _FakeStreamResponse:
    """Le flux TTS : des morceaux PCM 24 kHz, chacun précédé de sa taille sur 4 octets."""

    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = chunks

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def aiter_bytes(self):
        for c in self._chunks:
            yield len(c).to_bytes(4, "big") + c


class FakeHttp:
    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = chunks
        self.posted: list[str] = []

    def stream(self, method, url, json=None):
        self.posted.append(url)
        return _FakeStreamResponse(self._chunks)


def _pcm24k(seconds: float) -> bytes:
    n = int(audio.TTS_RATE * seconds)
    return struct.pack("<" + "h" * n, *([800, -800] * (n // 2)))


def _speaker(state, ws, http, streaming=True):
    return TelnyxSpeaker(
        ws=ws,
        http=http,
        state=state,
        tts_url="tts",
        tts_stream_url="tts/stream",
        streaming=streaming,
    )


@pytest.fixture(autouse=True)
def _pas_d_attente(monkeypatch):
    """`send_frames` attend la durée réelle de l'audio : inutile en test."""
    import asyncio

    real_sleep = asyncio.sleep

    async def instant(_):
        await real_sleep(0)

    monkeypatch.setattr("telnyx_pipeline.speaker.asyncio.sleep", instant)

    async def no_wait(awaitable, **_kwargs):
        if hasattr(awaitable, "close"):
            awaitable.close()
        raise TimeoutError

    monkeypatch.setattr("telnyx_pipeline.speaker.asyncio.wait_for", no_wait)


async def test_une_phrase_en_flux_part_vers_telnyx_en_messages_media():
    state, ws = CallState(), FakeWs()
    speaker = _speaker(state, ws, FakeHttp([_pcm24k(1.0), _pcm24k(1.0)]))
    await speaker.say("Bonjour.")
    medias = [m for m in ws.sent if m["event"] == "media"]
    assert len(medias) == 2
    payload = base64.b64decode(medias[0]["media"]["payload"])
    assert len(payload) % audio.ULAW_FRAME_BYTES == 0
    assert len(payload) >= 40 * audio.ULAW_FRAME_BYTES  # ~1 s d'audio = 50 trames
    assert state.speaking is False


async def test_apres_une_interruption_le_bot_reparle_a_la_phrase_suivante():
    """Bug réel du 2026-10-03 21:49 : après un barge-in, toutes les phrases suivantes
    faisaient 0 trame — le bot était muet, le client disait « Allô ? Allô ? »."""
    state, ws = CallState(), FakeWs()
    speaker = _speaker(state, ws, FakeHttp([_pcm24k(1.0)]))
    await speaker.interrupt()  # le client coupe le bot
    assert state.stop_requested is True
    await speaker.say("À quelle heure souhaitez-vous venir ?")
    assert any(m["event"] == "media" for m in ws.sent), "le bot doit reparler"


async def test_l_interruption_vide_la_file_audio_de_telnyx():
    state, ws = CallState(), FakeWs()
    speaker = _speaker(state, ws, FakeHttp([]))
    await speaker.interrupt()
    assert ws.sent[-1] == {"event": "clear"}


async def test_coupe_en_cours_de_phrase_l_historique_ne_garde_que_ce_qui_a_ete_entendu():
    state, ws = CallState(), FakeWs()
    texte = (
        "Nous vous accueillons de dix-neuf heures à vingt-trois heures. "
        "Je vous propose vingt heures, ça vous convient ?"
    )
    state.history.append({"role": "assistant", "content": texte})
    speaker = _speaker(state, ws, FakeHttp([_pcm24k(1.0), _pcm24k(1.0), _pcm24k(1.0)]))

    original = speaker.send_frames

    async def send_then_interrupt(frames):
        await original(frames)
        state.stop_requested = True  # le client coupe après le premier morceau

    speaker.send_frames = send_then_interrupt
    await speaker.say(texte)
    assert state.history[-1]["content"].endswith("(coupé par le client)")
    assert len(state.history[-1]["content"]) < len(texte)


async def test_sans_flux_la_synthese_complete_est_utilisee():
    class Http:
        async def post(self, url, json=None):
            class R:
                content = _pcm24k(0.5)
                headers = {"X-Sample-Rate": "24000"}

            return R()

    state, ws = CallState(), FakeWs()
    speaker = _speaker(state, ws, Http(), streaming=False)
    await speaker.say("Oui.")
    assert sum(1 for m in ws.sent if m["event"] == "media") == 1
