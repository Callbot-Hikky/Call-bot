from __future__ import annotations

import asyncio
import struct
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import time
from typing import Any

import pytest

from hikky.adapters.telephony.asterisk_audiosocket_adapter import (
    AsteriskAudioSocketAdapter,
)
from hikky.adapters.telephony.asterisk_audiosocket_server import (
    AudioSocketServerConfig,
    AudioSocketServerDeps,
    run_ai_loop,
)
from hikky.adapters.telephony.audiosocket_protocol import (
    AUDIOSOCKET_CHUNK_20MS_BYTES,
    MessageType,
    encode_audio,
    encode_hangup,
)
from hikky.domain.outcomes import CallOutcome
from hikky.domain.restaurant_context import (
    OpeningHours,
    RestaurantContext,
    RestaurantRules,
)

CALL_ID = "call-1"


def _ctx() -> RestaurantContext:
    return RestaurantContext(
        id="r-1",
        name="Le Petit Sud",
        greeting="Bonjour, Le Petit Sud.",
        opening_hours=[
            OpeningHours(weekday=i, opens=time(9, 0), closes=time(23, 0))
            for i in range(7)
        ],
        total_capacity=40,
        rules=RestaurantRules(),
        transfer_number=None,
        fallback_message="Je vous rappelle.",
    )


def _loud_frame() -> bytes:
    return struct.pack("<160h", *([8000] * 160))


def _silent_frame() -> bytes:
    return b"\x00" * AUDIOSOCKET_CHUNK_20MS_BYTES


class _ScriptedReader:
    def __init__(self, payload: bytes) -> None:
        self._data = payload
        self._pos = 0

    async def readexactly(self, n: int) -> bytes:
        if self._pos + n > len(self._data):
            raise asyncio.IncompleteReadError(self._data[self._pos :], n)
        chunk = self._data[self._pos : self._pos + n]
        self._pos += n
        return chunk


class _CapturingWriter:
    def __init__(self) -> None:
        self.written = bytearray()

    def write(self, data: bytes) -> None:
        self.written.extend(data)

    async def drain(self) -> None:
        return None

    def close(self) -> None:
        return None

    async def wait_closed(self) -> None:
        return None


class _FakeSTT:
    def __init__(self, transcripts: list[str]) -> None:
        self._transcripts = list(transcripts)
        self.received_audio: list[bytes] = []

    async def transcribe(self, audio_chunks: AsyncIterator[bytes]) -> AsyncIterator[str]:
        buffer = bytearray()
        async for chunk in audio_chunks:
            buffer.extend(chunk)
        self.received_audio.append(bytes(buffer))
        text = self._transcripts.pop(0) if self._transcripts else ""

        async def _stream() -> AsyncIterator[str]:
            if text:
                yield text

        return _stream()


class _FakeTTS:
    def __init__(self, sample_rate: int = 22050) -> None:
        self.spoken: list[str] = []
        self._sample_rate = sample_rate

    async def synthesize(self, text: str) -> AsyncIterator[bytes]:
        self.spoken.append(text)
        payload = struct.pack("<2205h", *([1000] * 2205))

        async def _stream() -> AsyncIterator[bytes]:
            yield payload

        return _stream()


class _FakeIntent:
    def __init__(self, complete: bool) -> None:
        self._complete = complete
        self.date_time = None
        self.party_size = 4
        self.customer_name = "Dupont"

    def is_complete(self) -> bool:
        return self._complete


@dataclass
class _TurnResult:
    bot_says: str
    updated_intent: object = None


class _FakeSession:
    def __init__(
        self,
        *,
        bot_says: str = "Pour combien de personnes ?",
        fallback: object | None = None,
        finalize_outcome: CallOutcome | None = None,
        complete_intent: bool = False,
    ) -> None:
        self.context = _ctx()
        self.turns: list[tuple[str, dict[str, Any]]] = []
        self.ended_with: CallOutcome | None = None
        self.finalize_calls = 0
        self._bot_says = bot_says
        self._fallback = fallback
        self._finalize_outcome = finalize_outcome
        self.intent = _FakeIntent(complete_intent)

    async def begin(self) -> None:
        return None

    async def process_user_turn(self, user_text: str, slot_updates: dict[str, Any]):
        self.turns.append((user_text, slot_updates))
        return _TurnResult(bot_says=self._bot_says)

    def check_fallback(self, *, user_requested_human: bool, group_size: Any):
        return self._fallback

    async def request_callback(self, **kwargs: Any) -> str:
        return "cb-1"

    async def finalize_if_complete(self, customer_phone: str | None):
        self.finalize_calls += 1
        return self._finalize_outcome

    async def end_with(self, outcome: CallOutcome) -> None:
        self.ended_with = outcome


class _FakeSlotExtractor:
    def __init__(self, slots: dict[str, Any] | None = None) -> None:
        self._slots = slots or {}

    async def extract(self, user_text: str) -> dict[str, Any]:
        return dict(self._slots)


def _config(**overrides: Any) -> AudioSocketServerConfig:
    base = {
        "realtime_playback": False,
        "silence_frames_to_end": 3,
        "min_speech_frames": 2,
        "tts_sample_rate": 22050,
    }
    base.update(overrides)
    return AudioSocketServerConfig(**base)


def _deps(stt: Any, tts: Any, slot_extractor: Any = None) -> AudioSocketServerDeps:
    return AudioSocketServerDeps(
        session_factory=lambda *_: None,
        restaurant_context_port=None,
        stt_adapter=stt,
        tts_adapter=tts,
        slot_extractor=slot_extractor or _FakeSlotExtractor(),
    )


def _decode_audio_payload(written: bytes) -> bytes:
    out = bytearray()
    pos = 0
    while pos + 3 <= len(written):
        msg_type = written[pos]
        (length,) = struct.unpack(">H", written[pos + 1 : pos + 3])
        payload = written[pos + 3 : pos + 3 + length]
        if msg_type == MessageType.AUDIO_PCM_8K:
            out.extend(payload)
        pos += 3 + length
    return bytes(out)


async def _run(reader_payload: bytes, session: _FakeSession, deps, config):
    adapter = AsteriskAudioSocketAdapter()
    writer = _CapturingWriter()
    adapter.bind_call(CALL_ID, writer)
    await run_ai_loop(
        _ScriptedReader(reader_payload),
        adapter,
        CALL_ID,
        session,
        deps,
        config,
    )
    return writer


async def test_greeting_is_spoken_when_call_starts():
    tts = _FakeTTS()
    session = _FakeSession()
    writer = await _run(encode_hangup(), session, _deps(_FakeSTT([]), tts), _config())
    assert tts.spoken[0] == "Bonjour, Le Petit Sud."
    assert len(_decode_audio_payload(bytes(writer.written))) > 0


async def test_greeting_audio_is_resampled_to_telephony_rate():
    tts = _FakeTTS()
    session = _FakeSession()
    writer = await _run(encode_hangup(), session, _deps(_FakeSTT([]), tts), _config())
    audio = _decode_audio_payload(bytes(writer.written))
    assert len(audio) == pytest.approx(1600, abs=32)


async def test_utterance_is_transcribed_after_silence():
    stt = _FakeSTT(["Bonjour je voudrais reserver"])
    session = _FakeSession()
    payload = b"".join(
        [encode_audio(_loud_frame()) for _ in range(4)]
        + [encode_audio(_silent_frame()) for _ in range(4)]
        + [encode_hangup()]
    )
    await _run(payload, session, _deps(stt, _FakeTTS()), _config())
    assert session.turns[0][0] == "Bonjour je voudrais reserver"


async def test_transcribed_audio_is_upsampled_for_whisper():
    stt = _FakeSTT(["oui"])
    session = _FakeSession()
    payload = b"".join(
        [encode_audio(_loud_frame()) for _ in range(4)]
        + [encode_audio(_silent_frame()) for _ in range(4)]
        + [encode_hangup()]
    )
    await _run(payload, session, _deps(stt, _FakeTTS()), _config())
    assert len(stt.received_audio[0]) > 4 * AUDIOSOCKET_CHUNK_20MS_BYTES


async def test_bot_response_is_spoken_back():
    tts = _FakeTTS()
    session = _FakeSession(bot_says="Pour combien de personnes ?")
    payload = b"".join(
        [encode_audio(_loud_frame()) for _ in range(4)]
        + [encode_audio(_silent_frame()) for _ in range(4)]
        + [encode_hangup()]
    )
    await _run(payload, session, _deps(_FakeSTT(["bonjour"]), tts), _config())
    assert "Pour combien de personnes ?" in tts.spoken


async def test_slot_updates_are_passed_to_session():
    session = _FakeSession()
    extractor = _FakeSlotExtractor({"party_size": 4})
    payload = b"".join(
        [encode_audio(_loud_frame()) for _ in range(4)]
        + [encode_audio(_silent_frame()) for _ in range(4)]
        + [encode_hangup()]
    )
    await _run(
        payload, session, _deps(_FakeSTT(["on serait 4"]), _FakeTTS(), extractor), _config()
    )
    assert session.turns[0][1] == {"party_size": 4}


async def test_empty_transcription_does_not_trigger_a_turn():
    session = _FakeSession()
    payload = b"".join(
        [encode_audio(_loud_frame()) for _ in range(4)]
        + [encode_audio(_silent_frame()) for _ in range(4)]
        + [encode_hangup()]
    )
    await _run(payload, session, _deps(_FakeSTT([""]), _FakeTTS()), _config())
    assert session.turns == []


async def test_pure_silence_never_triggers_transcription():
    stt = _FakeSTT(["ne devrait pas etre appele"])
    session = _FakeSession()
    payload = b"".join(
        [encode_audio(_silent_frame()) for _ in range(20)] + [encode_hangup()]
    )
    await _run(payload, session, _deps(stt, _FakeTTS()), _config())
    assert stt.received_audio == []
    assert session.turns == []


async def test_hangup_frame_stops_the_loop():
    session = _FakeSession()
    payload = encode_hangup() + encode_audio(_loud_frame()) * 10
    await _run(payload, session, _deps(_FakeSTT(["jamais"]), _FakeTTS()), _config())
    assert session.turns == []


async def test_peer_disconnect_stops_the_loop_cleanly():
    session = _FakeSession()
    await _run(b"", session, _deps(_FakeSTT([]), _FakeTTS()), _config())
    assert session.turns == []


async def test_fallback_ends_the_call():
    @dataclass
    class _Decision:
        outcome: CallOutcome
        reason: str

    session = _FakeSession(
        fallback=_Decision(outcome=CallOutcome.TRANSFERRED, reason="3 echecs")
    )
    tts = _FakeTTS()
    payload = b"".join(
        [encode_audio(_loud_frame()) for _ in range(4)]
        + [encode_audio(_silent_frame()) for _ in range(4)]
        + [encode_audio(_loud_frame()) for _ in range(4)]
        + [encode_audio(_silent_frame()) for _ in range(4)]
        + [encode_hangup()]
    )
    await _run(payload, session, _deps(_FakeSTT(["a", "b"]), tts), _config())
    assert session.ended_with == CallOutcome.TRANSFERRED
    assert "Je vous rappelle." in tts.spoken
    assert len(session.turns) == 1


async def test_complete_intent_asks_confirmation_without_ending():
    """Slots complets => recapitulatif, pas de reservation ni raccrochage."""
    session = _FakeSession(
        bot_says="C'est note, merci !",
        finalize_outcome=CallOutcome.RESERVATION_CREATED,
        complete_intent=True,
    )
    tts = _FakeTTS()
    payload = b"".join(
        [encode_audio(_loud_frame()) for _ in range(4)]
        + [encode_audio(_silent_frame()) for _ in range(4)]
        + [encode_audio(_loud_frame()) for _ in range(4)]
        + [encode_audio(_silent_frame()) for _ in range(4)]
        + [encode_hangup()]
    )
    await _run(payload, session, _deps(_FakeSTT(["a", "b"]), tts), _config())
    assert any("récapitule" in t for t in tts.spoken), tts.spoken
    assert session.finalize_calls == 0, "reserve sans accord du client"
    assert len(session.turns) == 2, "l'appel doit rester ouvert"


async def test_loop_refuses_to_run_without_stt_or_tts():
    session = _FakeSession()
    await _run(encode_hangup(), session, _deps(None, None), _config())
    assert session.turns == []


# ── Regroupement des trames sortantes ───────────────────────────────────
#
# Appel reel : 28 secondes de silence pendant que le bot « parlait ».
# Chaque trame de 20 ms declenchait un write + drain reseau a travers le
# tunnel vers la Suede. 218 trames x ~100 ms d aller-retour = ~22 s de
# retard. Asterisk sait tamponner : on envoie moins souvent, plus gros.

async def test_outbound_packets_are_always_20ms_frames():
    tts = _FakeTTS()
    session = _FakeSession()
    writer = await _run(encode_hangup(), session, _deps(_FakeSTT([]), tts), _config())
    packets = _count_audio_packets(bytes(writer.written))
    audio_bytes = len(_decode_audio_payload(bytes(writer.written)))
    frames = audio_bytes // AUDIOSOCKET_CHUNK_20MS_BYTES
    assert frames >= 4, "echantillon trop court pour le test"
    assert packets == frames, (
        f"{packets} paquets pour {frames} trames : Asterisk exige du 20 ms"
    )


async def test_no_packet_exceeds_the_20ms_frame_size():
    tts = _FakeTTS()
    session = _FakeSession()
    writer = await _run(encode_hangup(), session, _deps(_FakeSTT([]), tts), _config())
    for size in _audio_payload_sizes(bytes(writer.written)):
        assert size == AUDIOSOCKET_CHUNK_20MS_BYTES, (
            f"paquet de {size} octets : Asterisk n accepte que des trames de 20 ms"
        )


def _count_audio_packets(written: bytes) -> int:
    return len(_audio_payload_sizes(written))


def _audio_payload_sizes(written: bytes) -> list[int]:
    sizes = []
    pos = 0
    while pos + 3 <= len(written):
        msg_type = written[pos]
        (length,) = struct.unpack(">H", written[pos + 1 : pos + 3])
        if msg_type == MessageType.AUDIO_PCM_8K:
            sizes.append(length)
        pos += 3 + length
    return sizes


async def test_playback_warns_when_it_falls_behind_realtime():
    """Si l'envoi ne tient pas la cadence, Asterisk recoit par a-coups et
    grésille. On veut le savoir dans les logs plutot que de le deviner."""
    import logging

    from hikky.adapters.telephony import asterisk_audiosocket_server as srv

    records = []

    class _Capture(logging.Handler):
        def emit(self, record):
            records.append(record.getMessage())

    handler = _Capture()
    srv.logger.addHandler(handler)
    try:
        adapter = AsteriskAudioSocketAdapter()
        writer = _CapturingWriter()
        adapter.bind_call(CALL_ID, writer)

        async def _slow(call_id, frames):
            await asyncio.sleep(0.15)  # bien au-dela du budget des trames
            for f in frames:
                writer.write(encode_audio(f))

        adapter.send_audio_frames = _slow
        config = _config(realtime_playback=True, frames_per_batch=2)
        await srv._speak(
            adapter, CALL_ID, "bonjour", _deps(_FakeSTT([]), _FakeTTS()), config
        )
    finally:
        srv.logger.removeHandler(handler)

    assert any("retard" in m.lower() for m in records), records


async def test_speak_sends_audio_before_the_tts_stream_is_exhausted():
    """XTTS livre son premier morceau en 364 ms mais met 4152 ms pour une
    phrase entiere. `_speak` attendait TOUT : le flux ne servait a rien.
    """
    import asyncio

    from hikky.adapters.telephony import asterisk_audiosocket_server as srv

    derniere_synthese = []
    premier_envoi = []

    class _SlowTTS:
        sample_rate = 24000

        async def synthesize(self, text):
            async def _stream():
                for i in range(4):
                    await asyncio.sleep(0.05)
                    derniere_synthese.append(i)
                    yield struct.pack("<2400h", *([1500] * 2400))

            return _stream()

    adapter = AsteriskAudioSocketAdapter()
    writer = _CapturingWriter()
    adapter.bind_call(CALL_ID, writer)
    original = adapter.send_audio_frames

    async def _watch(call_id, frames):
        if not premier_envoi:
            premier_envoi.append(len(derniere_synthese))
        await original(call_id, frames)

    adapter.send_audio_frames = _watch

    deps = _deps(_FakeSTT([]), _SlowTTS())
    await srv._speak(adapter, CALL_ID, "bonjour", deps, _config(realtime_playback=False))

    assert premier_envoi, "aucun audio envoye"
    assert premier_envoi[0] < 4, (
        f"premier envoi apres {premier_envoi[0]}/4 morceaux : tout est bufferise"
    )
