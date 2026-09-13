"""Tests unitaires de `AsteriskAudioSocketAdapter`.

On mocke le `asyncio.StreamWriter` — pas de vrai socket. Ces tests valident
que l'adapter émet les bons paquets AudioSocket et gère bien le
cycle de vie des appels.
"""

from __future__ import annotations

import pytest

from hikky.adapters.telephony.asterisk_audiosocket_adapter import AsteriskAudioSocketAdapter
from hikky.adapters.telephony.audiosocket_protocol import (
    AudioFrame,
    HangupFrame,
    decode_packet,
)
from hikky.exceptions import TelephonyError


class FakeWriter:
    """Faux `asyncio.StreamWriter` qui accumule les octets écrits."""

    def __init__(self) -> None:
        self.written = bytearray()
        self.drained = 0
        self.closed = False
        self.wait_closed_called = False

    def write(self, data: bytes) -> None:
        if self.closed:
            raise RuntimeError("writer is closed")
        self.written.extend(data)

    async def drain(self) -> None:
        self.drained += 1

    def close(self) -> None:
        self.closed = True

    async def wait_closed(self) -> None:
        self.wait_closed_called = True


async def test_send_audio_encodes_pcm_and_writes_on_bound_writer():
    adapter = AsteriskAudioSocketAdapter()
    writer = FakeWriter()
    adapter.bind_call("c-1", writer)

    pcm = b"\x00\x01\x02\x03"
    await adapter.send_audio("c-1", pcm)

    assert writer.drained == 1
    packet = bytes(writer.written)
    frame = decode_packet(packet)
    assert isinstance(frame, AudioFrame)
    assert frame.audio == pcm


async def test_send_audio_on_unknown_call_raises():
    adapter = AsteriskAudioSocketAdapter()
    with pytest.raises(TelephonyError):
        await adapter.send_audio("ghost", b"\x00\x01")


async def test_hang_up_sends_hangup_packet_and_closes_writer():
    adapter = AsteriskAudioSocketAdapter()
    writer = FakeWriter()
    adapter.bind_call("c-1", writer)

    await adapter.hang_up("c-1")

    assert writer.closed is True
    assert writer.wait_closed_called is True
    frame = decode_packet(bytes(writer.written))
    assert isinstance(frame, HangupFrame)

    # Call est délié : un send_audio ultérieur lève
    with pytest.raises(TelephonyError):
        await adapter.send_audio("c-1", b"\x00\x00")


async def test_hang_up_on_unknown_call_is_a_noop():
    adapter = AsteriskAudioSocketAdapter()
    await adapter.hang_up("ghost")  # ne lève pas


async def test_transfer_is_not_supported_by_audiosocket_protocol():
    adapter = AsteriskAudioSocketAdapter()
    with pytest.raises(NotImplementedError):
        await adapter.transfer("c-1", "+33100000099")


async def test_multiple_calls_are_isolated():
    adapter = AsteriskAudioSocketAdapter()
    w1 = FakeWriter()
    w2 = FakeWriter()
    adapter.bind_call("c-1", w1)
    adapter.bind_call("c-2", w2)

    await adapter.send_audio("c-1", b"\xaa\xbb")
    await adapter.send_audio("c-2", b"\xcc\xdd")

    frame1 = decode_packet(bytes(w1.written))
    frame2 = decode_packet(bytes(w2.written))
    assert isinstance(frame1, AudioFrame) and frame1.audio == b"\xaa\xbb"
    assert isinstance(frame2, AudioFrame) and frame2.audio == b"\xcc\xdd"


async def test_unbind_call_removes_state():
    adapter = AsteriskAudioSocketAdapter()
    writer = FakeWriter()
    adapter.bind_call("c-1", writer)
    adapter.unbind_call("c-1")

    with pytest.raises(TelephonyError):
        await adapter.send_audio("c-1", b"\x00")
    # unbind idempotent
    adapter.unbind_call("c-1")


# ── Envoi groupe : trames de 20 ms, un seul drain ───────────────────────
#
# Asterisk attend des trames de 20 ms : un paquet plus gros casse la
# lecture. Mais un drain reseau toutes les 20 ms a travers un tunnel
# distant coute ~100 ms par trame, soit 28 s pour 4 s de parole. On
# ecrit donc plusieurs trames d affilee, avec un seul drain.

class _CountingWriter:
    def __init__(self) -> None:
        self.written = bytearray()
        self.drains = 0

    def write(self, data: bytes) -> None:
        self.written.extend(data)

    async def drain(self) -> None:
        self.drains += 1

    def close(self) -> None:
        return None

    async def wait_closed(self) -> None:
        return None


async def test_send_frames_emits_one_packet_per_frame():
    import struct

    from hikky.adapters.telephony.audiosocket_protocol import MessageType

    adapter = AsteriskAudioSocketAdapter()
    writer = _CountingWriter()
    adapter.bind_call("c1", writer)
    frames = [b"\x01\x02" * 160 for _ in range(5)]
    await adapter.send_audio_frames("c1", frames)

    data = bytes(writer.written)
    pos, count = 0, 0
    while pos + 3 <= len(data):
        msg_type = data[pos]
        (length,) = struct.unpack(">H", data[pos + 1 : pos + 3])
        if msg_type == MessageType.AUDIO_PCM_8K:
            assert length == 320, f"trame de {length} octets, Asterisk attend 320"
            count += 1
        pos += 3 + length
    assert count == 5


async def test_send_frames_drains_once_for_the_whole_batch():
    adapter = AsteriskAudioSocketAdapter()
    writer = _CountingWriter()
    adapter.bind_call("c1", writer)
    await adapter.send_audio_frames("c1", [b"\x00" * 320 for _ in range(25)])
    assert writer.drains == 1, f"{writer.drains} drains pour 25 trames"


async def test_send_frames_on_unknown_call_raises():
    import pytest

    from hikky.exceptions import TelephonyError

    adapter = AsteriskAudioSocketAdapter()
    with pytest.raises(TelephonyError):
        await adapter.send_audio_frames("inconnu", [b"\x00" * 320])
