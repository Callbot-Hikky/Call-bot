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
