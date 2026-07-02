"""Tests unitaires du (dé)codage du protocole AudioSocket.

Ces tests sont "unit" par nature (pas d'I/O, pas de socket), mais on
les range dans `integration/telephony` par symétrie avec `test_twilio_protocol.py`.
"""

from __future__ import annotations

import asyncio
import io
import struct
from uuid import UUID

import pytest

from hikky.adapters.telephony.audiosocket_protocol import (
    AUDIOSOCKET_CHUNK_20MS_BYTES,
    AUDIOSOCKET_SAMPLE_RATE_HZ,
    AudioFrame,
    DTMFFrame,
    ErrorFrame,
    HangupFrame,
    MessageType,
    ProtocolError,
    UuidFrame,
    decode_packet,
    encode_audio,
    encode_error,
    encode_hangup,
    read_packet,
)


# ── Décodage ────────────────────────────────────────────────────────────────


def test_decode_hangup_packet():
    packet = bytes([MessageType.HANGUP, 0x00, 0x00])
    frame = decode_packet(packet)
    assert isinstance(frame, HangupFrame)


def test_decode_uuid_packet_returns_uuid_object():
    call_uuid = UUID("11112222-3333-4444-5555-666677778888")
    packet = bytes([MessageType.UUID_ID, 0x00, 0x10]) + call_uuid.bytes
    frame = decode_packet(packet)
    assert isinstance(frame, UuidFrame)
    assert frame.call_uuid == call_uuid


def test_decode_uuid_packet_rejects_wrong_length():
    packet = bytes([MessageType.UUID_ID, 0x00, 0x08]) + b"\x00" * 8
    with pytest.raises(ProtocolError):
        decode_packet(packet)


def test_decode_dtmf_packet():
    packet = bytes([MessageType.DTMF, 0x00, 0x01]) + b"5"
    frame = decode_packet(packet)
    assert isinstance(frame, DTMFFrame)
    assert frame.digit == "5"


def test_decode_dtmf_packet_rejects_wrong_length():
    packet = bytes([MessageType.DTMF, 0x00, 0x02]) + b"55"
    with pytest.raises(ProtocolError):
        decode_packet(packet)


def test_decode_audio_packet_returns_raw_pcm_bytes():
    pcm = b"\x00\x01\x02\x03\xff\xfe"
    packet = bytes([MessageType.AUDIO_PCM_8K, 0x00, 0x06]) + pcm
    frame = decode_packet(packet)
    assert isinstance(frame, AudioFrame)
    assert frame.audio == pcm


def test_decode_empty_audio_packet_is_valid():
    packet = bytes([MessageType.AUDIO_PCM_8K, 0x00, 0x00])
    frame = decode_packet(packet)
    assert isinstance(frame, AudioFrame)
    assert frame.audio == b""


def test_decode_error_packet_with_code():
    packet = bytes([MessageType.ERROR, 0x00, 0x01, 0x2A])
    frame = decode_packet(packet)
    assert isinstance(frame, ErrorFrame)
    assert frame.code == 0x2A


def test_decode_error_packet_without_payload_yields_zero_code():
    packet = bytes([MessageType.ERROR, 0x00, 0x00])
    frame = decode_packet(packet)
    assert isinstance(frame, ErrorFrame)
    assert frame.code == 0


def test_decode_short_packet_raises():
    with pytest.raises(ProtocolError):
        decode_packet(b"\x10\x00")  # 2 octets < header


def test_decode_truncated_payload_raises():
    # Header dit "10 octets" mais on n'en fournit que 3
    packet = bytes([MessageType.AUDIO_PCM_8K, 0x00, 0x0A]) + b"\x00\x01\x02"
    with pytest.raises(ProtocolError):
        decode_packet(packet)


def test_decode_unknown_type_raises():
    packet = bytes([0x77, 0x00, 0x00])
    with pytest.raises(ProtocolError):
        decode_packet(packet)


# ── Encodage ────────────────────────────────────────────────────────────────


def test_encode_audio_round_trip():
    pcm = bytes(range(256))
    packet = encode_audio(pcm)
    assert packet[0] == MessageType.AUDIO_PCM_8K
    (declared_len,) = struct.unpack(">H", packet[1:3])
    assert declared_len == len(pcm)
    assert packet[3:] == pcm

    frame = decode_packet(packet)
    assert isinstance(frame, AudioFrame)
    assert frame.audio == pcm


def test_encode_audio_rejects_oversized_buffer():
    huge = b"\x00" * (0xFFFF + 1)
    with pytest.raises(ProtocolError):
        encode_audio(huge)


def test_encode_hangup_round_trip():
    packet = encode_hangup()
    assert packet == bytes([MessageType.HANGUP, 0x00, 0x00])
    assert isinstance(decode_packet(packet), HangupFrame)


def test_encode_error_round_trip_with_code():
    packet = encode_error(0x2A)
    frame = decode_packet(packet)
    assert isinstance(frame, ErrorFrame)
    assert frame.code == 0x2A


def test_encode_error_round_trip_without_code():
    packet = encode_error(0)
    frame = decode_packet(packet)
    assert isinstance(frame, ErrorFrame)
    assert frame.code == 0


def test_encode_error_rejects_out_of_range_code():
    with pytest.raises(ProtocolError):
        encode_error(-1)
    with pytest.raises(ProtocolError):
        encode_error(256)


# ── read_packet (asyncio) ───────────────────────────────────────────────────


class _FakeReader:
    """Faux `asyncio.StreamReader` alimenté par un buffer en mémoire."""

    def __init__(self, data: bytes) -> None:
        self._buf = io.BytesIO(data)

    async def readexactly(self, n: int) -> bytes:
        chunk = self._buf.read(n)
        if len(chunk) != n:
            raise asyncio.IncompleteReadError(chunk, n)
        return chunk


async def test_read_packet_sequence():
    call_uuid = UUID("11112222-3333-4444-5555-666677778888")
    pcm = b"\x00\x01\x02"
    stream = (
        # UUID
        bytes([MessageType.UUID_ID, 0x00, 0x10]) + call_uuid.bytes
        # Audio chunk
        + bytes([MessageType.AUDIO_PCM_8K, 0x00, 0x03]) + pcm
        # Hangup
        + bytes([MessageType.HANGUP, 0x00, 0x00])
    )
    reader = _FakeReader(stream)

    frame1 = await read_packet(reader)
    frame2 = await read_packet(reader)
    frame3 = await read_packet(reader)

    assert isinstance(frame1, UuidFrame)
    assert frame1.call_uuid == call_uuid
    assert isinstance(frame2, AudioFrame)
    assert frame2.audio == pcm
    assert isinstance(frame3, HangupFrame)


async def test_read_packet_raises_incomplete_read_on_socket_close_mid_header():
    reader = _FakeReader(b"\x10\x00")  # header incomplet (2 octets)
    with pytest.raises(asyncio.IncompleteReadError):
        await read_packet(reader)


async def test_read_packet_raises_incomplete_read_on_socket_close_mid_payload():
    # Header annonce 100 octets, on n'en fournit que 3
    reader = _FakeReader(bytes([MessageType.AUDIO_PCM_8K, 0x00, 0x64]) + b"\x00\x00\x00")
    with pytest.raises(asyncio.IncompleteReadError):
        await read_packet(reader)


# ── Constantes ──────────────────────────────────────────────────────────────


def test_sample_rate_constant_matches_asterisk_default():
    assert AUDIOSOCKET_SAMPLE_RATE_HZ == 8000


def test_chunk_20ms_size_matches_8khz_16bit_mono():
    # 20 ms * 8000 samples/s * 2 bytes/sample * 1 channel = 320 bytes
    assert AUDIOSOCKET_CHUNK_20MS_BYTES == 320
