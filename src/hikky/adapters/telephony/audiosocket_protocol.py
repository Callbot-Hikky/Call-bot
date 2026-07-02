"""Parseur/sérializer du protocole Asterisk AudioSocket.

Référence : https://docs.asterisk.org/Configuration/Channel-Drivers/AudioSocket/

Chaque paquet AudioSocket = header 3 octets + payload variable :

    [type: 1B] [length: 2B big-endian] [payload: length bytes]

Types utilisés :

- 0x00  Hangup           payload vide
- 0x01  UUID             payload = 16 octets binaires (UUID de l'appel)
- 0x03  DTMF             payload = 1 octet ASCII ('0'..'9', '*', '#', ...)
- 0x10  Audio PCM 8 kHz  payload = PCM signed 16-bit mono little-endian
- 0xff  Error            payload = code d'erreur applicatif (optionnel)

On implémente uniquement 0x00, 0x01, 0x03, 0x10 et 0xff — les sample rates
supérieurs (0x11..0x18) ne sont pas exposés par l'application dialplan
AudioSocket() en Asterisk 22 (8 kHz par défaut).

Ce module ne gère PAS le socket TCP — il ne fait que (dé)coder des paquets.
Le serveur asyncio vit dans `asterisk_audiosocket_server.py`.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from enum import IntEnum
from uuid import UUID

# Sample rate de l'AudioSocket "de base" (application dialplan Asterisk 22).
# À passer à Pipecat/Piper via `sample_rate` pour éviter tout resampling à tort.
AUDIOSOCKET_SAMPLE_RATE_HZ = 8000

# Un chunk PCM 20 ms à 8 kHz mono 16-bit signed = 320 octets. C'est l'unité
# usuelle d'un paquet audio en téléphonie — utile pour dimensionner les
# buffers et découper les réponses TTS en paquets envoyés progressivement.
AUDIOSOCKET_CHUNK_20MS_BYTES = 320


class MessageType(IntEnum):
    HANGUP = 0x00
    UUID_ID = 0x01
    DTMF = 0x03
    AUDIO_PCM_8K = 0x10
    ERROR = 0xFF


# ── Frames décodées ──────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class HangupFrame:
    pass


@dataclass(frozen=True, slots=True)
class UuidFrame:
    call_uuid: UUID


@dataclass(frozen=True, slots=True)
class DTMFFrame:
    digit: str  # un seul caractère ASCII


@dataclass(frozen=True, slots=True)
class AudioFrame:
    audio: bytes  # PCM signed 16-bit mono little-endian, 8 kHz


@dataclass(frozen=True, slots=True)
class ErrorFrame:
    code: int  # code d'erreur applicatif, 0 si payload absent


InboundFrame = HangupFrame | UuidFrame | DTMFFrame | AudioFrame | ErrorFrame


# ── Encodage / décodage ──────────────────────────────────────────────────────

_HEADER_LEN = 3


class ProtocolError(ValueError):
    """Payload AudioSocket mal formé (header incomplet, longueur incohérente...)."""


def decode_packet(packet: bytes) -> InboundFrame:
    """Décode un paquet AudioSocket complet (header + payload).

    Lève `ProtocolError` si le paquet est incomplet ou incohérent.
    """
    if len(packet) < _HEADER_LEN:
        raise ProtocolError(
            f"packet shorter than header: got {len(packet)} bytes, need >= {_HEADER_LEN}"
        )
    msg_type = packet[0]
    (payload_len,) = struct.unpack(">H", packet[1:3])
    payload = packet[_HEADER_LEN : _HEADER_LEN + payload_len]
    if len(payload) != payload_len:
        raise ProtocolError(
            f"payload truncated: header says {payload_len} bytes, got {len(payload)}"
        )

    if msg_type == MessageType.HANGUP:
        return HangupFrame()
    if msg_type == MessageType.UUID_ID:
        if payload_len != 16:
            raise ProtocolError(f"UUID payload must be 16 bytes, got {payload_len}")
        return UuidFrame(call_uuid=UUID(bytes=payload))
    if msg_type == MessageType.DTMF:
        if payload_len != 1:
            raise ProtocolError(f"DTMF payload must be 1 byte, got {payload_len}")
        return DTMFFrame(digit=payload.decode("ascii"))
    if msg_type == MessageType.AUDIO_PCM_8K:
        return AudioFrame(audio=payload)
    if msg_type == MessageType.ERROR:
        code = payload[0] if payload_len >= 1 else 0
        return ErrorFrame(code=code)
    raise ProtocolError(f"unknown message type 0x{msg_type:02x}")


def encode_audio(audio: bytes) -> bytes:
    """Emballe un buffer PCM 8 kHz mono 16-bit signed en paquet AudioSocket.

    Longueur max du payload = 65535 (2 octets big-endian). Un buffer plus
    long est refusé — appelant doit le découper en amont (usuellement en
    chunks de 20 ms = 320 octets, mais AudioSocket accepte tout ce qui
    tient dans un uint16).
    """
    if len(audio) > 0xFFFF:
        raise ProtocolError(
            f"audio payload too large for a single AudioSocket packet: {len(audio)} > 65535"
        )
    header = struct.pack(">BH", MessageType.AUDIO_PCM_8K, len(audio))
    return header + audio


def encode_hangup() -> bytes:
    """Demande à Asterisk de raccrocher (le socket sera fermé côté Asterisk)."""
    return struct.pack(">BH", MessageType.HANGUP, 0)


def encode_error(code: int = 0) -> bytes:
    """Signale une erreur applicative à Asterisk.

    En pratique on ferme juste le socket ; ce helper existe pour la
    complétude du protocole et pour les tests.
    """
    if not 0 <= code <= 0xFF:
        raise ProtocolError(f"error code must fit in one byte, got {code}")
    if code == 0:
        return struct.pack(">BH", MessageType.ERROR, 0)
    return struct.pack(">BH", MessageType.ERROR, 1) + bytes([code])


async def read_packet(reader) -> InboundFrame:
    """Lit un paquet AudioSocket depuis un `asyncio.StreamReader`.

    Renvoie la frame décodée. Lève `asyncio.IncompleteReadError` si le
    peer ferme le socket au milieu d'un paquet — l'appelant doit traiter
    ça comme une fin d'appel.
    """
    header = await reader.readexactly(_HEADER_LEN)
    (payload_len,) = struct.unpack(">H", header[1:3])
    payload = await reader.readexactly(payload_len) if payload_len else b""
    return decode_packet(header + payload)
