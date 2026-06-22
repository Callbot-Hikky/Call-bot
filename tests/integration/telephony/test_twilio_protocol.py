import base64

import pytest

from hikky.adapters.telephony.twilio_protocol import (
    ConnectedFrame,
    MediaFrame,
    StartFrame,
    StopFrame,
    decode_inbound,
    encode_outbound_media,
)


def test_decode_connected_event():
    frame = decode_inbound({"event": "connected", "protocol": "Call", "version": "1.0.0"})
    assert isinstance(frame, ConnectedFrame)


def test_decode_start_event_extracts_stream_and_call_sids():
    payload = {
        "event": "start",
        "sequenceNumber": "1",
        "start": {
            "streamSid": "MZxxx",
            "callSid": "CAxxx",
            "tracks": ["inbound"],
            "customParameters": {"restaurant_id": "r-1"},
            "mediaFormat": {"encoding": "audio/x-mulaw", "sampleRate": 8000, "channels": 1},
        },
        "streamSid": "MZxxx",
    }
    frame = decode_inbound(payload)
    assert isinstance(frame, StartFrame)
    assert frame.stream_sid == "MZxxx"
    assert frame.call_sid == "CAxxx"
    assert frame.custom_parameters == {"restaurant_id": "r-1"}


def test_decode_media_event_decodes_base64_payload():
    audio = b"\x00\x01\x02\x03"
    payload = {
        "event": "media",
        "sequenceNumber": "2",
        "media": {
            "track": "inbound",
            "chunk": "1",
            "timestamp": "100",
            "payload": base64.b64encode(audio).decode(),
        },
        "streamSid": "MZxxx",
    }
    frame = decode_inbound(payload)
    assert isinstance(frame, MediaFrame)
    assert frame.audio == audio
    assert frame.stream_sid == "MZxxx"


def test_decode_stop_event():
    payload = {
        "event": "stop",
        "sequenceNumber": "5",
        "stop": {"accountSid": "ACxxx", "callSid": "CAxxx"},
        "streamSid": "MZxxx",
    }
    frame = decode_inbound(payload)
    assert isinstance(frame, StopFrame)
    assert frame.stream_sid == "MZxxx"


def test_decode_unknown_event_raises():
    with pytest.raises(ValueError):
        decode_inbound({"event": "mystery"})


def test_encode_outbound_media_produces_expected_json_shape():
    audio = b"\xaa\xbb"
    out = encode_outbound_media("MZxxx", audio)
    assert out["event"] == "media"
    assert out["streamSid"] == "MZxxx"
    assert base64.b64decode(out["media"]["payload"]) == audio
