import base64
import json

import pytest

from hikky.adapters.telephony.twilio_adapter import TwilioMediaStreamsAdapter
from hikky.exceptions import TelephonyError


class FakeWebSocket:
    def __init__(self) -> None:
        self.sent: list[str] = []
        self.closed = False

    async def send_text(self, data: str) -> None:
        if self.closed:
            raise RuntimeError("WebSocket is closed")
        self.sent.append(data)

    async def close(self, code: int = 1000) -> None:
        self.closed = True


async def test_send_audio_encodes_and_sends_on_bound_ws():
    adapter = TwilioMediaStreamsAdapter()
    ws = FakeWebSocket()
    adapter.bind_call("c-1", ws, stream_sid="MZxxx")

    audio = b"\xaa\xbb"
    await adapter.send_audio("c-1", audio)

    assert len(ws.sent) == 1
    payload = json.loads(ws.sent[0])
    assert payload["event"] == "media"
    assert payload["streamSid"] == "MZxxx"
    assert base64.b64decode(payload["media"]["payload"]) == audio


async def test_send_audio_on_unknown_call_raises():
    adapter = TwilioMediaStreamsAdapter()
    with pytest.raises(TelephonyError):
        await adapter.send_audio("unknown", b"\x00")


async def test_hang_up_sends_clear_and_closes_ws():
    adapter = TwilioMediaStreamsAdapter()
    ws = FakeWebSocket()
    adapter.bind_call("c-1", ws, stream_sid="MZxxx")

    await adapter.hang_up("c-1")

    assert ws.closed is True
    assert len(ws.sent) == 1
    assert json.loads(ws.sent[0])["event"] == "clear"
    # Call est délié, un send_audio ultérieur lèverait
    with pytest.raises(TelephonyError):
        await adapter.send_audio("c-1", b"\x00")


async def test_hang_up_on_unknown_call_is_a_noop():
    adapter = TwilioMediaStreamsAdapter()
    await adapter.hang_up("ghost")  # ne lève pas


async def test_transfer_is_not_implemented_yet():
    adapter = TwilioMediaStreamsAdapter()
    with pytest.raises(NotImplementedError):
        await adapter.transfer("c-1", "+33100000099")
