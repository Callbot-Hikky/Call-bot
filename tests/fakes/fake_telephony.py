from hikky.ports.telephony import TelephonyPort


class FakeTelephony(TelephonyPort):
    def __init__(self) -> None:
        self.actions: list[tuple] = []

    async def send_audio(self, call_id: str, audio_chunk: bytes) -> None:
        self.actions.append(("send_audio", call_id, len(audio_chunk)))

    async def transfer(self, call_id: str, destination_number: str) -> None:
        self.actions.append(("transfer", call_id, destination_number))

    async def hang_up(self, call_id: str) -> None:
        self.actions.append(("hang_up", call_id))
