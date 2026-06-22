from abc import ABC, abstractmethod


class TelephonyPort(ABC):
    @abstractmethod
    async def send_audio(self, call_id: str, audio_chunk: bytes) -> None: ...

    @abstractmethod
    async def transfer(self, call_id: str, destination_number: str) -> None: ...

    @abstractmethod
    async def hang_up(self, call_id: str) -> None: ...
