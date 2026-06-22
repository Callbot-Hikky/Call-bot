from hikky.ports.language_model import LanguageModelPort


class FakeLanguageModel(LanguageModelPort):
    def __init__(self, replies: list[str]) -> None:
        self._replies = list(replies)
        self.calls: list[list[dict[str, str]]] = []

    async def complete(self, messages: list[dict[str, str]]) -> str:
        self.calls.append(messages)
        if not self._replies:
            raise RuntimeError("FakeLanguageModel ran out of scripted replies")
        return self._replies.pop(0)
