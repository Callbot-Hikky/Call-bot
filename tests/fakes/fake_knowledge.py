from hikky.domain.knowledge import KnowledgePassage
from hikky.ports.knowledge import KnowledgePort


class FakeKnowledge(KnowledgePort):
    def __init__(self, passages: list[KnowledgePassage] | None = None) -> None:
        self._passages = list(passages or [])
        self.searched: list[str] = []
        self.reported: list[str] = []

    async def search(self, question: str) -> list[KnowledgePassage]:
        self.searched.append(question)
        return list(self._passages)

    async def report_unanswered(self, question: str) -> None:
        self.reported.append(question)
