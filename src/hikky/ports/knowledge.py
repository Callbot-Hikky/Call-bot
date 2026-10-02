from abc import ABC, abstractmethod

from hikky.domain.knowledge import KnowledgePassage


class KnowledgePort(ABC):
    @abstractmethod
    async def search(self, question: str) -> list[KnowledgePassage]:
        """Passages les plus proches de la question, les meilleurs d'abord.

        Ne lève jamais : une base injoignable vaut une base vide. Un appel
        ne doit pas tomber parce qu'une réponse annexe est indisponible.
        """

    @abstractmethod
    async def report_unanswered(self, question: str) -> None:
        """Signale une question restée sans réponse. Ne lève jamais."""
