from abc import ABC, abstractmethod


class LanguageModelPort(ABC):
    @abstractmethod
    async def complete(self, messages: list[dict[str, str]]) -> str:
        """messages: list of {role, content}. Renvoie la réponse complète."""
