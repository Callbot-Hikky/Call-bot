"""Exceptions de domaine remontées par les ports."""


class HikkyDomainError(Exception):
    """Base des erreurs de domaine."""


class BackUnavailable(HikkyDomainError):
    """Le Back ne répond pas ou répond en erreur."""


class STTTimeout(HikkyDomainError):
    """Le STT n'a rien transcrit dans le délai imparti."""


class LLMOverloaded(HikkyDomainError):
    """Le LLM dépasse le timeout dur."""


class UnknownRestaurant(HikkyDomainError):
    """Le numéro appelé n'est rattaché à aucun restaurant."""


class TelephonyError(HikkyDomainError):
    """Erreur d'I/O côté téléphonie (WebSocket coupée, etc.)."""
