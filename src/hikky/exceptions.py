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


class ReservationConflict(HikkyDomainError):
    """Le Back refuse la réservation pour conflit (HTTP 409).

    Avec une clé d'idempotence désormais unique par appel, un 409 n'est plus
    une collision de clé : c'est un vrai conflit métier (créneau qui vient
    d'être pris, réservation déjà en attente pour ce client…). Le bot doit
    l'annoncer honnêtement, PAS le présenter comme un succès.
    """


class TelephonyError(HikkyDomainError):
    """Erreur d'I/O côté téléphonie (WebSocket coupée, etc.)."""
