"""Ce que le restaurateur a écrit ou validé, retrouvé pour une question."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class KnowledgePassage:
    """Un passage de la base de connaissances du restaurant.

    `score` est une similarité entre 0 et 1, calculée par le backend : plus
    il est haut, plus le passage ressemble à la question posée.
    """

    title: str
    content: str
    score: float
