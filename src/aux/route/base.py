"""Choosing the blend weight from the query.

The best weight is not a constant: similarity queries want audio, "songs about X" wants
lyrics, and using the wrong one costs 0.28-0.37 NDCG@10. An oracle picking per query gains
only +0.042 over the best fixed weight, so that is the ceiling here. Three implementations
are measured against it; none is assumed best.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Route:
    """A routing decision.

    `alpha` is the audio share, matching `Recommender.score`: 1.0 is audio-only, 0.0 is
    lyrics-only. `reason` is displayed to the user, so it must be phrased for a listener
    rather than naming internals.
    """

    alpha: float
    reason: str
    router: str

    def __post_init__(self) -> None:
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {self.alpha}")

    @property
    def label(self) -> str:
        """How the choice reads in the interface."""
        if self.alpha >= 0.85:
            return "sound"
        if self.alpha <= 0.15:
            return "lyrics"
        return "sound and lyrics"


class Router(ABC):
    """Maps a text query to a fusion weight."""

    name: str = "router"

    @abstractmethod
    def route(self, query: str) -> Route:
        ...

    def route_all(self, queries: list[str]) -> list[Route]:
        """Overridden where batching is cheaper than one call per query."""
        return [self.route(q) for q in queries]
