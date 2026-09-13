"""Choosing the fusion weight from the query.

The evaluation found that the best weight is not a constant. Track-to-track similarity and
genre-style queries want pure audio (NDCG@10 0.832 against 0.555 for lyrics); "songs about X"
wants pure lyrics (0.734 against 0.367). Applying either family's best weight to the other
costs 0.28-0.37, so a system serving both needs to decide per query.

An oracle allowed to pick the better modality per query scores +0.042 over the best single
weight. That is the ceiling for everything in this package, and it is deliberately small: a
router is worth building only if it captures most of it for less than it costs. Three
implementations are measured against that bound rather than one being assumed best, which is
the same treatment the query planner got — and the planner lost, and is off by default.
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
