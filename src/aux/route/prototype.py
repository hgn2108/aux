"""The middle baseline: let the text encoder decide, with no rules and no API call.

If the embedding space already separates "describe a sound" from "describe a meaning", then
routing is free and needs neither a hand-written vocabulary that will miss phrasings nor a
model call that costs latency. This measures whether it does.

A query is embedded once and compared against two small sets of prototype queries, one per
family. The gap between the two mean similarities is mapped to a weight through a logistic,
so the output is graded rather than binary: a query sitting between the prototypes lands
between the weights, which is what a compound query should do.

**Prototypes are not the evaluation queries.** They are written separately and generically;
reusing evaluation queries as prototypes would measure memorisation.
"""

from __future__ import annotations

import numpy as np

from .base import Route, Router

ACOUSTIC_PROTOTYPES = (
    "music with a fast tempo and loud drums",
    "a quiet acoustic recording with sparse instrumentation",
    "bright synthesizers over a heavy bassline",
    "warm analogue production with live instruments",
    "a dense mix with distorted guitars",
    "slow ambient textures with long reverb tails",
)

SEMANTIC_PROTOTYPES = (
    "songs about losing someone you love",
    "lyrics about growing up in a difficult place",
    "tracks about celebrating success with friends",
    "songs whose words describe regret",
    "music about falling in love for the first time",
    "songs that tell a story about leaving home",
)

#: Controls how sharply the similarity gap maps to a weight. Fitted by hand on the prototype
#: sets alone, never on the evaluation queries: large enough that a clear query reaches the
#: ends of the range, small enough that a mixed query stays in the middle.
SHARPNESS = 40.0


class PrototypeRouter(Router):
    name = "prototype"

    def __init__(self, encoder) -> None:
        """`encoder` needs only `embed_text`, so either tower can drive this."""
        self.encoder = encoder
        vectors = encoder.embed_text(list(ACOUSTIC_PROTOTYPES) + list(SEMANTIC_PROTOTYPES))
        vectors = np.asarray(vectors, dtype=np.float32)
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        split = len(ACOUSTIC_PROTOTYPES)
        self.acoustic = vectors[:split]
        self.semantic = vectors[split:]

    def _alphas(self, queries: list[str]) -> np.ndarray:
        vectors = np.asarray(self.encoder.embed_text(queries), dtype=np.float32)
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        gap = (vectors @ self.acoustic.T).mean(1) - (vectors @ self.semantic.T).mean(1)
        return 1.0 / (1.0 + np.exp(-SHARPNESS * gap))

    def route(self, query: str) -> Route:
        return self.route_all([query])[0]

    def route_all(self, queries: list[str]) -> list[Route]:
        out = []
        for query, alpha in zip(queries, self._alphas(queries)):
            alpha = float(np.clip(alpha, 0.0, 1.0))
            if alpha >= 0.85:
                reason = "reads as a description of sound"
            elif alpha <= 0.15:
                reason = "reads as a description of meaning"
            else:
                reason = "reads as both sound and meaning"
            out.append(Route(alpha, reason, self.name))
        return out
