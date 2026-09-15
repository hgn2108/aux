"""Combining two rankings, DEC-013's rung 3.

E2 left a specific gap: the baseline is sharper at the very top (K=5) while the planner is
better from K=10 outward. If those are complementary rather than redundant, taking both
should beat either.

**Fused at rank level, not score level.** DESIGN.md fixed this and the reason applies
exactly here: the two rankings come from different query embeddings, so their cosine scores
are not on a common scale. Averaging them silently weights whichever has more spread.
Reciprocal rank fusion only looks at position, so it cannot be fooled that way.

    RRF(track) = sum over systems of 1 / (k + rank)

`k` damps the top: without it, a first place in one system would dominate everything else.
60 is the value from the original RRF paper and is left alone, tuning it on 18 pairs would
fit the evaluation rather than the problem.
"""

from __future__ import annotations

import numpy as np

RRF_K = 60


def reciprocal_rank_fusion(score_lists: list[np.ndarray], *, k: int = RRF_K,
                           weights: list[float] | None = None) -> np.ndarray:
    """Fuse several scorings of the same items into one, using ranks alone."""
    if not score_lists:
        raise ValueError("nothing to fuse")
    n = len(score_lists[0])
    if any(len(s) != n for s in score_lists):
        raise ValueError("all scorings must cover the same items")
    weights = weights or [1.0] * len(score_lists)
    if len(weights) != len(score_lists):
        raise ValueError("one weight per scoring")

    fused = np.zeros(n, dtype=float)
    for scores, weight in zip(score_lists, weights):
        # rank 1 is the highest score
        ranks = np.empty(n, dtype=float)
        ranks[np.argsort(-np.asarray(scores))] = np.arange(1, n + 1)
        fused += weight / (k + ranks)
    return fused
