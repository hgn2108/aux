"""Combining two rankings with reciprocal rank fusion.

    RRF(track) = sum over systems of 1 / (k + rank)

Fused on rank, not score: the two rankings come from different query embeddings, so their
cosines are not on a common scale and averaging would weight whichever spreads wider.

`k` damps the top, or one first place would dominate. 60 is the value from the original
paper; tuning it on 18 pairs would fit the evaluation, not the problem.

Superseded by the weighted blend in `aux.recommend`, which RRF cannot do: it weights its
inputs equally by construction, so there is nothing to ablate.
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
