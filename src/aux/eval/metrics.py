"""Retrieval metrics and space diagnostics.

In the package rather than in a script because every later slice is judged with these, and
a metric that is only defined inside a one-off script cannot be unit-tested or reused.

Ranks are **1-based** throughout: the correct item ranked first has rank 1, so Recall@1 is
`rank <= 1` and MRR is `mean(1/rank)`.
"""

from __future__ import annotations

from collections import Counter

import numpy as np


def ranks_of_truth(scores: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """1-based rank of each query's correct item under descending score.

    Ties are broken by argsort order rather than optimistically or pessimistically. With
    float cosine scores exact ties are vanishingly rare; if a future encoder produces
    quantised scores this needs revisiting, because optimistic tie-breaking silently
    inflates Recall@1.
    """
    order = np.argsort(-scores, axis=1)
    return (order == np.asarray(truth)[:, None]).argmax(axis=1) + 1


def retrieval_metrics(ranks: np.ndarray, n_candidates: int) -> dict:
    """Recall@K, rank statistics and MRR for single-relevant-item retrieval."""
    ranks = np.asarray(ranks, dtype=float)
    if ranks.size == 0:
        raise ValueError("no ranks to score")
    return {
        "n_queries": int(ranks.size),
        "n_candidates": int(n_candidates),
        "recall@1": float((ranks <= 1).mean()),
        "recall@5": float((ranks <= 5).mean()),
        "recall@10": float((ranks <= 10).mean()),
        "median_rank": float(np.median(ranks)),
        "mean_rank": float(ranks.mean()),
        "mrr": float((1.0 / ranks).mean()),
    }


def random_baseline(n_candidates: int) -> dict:
    """What the same task yields by chance.

    Reported next to every measured number so that "above random" is arithmetic rather
    than an assertion -- on a 706-candidate benchmark, chance Recall@10 is already 1.4%.
    """
    return {
        "recall@1": 1 / n_candidates,
        "recall@5": 5 / n_candidates,
        "recall@10": 10 / n_candidates,
        "median_rank": (n_candidates + 1) / 2,
    }


def space_diagnostics(
    vectors: np.ndarray,
    top_k_indices: np.ndarray,
    segment_cosines: list[float] | None = None,
) -> dict:
    """Health checks on the embedding space itself, independent of retrieval quality.

    Two Slice 0 risks are only visible here:

    - **collapse** -- if track-track cosine is high with little spread, the encoder is not
      separating this corpus. That is a representation failure and no amount of ranking
      work recovers it, so it must be distinguished from a merely poor recall number.
    - **hubness** -- contrastive spaces reliably produce a few vectors that are nearest
      neighbour to almost everything. A hub caps achievable recall while presenting as a
      ranking problem.
    """
    n = vectors.shape[0]
    sims = vectors @ vectors.T
    off = ~np.eye(n, dtype=bool)

    counts = Counter(int(i) for i in np.asarray(top_k_indices).reshape(-1))
    appearances = np.array([counts.get(i, 0) for i in range(n)], dtype=float)
    total = appearances.sum() or 1.0
    share = np.sort(appearances)[::-1] / total

    return {
        "track_cosine_mean": float(sims[off].mean()),
        "track_cosine_std": float(sims[off].std()),
        "track_cosine_p95": float(np.percentile(sims[off], 95)),
        "hubness_top1pct_share": float(share[: max(1, n // 100)].sum()),
        "hubness_max_track_share": float(share[0]),
        "tracks_never_retrieved": int((appearances == 0).sum()),
        "within_track_segment_cosine_mean": (
            float(np.mean(segment_cosines)) if segment_cosines else None
        ),
        "within_track_segment_cosine_min": (
            float(np.min(segment_cosines)) if segment_cosines else None
        ),
    }
