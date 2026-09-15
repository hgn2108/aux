"""Graded-relevance metrics for human-rated retrieval.

Ratings are 1-5 where 1 means "not relevant at all", so gains are `rating - 1`. Five 1s
should score zero, not score for having returned five things.

Lives in the package rather than the script that used it, because a metric that only exists
inside a one-off script cannot be tested.
"""

from __future__ import annotations

import numpy as np

MAX_RATING = 5


def gains(ratings: np.ndarray) -> np.ndarray:
    """Map 1-5 ratings to 0-4 gains. A rating of 1 is worth nothing."""
    return np.clip(np.asarray(ratings, dtype=float) - 1.0, 0.0, None)


def dcg(ratings_in_rank_order: np.ndarray) -> float:
    """Discounted cumulative gain with exponential gain, ratings given in system order."""
    g = gains(ratings_in_rank_order)
    discounts = np.log2(np.arange(2, g.size + 2))
    return float(((2**g - 1) / discounts).sum())


def ndcg(ratings_in_rank_order: np.ndarray) -> float:
    """NDCG over one query's returned list.

    Note the scope: with only the system's own top-K rated, this measures whether the
    system ordered its own results well, not whether it found the right ones at all. A
    system returning five mediocre tracks in the perfect order scores 1.0. Mean relevance
    is the metric that speaks to the second question.
    """
    ideal = np.sort(np.asarray(ratings_in_rank_order))[::-1]
    best = dcg(ideal)
    return float(dcg(ratings_in_rank_order) / best) if best > 0 else 0.0


def success_at_k(ratings_in_rank_order: np.ndarray, k: int = 5, threshold: int = 4) -> bool:
    """Did at least one clearly-relevant result appear in the top k?

    The product question is closer to "did the user find something usable" than to "was the
    whole list good", so a threshold hit matters more than the average.
    """
    return bool((np.asarray(ratings_in_rank_order)[:k] >= threshold).any())


def random_ordering_ndcg(ratings: np.ndarray, trials: int = 2000, seed: int = 0) -> float:
    """Expected NDCG if the same items were returned in random order.

    This is the honest baseline for the ordering question, and only for that: it holds the
    retrieved set fixed and shuffles it. It says nothing about whether retrieval found the
    right items, answering that would require rating randomly drawn tracks, which this
    rating set does not contain.
    """
    rng = np.random.default_rng(seed)
    ratings = np.asarray(ratings)
    if ratings.size < 2:
        return 1.0
    return float(np.mean([ndcg(rng.permutation(ratings)) for _ in range(trials)]))


def bootstrap_ci(values: list[float], trials: int = 5000, seed: int = 0,
                 alpha: float = 0.05) -> tuple[float, float]:
    """Percentile bootstrap interval over per-query values.

    Reported on every category because ~8 queries per category is few, and a mean without
    an interval invites reading a gap that the data does not support.
    """
    arr = np.asarray(values, dtype=float)
    if arr.size < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = arr[rng.integers(0, arr.size, size=(trials, arr.size))].mean(axis=1)
    return (float(np.percentile(means, 100 * alpha / 2)),
            float(np.percentile(means, 100 * (1 - alpha / 2))))
