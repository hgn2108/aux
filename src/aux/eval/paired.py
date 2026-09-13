"""Paired comparison of two evaluation runs.

Two configurations scored on the *same* query set are not two independent samples, and
treating them as such throws away most of the available power: the queries that behaved
identically under both carry no information about which is better, and they are the
majority. McNemar's exact test conditions on the discordant queries only.

In the package rather than in a script for the same reason as `metrics`: a test that
decides whether a component ships has to itself be tested.
"""

from __future__ import annotations

from math import comb

import numpy as np


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact binomial p-value on discordant pairs.

    ``b`` is the count that the baseline got right and the treatment got wrong; ``c`` the
    reverse. Under the null the discordant outcomes split 50/50, so the p-value is the
    two-sided binomial tail. Exact rather than the chi-square approximation, which is
    unreliable when ``b + c`` is small -- exactly the regime where a marginal result would
    otherwise be over-read.
    """
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(comb(n, i) for i in range(k + 1)) / (2**n)
    return min(1.0, 2 * tail)


def compare_at_k(baseline_ranks: np.ndarray, treatment_ranks: np.ndarray, k: int) -> dict:
    """Paired Recall@K comparison between two runs over the same queries."""
    base = np.asarray(baseline_ranks)
    treat = np.asarray(treatment_ranks)
    if base.shape != treat.shape:
        raise ValueError("paired comparison requires the same queries in both runs")

    base_hit, treat_hit = base <= k, treat <= k
    gained = int((~base_hit & treat_hit).sum())
    lost = int((base_hit & ~treat_hit).sum())
    return {
        "k": k,
        "baseline": float(base_hit.mean()),
        "treatment": float(treat_hit.mean()),
        "delta": float(treat_hit.mean() - base_hit.mean()),
        "gained": gained,
        "lost": lost,
        "p_value": mcnemar_exact(lost, gained),
    }


def sign_test_on_ranks(baseline_ranks: np.ndarray, treatment_ranks: np.ndarray) -> dict:
    """Did ranks move, overall? Uses every query, not only those crossing a K boundary."""
    base = np.asarray(baseline_ranks)
    treat = np.asarray(treatment_ranks)
    improved = int((treat < base).sum())
    worsened = int((treat > base).sum())
    return {
        "improved": improved,
        "worsened": worsened,
        "unchanged": int(base.size - improved - worsened),
        "p_value": mcnemar_exact(worsened, improved),
    }


def bonferroni_threshold(n_tests: int, alpha: float = 0.05) -> float:
    """Significance threshold once ``n_tests`` comparisons are run.

    Recorded explicitly because running Recall@1/5/10 across three configurations is nine
    tests, and a p of 0.018 among nine is not the same claim as a p of 0.018 alone.
    """
    return alpha / max(1, n_tests)


def permutation_test(a: np.ndarray, b: np.ndarray, *, n_resamples: int = 20000,
                     seed: int = 0) -> dict:
    """Paired two-sided permutation test on the mean difference between two systems.

    Used where the paired values are continuous — per-query NDCG, say — rather than the
    win/loss counts `mcnemar_exact` takes. Under the null the two systems are
    interchangeable, so each query's difference is equally likely to carry either sign;
    the reference distribution flips those signs at random.

    Exact enumeration is used when there are few enough pairs for it, which is common here:
    a 24-query set has 2**24 sign assignments, so sampling is the practical choice, but a
    small subset can be enumerated outright and is not left to chance.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    diff = a - b
    observed = float(diff.mean())
    n = diff.size
    if n == 0:
        return {"observed": 0.0, "p_value": 1.0, "n": 0, "exact": True}

    if n <= 20:
        signs = np.array(np.meshgrid(*[[1, -1]] * n)).T.reshape(-1, n)
        means = (signs * diff).mean(axis=1)
        exact = True
    else:
        rng = np.random.default_rng(seed)
        signs = rng.choice([1.0, -1.0], size=(n_resamples, n))
        means = (signs * diff).mean(axis=1)
        exact = False

    p = float((np.abs(means) >= abs(observed) - 1e-12).mean())
    return {"observed": observed, "p_value": p, "n": int(n), "exact": exact}
