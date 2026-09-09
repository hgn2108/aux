"""Graded-relevance metric tests."""

from __future__ import annotations

import numpy as np
import pytest

from aux.eval import bootstrap_ci, dcg, gains, ndcg, random_ordering_ndcg, success_at_k


def test_rating_of_one_is_worth_nothing():
    """1 means "not relevant at all", so a list of 1s must not score for existing."""
    assert gains(np.array([1, 1, 1])).tolist() == [0.0, 0.0, 0.0]
    assert dcg(np.array([1, 1, 1])) == 0.0
    assert ndcg(np.array([1, 1, 1])) == 0.0


def test_perfect_order_scores_one():
    assert ndcg(np.array([5, 4, 3, 2, 1])) == pytest.approx(1.0)


def test_reversed_order_scores_less_than_perfect():
    assert ndcg(np.array([1, 2, 3, 4, 5])) < ndcg(np.array([5, 4, 3, 2, 1]))


def test_ndcg_is_order_sensitive_not_set_sensitive():
    """Same items, different order — the whole point of the metric."""
    assert ndcg(np.array([5, 1])) > ndcg(np.array([1, 5]))


def test_uniform_ratings_score_one_regardless_of_order():
    """Nothing to order well or badly."""
    assert ndcg(np.array([4, 4, 4])) == pytest.approx(1.0)


def test_ndcg_bounded():
    rng = np.random.default_rng(0)
    for _ in range(50):
        r = rng.integers(1, 6, size=5)
        assert 0.0 <= ndcg(r) <= 1.0 + 1e-9


def test_success_at_k_threshold_and_cutoff():
    assert success_at_k(np.array([1, 1, 4, 1, 1]), k=5)
    assert not success_at_k(np.array([1, 1, 4, 1, 1]), k=2)
    assert not success_at_k(np.array([3, 3, 3]), k=3)
    assert success_at_k(np.array([5, 1]), k=1)


def test_random_ordering_baseline_sits_below_perfect_and_above_worst():
    ratings = np.array([5, 4, 3, 2, 1])
    baseline = random_ordering_ndcg(ratings, trials=500)
    assert ndcg(np.array([1, 2, 3, 4, 5])) < baseline < 1.0


def test_random_ordering_baseline_is_one_when_all_equal():
    assert random_ordering_ndcg(np.array([3, 3, 3]), trials=100) == pytest.approx(1.0)


def test_bootstrap_ci_brackets_the_mean():
    values = [3.0, 4.0, 5.0, 2.0, 4.0, 3.0, 5.0, 4.0]
    lo, hi = bootstrap_ci(values, trials=2000)
    assert lo < float(np.mean(values)) < hi


def test_bootstrap_ci_needs_more_than_one_value():
    lo, hi = bootstrap_ci([3.0])
    assert np.isnan(lo) and np.isnan(hi)
