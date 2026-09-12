"""Ranking-metric tests, on hand-checkable examples."""

from __future__ import annotations

import numpy as np
import pytest

from aux.eval import (
    evaluate_ranking,
    hit_rate_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)

PERFECT = np.array([1, 1, 1, 0, 0], dtype=bool)
WORST = np.array([0, 0, 1, 1, 1], dtype=bool)


def test_precision_counts_relevant_in_the_top_k():
    assert precision_at_k(PERFECT, 3) == 1.0
    assert precision_at_k(PERFECT, 5) == pytest.approx(0.6)
    assert precision_at_k(WORST, 2) == 0.0


def test_recall_is_against_all_relevant_not_k():
    assert recall_at_k(PERFECT, 3, total_relevant=3) == 1.0
    assert recall_at_k(PERFECT, 3, total_relevant=6) == pytest.approx(0.5)


def test_recall_with_no_relevant_items_is_zero_not_nan():
    assert recall_at_k(PERFECT, 3, total_relevant=0) == 0.0


def test_hit_rate_is_binary():
    assert hit_rate_at_k(WORST, 3) == 1.0
    assert hit_rate_at_k(WORST, 2) == 0.0


def test_ndcg_rewards_putting_relevant_items_first():
    assert ndcg_at_k(PERFECT, 5, 3) == pytest.approx(1.0)
    assert ndcg_at_k(WORST, 5, 3) < 1.0
    assert ndcg_at_k(PERFECT, 5, 3) > ndcg_at_k(WORST, 5, 3)


def test_ndcg_ideal_accounts_for_scarce_relevant_items():
    """One relevant item ranked first is a perfect ranking, not 1/k of one."""
    ranked = np.array([1, 0, 0, 0, 0], dtype=bool)
    assert ndcg_at_k(ranked, 5, total_relevant=1) == pytest.approx(1.0)


def test_ndcg_known_value():
    # relevant at positions 1 and 3 -> DCG = 1/log2(2) + 1/log2(4) = 1 + 0.5
    # ideal with 2 relevant      -> 1/log2(2) + 1/log2(3) = 1 + 0.6309
    ranked = np.array([1, 0, 1, 0], dtype=bool)
    assert ndcg_at_k(ranked, 4, 2) == pytest.approx(1.5 / (1 + 1 / np.log2(3)), rel=1e-6)


# --- whole-system evaluation ---------------------------------------------------------

def test_a_track_is_never_its_own_recommendation():
    scores = np.eye(4)                      # every track scores itself highest
    relevant = ~np.eye(4, dtype=bool)
    out = evaluate_ranking(scores, relevant, ks=(1,))
    assert out["precision@1"] == 1.0        # would be 0 if self-matches were kept


def test_queries_with_no_relevant_item_are_skipped():
    relevant = np.zeros((3, 3), dtype=bool)
    relevant[0, 1] = True
    out = evaluate_ranking(np.random.default_rng(0).random((3, 3)), relevant, ks=(1,))
    assert out["n_queries"] == 1
    assert out["n_skipped"] == 2


def test_exclusion_removes_pairs_from_both_scores_and_labels():
    """Same-artist exclusion must not leave the excluded pairs counted as relevant."""
    relevant = np.ones((3, 3), dtype=bool)
    np.fill_diagonal(relevant, False)
    exclude = np.zeros((3, 3), dtype=bool)
    exclude[0, 1] = exclude[1, 0] = True
    out = evaluate_ranking(np.ones((3, 3)), relevant, ks=(1,), exclude=exclude)
    assert out["mean_relevant_per_query"] < 2.0


def test_a_perfect_system_scores_one():
    relevant = np.zeros((4, 4), dtype=bool)
    relevant[:, 0] = True
    np.fill_diagonal(relevant, False)
    scores = np.zeros((4, 4))
    scores[:, 0] = 1.0                      # always rank track 0 first
    out = evaluate_ranking(scores, relevant, ks=(1,))
    assert out["precision@1"] == 1.0
    assert out["ndcg@1"] == 1.0


def test_random_scores_land_near_the_base_rate():
    rng = np.random.default_rng(0)
    n = 200
    relevant = rng.random((n, n)) < 0.25
    np.fill_diagonal(relevant, False)
    out = evaluate_ranking(rng.random((n, n)), relevant, ks=(10,))
    assert 0.20 < out["precision@10"] < 0.30
