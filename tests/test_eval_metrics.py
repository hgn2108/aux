"""Metric tests.

These matter more than they look. A silently wrong Recall@K or an off-by-one in ranking
would not crash anything -- it would produce a plausible number that every later decision
is then made from.
"""

from __future__ import annotations

import numpy as np
import pytest

from aux.eval import random_baseline, ranks_of_truth, retrieval_metrics, space_diagnostics


def test_perfect_ranking_gives_rank_one():
    scores = np.array([[0.9, 0.1, 0.2], [0.1, 0.8, 0.3]])
    assert ranks_of_truth(scores, [0, 1]).tolist() == [1, 1]


def test_worst_ranking_gives_last_rank():
    scores = np.array([[0.1, 0.5, 0.9]])
    assert ranks_of_truth(scores, [0]).tolist() == [3]


def test_ranks_are_one_based():
    """An off-by-one here would inflate Recall@1 by counting rank 0 as a hit."""
    assert ranks_of_truth(np.array([[1.0, 0.0]]), [0]).min() == 1


def test_metrics_on_a_known_rank_vector():
    m = retrieval_metrics(np.array([1, 2, 6, 11]), n_candidates=100)
    assert m["recall@1"] == pytest.approx(0.25)
    assert m["recall@5"] == pytest.approx(0.50)   # ranks 1, 2
    assert m["recall@10"] == pytest.approx(0.75)  # ranks 1, 2, 6
    assert m["median_rank"] == pytest.approx(4.0)
    assert m["mrr"] == pytest.approx(np.mean([1, 1 / 2, 1 / 6, 1 / 11]))


def test_recall_at_k_is_inclusive_of_k():
    assert retrieval_metrics(np.array([5]), 100)["recall@5"] == 1.0
    assert retrieval_metrics(np.array([6]), 100)["recall@5"] == 0.0


def test_empty_ranks_raise_rather_than_return_nan():
    with pytest.raises(ValueError):
        retrieval_metrics(np.array([]), 10)


def test_random_baseline_matches_chance():
    b = random_baseline(706)
    assert b["recall@10"] == pytest.approx(10 / 706)
    assert b["median_rank"] == pytest.approx(353.5)


def test_diagnostics_detect_a_collapsed_space():
    """Near-identical vectors must show high mean cosine and low spread."""
    base = np.ones((20, 8), dtype=np.float32)
    base += np.random.default_rng(0).normal(0, 1e-3, base.shape)
    base /= np.linalg.norm(base, axis=1, keepdims=True)
    d = space_diagnostics(base, np.zeros((20, 10), dtype=int))
    assert d["track_cosine_mean"] > 0.99
    assert d["track_cosine_std"] < 0.01


def test_diagnostics_on_a_spread_space():
    vectors = np.eye(16, dtype=np.float32)
    d = space_diagnostics(vectors, np.arange(16 * 10).reshape(16, 10) % 16)
    assert d["track_cosine_mean"] == pytest.approx(0.0, abs=1e-6)


def test_hubness_is_detected():
    """One track taking every retrieval slot must show as a dominant share."""
    vectors = np.eye(50, dtype=np.float32)
    all_hub = np.zeros((50, 10), dtype=int)
    d = space_diagnostics(vectors, all_hub)
    assert d["hubness_max_track_share"] == pytest.approx(1.0)
    assert d["tracks_never_retrieved"] == 49


def test_segment_cosines_are_optional():
    vectors = np.eye(4, dtype=np.float32)
    assert space_diagnostics(vectors, np.zeros((4, 2), int))["within_track_segment_cosine_mean"] is None
    d = space_diagnostics(vectors, np.zeros((4, 2), int), segment_cosines=[0.9, 0.5])
    assert d["within_track_segment_cosine_mean"] == pytest.approx(0.7)
    assert d["within_track_segment_cosine_min"] == pytest.approx(0.5)
