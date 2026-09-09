"""Rank-fusion tests."""

from __future__ import annotations

import numpy as np
import pytest

from aux.rank import reciprocal_rank_fusion as rrf


def order(scores):
    return np.argsort(-np.asarray(scores)).tolist()


def test_fusing_one_scoring_preserves_its_order():
    s = np.array([0.1, 0.9, 0.5])
    assert order(rrf([s])) == order(s)


def test_agreement_is_preserved():
    a = np.array([0.9, 0.5, 0.1])
    b = np.array([0.8, 0.4, 0.2])
    assert order(rrf([a, b])) == [0, 1, 2]


def test_scale_differences_cannot_dominate():
    """The reason DESIGN.md fuses on rank: raw scores are not on a common scale."""
    modest = np.array([0.10, 0.09, 0.08])          # small spread, prefers item 0
    huge = np.array([0.0, 100.0, 0.0])             # vast spread, prefers item 1
    # A score-level average would be decided entirely by `huge`.
    assert order((modest + huge) / 2)[0] == 1
    # Rank fusion gives each system one vote per position.
    fused = rrf([modest, huge])
    assert fused[0] > fused[2]


def test_an_item_ranked_well_by_both_beats_one_ranked_well_by_either():
    a = np.array([0.9, 0.8, 0.1])   # 0 then 1
    b = np.array([0.1, 0.9, 0.8])   # 1 then 2
    fused = rrf([a, b])
    assert order(fused)[0] == 1     # only item 1 is high in both


def test_weights_shift_the_balance():
    a = np.array([0.9, 0.1])
    b = np.array([0.1, 0.9])
    assert order(rrf([a, b], weights=[3.0, 1.0])) == [0, 1]
    assert order(rrf([a, b], weights=[1.0, 3.0])) == [1, 0]


def test_k_damps_the_top():
    """A large k flattens the advantage of first place."""
    a = np.array([0.9, 0.8, 0.7])
    b = np.array([0.1, 0.2, 0.3])
    spread_small_k = np.ptp(rrf([a, b], k=1))
    spread_large_k = np.ptp(rrf([a, b], k=1000))
    assert spread_large_k < spread_small_k


def test_mismatched_lengths_raise():
    with pytest.raises(ValueError):
        rrf([np.array([1.0, 2.0]), np.array([1.0])])


def test_empty_input_raises():
    with pytest.raises(ValueError):
        rrf([])


def test_wrong_number_of_weights_raises():
    with pytest.raises(ValueError):
        rrf([np.array([1.0]), np.array([2.0])], weights=[1.0])
