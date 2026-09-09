"""Confidence-measure tests."""

from __future__ import annotations

import numpy as np
import pytest

from aux.eval import all_measures, crowding, relative_gap, top_k_entropy, z_top

rng = np.random.default_rng(0)
FLAT = np.full(200, 0.5)                                    # nothing stands out
NOISE = rng.normal(0.3, 0.05, 200)                          # no clear winner
STANDOUT = np.concatenate([[0.95], rng.normal(0.3, 0.05, 199)])  # one clear winner


def test_a_clear_winner_scores_more_confident_than_noise():
    assert z_top(STANDOUT) > z_top(NOISE)
    assert relative_gap(STANDOUT) > relative_gap(NOISE)
    assert crowding(STANDOUT) < crowding(NOISE)
    assert top_k_entropy(STANDOUT) < top_k_entropy(NOISE)


def test_a_flat_library_is_maximally_unconfident():
    assert z_top(FLAT) == 0.0
    assert relative_gap(FLAT) == 0.0
    assert crowding(FLAT) == pytest.approx(1.0)
    assert top_k_entropy(FLAT) == pytest.approx(1.0)


def test_scale_free_measures_ignore_a_constant_shift():
    """The property that should let a threshold transfer between libraries."""
    for fn in (relative_gap, top_k_entropy):
        assert fn(STANDOUT) == pytest.approx(fn(STANDOUT + 10.0), abs=1e-9)


def test_scale_free_measures_ignore_a_scale_change():
    for fn in (relative_gap, top_k_entropy):
        assert fn(STANDOUT) == pytest.approx(fn(STANDOUT * 7.0), abs=1e-9)


def test_z_top_is_also_shift_and_scale_invariant():
    """z_top's drift between libraries is about sample size, not units."""
    assert z_top(STANDOUT) == pytest.approx(z_top(STANDOUT * 3.0 + 1.0), abs=1e-9)


def test_z_top_grows_with_library_size_for_the_same_shape():
    """The reason a fixed z_top threshold may not transfer."""
    small = rng.normal(0, 1, 30)
    large = rng.normal(0, 1, 8000)
    assert z_top(large) > z_top(small)


def test_crowding_handles_negative_scores():
    """Cosine scores are routinely negative; a naive fraction-of-top would break."""
    negative = np.array([-0.1, -0.5, -0.6, -0.7])
    assert 0.0 < crowding(negative) <= 1.0


def test_entropy_is_bounded():
    for scores in (FLAT, NOISE, STANDOUT):
        assert 0.0 <= top_k_entropy(scores) <= 1.0


def test_measures_handle_a_tiny_library():
    tiny = np.array([0.5, 0.2])
    vals = all_measures(tiny)
    assert all(np.isfinite(v) for v in vals.values())


def test_all_measures_returns_every_measure():
    assert set(all_measures(NOISE)) == {"z_top", "relative_gap", "crowding", "top_k_entropy"}
