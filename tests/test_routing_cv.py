"""Cross-validated routing: the selection must not see the scores it is judged on.

The earlier analysis chose the best weight per family on the same queries it then scored.
These tests are built so that a return to that would fail, rather than checking the shape of
the output.
"""

from __future__ import annotations

import numpy as np
import pytest

from aux.eval import cross_validate_routing, stratified_folds


def test_folds_are_deterministic_and_keep_families_together():
    labels = np.array(["a"] * 20 + ["b"] * 20)
    first = stratified_folds(labels, n_splits=5, seed=7)
    assert np.array_equal(first, stratified_folds(labels, n_splits=5, seed=7))
    assert not np.array_equal(first, stratified_folds(labels, n_splits=5, seed=8))

    # Every family appears in every fold, so no fold has to fall back to the global weight.
    for fold in range(5):
        present = set(labels[first == fold])
        assert present == {"a", "b"}


def test_every_query_is_held_out_exactly_once():
    rng = np.random.default_rng(0)
    table = rng.random((40, 6))
    families = np.array(["a"] * 20 + ["b"] * 20)

    cv = cross_validate_routing(table, np.linspace(0, 1, 6), families, n_splits=4, seed=0)

    assert cv.fixed.size == 40
    assert cv.routed.size == 40
    assert np.bincount(cv.fold_of).tolist() == [10, 10, 10, 10]


def test_selection_cannot_see_the_held_out_scores():
    """The decisive test: make the held-out optimum differ from the training optimum.

    Two weights. On training rows weight 0 is better; on held-out rows weight 1 is better.
    An honest procedure carries the training choice over and scores the held-out rows at
    weight 0. A leaking one would pick weight 1 and score higher.
    """
    n, families = 20, np.array(["a"] * 20)
    folds = stratified_folds(families, n_splits=2, seed=0)
    held_out = folds == 0

    table = np.zeros((n, 2))
    table[~held_out, 0] = 1.0        # training rows prefer weight 0
    table[~held_out, 1] = 0.0
    table[held_out, 0] = 0.0         # held-out rows prefer weight 1
    table[held_out, 1] = 1.0

    cv = cross_validate_routing(table, np.array([0.0, 1.0]), families, n_splits=2, seed=0)

    # Fold 0's training rows are fold 1's, which prefer weight 0, so fold 0's held-out
    # rows are scored at weight 0 and get 0.0. Symmetrically for fold 1. A procedure that
    # peeked would score 1.0 everywhere.
    assert cv.routed.mean() == pytest.approx(0.0)
    assert cv.fixed.mean() == pytest.approx(0.0)


def test_family_weights_come_only_from_that_family_in_training():
    """Family b's training rows favour a weight that family a's rows never would."""
    families = np.array(["a"] * 10 + ["b"] * 10)
    grid = np.array([0.0, 1.0])
    table = np.zeros((20, 2))
    table[families == "a", 0] = 1.0        # a prefers weight 0
    table[families == "b", 1] = 1.0        # b prefers weight 1

    cv = cross_validate_routing(table, grid, families, n_splits=2, seed=0)

    for fold_choice in cv.family_alpha:
        assert fold_choice["a"] == 0.0
        assert fold_choice["b"] == 1.0
    # Each family is then scored at its own weight, so every held-out query scores 1.0,
    # while a single global weight can only satisfy half of them.
    assert cv.routed.mean() == pytest.approx(1.0)
    assert cv.fixed.mean() == pytest.approx(0.5)


def test_delta_is_paired_and_aligned():
    rng = np.random.default_rng(1)
    table = rng.random((30, 5))
    families = np.array(["a"] * 15 + ["b"] * 15)

    cv = cross_validate_routing(table, np.linspace(0, 1, 5), families, n_splits=5, seed=3)

    assert cv.delta.shape == cv.fixed.shape == cv.routed.shape
    assert np.allclose(cv.delta, cv.routed - cv.fixed)


def test_a_family_missing_from_training_falls_back_to_the_global_weight():
    """A family with fewer members than folds cannot appear in every training split."""
    families = np.array(["a"] * 10 + ["rare"])
    grid = np.array([0.0, 1.0])
    table = np.zeros((11, 2))
    table[:10, 0] = 1.0          # the common family, and so the global choice, prefers 0
    table[10, 1] = 1.0           # the rare query would prefer 1, but cannot say so

    cv = cross_validate_routing(table, grid, families, n_splits=5, seed=0)

    assert cv.fixed.size == 11
    # The rare query is scored at the global weight, so it scores 0 rather than its own 1.
    assert cv.routed.sum() == pytest.approx(cv.fixed.sum())
