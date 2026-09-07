"""Paired-comparison tests.

The test that decides whether a component ships must itself be tested. A p-value that is
wrong in the conservative direction loses a real improvement; wrong in the liberal
direction ships a change that did nothing. Both are silent.
"""

from __future__ import annotations

import numpy as np
import pytest

from aux.eval import bonferroni_threshold, compare_at_k, mcnemar_exact, sign_test_on_ranks


def test_no_discordant_pairs_is_not_significant():
    """Identical runs must give p = 1, never a spuriously small number."""
    assert mcnemar_exact(0, 0) == 1.0


def test_symmetric_discordance_is_not_significant():
    assert mcnemar_exact(10, 10) == pytest.approx(1.0)


def test_known_binomial_values():
    # b=0, c=1: two-sided p = 2 * (1/2)^1 = 1.0
    assert mcnemar_exact(0, 1) == pytest.approx(1.0)
    # b=0, c=5: two-sided p = 2 * (1/2)^5 = 0.0625
    assert mcnemar_exact(0, 5) == pytest.approx(0.0625)
    # b=0, c=10: 2 * (1/2)^10
    assert mcnemar_exact(0, 10) == pytest.approx(2 / 1024)


def test_p_value_is_symmetric_in_its_arguments():
    """Direction of the effect must not change its significance."""
    assert mcnemar_exact(3, 20) == pytest.approx(mcnemar_exact(20, 3))


def test_p_value_never_exceeds_one():
    for b in range(6):
        for c in range(6):
            assert 0.0 <= mcnemar_exact(b, c) <= 1.0


def test_larger_imbalance_gives_smaller_p():
    assert mcnemar_exact(5, 20) > mcnemar_exact(2, 23)


def test_compare_at_k_counts_gains_and_losses():
    base = np.array([1, 20, 3, 50])
    treat = np.array([30, 2, 3, 60])   # query 0 lost, query 1 gained, 2 and 3 unchanged
    r = compare_at_k(base, treat, k=10)
    assert r["gained"] == 1
    assert r["lost"] == 1
    assert r["baseline"] == pytest.approx(0.5)
    assert r["treatment"] == pytest.approx(0.5)
    assert r["delta"] == pytest.approx(0.0)


def test_compare_at_k_uses_only_discordant_queries():
    """Queries that behaved identically carry no information and must not shift p."""
    base = np.array([1, 1, 1, 50])
    treat = np.array([1, 1, 1, 5])
    lots_of_agreement = compare_at_k(base, treat, k=10)
    minimal = compare_at_k(np.array([50]), np.array([5]), k=10)
    assert lots_of_agreement["p_value"] == pytest.approx(minimal["p_value"])


def test_mismatched_lengths_raise():
    with pytest.raises(ValueError):
        compare_at_k(np.array([1, 2]), np.array([1]), k=5)


def test_sign_test_partitions_every_query():
    base = np.array([10, 5, 7, 7])
    treat = np.array([3, 9, 7, 7])
    st = sign_test_on_ranks(base, treat)
    assert st["improved"] == 1
    assert st["worsened"] == 1
    assert st["unchanged"] == 2
    assert st["improved"] + st["worsened"] + st["unchanged"] == base.size


def test_bonferroni_threshold():
    assert bonferroni_threshold(9) == pytest.approx(0.05 / 9)
    assert bonferroni_threshold(1) == pytest.approx(0.05)
    assert bonferroni_threshold(0) == pytest.approx(0.05)


def test_the_e1_headline_result_reproduces():
    """Guards the number DEC-011 rests on: 104 gained, 43 lost at K=10."""
    assert mcnemar_exact(43, 104) < 1e-6


def test_the_3_vs_5_result_stays_below_correction():
    """DEC-011 records 3-vs-5 as unseparated; that must not silently become significant."""
    p = mcnemar_exact(36, 60)
    assert p > bonferroni_threshold(9)
