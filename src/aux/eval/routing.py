"""Cross-validated comparison of family-conditioned routing against a fixed weight.

The earlier version of this analysis picked the best weight per query family and the best
global weight on the whole query set, then scored both on that same set. Selection and
evaluation shared the examples, and routing was fitting one weight per family against the
baseline's one weight overall, so it had more freedom on the same data. Whatever gap that
produced was partly the extra fitting.

Here each weight is chosen on a training split and applied unchanged to a held-out split.
Every query is held out exactly once, so the paired test runs on scores no selection step
ever saw.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def stratified_folds(labels: np.ndarray, n_splits: int = 5, seed: int = 0) -> np.ndarray:
    """Assign each item a fold, keeping label proportions even across folds.

    Written here rather than pulled from scikit-learn to avoid a dependency for one
    function. Each label group is shuffled with a seeded generator and dealt round-robin,
    which keeps every family present in every fold as long as it has at least `n_splits`
    members.
    """
    labels = np.asarray(labels)
    folds = np.empty(labels.size, dtype=int)
    rng = np.random.default_rng(seed)
    for label in sorted(set(labels.tolist())):
        idx = np.flatnonzero(labels == label)
        rng.shuffle(idx)
        folds[idx] = np.arange(idx.size) % n_splits
    return folds


@dataclass(frozen=True, slots=True)
class RoutingCV:
    """Held-out per-query scores for both systems, plus what each fold chose."""

    fixed: np.ndarray
    routed: np.ndarray
    fold_of: np.ndarray
    fixed_alpha: list[float] = field(default_factory=list)
    family_alpha: list[dict[str, float]] = field(default_factory=list)

    @property
    def delta(self) -> np.ndarray:
        """Per-query held-out difference, routed minus fixed. What the paired test takes."""
        return self.routed - self.fixed


def cross_validate_routing(table: np.ndarray, grid: np.ndarray, families: np.ndarray,
                           n_splits: int = 5, seed: int = 0) -> RoutingCV:
    """Choose weights on training folds, score on held-out folds.

    `table[i, j]` is query i's score at weight `grid[j]`, so no retrieval reruns here: the
    per-query scores at every weight are already known and cross-validation is a question
    of which rows may be looked at when.

    A family absent from a training fold falls back to that fold's global weight, which is
    the honest behaviour for a family the training data says nothing about.
    """
    table = np.asarray(table, dtype=float)
    families = np.asarray(families)
    fold_of = stratified_folds(families, n_splits, seed)

    fixed = np.full(table.shape[0], np.nan)
    routed = np.full(table.shape[0], np.nan)
    fixed_alpha: list[float] = []
    family_alpha: list[dict[str, float]] = []

    for fold in range(n_splits):
        test = fold_of == fold
        train = ~test
        if not train.any() or not test.any():
            continue

        global_col = int(np.argmax(table[train].mean(axis=0)))
        fixed[test] = table[test, global_col]
        fixed_alpha.append(float(grid[global_col]))

        chosen: dict[str, float] = {}
        for family in sorted(set(families.tolist())):
            in_train = train & (families == family)
            col = (int(np.argmax(table[in_train].mean(axis=0))) if in_train.any()
                   else global_col)
            chosen[family] = float(grid[col])
            in_test = test & (families == family)
            if in_test.any():
                routed[in_test] = table[in_test, col]
        family_alpha.append(chosen)

    scored = ~np.isnan(fixed)
    return RoutingCV(fixed=fixed[scored], routed=routed[scored],
                     fold_of=fold_of[scored], fixed_alpha=fixed_alpha,
                     family_alpha=family_alpha)
