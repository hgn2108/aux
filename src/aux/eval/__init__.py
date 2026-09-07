"""Evaluation metrics and paired comparison, shared across slices."""

from .metrics import random_baseline, ranks_of_truth, retrieval_metrics, space_diagnostics
from .paired import bonferroni_threshold, compare_at_k, mcnemar_exact, sign_test_on_ranks

__all__ = [
    "bonferroni_threshold",
    "compare_at_k",
    "mcnemar_exact",
    "random_baseline",
    "ranks_of_truth",
    "retrieval_metrics",
    "sign_test_on_ranks",
    "space_diagnostics",
]
