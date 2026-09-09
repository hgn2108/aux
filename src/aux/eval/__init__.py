"""Evaluation metrics, paired comparison and graded relevance, shared across slices."""

from .intent import INTENT_TERMS, intent_match, terms_in
from .metrics import random_baseline, ranks_of_truth, retrieval_metrics, space_diagnostics
from .paired import bonferroni_threshold, compare_at_k, mcnemar_exact, sign_test_on_ranks
from .relevance import (
    bootstrap_ci,
    dcg,
    gains,
    ndcg,
    random_ordering_ndcg,
    success_at_k,
)

__all__ = [
    "INTENT_TERMS",
    "bonferroni_threshold",
    "bootstrap_ci",
    "compare_at_k",
    "intent_match",
    "dcg",
    "gains",
    "mcnemar_exact",
    "ndcg",
    "random_baseline",
    "random_ordering_ndcg",
    "ranks_of_truth",
    "retrieval_metrics",
    "sign_test_on_ranks",
    "space_diagnostics",
    "success_at_k",
    "terms_in",
]
