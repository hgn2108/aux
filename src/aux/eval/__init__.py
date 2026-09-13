"""Evaluation metrics, paired comparison and graded relevance, shared across slices."""

from .confidence import (
    HIGHER_MEANS_MORE_CONFIDENT,
    MEASURES,
    all_measures,
    crowding,
    relative_gap,
    top_k_entropy,
    z_top,
)
from .intent import INTENT_TERMS, intent_match, terms_in
from .metrics import random_baseline, ranks_of_truth, retrieval_metrics, space_diagnostics
from .ranking import (
    evaluate_ranking,
    hit_rate_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from .ranking import random_baseline as random_ranking_baseline
from .paired import permutation_test, bonferroni_threshold, compare_at_k, mcnemar_exact, sign_test_on_ranks
from .relevance import (
    bootstrap_ci,
    dcg,
    gains,
    ndcg,
    random_ordering_ndcg,
    success_at_k,
)

__all__ = [
    "HIGHER_MEANS_MORE_CONFIDENT",
    "INTENT_TERMS",
    "MEASURES",
    "all_measures",
    "crowding",
    "relative_gap",
    "top_k_entropy",
    "z_top",
    "bonferroni_threshold",
    "bootstrap_ci",
    "compare_at_k",
    "intent_match",
    "dcg",
    "gains",
    "mcnemar_exact",
    "ndcg",
    "evaluate_ranking",
    "hit_rate_at_k",
    "ndcg_at_k",
    "precision_at_k",
    "permutation_test",
    "random_baseline",
    "random_ranking_baseline",
    "recall_at_k",
    "random_ordering_ndcg",
    "ranks_of_truth",
    "retrieval_metrics",
    "sign_test_on_ranks",
    "space_diagnostics",
    "success_at_k",
    "terms_in",
]
