"""Ranking metrics for recommendation against binary relevance labels.

Separate from `relevance.py`, which handles graded 1-5 human ratings. Here relevance is
binary and comes from metadata — same genre, same artist, same album — so the metrics are
the standard retrieval set rather than graded NDCG.

All four are reported because they answer different questions, and a recommender can look
good on one while failing another:

- **Precision@K** — of what was shown, how much was relevant. The user-facing quality of a
  short list.
- **Recall@K** — of everything relevant, how much was found. Meaningless when a query has
  249 relevant items and K is 10, which is exactly the case for the genre label, so it is
  reported but not used to compare systems there.
- **HitRate@K** — did *anything* relevant appear. The right metric when relevant items are
  scarce, as with the album label.
- **NDCG@K** — rewards putting relevant items higher, not merely including them.
"""

from __future__ import annotations

import numpy as np


def precision_at_k(ranked_relevant: np.ndarray, k: int) -> float:
    return float(ranked_relevant[:k].sum() / k)


def recall_at_k(ranked_relevant: np.ndarray, k: int, total_relevant: int) -> float:
    if total_relevant <= 0:
        return 0.0
    return float(ranked_relevant[:k].sum() / total_relevant)


def hit_rate_at_k(ranked_relevant: np.ndarray, k: int) -> float:
    return float(bool(ranked_relevant[:k].any()))


def ndcg_at_k(ranked_relevant: np.ndarray, k: int, total_relevant: int) -> float:
    """NDCG with binary gains.

    The ideal ranking places `min(k, total_relevant)` relevant items first, so a query with
    fewer relevant items than K is not penalised for the shortfall — without that, a query
    with one relevant item could never score 1.0.
    """
    gains = np.asarray(ranked_relevant[:k], dtype=float)
    discounts = 1.0 / np.log2(np.arange(2, gains.size + 2))
    dcg = float((gains * discounts).sum())
    ideal_n = int(min(k, total_relevant))
    if ideal_n <= 0:
        return 0.0
    idcg = float(discounts[:ideal_n].sum())
    return dcg / idcg if idcg > 0 else 0.0


def evaluate_ranking(scores: np.ndarray, relevant: np.ndarray, ks: tuple[int, ...] = (5, 10, 20),
                     exclude: np.ndarray | None = None) -> dict:
    """Score one system over every query.

    `scores[i, j]` is how strongly track j is recommended for query i; `relevant[i, j]` is
    whether it should be. `exclude` masks pairs out of both ranking and labels — used to
    drop same-artist pairs when scoring the genre label, which otherwise rewards a model for
    recognising an album's production rather than its genre.

    Queries with no relevant item are skipped: they cannot distinguish a good system from a
    bad one, and including them just dilutes every metric by a constant.
    """
    scores = np.array(scores, dtype=float, copy=True)
    relevant = np.asarray(relevant, dtype=bool)
    n = scores.shape[0]

    # A track is never its own recommendation.
    np.fill_diagonal(scores, -np.inf)
    if exclude is not None:
        scores[np.asarray(exclude, dtype=bool)] = -np.inf
        relevant = relevant & ~np.asarray(exclude, dtype=bool)

    totals = relevant.sum(axis=1)
    usable = totals > 0
    out: dict = {"n_queries": int(usable.sum()), "n_skipped": int((~usable).sum()),
                 "mean_relevant_per_query": float(totals[usable].mean()) if usable.any() else 0.0}

    order = np.argsort(-scores, axis=1)
    for k in ks:
        p, r, h, nd = [], [], [], []
        for i in np.flatnonzero(usable):
            ranked = relevant[i, order[i]]
            p.append(precision_at_k(ranked, k))
            r.append(recall_at_k(ranked, k, int(totals[i])))
            h.append(hit_rate_at_k(ranked, k))
            nd.append(ndcg_at_k(ranked, k, int(totals[i])))
        out[f"precision@{k}"] = float(np.mean(p))
        out[f"recall@{k}"] = float(np.mean(r))
        out[f"hit_rate@{k}"] = float(np.mean(h))
        out[f"ndcg@{k}"] = float(np.mean(nd))
    return out


def random_baseline(relevant: np.ndarray, ks: tuple[int, ...] = (5, 10, 20),
                    exclude: np.ndarray | None = None, seed: int = 0) -> dict:
    """What random ranking scores on the same labels.

    Computed rather than derived, so it carries the same query-skipping and exclusion rules
    as the systems it is compared against.
    """
    rng = np.random.default_rng(seed)
    return evaluate_ranking(rng.random(np.asarray(relevant).shape), relevant, ks, exclude)
