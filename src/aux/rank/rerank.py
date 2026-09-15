"""Reranking a retrieved candidate set.

Ordering within the top 5 was no better than chance (NDCG 0.866 against 0.863 shuffled),
while success@1 at 67% trailed success@5 at 89%. Relevant tracks were being retrieved but
not put first. These methods try to close that gap; none changes what is retrieved.

One parameter at most, on purpose: 36 rated queries will fit anything with real capacity.
"""

from __future__ import annotations

import numpy as np


def cosine(query_vec: np.ndarray, track_vecs: np.ndarray) -> np.ndarray:
    """The current behaviour: rank by cosine to the pooled track vector."""
    return track_vecs @ query_vec


def max_segment(query_vec: np.ndarray, segment_vecs: list[np.ndarray]) -> np.ndarray:
    """Rank by a track's single best-matching segment.

    Mean pooling asks "is this track as a whole about the query". A listener asking for
    "a saxophone solo" may be satisfied by a track containing one, even if most of it is
    something else. Max is the natural counterpart to the pooling decision E1 settled, and
    it costs nothing extra, the segment vectors already exist at index time.
    """
    return np.array([float((segs @ query_vec).max()) for segs in segment_vecs])


def mean_plus_max(query_vec: np.ndarray, track_vecs: np.ndarray,
                  segment_vecs: list[np.ndarray], alpha: float = 0.5) -> np.ndarray:
    """Blend whole-track and best-segment evidence."""
    return (1 - alpha) * cosine(query_vec, track_vecs) + alpha * max_segment(
        query_vec, segment_vecs)


def csls(query_vec: np.ndarray, track_vecs: np.ndarray, *, k: int = 10) -> np.ndarray:
    """Cross-domain similarity local scaling, a hubness correction.

    Subtracts each track's average similarity to its own k nearest neighbours, so a track
    in a dense region must clear a higher bar. Used here because hubness was measured in
    this space, not because it is a generic ranking trick.
    """
    sims = track_vecs @ track_vecs.T
    np.fill_diagonal(sims, -np.inf)
    kk = min(k, max(1, track_vecs.shape[0] - 1))
    local_density = np.sort(sims, axis=1)[:, -kk:].mean(axis=1)
    return cosine(query_vec, track_vecs) - local_density


def query_z(query_vec: np.ndarray, track_vecs: np.ndarray,
            all_query_vecs: np.ndarray) -> np.ndarray:
    """Standardise each track's score against how it scores for other queries.

    A track that scores high for every query carries little information when it scores high
    for this one. Requires the other queries, so it is only available in batch evaluation --
    noted because a method that cannot run at serving time is not shippable however well it
    scores.
    """
    per_track = track_vecs @ all_query_vecs.T
    mu = per_track.mean(axis=1)
    sd = per_track.std(axis=1)
    return (cosine(query_vec, track_vecs) - mu) / np.maximum(sd, 1e-9)
