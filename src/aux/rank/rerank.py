"""Reranking a retrieved candidate set.

Eval 1 found that ordering within the returned top 5 was no better than chance, NDCG 0.866
against 0.863 for the same items shuffled, while success@1 (67%) trailed success@5 (89%).
A clearly-relevant track was often retrieved and not placed first. That gap is the headroom
these methods try to close, and none of them changes *which* tracks are retrieved.

Each method is parameter-free or carries a single parameter, deliberately: with 36 rated
queries, anything with real capacity would fit the evaluation rather than the problem.
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

    Eval 0C measured real hubness in this space: a few tracks are near neighbours of almost
    everything, and a hub is retrieved because it sits centrally, not because it matches.
    CSLS subtracts each track's average similarity to its own k nearest neighbours, so a
    track sitting in a dense region must clear a higher bar.

    Principled here rather than borrowed: the correction targets a property that was
    independently measured in this exact space, instead of being a generic ranking trick.
    """
    sims = track_vecs @ track_vecs.T
    np.fill_diagonal(sims, -np.inf)
    kk = min(k, max(1, track_vecs.shape[0] - 1))
    local_density = np.sort(sims, axis=1)[:, -kk:].mean(axis=1)
    return cosine(query_vec, track_vecs) - local_density


def query_z(query_vec: np.ndarray, track_vecs: np.ndarray,
            all_query_vecs: np.ndarray) -> np.ndarray:
    """Standardise each track's score against how it scores for *other* queries.

    A track that scores high for every query carries little information when it scores high
    for this one. Requires the other queries, so it is only available in batch evaluation --
    noted because a method that cannot run at serving time is not shippable however well it
    scores.
    """
    per_track = track_vecs @ all_query_vecs.T
    mu = per_track.mean(axis=1)
    sd = per_track.std(axis=1)
    return (cosine(query_vec, track_vecs) - mu) / np.maximum(sd, 1e-9)
