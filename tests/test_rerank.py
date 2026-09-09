"""Reranking method tests."""

from __future__ import annotations

import numpy as np
import pytest

from aux.rank import cosine, csls, max_segment, mean_plus_max, query_z


def unit(x):
    x = np.asarray(x, dtype=float)
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def test_cosine_ranks_the_closest_track_first():
    q = unit([1.0, 0.0])
    tracks = unit(np.array([[0.0, 1.0], [1.0, 0.1]]))
    assert int(np.argmax(cosine(q, tracks))) == 1


def test_max_segment_rewards_a_single_strong_match():
    """A track that is mostly unrelated but contains one matching section."""
    q = unit([1.0, 0.0])
    segs = [unit(np.array([[0.0, 1.0], [1.0, 0.0]])),   # one perfect segment
            unit(np.array([[0.6, 0.8], [0.6, 0.8]]))]   # uniformly mediocre
    scores = max_segment(q, segs)
    assert scores[0] > scores[1]


def test_max_segment_differs_from_mean_pooling():
    """The case the two methods disagree on, which is the reason to try max at all.

    Track A is half perfect and half irrelevant; track B is uniformly close but never
    exact. Mean pooling prefers B, best-segment prefers A.
    """
    q = unit([1.0, 0.0])
    segs = [unit(np.array([[1.0, 0.0], [0.0, 1.0]])),      # one exact segment
            unit(np.array([[0.9, 0.44], [0.9, 0.44]]))]    # uniformly close
    pooled = unit(np.array([s.mean(axis=0) for s in segs]))
    assert int(np.argmax(cosine(q, pooled))) == 1
    assert int(np.argmax(max_segment(q, segs))) == 0


def test_mean_plus_max_interpolates():
    q = unit([1.0, 0.0])
    segs = [unit(np.array([[0.0, 1.0], [1.0, 0.0]])),
            unit(np.array([[0.6, 0.8], [0.6, 0.8]]))]
    pooled = unit(np.array([s.mean(axis=0) for s in segs]))
    at0 = mean_plus_max(q, pooled, segs, alpha=0.0)
    at1 = mean_plus_max(q, pooled, segs, alpha=1.0)
    assert np.allclose(at0, cosine(q, pooled))
    assert np.allclose(at1, max_segment(q, segs))


def test_csls_penalises_a_hub():
    """A track sitting among near-duplicates must clear a higher bar than an isolated one."""
    cluster = unit(np.array([[1.0, 0.02], [1.0, 0.01], [1.0, 0.03], [1.0, 0.0]]))
    lonely = unit(np.array([[0.72, 0.69]]))
    tracks = np.vstack([cluster, lonely])
    q = unit([1.0, 0.0])
    plain = cosine(q, tracks)
    corrected = csls(q, tracks, k=2)
    # The hub leads on raw cosine; the correction closes the gap against the isolated track.
    assert plain[0] > plain[-1]
    assert (corrected[0] - corrected[-1]) < (plain[0] - plain[-1])


def test_csls_handles_a_tiny_index():
    tracks = unit(np.array([[1.0, 0.0], [0.0, 1.0]]))
    assert np.all(np.isfinite(csls(unit([1.0, 0.0]), tracks, k=10)))


def test_query_z_penalises_a_track_that_matches_everything():
    q = unit([1.0, 0.0])
    others = unit(np.array([[0.0, 1.0], [0.7, 0.7]]))
    generalist = unit([0.58, 0.58])          # decent against every query
    specialist = unit([1.0, 0.0])            # strong here, weak elsewhere
    tracks = np.vstack([generalist, specialist])
    z = query_z(q, tracks, np.vstack([q, others]))
    assert z[1] > z[0]


def test_all_methods_return_one_score_per_track():
    q = unit([1.0, 0.0])
    tracks = unit(np.array([[1.0, 0.1], [0.1, 1.0], [0.7, 0.7]]))
    segs = [unit(np.array([t, t])) for t in tracks]
    for scores in (cosine(q, tracks), max_segment(q, segs),
                   mean_plus_max(q, tracks, segs), csls(q, tracks),
                   query_z(q, tracks, np.vstack([q, unit([0.0, 1.0])]))):
        assert np.asarray(scores).shape == (3,)
