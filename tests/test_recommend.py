"""Recommender tests."""

from __future__ import annotations

import numpy as np
import pytest

from aux.recommend import Recommender


def unit(x):
    x = np.asarray(x, dtype=np.float32)
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


@pytest.fixture
def rec():
    # audio: 0 and 1 are close; lyrics: 0 and 2 are close. So the modalities disagree,
    # which is the only configuration that makes a fusion test meaningful.
    audio = unit(np.array([[1, 0], [0.99, 0.14], [0, 1], [0.1, 1]]))
    lyrics = unit(np.array([[1, 0], [0, 1], [0.99, 0.14], [0, 1]]))
    return Recommender(audio, lyric_vectors=lyrics, has_lyrics=np.array([1, 1, 1, 1], bool))


def test_query_track_is_excluded(rec):
    for modality in ("audio", "lyrics", "fused"):
        assert all(r.index != 0 for r in rec.recommend(0, modality=modality))


def test_returns_the_requested_number(rec):
    assert len(rec.recommend(0, top_k=2)) == 2
    assert len(rec.recommend(0, top_k=99)) == len(rec) - 1


def test_scores_descend(rec):
    for modality in ("audio", "lyrics", "fused"):
        scores = [r.score for r in rec.recommend(0, modality=modality)]
        assert scores == sorted(scores, reverse=True)


def test_ranks_are_sequential_from_one(rec):
    assert [r.rank for r in rec.recommend(0, top_k=3)] == [1, 2, 3]


# --- fusion behaviour ---------------------------------------------------------------

def test_alpha_one_reproduces_audio_only(rec):
    fused = [r.index for r in rec.recommend(0, modality="fused", alpha=1.0)]
    audio = [r.index for r in rec.recommend(0, modality="audio")]
    assert fused == audio


def test_alpha_zero_reproduces_lyrics_only(rec):
    fused = [r.index for r in rec.recommend(0, modality="fused", alpha=0.0)]
    lyrics = [r.index for r in rec.recommend(0, modality="lyrics")]
    assert fused == lyrics


def test_the_modalities_actually_disagree(rec):
    """Guards the fixture: if both modalities agreed, the two tests above prove nothing."""
    assert rec.recommend(0, modality="audio")[0].index != \
           rec.recommend(0, modality="lyrics")[0].index


def test_alpha_interpolates_between_them(rec):
    top = {a: rec.recommend(0, modality="fused", alpha=a)[0].index for a in (0.0, 0.5, 1.0)}
    assert top[0.0] != top[1.0]
    assert top[0.5] in (top[0.0], top[1.0])


def test_per_query_normalisation_keeps_alpha_meaningful():
    """Raw cosines with different spreads would let one modality dominate regardless of alpha."""
    audio = unit(np.array([[1, 0], [0.9, 0.1], [0.8, 0.2]]))
    lyrics = unit(np.array([[1, 0], [0.1, 0.9], [0.99, 0.1]]))
    r = Recommender(audio, lyric_vectors=lyrics, has_lyrics=np.ones(3, bool))
    scores = r.score(0, modality="fused", alpha=0.5)
    assert np.isfinite(scores).all()
    assert scores.max() <= 1.0 + 1e-6


# --- missing modality ---------------------------------------------------------------

def test_a_candidate_without_lyrics_keeps_its_audio_score():
    """Zeroing it would make fusion a vocal-music filter, sinking every instrumental."""
    audio = unit(np.array([[1, 0], [0.99, 0.1], [0, 1]]))
    lyrics = unit(np.array([[1, 0], [0, 1], [0, 1]]))
    r = Recommender(audio, lyric_vectors=lyrics, has_lyrics=np.array([True, False, True]))
    top = r.recommend(0, modality="fused", alpha=0.5)
    assert top[0].index == 1                      # acoustically nearest, despite no lyrics
    assert top[0].lyric_score is None             # and says so rather than faking a zero


def test_a_query_without_lyrics_falls_back_to_audio():
    audio = unit(np.array([[1, 0], [0.99, 0.1], [0, 1]]))
    lyrics = unit(np.array([[1, 0], [0, 1], [0, 1]]))
    r = Recommender(audio, lyric_vectors=lyrics, has_lyrics=np.array([False, True, True]))
    fused = [x.index for x in r.recommend(0, modality="fused")]
    assert fused == [x.index for x in r.recommend(0, modality="audio")]


def test_lyrics_mode_without_lyric_vectors_raises():
    with pytest.raises(ValueError):
        Recommender(unit(np.eye(3))).recommend(0, modality="lyrics")


def test_unknown_modality_raises(rec):
    with pytest.raises(ValueError):
        rec.recommend(0, modality="telepathy")


# --- text-to-track and explanations --------------------------------------------------

def test_text_query_ranks_by_the_audio_space(rec):
    out = rec.recommend_from_text(unit([1, 0]), top_k=2)
    assert out[0].index in (0, 1)
    assert out[0].rank == 1


def test_explanations_are_deterministic_and_mention_both_modalities(rec):
    r = rec.recommend(0, modality="fused")[0]
    assert r.explain() == r.explain()
    assert "acoustic" in r.explain() and "lyrical" in r.explain()


def test_explanation_says_when_lyrics_are_missing():
    audio = unit(np.array([[1, 0], [0.9, 0.1]]))
    lyrics = unit(np.array([[1, 0], [0, 1]]))
    r = Recommender(audio, lyric_vectors=lyrics, has_lyrics=np.array([True, False]))
    assert "no lyrics available" in r.recommend(0, modality="fused")[0].explain()


def test_score_matrix_shape(rec):
    assert rec.score_matrix(modality="fused", alpha=0.5).shape == (len(rec), len(rec))


def test_normaliser_choice_changes_fused_ranking_but_not_the_endpoints():
    """z-score and min-max must agree where alpha collapses onto one modality."""
    audio = unit(np.array([[1.0, 0.0], [0.9, 0.4], [0.0, 1.0], [-1.0, 0.2]]))
    lyrics = unit(np.array([[0.0, 1.0], [0.1, 1.0], [1.0, 0.1], [1.0, 0.0]]))
    has = np.ones(4, dtype=bool)

    z = Recommender(audio, lyric_vectors=lyrics, has_lyrics=has, normaliser="zscore")
    m = Recommender(audio, lyric_vectors=lyrics, has_lyrics=has, normaliser="minmax")

    # alpha=1 is audio-only and alpha=0 is lyrics-only, so normalisation is a monotone
    # transform of a single channel and cannot reorder either endpoint.
    for alpha in (0.0, 1.0):
        zo = np.argsort(-z.score(0, modality="fused", alpha=alpha))
        mo = np.argsort(-m.score(0, modality="fused", alpha=alpha))
        assert list(zo) == list(mo)

    # In between, the two normalisers weight the channels differently.
    assert not np.allclose(z.score(0, modality="fused", alpha=0.5),
                           m.score(0, modality="fused", alpha=0.5))


def test_unknown_normaliser_is_rejected():
    with pytest.raises(ValueError, match="unknown normaliser"):
        Recommender(unit(np.eye(3)), normaliser="softmax")


def test_displayed_scores_stay_bounded_under_zscore_ranking():
    """`explain()` bands scores into low/moderate/high, so display must stay in [0, 1]."""
    audio = unit(np.array([[1.0, 0.0], [0.8, 0.6], [0.0, 1.0]]))
    lyrics = unit(np.array([[0.0, 1.0], [0.5, 0.9], [1.0, 0.0]]))
    rec = Recommender(audio, lyric_vectors=lyrics, has_lyrics=np.ones(3, bool),
                      normaliser="zscore")
    for r in rec.recommend(0, modality="fused", top_k=2, alpha=0.5):
        assert 0.0 <= r.audio_score <= 1.0
        assert r.lyric_score is None or 0.0 <= r.lyric_score <= 1.0
        assert "similarity" in r.explain()


def test_blend_is_the_only_implementation_of_the_rule():
    """The blend was written out five times before this existed.

    Two of those copies were in the app and two in evaluation scripts, so the published
    numbers and the shipped behaviour could have drifted apart without anything failing.
    This checks the duplicates have not come back.
    """
    import re
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    sources = [root / "app.py", *(root / "scripts").glob("*.py"),
               *(root / "src" / "aux").rglob("*.py")]
    # The arithmetic that defines the blend. Only recommend.py should contain it.
    pattern = re.compile(r"alpha\s*\*\s*\w+\s*\+\s*\(1\s*-\s*alpha\)\s*\*")
    offenders = [p.name for p in sources
                 if pattern.search(p.read_text()) and p.name != "recommend.py"]
    assert offenders == [], f"blend reimplemented in: {offenders}"
