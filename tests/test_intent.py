"""Intent-match tests."""

from __future__ import annotations

import numpy as np
import pytest

from aux.eval import INTENT_TERMS, intent_match, terms_in

FEATURES = ("loudness", "onset", "brightness")


def test_finds_terms_and_their_direction():
    found = dict((t, (f, d)) for t, f, d in terms_in("slow and quiet piano"))
    assert found["slow"] == ("onset", -1)
    assert found["quiet"] == ("loudness", -1)


def test_ignores_words_with_no_physical_direction():
    """"nostalgic" has no honest waveform mapping, so it must not be scored."""
    assert terms_in("warm nostalgic dreamy music") == [("warm", "brightness", -1)]


def test_no_measurable_terms_returns_none_not_zero():
    """"No opinion" and "no effect" are different and must not be conflated."""
    assert intent_match("songs about yearning", np.zeros((3, 3))) is None


def test_audio_matching_the_words_scores_positive():
    """Query asks for quiet; retrieved tracks are quiet."""
    retrieved = np.array([[-1.5, 0.0, 0.0], [-1.2, 0.0, 0.0]])  # low loudness z
    assert intent_match("quiet music", retrieved, FEATURES) > 0


def test_audio_contradicting_the_words_scores_negative():
    retrieved = np.array([[1.5, 0.0, 0.0], [1.2, 0.0, 0.0]])   # loud
    assert intent_match("quiet music", retrieved, FEATURES) < 0


def test_audio_ignoring_the_words_scores_about_zero():
    retrieved = np.zeros((4, 3))
    assert intent_match("fast loud music", retrieved, FEATURES) == pytest.approx(0.0)


def test_opposed_terms_are_averaged_not_double_counted():
    """"fast" and "driving" both point at onset; two words are not twice the evidence."""
    retrieved = np.array([[0.0, 1.0, 0.0]])
    one = intent_match("fast music", retrieved, FEATURES)
    two = intent_match("fast driving music", retrieved, FEATURES)
    assert one == pytest.approx(two)


def test_terms_across_different_features_combine():
    retrieved = np.array([[-1.0, -1.0, 0.0]])   # quiet and slow
    assert intent_match("quiet slow music", retrieved, FEATURES) == pytest.approx(1.0)


def test_substring_matches_do_not_fire():
    """"slowly" and "brightness" must not trigger "slow" and "bright"."""
    assert terms_in("the track unfolds slowly") == []


def test_every_term_maps_to_a_known_feature():
    assert all(f in FEATURES for f, _ in INTENT_TERMS.values())
    assert all(d in (-1, 1) for _, d in INTENT_TERMS.values())
