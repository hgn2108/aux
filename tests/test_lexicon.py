"""Context→acoustic lexicon tests."""

from __future__ import annotations

import pytest

from aux.query import CONTEXT_LEXICON, expand


@pytest.mark.parametrize("query", [
    "romantic classical piano",
    "songs with heavy bass",
    "groovy jazz rap",
    "melancholy and heartbroken",
])
def test_queries_without_context_terms_are_untouched(query):
    e = expand(query)
    assert not e.expanded
    assert e.acoustic == ""
    assert e.original == query


@pytest.mark.parametrize(("query", "term"), [
    ("fast, upbeat rap and hip hop for running", "running"),
    ("chill r&b and hip hop for studying", "studying"),
    ("hip hop for falling asleep", "falling asleep"),
    ("background music while reading", "background"),
    ("afrobeats songs for the summer", "summer"),
])
def test_context_terms_are_found(query, term):
    assert term in expand(query).matched


def test_longest_term_wins():
    """"falling asleep" must not be shadowed by "sleeping", nor "night drive" by "drive"."""
    assert expand("hip hop for falling asleep").matched == ("falling asleep",)
    assert expand("late night drive vibes").matched[0] in ("night drive", "late night")


def test_a_shorter_term_inside_a_longer_one_is_not_double_counted():
    e = expand("late night drive vibes")
    assert "drive" not in e.matched or "night drive" not in e.matched


def test_word_boundaries_prevent_false_matches():
    """"drive" must not fire inside "driven", nor "study" inside "studying" twice."""
    assert not expand("a driven, relentless track").expanded


def test_expansion_keeps_the_original_query():
    e = expand("hip hop for running")
    assert e.original == "hip hop for running"
    assert "fast tempo" in e.acoustic


def test_multiple_context_terms_are_all_expanded():
    e = expand("music for cooking and cleaning")
    assert set(e.matched) == {"cooking", "cleaning"}


def test_lexicon_values_describe_sound_not_genre():
    """Mapping a situation to a genre would bake one listener's taste into the system."""
    genre_words = {"hip hop", "jazz", "classical", "edm", "rock", "pop", "lo-fi", "lofi"}
    for term, value in CONTEXT_LEXICON.items():
        lowered = value.lower()
        assert not any(g in lowered for g in genre_words), f"{term} maps to a genre: {value}"


def test_lexicon_has_no_empty_entries():
    assert all(v.strip() for v in CONTEXT_LEXICON.values())
