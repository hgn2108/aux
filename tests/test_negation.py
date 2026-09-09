"""Negation-parsing tests."""

from __future__ import annotations

import pytest

from aux.query import parse


@pytest.mark.parametrize("query", [
    "solo piano",
    "romantic classical piano",
    "hip hop for running",
    "something warm and nostalgic",
])
def test_queries_without_negation_pass_through(query):
    p = parse(query)
    assert p.positive == query
    assert p.negatives == ()
    assert not p.has_negation


@pytest.mark.parametrize(("query", "positive", "negatives"), [
    ("solo piano, no vocals", "solo piano", ("vocals",)),
    ("acoustic guitar, no vocals", "acoustic guitar", ("vocals",)),
    ("hip hop but not aggressive", "hip hop", ("aggressive",)),
    ("jazz without drums", "jazz", ("drums",)),
    ("upbeat music, not sad", "upbeat music", ("sad",)),
    ("dance music excluding techno", "dance music", ("techno",)),
])
def test_basic_negation_forms(query, positive, negatives):
    p = parse(query)
    assert p.positive == positive
    assert p.negatives == negatives


def test_multiple_negations_are_separated():
    """A negated phrase ends at the next clause boundary, not the end of the query."""
    p = parse("jazz, no vocals and no drums")
    assert p.positive == "jazz"
    assert p.negatives == ("vocals", "drums")


def test_word_boundary_prevents_false_matches():
    """"piano" contains "no" — matching inside a word would wreck ordinary queries."""
    p = parse("piano and snow patrol vibes")
    assert not p.has_negation
    assert p.positive == "piano and snow patrol vibes"


def test_longest_marker_wins():
    p = parse("hip hop but not mumble rap")
    assert p.positive == "hip hop"
    assert p.negatives == ("mumble rap",)


def test_bare_negation_gets_something_to_retrieve_against():
    """"no vocals" alone still has to rank the library somehow."""
    p = parse("no vocals")
    assert p.positive == "music"
    assert p.negatives == ("vocals",)


def test_parsing_is_idempotent_on_its_own_positive():
    p = parse("solo piano, no vocals")
    assert parse(p.positive).positive == "solo piano"
