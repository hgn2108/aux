"""Tests for the semantic benchmark: label agreement, fusion, and the oracle bound."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from eval_semantic import fuse, per_query_metrics, score_queries  # noqa: E402
from rate_themes import cohen_kappa, sample_pairs  # noqa: E402


# --- agreement -----------------------------------------------------------------------

def test_kappa_is_zero_for_a_labeller_that_always_says_no():
    """Raw agreement flatters an always-negative labeller; kappa must not."""
    judged = [(False, False)] * 38 + [(False, True)] * 2
    kappa, agree = cohen_kappa(judged)
    assert agree == pytest.approx(0.95)
    assert kappa == pytest.approx(0.0, abs=1e-9)


def test_kappa_is_one_for_perfect_agreement_and_negative_for_anticorrelation():
    assert cohen_kappa([(True, True), (False, False)] * 5)[0] == pytest.approx(1.0)
    assert cohen_kappa([(True, False), (False, True)] * 5)[0] < 0


def test_sample_is_balanced_on_the_model_label_not_the_human_one():
    labels = {f"h{i}": {"themes": ["money"] if i % 2 == 0 else [], "confident": True}
              for i in range(40)}
    transcripts = {f"h{i}": "words" for i in range(40)}
    pairs = sample_pairs(labels, transcripts, n=20)

    assert len(pairs) == 20
    positive = sum(1 for h, theme in pairs if theme in labels[h]["themes"])
    assert positive == 10          # exactly half, though positives are 1/24 of the pool


def test_sample_tops_up_rather_than_shrinking_when_positives_are_scarce():
    """A thinly labelled theme set must not quietly reduce the size of the check."""
    labels = {f"h{i}": {"themes": ["money"] if i < 3 else [], "confident": True}
              for i in range(40)}
    pairs = sample_pairs(labels, {f"h{i}": "x" for i in range(40)}, n=20)

    assert len(pairs) == 20
    positive = sum(1 for h, theme in pairs if theme in labels[h]["themes"])
    assert positive == 3           # every positive there is, the rest made up from negatives


def test_sample_skips_tracks_the_labeller_was_not_confident_about():
    labels = {"a": {"themes": ["money"], "confident": False},
              "b": {"themes": ["money"], "confident": True}}
    pairs = sample_pairs(labels, {"a": "x", "b": "x"}, n=10)
    assert {h for h, _ in pairs} == {"b"}


def test_sample_is_deterministic():
    labels = {f"h{i}": {"themes": ["money"] if i % 3 == 0 else [], "confident": True}
              for i in range(30)}
    transcripts = {f"h{i}": "x" for i in range(30)}
    assert sample_pairs(labels, transcripts, 12) == sample_pairs(labels, transcripts, 12)


# --- scoring -------------------------------------------------------------------------

def test_score_queries_marks_tracks_without_the_modality_unreachable():
    queries = np.array([[1.0, 0.0]], dtype=np.float32)
    items = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
    usable = np.array([True, True, False])
    scores = score_queries(queries, items, usable)
    assert scores[0, 2] == -np.inf
    assert scores[0, 0] == pytest.approx(1.0)


def test_fusion_falls_back_to_audio_where_a_track_has_no_lyrics():
    """Zeroing a missing modality would push every instrumental to the bottom."""
    audio = np.array([[0.9, 0.1, 0.5]])
    lyric = np.array([[0.2, 0.8, 0.0]])
    has = np.array([True, True, False])
    norm = lambda x: x  # noqa: E731 - identity keeps the assertion readable

    fused = fuse(audio, lyric, 0.5, has, norm)
    assert fused[0, 0] == pytest.approx(0.5 * 0.9 + 0.5 * 0.2)
    assert fused[0, 2] == pytest.approx(0.5)      # audio score, untouched by the blend


def test_fusion_collapses_onto_each_modality_at_the_ends_of_the_sweep():
    audio = np.array([[0.9, 0.1, 0.4]])
    lyric = np.array([[0.2, 0.8, 0.3]])
    has = np.ones(3, dtype=bool)
    norm = lambda x: x  # noqa: E731

    assert np.allclose(fuse(audio, lyric, 1.0, has, norm), audio)
    assert np.allclose(fuse(audio, lyric, 0.0, has, norm), lyric)


def test_oracle_is_an_upper_bound_on_both_modalities():
    relevant = np.array([[True, False, False, True],
                         [False, True, True, False]])
    audio = np.array([[0.9, 0.8, 0.1, 0.2], [0.9, 0.1, 0.2, 0.8]])
    lyric = np.array([[0.1, 0.9, 0.8, 0.2], [0.1, 0.9, 0.8, 0.2]])

    nd_audio = per_query_metrics(audio, relevant, 2)
    nd_lyric = per_query_metrics(lyric, relevant, 2)
    oracle = np.maximum(nd_audio, nd_lyric)

    assert oracle.mean() >= nd_audio.mean()
    assert oracle.mean() >= nd_lyric.mean()
    # Each modality wins one query here, so the oracle must strictly beat both.
    assert oracle.mean() > max(nd_audio.mean(), nd_lyric.mean())
