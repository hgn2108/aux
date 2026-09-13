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


# --- routers -------------------------------------------------------------------------

def test_rule_router_separates_the_two_query_families():
    from aux.route import RuleRouter

    router = RuleRouter()
    assert router.route("songs about heartbreak").alpha < 0.2
    assert router.route("smooth jazz with brushed drums").alpha > 0.8
    # Both kinds of signal present: neither channel should be switched off.
    assert 0.2 < router.route("jazz songs about money").alpha < 0.8


def test_rule_router_defaults_to_audio_when_it_recognises_nothing():
    from aux.route import RuleRouter
    from aux.route.rules import ACOUSTIC_ALPHA

    route = RuleRouter().route("qwerty zxcvb")
    assert route.alpha == ACOUSTIC_ALPHA
    assert "no clear signal" in route.reason


def test_route_rejects_a_weight_outside_the_unit_interval():
    from aux.route import Route

    with pytest.raises(ValueError, match=r"alpha must be in \[0, 1\]"):
        Route(1.4, "why", "test")


def test_route_label_reads_for_a_listener():
    from aux.route import Route

    assert Route(1.0, "", "t").label == "sound"
    assert Route(0.0, "", "t").label == "lyrics"
    assert Route(0.5, "", "t").label == "sound and lyrics"


def test_permutation_test_is_exact_for_small_samples_and_symmetric():
    from aux.eval import permutation_test

    a = np.array([0.5, 0.6, 0.7, 0.8])
    b = np.array([0.1, 0.2, 0.3, 0.4])
    forward = permutation_test(a, b)
    assert forward["exact"] is True
    assert forward["observed"] == pytest.approx(0.4)
    # Two-sided, so swapping the arguments changes only the sign of the difference.
    assert permutation_test(b, a)["p_value"] == pytest.approx(forward["p_value"])


def test_permutation_test_reports_no_effect_for_identical_systems():
    from aux.eval import permutation_test

    x = np.array([0.2, 0.4, 0.6, 0.8, 0.5])
    assert permutation_test(x, x)["p_value"] == pytest.approx(1.0)


# --- demo app corpus gating ----------------------------------------------------------

def test_personal_library_is_unavailable_on_a_public_instance(monkeypatch):
    """The personal corpus must never be offered where it could be served to others."""
    from aux.app import data

    monkeypatch.setenv("AUX_PUBLIC", "1")
    assert data.is_public() is True
    assert data.available_corpora() == ["fma"]

    with pytest.raises(RuntimeError, match="not available on this instance"):
        data.load_corpus("personal", encoder=None)


def test_public_flag_treats_falsey_strings_as_local(monkeypatch):
    from aux.app import data

    for value in ("", "0", "false", "False"):
        monkeypatch.setenv("AUX_PUBLIC", value)
        assert data.is_public() is False


def test_personal_library_needs_audio_present_not_just_a_directory(monkeypatch, tmp_path):
    from aux.app import data

    monkeypatch.delenv("AUX_PUBLIC", raising=False)
    monkeypatch.setattr(data, "MUSIC", tmp_path / "music")
    assert data.available_corpora() == ["fma"]      # missing entirely

    (tmp_path / "music").mkdir()
    assert data.available_corpora() == ["fma"]      # present but empty

    (tmp_path / "music" / "a.mp3").write_bytes(b"")
    assert data.available_corpora() == ["fma", "personal"]
