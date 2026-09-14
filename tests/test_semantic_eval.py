"""Tests for the semantic benchmark: label agreement, fusion, and the oracle bound."""

from __future__ import annotations

import sys
from pathlib import Path

import json

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

def test_public_instance_serves_the_bundle_and_never_the_audio():
    """A deployment may rank the personal corpus, but must not be able to play it.

    Serving the embeddings is what lets a public demo show the lyric modality at all --
    the Creative Commons corpus cannot. Serving the audio would be redistribution.
    """
    from aux.app import data

    if not data._bundle_present():
        pytest.skip("no exported bundle; run scripts/export_demo_corpus.py")

    corpus = data.load_personal_bundle()
    assert corpus.playable is False
    assert corpus.supports_lyrics is True
    assert all(str(t.path) in ("", ".") for t in corpus.tracks)


def test_bundle_carries_no_audio_paths_or_transcripts():
    from aux.app import data

    if not data._bundle_present():
        pytest.skip("no exported bundle; run scripts/export_demo_corpus.py")

    blob = (data.DEMO / "personal_manifest.json").read_text()
    assert ".mp3" not in blob
    assert "/Users/" not in blob
    record = json.loads(blob)["tracks"][0]
    assert set(record) == {"track_id", "title", "artist", "genre", "has_lyrics"}


def test_public_flag_treats_falsey_strings_as_local(monkeypatch):
    from aux.app import data

    for value in ("", "0", "false", "False"):
        monkeypatch.setenv("AUX_PUBLIC", value)
        assert data.is_public() is False


def test_personal_library_needs_audio_or_a_bundle(monkeypatch, tmp_path):
    """Offering a corpus whose files are absent crashes on selection, so check first."""
    from aux.app import data

    monkeypatch.delenv("AUX_PUBLIC", raising=False)
    monkeypatch.setattr(data, "MUSIC", tmp_path / "music")
    monkeypatch.setattr(data, "DEMO", tmp_path / "demo")
    assert data.available_corpora() == ["fma"]      # neither audio nor bundle

    (tmp_path / "music").mkdir()
    assert data.available_corpora() == ["fma"]      # directory present but empty

    (tmp_path / "music" / "a.mp3").write_bytes(b"")
    assert data.available_corpora() == ["fma", "personal"]


def test_bundle_alone_is_enough_to_offer_the_corpus(monkeypatch, tmp_path):
    from aux.app import data

    monkeypatch.setattr(data, "MUSIC", tmp_path / "nothing")
    monkeypatch.setattr(data, "DEMO", tmp_path / "demo")
    (tmp_path / "demo").mkdir()
    (tmp_path / "demo" / "personal_manifest.json").write_text("{}")
    (tmp_path / "demo" / "personal_vectors.npz").write_bytes(b"")
    assert data.available_corpora() == ["fma", "personal"]


def test_personal_corpus_label_condition_matches_the_loader(monkeypatch, tmp_path):
    """The sidebar label and the loader must agree on whether audio is available.

    They were written separately and drifted: the label said "no audio" on a machine where
    playback works, so the corpus looked broken to its own owner.
    """
    from aux.app import data

    monkeypatch.delenv("AUX_PUBLIC", raising=False)
    monkeypatch.setattr(data, "MUSIC", tmp_path / "music")
    (tmp_path / "music").mkdir()
    assert data.personal_is_local() is True       # audio present, not deployed

    monkeypatch.setenv("AUX_PUBLIC", "1")
    assert data.personal_is_local() is False      # deployed: bundle only

    monkeypatch.delenv("AUX_PUBLIC", raising=False)
    monkeypatch.setattr(data, "MUSIC", tmp_path / "gone")
    assert data.personal_is_local() is False      # no audio on disk: bundle only


def test_display_shows_written_artist_names_not_the_matching_key():
    """`TrackMeta.artist` is case-folded because it decides same-artist relevance.

    Showing it directly put "a boogie wit da hoodie" on screen, so display names are
    resolved separately -- and a filename with no separable artist must not print the same
    text twice.
    """
    from aux.app.data import Corpus

    class T:
        def __init__(self, title, artist, genre):
            self.title, self.artist, self.genre = title, artist, genre
            self.track_id, self.path = 0, Path()

    tracks = [T("Skeezers", "a boogie wit da hoodie", "hiphop_rnb"),
              T("Some Live Set 2023", "Some   Live Set 2023", "hiphop_rnb")]
    corpus = Corpus("test", tracks, np.zeros((2, 4)), None, None, playable=False,
                    anonymous=False, note="", artists=["A Boogie Wit Da Hoodie",
                                                       "Some   Live Set 2023"])

    assert corpus.display(0) == ("Skeezers", "A Boogie Wit Da Hoodie · hiphop_rnb")
    # Artist and title are the same string up to whitespace: show the genre alone.
    assert corpus.display(1) == ("Some Live Set 2023", "hiphop_rnb")


def test_clean_display_title_strips_filename_debris_only():
    from aux.app.data import clean_display_title

    assert clean_display_title("05 A-Trak - Me & My Sneakers.mp3") == "A-Trak - Me & My Sneakers"
    assert clean_display_title("12. Crossover") == "Crossover"
    # A real title is left alone, including one that opens with a number.
    assert clean_display_title("broke Pimpin") == "broke Pimpin"
    assert clean_display_title("Just a Feelin'") == "Just a Feelin'"
    assert clean_display_title("") == ""


def test_explanation_is_hidden_where_only_one_modality_exists():
    """On a sound-only corpus every line would read "no lyrics available"."""
    from aux.recommend import Recommendation

    both = Recommendation(index=1, rank=1, score=0.9, audio_score=0.9, lyric_score=0.4)
    assert "lyrical similarity" in both.explain()

    sound_only = Recommendation(index=1, rank=1, score=0.9, audio_score=0.9, lyric_score=None)
    assert "no lyrics available" in sound_only.explain()


def test_fma_falls_back_to_the_bundle_when_the_corpus_is_absent(monkeypatch, tmp_path):
    """A deployment has neither the 7.4 GB of audio nor the 248 MB metadata CSV.

    Without this fallback the app crashed on the default corpus the moment it was hosted.
    """
    from aux.app import data

    monkeypatch.setattr(data, "DEFAULT_AUDIO", tmp_path / "absent")
    monkeypatch.setattr(data, "DEFAULT_METADATA", tmp_path / "absent.csv")
    assert data.fma_is_local() is False

    if not data._bundle_present("fma"):
        pytest.skip("no exported bundle; run scripts/export_demo_corpus.py --corpus fma")

    corpus = data.load_corpus("fma", encoder=None)
    assert len(corpus.tracks) > 0
    assert corpus.playable is True          # Creative Commons: the audio ships with it
    assert corpus.supports_lyrics is False  # and it has no lyric channel
    assert all(t.path.exists() for t in corpus.tracks)


def test_a_bundled_corpus_needs_no_encoder(monkeypatch, tmp_path):
    """Loading stored vectors must not pull in a 2.5GB model.

    It did: the encoder was passed to `load_corpus` unconditionally, and Python evaluates
    arguments before the call, so a deployment -- where both corpora are bundles -- loaded
    MuQ-MuLan before rendering a single track.
    """
    from aux.app import data

    monkeypatch.setattr(data, "DEFAULT_AUDIO", tmp_path / "absent")
    monkeypatch.setattr(data, "DEFAULT_METADATA", tmp_path / "absent.csv")
    monkeypatch.setattr(data, "MUSIC", tmp_path / "absent")

    assert data.needs_encoder("fma") is False
    assert data.needs_encoder("personal") is False

    # And with the corpora present, it does.
    monkeypatch.setattr(data, "DEFAULT_AUDIO", tmp_path)
    monkeypatch.setattr(data, "DEFAULT_METADATA", tmp_path)
    assert data.needs_encoder("fma") is True
