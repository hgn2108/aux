"""Search entry-point tests, with a stub encoder so no model or audio is needed."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from aux.search import Library, SearchResponse


class StubEncoder:
    """Maps a handful of known words to fixed directions in a 3-d space."""

    name, version = "stub", "test"
    sample_rate, segment_seconds, embedding_dim = 48_000, 10.0, 3
    DIRECTIONS = {"piano": [1, 0, 0], "drums": [0, 1, 0], "vocals": [0, 0, 1]}

    def embed_text(self, texts):
        out = []
        for t in texts:
            v = np.zeros(3, dtype=np.float32)
            for word, d in self.DIRECTIONS.items():
                if word in t.lower():
                    v += np.array(d, dtype=np.float32)
            out.append(v if v.any() else np.array([0.4, 0.4, 0.4], np.float32))
        return np.stack(out)


@pytest.fixture
def library(monkeypatch, tmp_path):
    # "d" is drums with heavy vocals and "e" is quieter drums with none. Negation has to
    # reorder those two to be doing anything; the earlier fixture had gaps too wide for any
    # penalty to cross, so it passed while testing nothing.
    paths = [tmp_path / f"{n}.mp3" for n in "abcde"]
    vectors = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1],
                        [0, 0.5, 0.87], [0, 0.4, 0]], dtype=np.float32)
    monkeypatch.setattr("aux.search.build_index", lambda *a, **k: (vectors, paths, []))
    return Library(tmp_path, StubEncoder())


def test_search_ranks_the_matching_track_first(library):
    r = library.search("piano")
    assert r.results[0].path.name == "a.mp3"
    assert r.results[0].rank == 1


def test_planner_is_off_by_default(library):
    """The measured default: applied indiscriminately the planner nets to nothing."""
    assert library.search("piano").used_planner is False
    assert library.search("piano").rewritten is None


def test_negation_is_always_applied(library):
    """Negation needs no flag, it is structural, and shipped (DEC-014)."""
    v_plain = [r.path.name for r in library.search("drums").results]
    v_neg = [r.path.name for r in library.search("drums, no vocals").results]
    assert v_plain[0] == v_neg[0] == "b.mp3"                    # still finds the drums track
    assert v_plain.index("d.mp3") < v_plain.index("e.mp3")      # vocal-heavy ranks above
    assert v_neg.index("d.mp3") > v_neg.index("e.mp3")          # and below, once negated


def test_a_planner_failure_degrades_to_the_plain_path(library):
    class FailingPlanner:
        version = "broken"

        def plan(self, q):
            from aux.plan import passthrough
            return passthrough(q), {"fallback": True, "error": "boom"}

    r = library.search("piano", use_planner=True, planner=FailingPlanner())
    assert r.used_planner is False
    assert r.results[0].path.name == "a.mp3"


def test_low_coverage_flags_a_weak_best_match(library):
    """Phase B: raw top score detected a removed genre in 20 of 22 cases."""
    # top score here is 1.0, so a threshold above it flags low coverage and one below does not.
    assert library.search("piano", low_coverage_score=1.5).low_coverage is True
    assert library.search("piano", low_coverage_score=0.1).low_coverage is False


def test_k_limits_results(library):
    assert len(library.search("piano", k=2).results) == 2


def test_similar_to_excludes_the_reference_track(library):
    """Regression: with a sentinel score and k larger than the library, the reference
    survived the slice and came back as its own neighbour."""
    r = library.similar_to(library.paths[0], k=10)
    assert all(res.path != library.paths[0] for res in r.results)
    assert len(r.results) == len(library) - 1


def test_similar_to_rejects_an_unknown_track(library):
    with pytest.raises(KeyError):
        library.similar_to(Path("/nowhere/x.mp3"))


def test_response_carries_the_confidence_signals(library):
    r = library.search("piano")
    assert isinstance(r, SearchResponse)
    assert r.top_score > 0
    assert np.isfinite(r.z_top)
