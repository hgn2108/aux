"""Headless smoke tests for the demo app.

`AppTest` runs the script in-process, so a broken import, a bad widget key or an exception
on a tab fails here rather than in front of whoever opened the link. These do not assert on
recommendations -- the evaluation scripts do that -- only that every corpus and mode renders.
"""

from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
APP = str(ROOT / "app.py")

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402


def _run(monkeypatch, *, public: bool, timeout: int = 180):
    if public:
        monkeypatch.setenv("AUX_PUBLIC", "1")
    else:
        monkeypatch.delenv("AUX_PUBLIC", raising=False)
    app = AppTest.from_file(APP, default_timeout=timeout).run()
    assert not app.exception, app.exception
    return app


@pytest.mark.slow
def test_app_starts_and_renders_every_tab(monkeypatch):
    app = _run(monkeypatch, public=True)
    headings = [h.value for h in app.subheader]
    assert any("Use your own music" in h for h in headings)
    assert any("What was measured" in h for h in headings)
    # The corpus picker governs only the browse tab, so it lives inside it.
    assert {w.key for w in app.segmented_control} >= {"corpus", "browse_mode"}


@pytest.mark.slow
def test_switching_to_the_lyric_corpus_keeps_the_app_alive(monkeypatch):
    """The bundled corpus loads from different code than the local one; exercise it."""
    app = _run(monkeypatch, public=True)
    picker = next(w for w in app.segmented_control if w.key == "corpus")
    assert picker.options == ["Demo library", "Creator's library (lyrics available)"]

    picker.set_value("personal").run()
    assert not app.exception, app.exception
    # Widget keys carry the corpus name, so they show which corpus actually loaded.
    keys = {w.key for w in app.pills} | {w.key for w in app.radio}
    assert any("personal_library" in k for k in keys if k)


@pytest.mark.slow
def test_every_match_mode_renders_on_the_lyric_corpus(monkeypatch):
    app = _run(monkeypatch, public=True)
    next(w for w in app.segmented_control if w.key == "corpus").set_value("personal").run()
    mode = app.radio[0]
    assert mode.options == ["Sound", "Lyrics", "Both"]
    for choice in mode.options:
        app.radio[0].set_value(choice).run()
        assert not app.exception, f"{choice}: {app.exception}"


@pytest.mark.slow
def test_browse_offers_both_ways_of_finding_tracks(monkeypatch):
    app = _run(monkeypatch, public=True)
    how = next(w for w in app.segmented_control if w.key == "browse_mode")
    assert how.options == ["By description", "By a track you like"]
    # Description search is the default: it needs nothing from the visitor.
    assert any("Describe what you want to hear" in str(m.value) for m in app.markdown)

    how.set_value("By a track you like").run()
    assert not app.exception, app.exception
    assert any("no listening history" in str(m.value) for m in app.markdown)
