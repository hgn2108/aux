"""Loading a corpus for the demo: vectors, metadata, and the modalities it supports.

Two corpora, and the difference between them is a finding rather than an inconvenience.
FMA is Creative Commons, so its audio can be played in a browser, but 56% of a sampled 75
clips are instrumental with a median transcript of 11 words — there is no lyric channel to
search. The personal library has both modalities but cannot be redistributed, so it runs
locally and its tracks are shown under stable pseudonyms.

Everything here is cached by Streamlit, so the encoder and the index load once per session
rather than once per interaction.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
CACHE = ROOT / ".cache"
RESULTS = ROOT / "results"


@dataclass(frozen=True, slots=True)
class Corpus:
    name: str
    tracks: list
    audio: np.ndarray
    lyrics: np.ndarray | None
    has_lyrics: np.ndarray | None
    playable: bool
    """Whether the audio may be served to a browser. False for anything not licensed for
    redistribution, which is enforced here rather than left to the page."""
    anonymous: bool
    """Whether track identity must be hidden. True for the personal library."""
    note: str

    @property
    def supports_lyrics(self) -> bool:
        return self.lyrics is not None

    def display(self, index: int) -> tuple[str, str]:
        """(title, subtitle) for one track, honouring the corpus's privacy setting."""
        track = self.tracks[index]
        if self.anonymous:
            return f"Track {track.track_id:03d}", track.genre
        return track.title or track.path.stem, f"{track.artist} · {track.genre}"


def _index(paths, cache_path: Path, encoder):
    from aux.index import build_index

    vectors, kept, _ = build_index(None, encoder, paths=paths, n_segments=5,
                                   cache_path=cache_path, progress_every=0)
    return vectors, {str(p) for p in kept}


def load_corpus(which: str, encoder, *, limit: int | None = None) -> Corpus:
    """Load one corpus. `limit` trims the track list before indexing, for a lighter demo."""
    if which == "fma":
        from aux.data import balanced_subset, load_tracks

        tracks = balanced_subset(load_tracks(), per_genre=250)
        if limit:
            # Keep the genre balance while trimming: take a fixed share from each.
            per = max(1, limit // len({t.genre for t in tracks}))
            seen: dict[str, int] = {}
            trimmed = []
            for t in tracks:
                if seen.get(t.genre, 0) < per:
                    seen[t.genre] = seen.get(t.genre, 0) + 1
                    trimmed.append(t)
            tracks = trimmed
        vectors, kept = _index([t.path for t in tracks], CACHE / "fma_audio_250.npz", encoder)
        tracks = [t for t in tracks if str(t.path) in kept]
        return Corpus("FMA small", tracks, vectors, None, None, playable=True,
                      anonymous=False,
                      note=("Creative Commons. 56% of a sampled 75 clips are instrumental "
                            "and transcripts run a median of 11 words, so this corpus has "
                            "no lyric channel to search."))

    from aux.data import load_personal_tracks

    tracks = load_personal_tracks()
    vectors, kept = _index([t.path for t in tracks], CACHE / "personal.npz", encoder)
    tracks = [t for t in tracks if str(t.path) in kept]

    import sys

    sys.path.insert(0, str(ROOT / "scripts"))
    from eval_recommendation import load_lyric_vectors

    lyrics, has_lyrics = load_lyric_vectors(tracks, "small", "personal")
    return Corpus("personal library", tracks, vectors, lyrics, has_lyrics,
                  playable=False, anonymous=True,
                  note=("Commercially released music: not redistributable, so audio is not "
                        "served and tracks are shown under stable pseudonyms. 79% have a "
                        "reliable transcript at a median of 376 words, which is why the "
                        "multimodal evaluation runs here."))


def load_results() -> dict:
    """Every committed results file, keyed by name, for the findings page."""
    out = {}
    for path in sorted(RESULTS.glob("*.json")):
        try:
            out[path.stem] = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
    return out
