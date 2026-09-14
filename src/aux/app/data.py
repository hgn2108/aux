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
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import os

ROOT = Path(__file__).resolve().parents[3]
CACHE = ROOT / ".cache"
RESULTS = ROOT / "results"
MUSIC = ROOT / "data" / "music"
DEMO = ROOT / "demo"

from aux.data.fma import DEFAULT_AUDIO, DEFAULT_METADATA  # noqa: E402


def is_public() -> bool:
    """Whether this instance is deployed rather than running on the owner's machine.

    Set AUX_PUBLIC=1 wherever the app is hosted. It gates the personal library, which must
    never be served to anyone else -- and which is not present on a deployment anyway,
    since the audio is not committed.
    """
    return os.environ.get("AUX_PUBLIC", "").strip() not in ("", "0", "false", "False")


def available_corpora() -> list[str]:
    """Corpora this instance can actually load, in display order.

    Offering one whose files are absent produces a crash on selection rather than a
    missing option, so availability is checked here instead of assumed.
    """
    out = ["fma"]
    if (MUSIC.exists() and any(MUSIC.rglob("*.mp3"))) or _bundle_present():
        out.append("personal")
    return out


def personal_is_local() -> bool:
    """Whether the personal corpus loads from its audio rather than from the bundle.

    The same condition `load_corpus` branches on, exported so the interface cannot label
    the corpus one way while the loader does the other -- which it did: the sidebar said
    "no audio" on a machine where playback works.
    """
    return not is_public() and MUSIC.exists()


def _bundle_present(prefix: str = "personal") -> bool:
    return ((DEMO / f"{prefix}_manifest.json").exists()
            and (DEMO / f"{prefix}_vectors.npz").exists())


def fma_is_local() -> bool:
    """Whether FMA loads from the downloaded corpus rather than the exported bundle."""
    return DEFAULT_AUDIO.exists() and DEFAULT_METADATA.exists()


def needs_encoder(which: str) -> bool:
    """Whether loading this corpus has to index audio, rather than read stored vectors.

    A bundle carries its vectors, so it needs no model. Asking for one anyway costs a
    2.5GB load before anything renders -- which is what a deployment does on every cold
    start, since a deployment is exactly where both corpora are bundles.
    """
    return fma_is_local() if which == "fma" else personal_is_local()


#: Some FMA titles are the uploader's filename rather than a title: a leading track
#: number, and a ".mp3" left on the end.
_FILENAME_TITLE = re.compile(r"^\s*\d{1,3}\s*[-._ ]\s*|\.(mp3|wav|flac|m4a|ogg)\s*$",
                             re.IGNORECASE)


def clean_display_title(title: str) -> str:
    """Strip filename debris from a title. Display only; nothing is matched on this."""
    cleaned = _FILENAME_TITLE.sub("", (title or "").strip())
    return re.sub(r"\s{2,}", " ", cleaned).strip(" -_")


@dataclass(frozen=True, slots=True)
class Corpus:
    name: str
    tracks: list
    audio: np.ndarray
    lyrics: np.ndarray | None
    has_lyrics: np.ndarray | None
    playable: bool
    """Whether the audio may be served. Creative Commons audio is always playable. The
    personal library is playable only on the owner's own machine: listening to your own
    files locally is not redistribution, and being unable to hear the results makes the
    recommender impossible to inspect. A deployment never loads it at all."""
    anonymous: bool
    """Whether track identity is hidden. Off locally, where the point is to recognise the
    tracks, and irrelevant on a deployment, which cannot load this corpus. Published
    artefacts -- results files, the README -- stay pseudonymous regardless."""
    note: str
    artists: list[str] | None = None
    """Artist names as written, for display.

    `TrackMeta.artist` is a case-folded matching key -- it decides which pairs count as
    same-artist in the evaluation -- so showing it directly put "a boogie wit da hoodie" on
    screen. Display names are resolved once at load time instead.
    """

    @property
    def supports_lyrics(self) -> bool:
        return self.lyrics is not None

    def display(self, index: int) -> tuple[str, str]:
        """(title, subtitle) for one track, honouring the corpus's privacy setting."""
        track = self.tracks[index]
        if self.anonymous:
            return f"Track {track.track_id:03d}", track.genre
        title = clean_display_title(track.title) or track.path.stem
        artist = self.artists[index] if self.artists else track.artist
        # Some filenames carry no "Artist - Title" separator, so both fields fall back to
        # the whole stem. Printing it twice reads as a bug; show the genre alone.
        def flat(text: str) -> str:
            return " ".join(text.split()).casefold()

        if not artist or flat(artist) == flat(title):
            return title, track.genre
        return title, f"{artist} · {track.genre}"


def _index(paths, cache_path: Path, encoder):
    from aux.index import build_index

    vectors, kept, _ = build_index(None, encoder, paths=paths, n_segments=5,
                                   cache_path=cache_path, progress_every=0)
    return vectors, {str(p) for p in kept}


def load_corpus(which: str, encoder, *, limit: int | None = None) -> Corpus:
    """Load one corpus. `limit` trims the track list before indexing, for a lighter demo."""
    if which == "personal" and which not in available_corpora():
        raise RuntimeError("the personal library is not available on this instance")
    if which == "personal" and not personal_is_local():
        return load_personal_bundle()
    if which == "fma" and not fma_is_local():
        return load_fma_bundle()
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
                      note=("Creative Commons, so it plays here. 56% instrumental in a "
                            "sampled 75 clips, median transcript 11 words — no lyric "
                            "channel to search."))

    from aux.data import load_personal_tracks

    tracks = load_personal_tracks()
    vectors, kept = _index([t.path for t in tracks], CACHE / "personal.npz", encoder)
    tracks = [t for t in tracks if str(t.path) in kept]

    import sys

    sys.path.insert(0, str(ROOT / "scripts"))
    from eval_recommendation import load_lyric_vectors

    lyrics, has_lyrics = load_lyric_vectors(tracks, "small", "personal")
    from aux.data.personal import display_artist

    local = not is_public()
    return Corpus("personal library", tracks, vectors, lyrics, has_lyrics,
                  playable=local, anonymous=not local,
                  note=("The author's own music. 127 of 160 tracks have a reliable "
                        "transcript at a median of 376 words, which is why the multimodal "
                        "evaluation runs here."),
                  artists=[display_artist(t.path) for t in tracks])


def load_results() -> dict:
    """Every committed results file, keyed by name, for the findings page."""
    out = {}
    for path in sorted(RESULTS.glob("*.json")):
        try:
            out[path.stem] = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
    return out


@dataclass(frozen=True, slots=True)
class BundledTrack:
    """A track known only by its metadata. Stands in for `TrackMeta` where no file exists."""

    track_id: int
    title: str
    artist: str
    genre: str
    path: Path = Path()


def load_fma_bundle() -> Corpus:
    """Load the exported FMA slice: metadata, vectors, and audio that ships with it.

    Creative Commons, so unlike the personal library the recordings travel with the bundle
    and a deployed demo can play them.
    """
    manifest = json.loads((DEMO / "fma_manifest.json").read_text())
    blob = np.load(DEMO / "fma_vectors.npz")
    tracks = [BundledTrack(track_id=r["track_id"], title=r["title"], artist=r["artist"],
                           genre=r["genre"], path=DEMO / "audio" / r["file"])
              for r in manifest["tracks"]]
    return Corpus(manifest["name"], tracks, blob["audio"], None, None,
                  playable=True, anonymous=False, note=manifest["note"],
                  artists=[t.artist for t in tracks])


def load_personal_bundle() -> Corpus:
    """Load the exported personal corpus: metadata and vectors, no audio, no transcripts.

    Used wherever the audio is absent — every deployment. The lyric modality stays fully
    functional, because ranking needs the embeddings and not the text they came from, which
    is what lets a public instance demonstrate the feature at all.
    """
    manifest = json.loads((DEMO / "personal_manifest.json").read_text())
    blob = np.load(DEMO / "personal_vectors.npz")
    tracks = [BundledTrack(track_id=r["track_id"], title=r["title"], artist=r["artist"],
                           genre=r["genre"]) for r in manifest["tracks"]]
    return Corpus(manifest["name"], tracks, blob["audio"], blob["lyrics"],
                  blob["has_lyrics"], playable=False, anonymous=False,
                  note=manifest["note"], artists=[t.artist for t in tracks])
