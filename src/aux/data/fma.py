"""FMA metadata: track identity and the labels used as proxy relevance.

The Free Music Archive's `fma_small` subset is well suited to evaluating a recommender:
8,000 tracks balanced at exactly 1,000 per genre across 8 genres, with complete artist and
album metadata. That balance matters — an unbalanced corpus makes Precision@K partly a
measure of how common a genre is.

**These labels are proxies for musical similarity, not ground truth.** Two tracks sharing a
genre are not necessarily similar, and two similar tracks may sit in different genres. Three
definitions are provided rather than one, because each is wrong in a different direction and
agreement between them is more informative than any single number.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_METADATA = Path("data/raw/fma/fma_metadata/tracks.csv")
DEFAULT_AUDIO = Path("data/raw/fma/fma_small")


@dataclass(frozen=True, slots=True)
class TrackMeta:
    track_id: int
    path: Path
    title: str
    artist: str
    album: str
    genre: str


def track_path(audio_root: Path, track_id: int) -> Path:
    """FMA lays out audio as `<root>/<first 3 digits of zero-padded id>/<id>.mp3`."""
    padded = f"{track_id:06d}"
    return Path(audio_root) / padded[:3] / f"{padded}.mp3"


def load_tracks(metadata: Path = DEFAULT_METADATA, audio_root: Path = DEFAULT_AUDIO,
                subset: str = "small") -> list[TrackMeta]:
    """Load metadata for one FMA subset, keeping only tracks whose audio is on disk."""
    frame = pd.read_csv(metadata, index_col=0, header=[0, 1], low_memory=False)
    frame = frame[frame[("set", "subset")] == subset]

    out = []
    for track_id, row in frame.iterrows():
        path = track_path(audio_root, int(track_id))
        if not path.exists():
            continue
        out.append(TrackMeta(
            track_id=int(track_id), path=path,
            title=str(row[("track", "title")]),
            artist=str(row[("artist", "name")]),
            album=str(row[("album", "title")]),
            genre=str(row[("track", "genre_top")]),
        ))
    return out


def balanced_subset(tracks: list[TrackMeta], per_genre: int, *, seed: int = 20260912
                    ) -> list[TrackMeta]:
    """Take an equal number of tracks per genre, deterministically.

    Sampling per genre rather than uniformly keeps the evaluation balanced even when a
    subset is drawn, so a genre's Precision@K reflects the model rather than the genre's
    share of the corpus.
    """
    rng = np.random.default_rng(seed)
    by_genre: dict[str, list[TrackMeta]] = {}
    for t in tracks:
        by_genre.setdefault(t.genre, []).append(t)

    chosen: list[TrackMeta] = []
    for genre in sorted(by_genre):
        pool = sorted(by_genre[genre], key=lambda t: t.track_id)
        take = min(per_genre, len(pool))
        idx = rng.choice(len(pool), size=take, replace=False)
        chosen.extend(pool[i] for i in sorted(idx))
    return sorted(chosen, key=lambda t: t.track_id)


def relevance_matrix(tracks: list[TrackMeta], label: str) -> np.ndarray:
    """Boolean matrix where `[i, j]` is True when j is relevant to query i.

    The diagonal is False: a track is never its own recommendation.

    - **genre** — same top-level genre. Broad, and the weakest proxy: audio encoders
      represent genre strongly, so this partly measures genre classification.
    - **artist** — same artist. Strict, and not explainable by genre alone.
    - **album** — same album. Strictest, and the closest thing here to "these belong
      together", though it also rewards shared production and mastering rather than
      musical similarity as such.
    """
    values = {"genre": [t.genre for t in tracks],
              "artist": [t.artist for t in tracks],
              "album": [t.album for t in tracks]}
    if label not in values:
        raise ValueError(f"unknown label {label!r}; expected one of {sorted(values)}")
    arr = np.array(values[label])
    matrix = arr[:, None] == arr[None, :]
    np.fill_diagonal(matrix, False)
    return matrix


def same_artist_matrix(tracks: list[TrackMeta]) -> np.ndarray:
    """Used to exclude same-artist pairs when scoring the genre label.

    Without it, genre scores are inflated by the album effect: tracks from one album share
    production, mastering and instrumentation, so a model can rank them together without
    having learned anything about genre. Excluding them is standard practice in music
    information retrieval and turns a flattering number into an honest one.
    """
    return relevance_matrix(tracks, "artist")
