"""MagnaTagATune — human similarity judgements.

Used as a *perceptual evaluation set only*. MTAT audio is 16 kHz, 32 kbps mono,
too compressed to serve as a feature-extraction corpus; FMA stays the audio
source. What MTAT uniquely provides is what listeners actually heard as similar,
which no metadata proxy can substitute for.

Judgements come from the TagATune "odd one out" game: three clips, and players
vote for the one that sounds least like the other two.
"""

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from aux.config.settings import get_settings

COMPARISONS = "mtat/comparisons_final.csv"
CLIP_INFO = "mtat/clip_info_final.csv"

VOTE_COLUMNS = ["clip1_numvotes", "clip2_numvotes", "clip3_numvotes"]
CLIP_COLUMNS = ["clip1_id", "clip2_id", "clip3_id"]


def _path(name: str) -> Path:
    return get_settings().raw_dir / name


@lru_cache
def load_comparisons() -> pd.DataFrame:
    """Raw triplet comparisons: three clip ids and their vote counts."""
    return pd.read_csv(_path(COMPARISONS), sep="\t")


@lru_cache
def load_clip_info() -> pd.DataFrame:
    """Clip metadata, including the mp3 path each clip was cut from."""
    return pd.read_csv(_path(CLIP_INFO), sep="\t", index_col="clip_id")


def similarity_triplets(min_votes: int = 3, min_margin: int = 1) -> np.ndarray:
    """Triplets as ``(a, b, c)`` clip ids, where ``c`` is the voted outlier.

    Of 533 raw comparisons, 87 are ties with no agreed outlier and many carry
    only one or two votes. Both are filtered out rather than counted as signal:

    - ``min_votes``  — total votes the triplet must have received.
    - ``min_margin`` — how far ahead the outlier must be. A margin of 0 means
      listeners disagreed, so the triplet carries no usable judgement.

    Defaults keep 307 of 533. Report the count alongside any score, since
    statistical power at this size is modest.
    """
    return triplets_from(load_comparisons(), min_votes, min_margin)


def triplets_from(df: pd.DataFrame, min_votes: int = 3, min_margin: int = 1) -> np.ndarray:
    """Filter and order raw comparisons into ``(a, b, outlier)`` rows."""
    votes = df[VOTE_COLUMNS].to_numpy()
    clips = df[CLIP_COLUMNS].to_numpy()

    ordered = np.sort(votes, axis=1)
    keep = (votes.sum(axis=1) >= min_votes) & (ordered[:, 2] - ordered[:, 1] >= min_margin)

    votes, clips = votes[keep], clips[keep]

    # Reorder each row so the most-voted clip (the outlier) sits last.
    outlier = votes.argmax(axis=1)
    rows = np.arange(len(clips))
    others = np.array([[j for j in range(3) if j != o] for o in outlier])

    return np.column_stack(
        [clips[rows, others[:, 0]], clips[rows, others[:, 1]], clips[rows, outlier]]
    )


AUDIO_DIR = "mtat/audio"


def audio_paths(clip_ids: np.ndarray | None = None) -> dict[int, Path]:
    """Map clip ids to their mp3 files.

    ``clip_info`` stores a relative path per clip (``f/artist-album-track.mp3``);
    the archive unpacks to that same layout.
    """
    root = get_settings().raw_dir / AUDIO_DIR
    info = load_clip_info()
    if clip_ids is not None:
        info = info.loc[info.index.intersection(clip_ids)]

    return {
        int(cid): root / str(rel)
        for cid, rel in info["mp3_path"].items()
        if isinstance(rel, str) and rel.strip()
    }
