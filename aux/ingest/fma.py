"""Loaders for the FMA metadata release.

FMA ships CSVs with multi-row headers; these helpers hide that shape so callers
work with ordinary DataFrames.
"""

import zipfile
from functools import lru_cache
from pathlib import Path

import pandas as pd

from aux.config.settings import get_settings

ARCHIVE = "fma/fma_metadata.zip"


def _archive_path() -> Path:
    return get_settings().raw_dir / ARCHIVE


def _read(name: str, header: list[int]) -> pd.DataFrame:
    with zipfile.ZipFile(_archive_path()) as zf, zf.open(f"fma_metadata/{name}") as fh:
        return pd.read_csv(fh, index_col=0, header=header)


@lru_cache
def load_tracks() -> pd.DataFrame:
    """Track metadata: genre, license, subset membership."""
    return _read("tracks.csv", header=[0, 1])


@lru_cache
def load_features() -> pd.DataFrame:
    """Precomputed librosa features, one row per track."""
    return _read("features.csv", header=[0, 1, 2])


def subset(tracks: pd.DataFrame, name: str = "small") -> pd.Index:
    """Track ids belonging to an FMA subset ('small', 'medium', 'large')."""
    return tracks.index[tracks[("set", "subset")] == name]


AUDIO_DIR = "fma/fma_small"


def audio_path(track_id: int, root: Path | None = None) -> Path:
    """Path to a track's mp3.

    FMA lays audio out as ``<zero-padded 6-digit id>`` bucketed into directories
    named by its first three digits: track 2 lives at ``000/000002.mp3``.
    """
    root = root or (get_settings().raw_dir / AUDIO_DIR)
    name = f"{track_id:06d}"
    return root / name[:3] / f"{name}.mp3"
