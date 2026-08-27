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
