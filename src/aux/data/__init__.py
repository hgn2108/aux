"""Dataset loaders and proxy relevance labels."""

from .fma import (
    DEFAULT_AUDIO,
    DEFAULT_METADATA,
    TrackMeta,
    balanced_subset,
    load_tracks,
    relevance_matrix,
    same_artist_matrix,
    track_path,
)
from .personal import load_tracks as load_personal_tracks

__all__ = [
    "DEFAULT_AUDIO",
    "DEFAULT_METADATA",
    "TrackMeta",
    "balanced_subset",
    "load_personal_tracks",
    "load_tracks",
    "relevance_matrix",
    "same_artist_matrix",
    "track_path",
]
