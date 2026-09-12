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

__all__ = [
    "DEFAULT_AUDIO",
    "DEFAULT_METADATA",
    "TrackMeta",
    "balanced_subset",
    "load_tracks",
    "relevance_matrix",
    "same_artist_matrix",
    "track_path",
]
