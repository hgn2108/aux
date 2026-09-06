"""Slice 0 ingestion: ``SourceAdapter -> MediaProbe -> AudioDecoder -> AudioAsset``."""

from .asset import SUPPORTED_EXTENSIONS, AudioAsset, MediaInfo, content_hash
from .decode import decode
from .errors import FailureCategory, IngestError
from .probe import probe
from .source import DiscoveredFile, discover

__all__ = [
    "SUPPORTED_EXTENSIONS",
    "AudioAsset",
    "DiscoveredFile",
    "FailureCategory",
    "IngestError",
    "MediaInfo",
    "content_hash",
    "decode",
    "discover",
    "probe",
]
