"""The ingestion contract's data types.

DESIGN.md: ``SourceAdapter -> MediaProbe -> AudioDecoder -> AudioAsset``, where AudioAsset
preserves source path/provenance, duration, native sample rate, waveform/stream handle,
content hash, and media type.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

SUPPORTED_EXTENSIONS: frozenset[str] = frozenset(
    {".mp3", ".wav", ".flac", ".m4a", ".mp4"}
)
"""Formats the project claims to support (PROJECT.md, STATUS.md Slice 0 contract).

``.mp4`` is present because DESIGN.md treats video as a container: extract and decode the
audio stream, then run the identical downstream pipeline.
"""

_HASH_CHUNK_BYTES = 1 << 20


def content_hash(path: Path) -> str:
    """BLAKE2b digest of the file's bytes.

    Keys the embedding cache. Hashing *bytes* rather than decoded audio is deliberate: it
    makes a moved or renamed file a cache hit, which is the stated purpose, and it is
    cheap. The accepted consequence is that a re-encode of the same recording is a cache
    miss, correct behaviour anyway, since a different encode is different audio to the
    encoder.
    """
    digest = hashlib.blake2b(digest_size=16)
    with path.open("rb") as handle:
        while chunk := handle.read(_HASH_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class MediaInfo:
    """What MediaProbe learns without decoding.

    Cheap by design: probing reads container headers only, so the pipeline can categorise
    a file, report duration, and reject it before paying for a full decode.
    """

    path: Path
    media_type: str
    """Container format short name as reported by the demuxer (e.g. ``mov,mp4,m4a``)."""
    codec: str
    native_sample_rate: int
    native_channels: int
    duration_seconds: float
    has_video_stream: bool
    """True for MP4 carrying video. Recorded for provenance; the video is never decoded."""


@dataclass(frozen=True, slots=True)
class AudioAsset:
    """A decoded, pipeline-ready audio file.

    ``samples`` is mono float32 in [-1, 1] **at the file's native sample rate**. DESIGN.md
    forbids normalising the library to one universal rate: each encoder adapter resamples
    to its own contract downstream (CLAP and MuQ-MuLan differ), so baking one encoder's
    rate in here would degrade every other encoder's input and invalidate E0.
    """

    path: Path
    content_hash: str
    media_type: str
    codec: str
    sample_rate: int
    """Native rate, preserved. Not an encoder's rate."""
    native_channels: int
    """Channel count before the mono downmix below."""
    duration_seconds: float
    """Measured from the decoded sample count, not from container metadata."""
    samples: np.ndarray = field(repr=False)
    """Shape ``(n_samples,)``, dtype float32, mono."""

    @property
    def n_samples(self) -> int:
        return int(self.samples.shape[0])
