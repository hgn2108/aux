"""Failure taxonomy for ingestion.

Eval 0A's pass gate is ">=95% supported-media success **with failures categorized
cleanly**". A 5% failure rate that is entirely one cause is a different fact about the
system than 5% scattered across unrelated causes, so every failure carries a category
rather than only a message.
"""

from __future__ import annotations

from enum import Enum


class FailureCategory(str, Enum):
    """Why a file did not become an AudioAsset."""

    UNSUPPORTED_EXTENSION = "unsupported_extension"
    """Not one of the formats the project claims to support. Not counted against the gate."""

    FILE_UNREADABLE = "file_unreadable"
    """Missing, permission-denied, or zero bytes. An environment fault, not a codec one."""

    CONTAINER_UNPARSEABLE = "container_unparseable"
    """Demuxer could not open the container. Truncated download, or a mislabelled extension."""

    NO_AUDIO_STREAM = "no_audio_stream"
    """Opened fine but carries no audio -- e.g. a silent-video MP4."""

    DECODE_FAILED = "decode_failed"
    """Audio stream present but decoding raised. Includes DRM-protected M4A, which
    surfaces as a decoder error rather than as an identifiable DRM signal."""

    EMPTY_AUDIO = "empty_audio"
    """Decoded successfully to zero samples. Distinguished from DECODE_FAILED because it
    points at the file, not the decoder."""


class IngestError(Exception):
    """An ingestion failure carrying its Eval 0A category."""

    def __init__(self, category: FailureCategory, message: str) -> None:
        super().__init__(message)
        self.category = category
        self.message = message
