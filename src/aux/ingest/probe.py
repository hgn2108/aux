"""MediaProbe, identify a file without decoding it.

Reads container headers only. Cheap enough to run over a whole library to categorise it,
which is what Eval 0A's per-format breakdown needs.
"""

from __future__ import annotations

from pathlib import Path

import av

from .asset import SUPPORTED_EXTENSIONS, MediaInfo
from .errors import FailureCategory, IngestError


def probe(path: Path) -> MediaInfo:
    """Read container/stream metadata for ``path``.

    Raises IngestError with an Eval 0A category on any failure.
    """
    path = Path(path)

    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise IngestError(
            FailureCategory.UNSUPPORTED_EXTENSION,
            f"extension {path.suffix!r} is not a supported format",
        )
    if not path.is_file():
        raise IngestError(FailureCategory.FILE_UNREADABLE, f"not a file: {path}")
    try:
        if path.stat().st_size == 0:
            raise IngestError(FailureCategory.FILE_UNREADABLE, "file is zero bytes")
    except OSError as exc:
        raise IngestError(FailureCategory.FILE_UNREADABLE, str(exc)) from exc

    try:
        container = av.open(str(path))
    except (av.FFmpegError, OSError) as exc:
        raise IngestError(FailureCategory.CONTAINER_UNPARSEABLE, str(exc)) from exc

    with container:
        audio_streams = container.streams.audio
        if not audio_streams:
            raise IngestError(
                FailureCategory.NO_AUDIO_STREAM,
                "container has no audio stream",
            )

        # First audio stream only. A file with several (alternate language tracks, for
        # instance) is out of scope for Slice 0; taking the first keeps behaviour
        # deterministic rather than guessing which one is "the music".
        stream = audio_streams[0]

        # Demuxer selection is extension-driven, so a corrupt file with a media extension
        # can open "successfully" and present a stream whose sample rate is 0. Caught
        # here rather than downstream: a rate of 0 would otherwise produce an asset with
        # a nonsense duration and silently poison every measurement built on it.
        sample_rate = int(stream.codec_context.sample_rate or 0)
        if sample_rate <= 0:
            raise IngestError(
                FailureCategory.CONTAINER_UNPARSEABLE,
                "audio stream declares no sample rate; container is malformed",
            )

        return MediaInfo(
            path=path,
            media_type=container.format.name,
            codec=stream.codec_context.name,
            native_sample_rate=sample_rate,
            native_channels=int(getattr(stream.codec_context, "channels", 0) or 0),
            duration_seconds=_container_duration(container, stream),
            has_video_stream=bool(container.streams.video),
        )


def _container_duration(container: "av.container.InputContainer", stream) -> float:
    """Best-effort duration in seconds from header metadata.

    Header duration is advisory only, some MP3s report it wrongly, and a truncated file
    reports its intended length. AudioAsset.duration_seconds is measured from the decoded
    sample count instead; this value exists so probing alone can characterise a library.
    """
    if stream.duration is not None and stream.time_base:
        return float(stream.duration * stream.time_base)
    if container.duration is not None:
        return float(container.duration / av.time_base)
    return 0.0
