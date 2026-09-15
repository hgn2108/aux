"""AudioDecoder, decode any supported container to a mono float32 waveform.

Two decisions are load-bearing here and both come from DESIGN.md:

1. **Native sample rate is preserved.** The decoder never resamples. Each encoder adapter
   resamples to its own contract downstream, so that adding an encoder with a different
   input rate never requires re-ingesting the library and never feeds a second encoder
   audio that was already resampled for the first.
2. **MP4 is a container, not a special case.** The audio stream is extracted and decoded
   through the identical path as a standalone MP3; the video stream is never touched.
"""

from __future__ import annotations

from pathlib import Path

import av
import numpy as np
from av.audio.resampler import AudioResampler

from .asset import AudioAsset, content_hash
from .errors import FailureCategory, IngestError
from .probe import probe


def decode(path: Path, *, info=None) -> AudioAsset:
    """Decode ``path`` into an AudioAsset.

    ``info`` may carry a MediaInfo from an earlier probe to avoid opening the file twice.
    Raises IngestError with an Eval 0A category on any failure.
    """
    path = Path(path)
    info = info or probe(path)

    try:
        container = av.open(str(path))
    except (av.FFmpegError, OSError) as exc:
        raise IngestError(FailureCategory.CONTAINER_UNPARSEABLE, str(exc)) from exc

    with container:
        streams = container.streams.audio
        if not streams:
            raise IngestError(FailureCategory.NO_AUDIO_STREAM, "no audio stream")
        stream = streams[0]

        # Downmix to mono and convert to float32, holding the rate at the source's own.
        # Mono is what the joint encoders consume; doing it here keeps one waveform
        # convention across the pipeline rather than repeating the downmix per encoder.
        resampler = AudioResampler(
            format="fltp", layout="mono", rate=info.native_sample_rate or None
        )

        blocks: list[np.ndarray] = []
        try:
            for frame in container.decode(stream):
                for resampled in resampler.resample(frame):
                    blocks.append(resampled.to_ndarray().reshape(-1))
            # Flush the resampler's internal buffer, or the tail is silently dropped.
            for resampled in resampler.resample(None):
                blocks.append(resampled.to_ndarray().reshape(-1))
        except (av.FFmpegError, ValueError) as exc:
            raise IngestError(FailureCategory.DECODE_FAILED, str(exc)) from exc

    if not blocks:
        raise IngestError(FailureCategory.EMPTY_AUDIO, "decoded to zero samples")

    samples = np.concatenate(blocks).astype(np.float32, copy=False)
    if samples.size == 0:
        raise IngestError(FailureCategory.EMPTY_AUDIO, "decoded to zero samples")

    sample_rate = info.native_sample_rate or int(stream.codec_context.sample_rate or 0)
    if sample_rate <= 0:
        raise IngestError(
            FailureCategory.DECODE_FAILED, "could not determine sample rate"
        )

    return AudioAsset(
        path=path,
        content_hash=content_hash(path),
        media_type=info.media_type,
        codec=info.codec,
        sample_rate=sample_rate,
        native_channels=info.native_channels,
        # Measured, not taken from the header: a truncated file reports its intended
        # length, and some MP3 headers are simply wrong.
        duration_seconds=float(samples.size) / sample_rate,
        samples=samples,
    )
