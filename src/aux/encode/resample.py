"""Sample-rate conversion, owned by the encoder layer.

DESIGN.md places resampling here rather than in ingestion: the library keeps its source
rates, and each encoder adapter converts to its own contract. Adding an encoder with a
different input rate therefore never requires re-ingesting, and never feeds one encoder
audio that was already resampled for another.

Uses FFmpeg's resampler through PyAV rather than adding a second DSP dependency, so the
conversion is the same implementation that decoding already relies on.
"""

from __future__ import annotations

from fractions import Fraction

import av
import numpy as np
from av.audio.resampler import AudioResampler


def resample(samples: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    """Convert mono float32 ``samples`` from ``source_rate`` to ``target_rate``."""
    if source_rate == target_rate:
        return samples.astype(np.float32, copy=False)
    if source_rate <= 0 or target_rate <= 0:
        raise ValueError(f"invalid rates: {source_rate} -> {target_rate}")

    frame = av.AudioFrame.from_ndarray(
        samples.reshape(1, -1).astype(np.float32), format="flt", layout="mono"
    )
    frame.sample_rate = source_rate
    frame.time_base = Fraction(1, source_rate)
    frame.pts = 0

    resampler = AudioResampler(format="fltp", layout="mono", rate=target_rate)
    blocks = [f.to_ndarray().reshape(-1) for f in resampler.resample(frame)]
    # Flush, or the tail of the signal is silently discarded.
    blocks += [f.to_ndarray().reshape(-1) for f in resampler.resample(None)]

    if not blocks:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(blocks).astype(np.float32, copy=False)
