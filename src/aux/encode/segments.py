"""Deterministic segment selection.

A 3-5 minute track must become one vector, but encoders accept windows of seconds. The
choice of how to sample the track is E1's subject:

- **baseline**, one representative centre segment;
- **candidate**, 3-5 deterministic windows spread across the track, excluding obvious
  leading and trailing silence.

Everything here is deterministic. No random offsets: Eval 0A's repeatability guarantee has
to survive into embedding, or two runs over the same library produce different vectors and
the content-hash cache becomes unsound.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SILENCE_THRESHOLD = 1e-3
"""Absolute amplitude below which a window counts as silence (~-60 dBFS)."""


@dataclass(frozen=True, slots=True)
class Segment:
    start_sample: int
    samples: np.ndarray

    @property
    def n_samples(self) -> int:
        return int(self.samples.shape[0])


def trim_silence(
    samples: np.ndarray, sample_rate: int, *, threshold: float = SILENCE_THRESHOLD
) -> tuple[int, int]:
    """Return ``(start, end)`` sample indices with leading/trailing silence removed.

    Coarse by design, measured on 50 ms blocks rather than per sample, so a single
    non-zero sample in a fade cannot defeat it. Returns the full range if the track is
    silent throughout, leaving that judgement to the caller.
    """
    block = max(1, int(0.05 * sample_rate))
    n_blocks = samples.size // block
    if n_blocks == 0:
        return 0, samples.size

    peaks = np.abs(samples[: n_blocks * block].reshape(n_blocks, block)).max(axis=1)
    loud = np.flatnonzero(peaks > threshold)
    if loud.size == 0:
        return 0, samples.size
    return int(loud[0] * block), int(min((loud[-1] + 1) * block, samples.size))


def select_segments(
    samples: np.ndarray,
    sample_rate: int,
    *,
    n_segments: int = 1,
    segment_seconds: float = 10.0,
    trim: bool = True,
) -> list[Segment]:
    """Pick ``n_segments`` deterministic windows of ``segment_seconds`` from a track.

    One segment is taken from the centre, the least-bad single choice, since intros and
    outros are the least representative parts of a track. Several segments are spaced
    evenly across the trimmed region, each centred in its own equal share of the track, so
    coverage does not depend on track length.

    Tracks shorter than one window are zero-padded to the window length, which keeps the
    encoder's input shape fixed without inventing content.
    """
    if n_segments < 1:
        raise ValueError("n_segments must be >= 1")

    window = int(segment_seconds * sample_rate)
    start, end = trim_silence(samples, sample_rate) if trim else (0, samples.size)
    usable = end - start
    if usable <= 0:
        start, end, usable = 0, samples.size, samples.size

    if usable <= window:
        chunk = samples[start:end]
        padded = np.zeros(window, dtype=np.float32)
        padded[: chunk.size] = chunk
        return [Segment(start_sample=start, samples=padded)] * n_segments

    if n_segments == 1:
        offsets = [start + (usable - window) // 2]
    else:
        # Centre each window inside its own equal share of the usable region.
        share = usable / n_segments
        offsets = [
            int(start + i * share + (share - window) / 2) for i in range(n_segments)
        ]
        offsets = [min(max(o, start), end - window) for o in offsets]

    return [Segment(start_sample=o, samples=samples[o : o + window]) for o in offsets]
