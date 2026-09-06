"""Synthetic media fixtures.

Generated rather than committed: `data/**` is gitignored and personal audio is never
redistributed, so the test suite has to make its own inputs. Encoding a known signal also
gives the cross-format stability check in Eval 0A a ground truth to compare against.
"""

from __future__ import annotations

import math
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
import pytest

SAMPLE_RATE = 44_100
DURATION_S = 2.0
FREQ_HZ = 440.0


def _tone(sample_rate: int = SAMPLE_RATE, seconds: float = DURATION_S) -> np.ndarray:
    t = np.arange(int(sample_rate * seconds), dtype=np.float32) / sample_rate
    return (0.5 * np.sin(2 * math.pi * FREQ_HZ * t)).astype(np.float32)


def _encode(path: Path, signal: np.ndarray, sample_rate: int, codec: str, fmt: str) -> Path:
    with av.open(str(path), mode="w", format=fmt) as container:
        stream = container.add_stream(codec, rate=sample_rate)
        stream.layout = "mono"
        # Encoders take fixed frame sizes; chunk to whatever this codec asked for.
        frame_size = getattr(stream.codec_context, "frame_size", 0) or 1024
        for start in range(0, signal.size, frame_size):
            chunk = signal[start : start + frame_size]
            frame = av.AudioFrame.from_ndarray(
                chunk.reshape(1, -1), format="flt", layout="mono"
            )
            frame.sample_rate = sample_rate
            frame.pts = start
            frame.time_base = Fraction(1, sample_rate)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode(None):
            container.mux(packet)
    return path


@pytest.fixture(scope="session")
def tone() -> np.ndarray:
    return _tone()


@pytest.fixture(scope="session")
def media_dir(tmp_path_factory, tone) -> Path:
    """One directory holding the same 440 Hz tone in every supported format."""
    root = tmp_path_factory.mktemp("media")
    _encode(root / "tone.wav", tone, SAMPLE_RATE, "pcm_s16le", "wav")
    _encode(root / "tone.flac", tone, SAMPLE_RATE, "flac", "flac")
    _encode(root / "tone.mp3", tone, SAMPLE_RATE, "mp3", "mp3")
    _encode(root / "tone.m4a", tone, SAMPLE_RATE, "aac", "ipod")
    _encode(root / "tone.mp4", tone, SAMPLE_RATE, "aac", "mp4")

    # Negative cases, one per failure category we can construct cheaply.
    (root / "notes.txt").write_text("not media")
    (root / "empty.mp3").write_bytes(b"")
    (root / "garbage.flac").write_bytes(b"\x00\x01\x02" * 512)
    return root
