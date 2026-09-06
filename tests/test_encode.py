"""Encoder-layer tests.

Deliberately model-free apart from the guard test: segment selection, resampling and
pooling are where correctness bugs would silently distort every embedding, and they can be
verified without downloading half a gigabyte of weights.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from aux.encode import l2_normalise, resample, select_segments, trim_silence
from aux.encode.base import EncoderAdapter
from aux.ingest import AudioAsset

SR = 44_100


def tone(seconds: float, sr: int = SR, freq: float = 440.0) -> np.ndarray:
    t = np.arange(int(sr * seconds), dtype=np.float32) / sr
    return (0.5 * np.sin(2 * math.pi * freq * t)).astype(np.float32)


class CountingAdapter(EncoderAdapter):
    """Stands in for a real encoder: returns a distinct fixed vector per segment."""

    sample_rate = 48_000
    segment_seconds = 10.0
    embedding_dim = 4

    def __init__(self, vectors: np.ndarray | None = None) -> None:
        self.name, self.version = "counting", "test"
        self.vectors = vectors
        self.calls = 0

    def embed_audio(self, waveforms):
        self.calls += 1
        if self.vectors is not None:
            return self.vectors[: len(waveforms)]
        return np.arange(len(waveforms) * 4, dtype=np.float32).reshape(len(waveforms), 4) + 1

    def embed_text(self, texts):
        return np.ones((len(texts), 4), dtype=np.float32)


def make_asset(samples: np.ndarray, sr: int = SR) -> AudioAsset:
    return AudioAsset(
        path=__import__("pathlib").Path("x.mp3"), content_hash="h", media_type="mp3",
        codec="mp3float", sample_rate=sr, native_channels=1,
        duration_seconds=samples.size / sr, samples=samples,
    )


# --- resampling -------------------------------------------------------------------

def test_resample_is_identity_at_the_same_rate():
    signal = tone(0.5)
    assert resample(signal, SR, SR) is not None
    assert np.array_equal(resample(signal, SR, SR), signal)


@pytest.mark.parametrize("target", [48_000, 24_000, 16_000])
def test_resample_changes_length_by_the_rate_ratio(target):
    out = resample(tone(1.0), SR, target)
    assert out.shape[0] == pytest.approx(target, rel=0.02)
    assert out.dtype == np.float32


def test_resample_preserves_signal_energy():
    """A resampler that drops the flush buffer or mangles scaling shows up here."""
    signal = tone(1.0)
    out = resample(signal, SR, 48_000)
    assert float(np.sqrt(np.mean(out**2))) == pytest.approx(
        float(np.sqrt(np.mean(signal**2))), rel=0.05
    )


# --- segment selection ------------------------------------------------------------

def test_trim_silence_finds_the_audible_region():
    padded = np.concatenate([np.zeros(SR, np.float32), tone(2.0), np.zeros(SR, np.float32)])
    start, end = trim_silence(padded, SR)
    assert start == pytest.approx(SR, abs=SR // 10)
    assert end == pytest.approx(3 * SR, abs=SR // 10)


def test_trim_silence_returns_full_range_when_silent():
    silent = np.zeros(SR, dtype=np.float32)
    assert trim_silence(silent, SR) == (0, silent.size)


def test_single_segment_comes_from_the_centre():
    signal = tone(60.0)
    (segment,) = select_segments(signal, SR, n_segments=1, segment_seconds=10.0)
    expected = (signal.size - 10 * SR) // 2
    assert segment.start_sample == pytest.approx(expected, abs=SR)


@pytest.mark.parametrize("n", [1, 3, 5])
def test_segment_count_and_window_length(n):
    segments = select_segments(tone(60.0), SR, n_segments=n, segment_seconds=10.0)
    assert len(segments) == n
    assert all(s.n_samples == 10 * SR for s in segments)


def test_multi_segments_are_ordered_and_spread():
    segments = select_segments(tone(120.0), SR, n_segments=5, segment_seconds=10.0)
    starts = [s.start_sample for s in segments]
    assert starts == sorted(starts)
    assert len(set(starts)) == 5


def test_segment_selection_is_deterministic():
    """Repeatability has to survive into embedding, or the content-hash cache is unsound."""
    signal = tone(90.0)
    a = select_segments(signal, SR, n_segments=3)
    b = select_segments(signal, SR, n_segments=3)
    assert [s.start_sample for s in a] == [s.start_sample for s in b]


def test_short_track_is_padded_not_truncated():
    segments = select_segments(tone(3.0), SR, n_segments=1, segment_seconds=10.0)
    assert segments[0].n_samples == 10 * SR
    assert np.count_nonzero(segments[0].samples) > 0


def test_segments_skip_leading_silence():
    signal = np.concatenate([np.zeros(30 * SR, np.float32), tone(30.0)])
    (segment,) = select_segments(signal, SR, n_segments=1, segment_seconds=10.0)
    assert segment.start_sample > 25 * SR


# --- pooling ----------------------------------------------------------------------

def test_l2_normalise_gives_unit_rows():
    normed = l2_normalise(np.array([[3.0, 4.0], [1.0, 0.0]], dtype=np.float32))
    assert np.allclose(np.linalg.norm(normed, axis=1), 1.0)


def test_l2_normalise_survives_a_zero_row():
    assert np.all(np.isfinite(l2_normalise(np.zeros((1, 4), dtype=np.float32))))


def test_track_vector_is_unit_length():
    adapter = CountingAdapter()
    vector, segments = adapter.embed_track(make_asset(tone(60.0)), n_segments=3)
    assert vector.shape == (4,)
    assert float(np.linalg.norm(vector)) == pytest.approx(1.0, abs=1e-5)
    assert segments.shape == (3, 4)


def test_segments_are_normalised_before_pooling():
    """A loud segment must not outweigh a quiet one through vector magnitude alone."""
    big, small = np.array([[10.0, 0, 0, 0]], np.float32), np.array([[0, 1.0, 0, 0]], np.float32)
    adapter = CountingAdapter(vectors=np.concatenate([big, small]))
    vector, _ = adapter.embed_track(make_asset(tone(60.0)), n_segments=2)
    # Equal contribution => equal components. Magnitude-weighted pooling would skew to dim 0.
    assert vector[0] == pytest.approx(vector[1], abs=1e-5)


def test_embed_track_is_deterministic():
    adapter = CountingAdapter()
    asset = make_asset(tone(60.0))
    first, _ = adapter.embed_track(asset, n_segments=3)
    second, _ = adapter.embed_track(asset, n_segments=3)
    assert np.array_equal(first, second)
