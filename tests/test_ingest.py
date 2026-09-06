"""Slice 0 ingestion tests.

These check the contract, not retrieval quality: does every supported container reach an
AudioAsset, is the source sample rate preserved, is decoding deterministic, and does every
failure carry an Eval 0A category.
"""

from __future__ import annotations

import numpy as np
import pytest

from aux.ingest import (
    SUPPORTED_EXTENSIONS,
    FailureCategory,
    IngestError,
    content_hash,
    decode,
    discover,
    probe,
)

from .conftest import DURATION_S, SAMPLE_RATE

SUPPORTED_FIXTURES = ["tone.wav", "tone.flac", "tone.mp3", "tone.m4a", "tone.mp4"]


def test_discover_finds_only_supported_media(media_dir):
    found = {f.path.name for f in discover(media_dir)}
    assert found == set(SUPPORTED_FIXTURES) | {"empty.mp3", "garbage.flac"}
    assert "notes.txt" not in found


def test_discover_is_deterministic(media_dir):
    first = [f.path for f in discover(media_dir)]
    second = [f.path for f in discover(media_dir)]
    assert first == second == sorted(first)


@pytest.mark.parametrize("name", SUPPORTED_FIXTURES)
def test_probe_reports_native_rate(media_dir, name):
    info = probe(media_dir / name)
    assert info.native_sample_rate == SAMPLE_RATE
    assert info.native_channels == 1


@pytest.mark.parametrize("name", SUPPORTED_FIXTURES)
def test_decode_every_supported_format(media_dir, name):
    asset = decode(media_dir / name)
    assert asset.samples.dtype == np.float32
    assert asset.samples.ndim == 1
    assert asset.n_samples > 0
    # Lossy codecs pad; only require the decode to cover the source, within a frame or so.
    assert DURATION_S <= asset.duration_seconds < DURATION_S + 0.2


@pytest.mark.parametrize("name", SUPPORTED_FIXTURES)
def test_decode_preserves_source_sample_rate(media_dir, name):
    """The load-bearing one: nothing in ingestion may normalise to a universal rate."""
    assert decode(media_dir / name).sample_rate == SAMPLE_RATE


def test_mp4_is_treated_as_a_container(media_dir):
    """A video container must reach the same AudioAsset as standalone audio."""
    from_mp4 = decode(media_dir / "tone.mp4")
    from_m4a = decode(media_dir / "tone.m4a")
    assert from_mp4.sample_rate == from_m4a.sample_rate
    assert from_mp4.samples.ndim == from_m4a.samples.ndim


def test_decode_is_deterministic(media_dir):
    """Eval 0A repeatability: same file twice, byte-identical waveform."""
    first = decode(media_dir / "tone.mp3")
    second = decode(media_dir / "tone.mp3")
    assert np.array_equal(first.samples, second.samples)
    assert first.content_hash == second.content_hash


def test_content_hash_follows_bytes_not_path(media_dir, tmp_path):
    """A moved or renamed file must be a cache hit."""
    original = media_dir / "tone.flac"
    moved = tmp_path / "renamed.flac"
    moved.write_bytes(original.read_bytes())
    assert content_hash(moved) == content_hash(original)
    assert decode(moved).content_hash == decode(original).content_hash


def test_cross_format_stability(media_dir, tone):
    """The same signal through different codecs must land in the same place.

    Lossy encoding shifts individual samples, so compare energy rather than waveforms --
    enough to catch a broken downmix, a wrong rate, or a silent decode.
    """
    rms = {
        name: float(np.sqrt(np.mean(decode(media_dir / name).samples ** 2)))
        for name in SUPPORTED_FIXTURES
    }
    source_rms = float(np.sqrt(np.mean(tone**2)))
    for name, value in rms.items():
        assert value == pytest.approx(source_rms, rel=0.15), f"{name} drifted: {rms}"


@pytest.mark.parametrize(
    ("name", "category"),
    [
        ("notes.txt", FailureCategory.UNSUPPORTED_EXTENSION),
        ("empty.mp3", FailureCategory.FILE_UNREADABLE),
        ("garbage.flac", FailureCategory.CONTAINER_UNPARSEABLE),
    ],
)
def test_failures_carry_a_category(media_dir, name, category):
    with pytest.raises(IngestError) as excinfo:
        probe(media_dir / name)
    assert excinfo.value.category is category


def test_supported_extensions_match_the_slice_contract():
    assert SUPPORTED_EXTENSIONS == {".mp3", ".wav", ".flac", ".m4a", ".mp4"}
