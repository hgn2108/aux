"""Audio feature extraction tests.

Synthetic-audio tests run anywhere. Tests needing the FMA metadata archive skip
when it is absent, so a fresh clone still gets a green suite.
"""

import numpy as np
import pytest
import soundfile as sf

from aux.config.settings import get_settings
from aux.models import features

FMA_METADATA = get_settings().raw_dir / "fma/fma_metadata.zip"
needs_fma = pytest.mark.skipif(not FMA_METADATA.exists(), reason="FMA metadata not downloaded")


@pytest.fixture
def tone(tmp_path):
    """A two-second chord, so chroma and tonnetz have real content to describe."""
    sr = 22050
    t = np.linspace(0, 2.0, int(sr * 2.0), endpoint=False)
    y = sum(0.3 * np.sin(2 * np.pi * f * t) for f in (261.6, 329.6, 392.0))

    path = tmp_path / "tone.wav"
    sf.write(path, y.astype(np.float32), sr)
    return path


def test_column_spec_is_518_dims() -> None:
    cols = features.feature_columns()
    assert len(cols) == 518
    assert cols.names == ["feature", "statistics", "number"]


@needs_fma
def test_column_spec_matches_fma_exactly() -> None:
    """The point of mirroring FMA: our features must be directly comparable."""
    from aux.ingest import fma

    assert list(features.feature_columns()) == list(fma.load_features().columns)


def test_extract_returns_full_vector(tone) -> None:
    vector = features.extract(tone)

    assert len(vector) == 518
    assert vector.notna().all(), "no feature may be NaN"
    assert np.isfinite(vector.to_numpy()).all()


def test_extract_is_deterministic(tone) -> None:
    a, b = features.extract(tone), features.extract(tone)
    np.testing.assert_allclose(a.to_numpy(), b.to_numpy())


def test_extract_distinguishes_different_audio(tone, tmp_path) -> None:
    """A different signal must produce a different vector — guards a constant bug."""
    sr = 22050
    rng = np.random.default_rng(0)
    noise_path = tmp_path / "noise.wav"
    sf.write(noise_path, rng.normal(scale=0.2, size=sr * 2).astype(np.float32), sr)

    chord = features.extract(tone).to_numpy()
    noise = features.extract(noise_path).to_numpy()

    assert not np.allclose(chord, noise)
    # Noise is spectrally flat and bright; a chord is not.
    centroid = ("spectral_centroid", "mean", "01")
    assert features.extract(noise_path)[centroid] > features.extract(tone)[centroid]


def test_extract_many_survives_unreadable_files(tone, tmp_path) -> None:
    """Corrupt audio is data, not a crash: FMA ships a few truncated mp3s."""
    broken = tmp_path / "broken.mp3"
    broken.write_bytes(b"not audio")

    out = features.extract_many({1: tone, 2: broken})

    assert list(out.index) == [1]
    assert out.shape == (1, 518)
