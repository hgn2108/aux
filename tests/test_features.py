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


def test_fixed_sample_rate_makes_corpora_commensurable(tmp_path) -> None:
    """Sources arrive at different rates; features must land on one frequency axis.

    FMA is 44.1 kHz and MagnaTagATune is 16 kHz. Loading each natively put their
    descriptors on different axes and made cross-corpus comparison meaningless.
    """
    paths = []
    for native_sr in (16000, 44100, 48000):
        t = np.linspace(0, 2.0, 2 * native_sr, endpoint=False)
        y = np.sin(2 * np.pi * 440.0 * t)
        p = tmp_path / f"tone_{native_sr}.wav"
        sf.write(p, y.astype(np.float32), native_sr)
        paths.append(p)

    centroid = ("spectral_centroid", "mean", "01")
    values = [features.extract(p)[centroid] for p in paths]

    # A 440 Hz tone must report the same centroid regardless of source rate.
    assert max(values) - min(values) < 0.05 * np.mean(values)


def test_sample_rate_is_passed_to_frequency_features(tone) -> None:
    """librosa defaults sr=22050 on S= calls; a 440 Hz tone pins the axis."""
    vector = features.extract(tone)
    centroid = vector[("spectral_centroid", "mean", "01")]

    # Lowest note of the test chord is 261.6 Hz; centroid sits above it but well
    # below Nyquist. A mislabelled axis would halve or double this.
    assert 200 < centroid < features.SAMPLE_RATE / 2
