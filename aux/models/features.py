"""Audio → feature descriptors (roadmap A4b).

Mirrors FMA's precomputed feature layout exactly — 11 families × 7 statistics =
518 columns — so our extraction can be compared column-for-column against theirs.
That reference is the whole reason A4a ran on precomputed features first: without
it we would only know our pipeline runs, not that it is right.

Frame-level descriptors are summarized to fixed length by seven statistics. This
is a "bag of frames" representation: it discards temporal order, which is a real
limitation and the main thing a learned encoder (roadmap A7) would improve on.
"""

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

# Ordered to match FMA's columns (alphabetical in both levels).
STATISTICS = ("kurtosis", "max", "mean", "median", "min", "skew", "std")

FAMILY_SIZES = {
    "chroma_cens": 12,
    "chroma_cqt": 12,
    "chroma_stft": 12,
    "mfcc": 20,
    "rmse": 1,
    "spectral_bandwidth": 1,
    "spectral_centroid": 1,
    "spectral_contrast": 7,
    "spectral_rolloff": 1,
    "tonnetz": 6,
    "zcr": 1,
}


def feature_columns() -> pd.MultiIndex:
    """The 518-column MultiIndex, matching FMA's ordering.

    FMA numbers coefficients as zero-padded strings ('01'), not integers. Matching
    that exactly is what lets the two frames be compared directly.
    """
    tuples = [
        (family, stat, f"{i + 1:02d}")
        for family, size in FAMILY_SIZES.items()
        for stat in STATISTICS
        for i in range(size)
    ]
    return pd.MultiIndex.from_tuples(tuples, names=["feature", "statistics", "number"])


def _summarize(values: np.ndarray) -> dict[str, np.ndarray]:
    """Collapse a (coefficients, frames) matrix to seven per-coefficient statistics.

    Skew and kurtosis are undefined for a constant sequence — zero variance gives
    0/0. That happens for real audio too (silence, a sustained tone), and a single
    NaN would propagate through standardization into every embedding. Constant
    input has no asymmetry and no tail, so both are reported as 0.
    """
    return {
        "kurtosis": np.nan_to_num(scipy_stats.kurtosis(values, axis=1)),
        "max": values.max(axis=1),
        "mean": values.mean(axis=1),
        "median": np.median(values, axis=1),
        "min": values.min(axis=1),
        "skew": np.nan_to_num(scipy_stats.skew(values, axis=1)),
        "std": values.std(axis=1),
    }


def _descriptors(y: np.ndarray, sr: int) -> dict[str, np.ndarray]:
    """Every frame-level descriptor family for one signal."""
    stft = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
    cqt = np.abs(librosa.cqt(y, sr=sr, hop_length=512, bins_per_octave=12, n_bins=7 * 12))

    chroma_cqt = librosa.feature.chroma_cqt(C=cqt, n_chroma=12, bins_per_octave=12)

    return {
        "chroma_cens": librosa.feature.chroma_cens(C=cqt, n_chroma=12, bins_per_octave=12),
        "chroma_cqt": chroma_cqt,
        "chroma_stft": librosa.feature.chroma_stft(S=stft**2, n_chroma=12),
        "mfcc": librosa.feature.mfcc(
            S=librosa.power_to_db(librosa.feature.melspectrogram(S=stft**2, sr=sr)),
            n_mfcc=20,
        ),
        "rmse": librosa.feature.rms(S=stft),
        "spectral_bandwidth": librosa.feature.spectral_bandwidth(S=stft),
        "spectral_centroid": librosa.feature.spectral_centroid(S=stft),
        "spectral_contrast": librosa.feature.spectral_contrast(S=stft, n_bands=6),
        "spectral_rolloff": librosa.feature.spectral_rolloff(S=stft),
        "tonnetz": librosa.feature.tonnetz(chroma=chroma_cqt),
        "zcr": librosa.feature.zero_crossing_rate(y, frame_length=2048, hop_length=512),
    }


def extract(path: str | Path) -> pd.Series:
    """Extract the full 518-dim descriptor vector from one audio file."""
    y, sr = librosa.load(path, sr=None, mono=True)

    values: dict[tuple[str, str, str], float] = {}
    for family, matrix in _descriptors(y, int(sr)).items():
        for stat, arr in _summarize(np.atleast_2d(matrix)).items():
            for i, value in enumerate(arr):
                values[(family, stat, f"{i + 1:02d}")] = float(value)

    return pd.Series(values).reindex(feature_columns())


def _extract_one(item: tuple[int, Path]) -> tuple[int, pd.Series | None]:
    track_id, path = item
    try:
        return track_id, extract(path)
    except Exception:  # noqa: BLE001 - corrupt audio is data, not a bug
        return track_id, None


def extract_many(paths: dict[int, Path], workers: int = 1) -> pd.DataFrame:
    """Extract for many files, skipping unreadable ones.

    FMA is known to contain truncated and corrupt mp3s, so a failure here is
    expected and must not abort a long run. Extraction is CPU-bound on the CQT,
    so ``workers > 1`` scales close to linearly.
    """
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(_extract_one, paths.items(), chunksize=4))
    else:
        results = [_extract_one(item) for item in paths.items()]

    rows = {tid: series for tid, series in results if series is not None}
    failed = [tid for tid, series in results if series is None]

    if failed:
        print(f"skipped {len(failed)} unreadable file(s): {failed[:5]}")

    return pd.DataFrame(rows).T.rename_axis("track_id")
