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

# Every corpus is resampled to this rate before extraction.
#
# Source material arrives at wildly different rates — FMA at 44100 Hz, MagnaTagATune
# at 16000 Hz — and loading each at its native rate puts their features on different
# frequency axes, making them silently incomparable. A fixed rate is what makes one
# feature space span several corpora.
#
# 22050 was chosen because FMA's own reference features are computed on that axis
# (their spectral centroids never exceed 7974 Hz and rolloff caps at 10451 Hz, both
# under the 11025 Hz Nyquist), it is librosa's default, and it is a common MIR
# convention. Audio arriving below this rate is upsampled: no information is added,
# and its spectrum is simply empty above its own Nyquist, which is honest about the
# source rather than hidden.
SAMPLE_RATE = 22050

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


# Rhythm descriptors, mirroring Essentia's `rhythm` group — the taxonomy
# AcousticBrainz ships its dumps in, so our features and the user's library will
# share one axis structure.
#
# The core 518 features contain no rhythm information at all: eleven families of
# spectral, tonal and energy descriptors and nothing about time. Bridge Finder's
# first named axis is tempo, so this is a gap in the product, not only the model.
#
# Kept as an additive block so the core 518 columns still line up with FMA's
# reference for the extraction check.
RHYTHM_SCALARS = ("tempo", "onset_rate", "pulse_clarity")
RHYTHM_ENVELOPE = "onset_strength"


# Feature families grouped into axes, following Essentia's descriptor taxonomy —
# the grouping AcousticBrainz ships its dumps in, so these carry over unchanged to
# the user's own library.
#
# Concatenating all families into one space is measurably harmful: a subspace of 10
# rhythm columns scores 2.69x chance on tempo while the combined 528-column space
# scores 1.08x, because column count decides a PCA's variance structure. Embed per
# axis, then fuse. See docs/results.md.
AXES = {
    "rhythm": ("tempo", "onset_rate", "pulse_clarity", "onset_strength"),
    "tonal": ("chroma_cens", "chroma_cqt", "chroma_stft", "tonnetz"),
    "timbre": (
        "mfcc",
        "spectral_contrast",
        "spectral_centroid",
        "spectral_bandwidth",
        "spectral_rolloff",
    ),
    "dynamics": ("rmse", "zcr"),
}


def axis_columns(frame: pd.DataFrame, axis: str) -> pd.Index:
    """Columns belonging to one axis, tolerating frames that predate a family.

    Cached extractions made before the rhythm block exists simply yield no rhythm
    columns rather than raising.
    """
    return frame.columns[frame.columns.get_level_values(0).isin(AXES[axis])]


def flatten_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Join the MultiIndex into single strings so the frame can be written to parquet."""
    flat = frame.copy()
    flat.columns = ["|".join(c) for c in frame.columns]
    return flat


def restore_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Inverse of flatten_columns, for reading a cached parquet back."""
    frame.columns = pd.MultiIndex.from_tuples(
        [tuple(c.split("|")) for c in frame.columns],
        names=["feature", "statistics", "number"],
    )
    return frame


def rhythm_columns() -> pd.MultiIndex:
    """MultiIndex for the rhythm block: three scalars plus a summarised envelope."""
    tuples = [(name, "mean", "01") for name in RHYTHM_SCALARS]
    tuples += [(RHYTHM_ENVELOPE, stat, "01") for stat in STATISTICS]
    return pd.MultiIndex.from_tuples(tuples, names=["feature", "statistics", "number"])


def _rhythm(y: np.ndarray, sr: int) -> dict[tuple[str, str, str], float]:
    """Tempo, onset density, pulse clarity, and the onset-strength envelope."""
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=512)
    tempo = librosa.feature.tempo(onset_envelope=onset_env, sr=sr, hop_length=512)
    onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, hop_length=512)

    # Pulse clarity: how sharply the tempogram peaks. A steady beat concentrates
    # energy at one lag; free or rubato playing spreads it out.
    tempogram = librosa.feature.tempogram(onset_envelope=onset_env, sr=sr, hop_length=512)
    profile = tempogram.mean(axis=1)
    clarity = float(profile.max() / profile.sum()) if profile.sum() > 0 else 0.0

    duration = len(y) / sr
    values: dict[tuple[str, str, str], float] = {
        ("tempo", "mean", "01"): float(tempo[0]),
        ("onset_rate", "mean", "01"): float(len(onsets) / duration) if duration else 0.0,
        ("pulse_clarity", "mean", "01"): clarity,
    }
    for stat, arr in _summarize(np.atleast_2d(onset_env)).items():
        values[(RHYTHM_ENVELOPE, stat, "01")] = float(arr[0])
    return values


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
        # sr must be passed with every S= call: librosa defaults to 22050 and
        # builds its frequency axis from that, silently mislabelling the spectrum
        # for 44.1 kHz audio.
        "chroma_stft": librosa.feature.chroma_stft(S=stft**2, sr=sr, n_chroma=12),
        "mfcc": librosa.feature.mfcc(
            S=librosa.power_to_db(librosa.feature.melspectrogram(S=stft**2, sr=sr)),
            n_mfcc=20,
        ),
        "rmse": librosa.feature.rms(S=stft),
        "spectral_bandwidth": librosa.feature.spectral_bandwidth(S=stft, sr=sr),
        "spectral_centroid": librosa.feature.spectral_centroid(S=stft, sr=sr),
        "spectral_contrast": librosa.feature.spectral_contrast(S=stft, sr=sr, n_bands=6),
        "spectral_rolloff": librosa.feature.spectral_rolloff(S=stft, sr=sr),
        "tonnetz": librosa.feature.tonnetz(chroma=chroma_cqt),
        "zcr": librosa.feature.zero_crossing_rate(y, frame_length=2048, hop_length=512),
    }


def extract(path: str | Path, rhythm: bool = True) -> pd.Series:
    """Extract descriptors from one audio file.

    Returns the core 518 columns, plus the rhythm block unless ``rhythm=False``.
    The extraction check against FMA's reference passes ``rhythm=False`` so the
    two frames line up column for column.
    """
    y, sr = librosa.load(path, sr=SAMPLE_RATE, mono=True)

    values: dict[tuple[str, str, str], float] = {}
    for family, matrix in _descriptors(y, int(sr)).items():
        for stat, arr in _summarize(np.atleast_2d(matrix)).items():
            for i, value in enumerate(arr):
                values[(family, stat, f"{i + 1:02d}")] = float(value)

    columns = feature_columns()
    if rhythm:
        values |= _rhythm(y, int(sr))
        columns = columns.append(rhythm_columns())

    return pd.Series(values).reindex(columns)


def _extract_one(item: tuple[int, Path, bool]) -> tuple[int, pd.Series | None]:
    track_id, path, rhythm = item
    try:
        return track_id, extract(path, rhythm=rhythm)
    except Exception:  # noqa: BLE001 - corrupt audio is data, not a bug
        return track_id, None


def extract_many(paths: dict[int, Path], workers: int = 1, rhythm: bool = True) -> pd.DataFrame:
    """Extract for many files, skipping unreadable ones.

    FMA is known to contain truncated and corrupt mp3s, so a failure here is
    expected and must not abort a long run. Extraction is CPU-bound on the CQT,
    so ``workers > 1`` scales close to linearly.
    """
    items = [(track_id, path, rhythm) for track_id, path in paths.items()]

    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(_extract_one, items, chunksize=4))
    else:
        results = [_extract_one(item) for item in items]

    rows = {tid: series for tid, series in results if series is not None}
    failed = [tid for tid, series in results if series is None]

    if failed:
        print(f"skipped {len(failed)} unreadable file(s): {failed[:5]}")

    return pd.DataFrame(rows).T.rename_axis("track_id")
