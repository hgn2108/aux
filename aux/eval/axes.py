"""Per-axis evaluation of acoustic similarity.

Model A answers *does this sound alike*, which — unlike perceived similarity — has
measurable ground truth for every track we hold. That removes the constraint that
had been throttling evaluation: perceptual benchmarks are scarce and small (307
triplets, exhausted), while acoustic axes are computable for all 8,000.

The PRD names Model A's axes directly: tempo, energy, key, spectral proximity.
Each is scored the same way, so results are comparable side by side:

    do a track's k nearest neighbours share this property
    more often than a randomly chosen pair does?

Scoring follows MIREX convention rather than exact match:

- **Tempo** — within ±4% (Accuracy1). Optionally accepting half, double and triple
  (Accuracy2), since estimators routinely land an octave out and listeners do not
  hear 90 BPM and 180 BPM as unrelated.
- **Key** — weighted, with partial credit: a perfect match scores 1, a fifth 0.5, a
  relative major/minor 0.3, a parallel mode 0.2. Exact-match scoring would count a
  fifth as no better than a tritone, which is musically wrong.

A note on circularity: tempo, loudness and brightness ground truth is derived from
the same audio the embedding is built from. That does not make the question empty —
the embedding is a lossy compression, and asking what it *discards* is exactly the
point — but it is weaker evidence than an independent annotation, and is reported
as such.
"""

import numpy as np

# Krumhansl–Kessler key profiles: how strongly each pitch class is expected to
# sound in a major or minor key. Correlating a track's averaged chroma against all
# 24 rotations gives its key.
KRUMHANSL_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
KRUMHANSL_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

PERFECT, FIFTH, RELATIVE, PARALLEL = 1.0, 0.5, 0.3, 0.2


def estimate_keys(chroma: np.ndarray) -> np.ndarray:
    """Estimate key per track from averaged chroma. Returns 0-23: 0-11 major, 12-23 minor."""
    profiles = np.vstack(
        [np.roll(KRUMHANSL_MAJOR, i) for i in range(12)]
        + [np.roll(KRUMHANSL_MINOR, i) for i in range(12)]
    )
    # Correlate each track's chroma against all 24 key profiles.
    a = chroma - chroma.mean(axis=1, keepdims=True)
    b = profiles - profiles.mean(axis=1, keepdims=True)
    correlation = (a @ b.T) / (
        np.linalg.norm(a, axis=1, keepdims=True) * np.linalg.norm(b, axis=1) + 1e-12
    )
    return np.asarray(correlation.argmax(axis=1))


def key_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """MIREX-weighted key agreement, giving partial credit for related keys."""
    tonic_a, tonic_b = a % 12, b % 12
    major_a, major_b = a < 12, b < 12

    interval = (tonic_a - tonic_b) % 12
    same_mode = major_a == major_b

    score = np.zeros(a.shape, dtype=float)
    score[same_mode & (interval == 0)] = PERFECT
    score[same_mode & np.isin(interval, (5, 7))] = FIFTH
    # Relative keys share a signature: C major <-> A minor. Measured as
    # (tonic_a - tonic_b) % 12, that is 3 looking from the major and 9 looking
    # from the minor — the two directions are not symmetric.
    score[(~same_mode) & major_a & (interval == 3)] = RELATIVE
    score[(~same_mode) & (~major_a) & (interval == 9)] = RELATIVE
    score[(~same_mode) & (interval == 0)] = PARALLEL
    return score


def tempo_similarity(
    a: np.ndarray, b: np.ndarray, tolerance: float = 0.04, octaves: bool = True
) -> np.ndarray:
    """Binary tempo agreement within a tolerance, optionally allowing octave errors."""
    ratios = [1.0, 0.5, 2.0, 1 / 3, 3.0] if octaves else [1.0]
    close = np.zeros(a.shape, dtype=bool)
    for ratio in ratios:
        close |= np.abs(a / (b * ratio + 1e-12) - 1) < tolerance
    return close.astype(float)


def proximity_similarity(a: np.ndarray, b: np.ndarray, tolerance: float) -> np.ndarray:
    """Binary agreement for a continuous quantity, in units of its own spread."""
    return (np.abs(a - b) < tolerance).astype(float)


def neighbourhood_coherence(
    distances: np.ndarray,
    values: np.ndarray,
    similarity,
    k: int = 5,
    seed: int = 0,
) -> tuple[float, float]:
    """Mean axis-similarity among k nearest neighbours, and the chance rate.

    Chance is estimated by sampling random pairs rather than assumed, because it
    differs per axis — tempo agreement is common by accident, an exact key match
    much less so.
    """
    d = distances.copy()
    np.fill_diagonal(d, np.inf)
    neighbours = np.argpartition(d, k, axis=1)[:, :k]

    observed = float(similarity(np.repeat(values, k), values[neighbours].ravel()).mean())

    rng = np.random.default_rng(seed)
    n = len(values) * k
    chance = float(
        similarity(
            values[rng.integers(0, len(values), n)], values[rng.integers(0, len(values), n)]
        ).mean()
    )
    return observed, chance
