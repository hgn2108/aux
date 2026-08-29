"""Per-axis metric tests.

The key metric encodes music theory, so it is tested against named intervals
rather than arbitrary values — a sign error there would silently reward
unrelated keys.
"""

import numpy as np

from aux.eval import axes

# Keys are encoded 0-23: 0-11 major by tonic, 12-23 minor by tonic.
C_MAJOR, F_MAJOR, G_MAJOR, FS_MAJOR = 0, 5, 7, 6
C_MINOR, A_MINOR = 12, 12 + 9


def _score(a: int, b: int) -> float:
    return float(axes.key_similarity(np.array([a]), np.array([b]))[0])


def test_key_similarity_perfect_match() -> None:
    assert _score(C_MAJOR, C_MAJOR) == axes.PERFECT
    assert _score(A_MINOR, A_MINOR) == axes.PERFECT


def test_key_similarity_fifths() -> None:
    """A fifth apart is the closest relationship after identity."""
    assert _score(C_MAJOR, G_MAJOR) == axes.FIFTH
    assert _score(C_MAJOR, F_MAJOR) == axes.FIFTH


def test_key_similarity_relative_major_minor() -> None:
    """C major and A minor share a key signature: relative, in both directions."""
    assert _score(C_MAJOR, A_MINOR) == axes.RELATIVE
    assert _score(A_MINOR, C_MAJOR) == axes.RELATIVE


def test_key_similarity_parallel_mode() -> None:
    """Same tonic, different mode."""
    assert _score(C_MAJOR, C_MINOR) == axes.PARALLEL


def test_key_similarity_unrelated() -> None:
    """A tritone apart scores nothing."""
    assert _score(C_MAJOR, FS_MAJOR) == 0.0


def test_tempo_similarity_tolerance_and_octaves() -> None:
    fast, slow = np.array([120.0]), np.array([60.0])

    assert axes.tempo_similarity(fast, np.array([122.0]))[0] == 1.0  # within 4%
    assert axes.tempo_similarity(fast, np.array([135.0]))[0] == 0.0  # beyond 4%
    assert axes.tempo_similarity(fast, slow)[0] == 1.0  # octave accepted
    assert axes.tempo_similarity(fast, slow, octaves=False)[0] == 0.0  # and rejected


def test_estimate_keys_recovers_a_planted_profile() -> None:
    """A chroma vector shaped like the C-major profile must be read as C major."""
    chroma = np.vstack([axes.KRUMHANSL_MAJOR, np.roll(axes.KRUMHANSL_MINOR, 9)])
    assert list(axes.estimate_keys(chroma)) == [C_MAJOR, A_MINOR]


def test_neighbourhood_coherence_detects_structure_and_its_absence() -> None:
    """Ordered values under a matching distance score high; shuffled values score at chance."""
    values = np.arange(200, dtype=float)
    distances = np.abs(values[:, None] - values[None, :])

    def close(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return axes.proximity_similarity(a, b, tolerance=5.0)

    observed, chance = axes.neighbourhood_coherence(distances, values, close, k=5)
    assert observed > 0.95
    assert chance < 0.1

    rng = np.random.default_rng(0)
    shuffled = rng.permutation(values)
    observed, chance = axes.neighbourhood_coherence(distances, shuffled, close, k=5)
    assert abs(observed - chance) < 0.05
