"""Head-to-head comparison of ways to build the embedding.

Fusion has only ever been compared against concatenation — which we separately
showed is broken, since ten rhythm columns among 528 contribute nothing to a
concatenated PCA's variance. Beating a baseline we proved defective is not
evidence that fusion is the right answer.

Compared here, all from the same features:

- **raw** — no projection, a reference point
- **concat** — one PCA over every column (the current code default)
- **axes_equal** — per-axis embeddings, distances standardised then averaged
- **axes_learned** — same, with weights fit on a training split

Scored against MagnaTagATune's human tags rather than our computed axes. Tempo
and loudness ground truth is derived from the same audio the embedding is built
from; nobody listened to a spectrogram to write "harpsichord". The tags are the
one target here that is independent of the representation being judged.

Tag similarity is cosine between binary tag vectors. Weights are fit on train
clips and every number reported comes from held-out clips, because a weighting
chosen on the data it is scored against is not a measurement.

Run with:  python -m aux.experiments.bakeoff
"""

import argparse

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from aux.config.settings import get_settings
from aux.eval.embeddings import agreement_rate, standardize_distances, triplet_correct
from aux.experiments import tracking
from aux.ingest import mtat
from aux.models.features import AXES, axis_columns, restore_columns

SEED = 0
K = 5
CACHE = "mtat_features.parquet"


def _embed(frame: pd.DataFrame, dims: int) -> np.ndarray:
    n_components = min(dims, frame.shape[1] - 1)
    pipeline = Pipeline(
        [
            ("scale", StandardScaler()),
            ("pca", PCA(n_components, whiten=True, random_state=SEED)),
        ]
    )
    return np.asarray(pipeline.fit_transform(frame.to_numpy()))


def _tag_coherence(distances: np.ndarray, tag_vectors: np.ndarray, k: int = K) -> float:
    """Mean tag-vector cosine among a clip's k nearest neighbours."""
    d = distances.copy()
    np.fill_diagonal(d, np.inf)
    neighbours = np.argpartition(d, k, axis=1)[:, :k]

    anchors = np.repeat(tag_vectors, k, axis=0)
    found = tag_vectors[neighbours.ravel()]
    return float((anchors * found).sum(axis=1).mean())


def _chance_coherence(tag_vectors: np.ndarray, seed: int = SEED) -> float:
    rng = np.random.default_rng(seed)
    n = len(tag_vectors) * K
    a = tag_vectors[rng.integers(0, len(tag_vectors), n)]
    b = tag_vectors[rng.integers(0, len(tag_vectors), n)]
    return float((a * b).sum(axis=1).mean())


def _learn_weights(
    axis_distances: dict[str, np.ndarray], tag_vectors: np.ndarray, train: np.ndarray
) -> dict[str, float]:
    """Fit non-negative axis weights maximising tag coherence on the training split.

    Parametrised through a softmax so weights stay positive and sum to one without
    constrained optimisation.
    """
    names = list(axis_distances)
    sub = {n: axis_distances[n][np.ix_(train, train)] for n in names}
    tags = tag_vectors[train]

    def negative_coherence(raw: np.ndarray) -> float:
        w = np.exp(raw - raw.max())
        w /= w.sum()
        blended = sum(w[i] * sub[n] for i, n in enumerate(names))
        return -_tag_coherence(np.asarray(blended), tags)

    result = minimize(
        negative_coherence,
        np.zeros(len(names)),
        method="Nelder-Mead",
        options={"maxiter": 200, "xatol": 1e-3, "fatol": 1e-5},
    )
    w = np.exp(result.x - result.x.max())
    w /= w.sum()
    return dict(zip(names, w.tolist(), strict=True))


def run(dims: int = 64, k: int = K) -> None:
    tracking.setup()

    frame = restore_columns(pd.read_parquet(get_settings().data_dir / "interim" / CACHE))
    tags = mtat.load_tags()

    shared = frame.index.intersection(tags.index)
    frame, tags = frame.loc[shared], tags.loc[shared]
    print(f"{len(shared)} clips with both features and tags | {tags.shape[1]} tags")

    raw = tags.to_numpy(dtype=float)
    tag_vectors = raw / (np.linalg.norm(raw, axis=1, keepdims=True) + 1e-12)

    # Per-axis distances, and the two whole-space baselines.
    axis_distances = {}
    for axis in AXES:
        columns = axis_columns(frame, axis)
        if len(columns) == 0:
            continue
        axis_distances[axis] = standardize_distances(
            pairwise_distances(_embed(frame[columns], dims), metric="cosine")
        )

    spaces = {
        "raw": standardize_distances(
            pairwise_distances(StandardScaler().fit_transform(frame.to_numpy()), metric="cosine")
        ),
        "concat": standardize_distances(pairwise_distances(_embed(frame, 128), metric="cosine")),
        "axes_equal": np.mean(list(axis_distances.values()), axis=0),
    }

    train, test = train_test_split(np.arange(len(frame)), test_size=0.3, random_state=SEED)
    weights = _learn_weights(axis_distances, tag_vectors, train)
    spaces["axes_learned"] = sum(w * axis_distances[n] for n, w in weights.items())

    print("\nlearned axis weights: " + "  ".join(f"{n} {w:.2f}" for n, w in weights.items()))

    # Everything is scored on held-out clips only.
    held = np.ix_(test, test)
    chance = _chance_coherence(tag_vectors[test])

    print(f"\ntag coherence on {len(test)} held-out clips (chance {chance:.3f})\n")
    print(f"{'space':<16}{'coherence':>12}{'lift':>9}")

    metrics: dict[str, float] = {"tag_chance": chance}
    for name, d in {**{f"axis:{a}": v for a, v in axis_distances.items()}, **spaces}.items():
        score = _tag_coherence(np.asarray(d)[held], tag_vectors[test], k=k)
        print(f"{name:<16}{score:>12.3f}{score / chance:>8.2f}x")
        metrics[f"{name.replace(':', '_')}_tag_coherence"] = score

    # The 307 human triplets, as a second opinion on a much smaller sample.
    position = {int(t): i for i, t in enumerate(frame.index)}
    triplets = np.array(
        [t for t in mtat.similarity_triplets() if all(int(x) in position for x in t)]
    )
    if len(triplets) > 20:
        indexed = np.vectorize(position.get)(triplets)
        print(f"\nhuman triplet agreement ({len(indexed)} triplets, chance 0.333)\n")
        for name, d in spaces.items():
            rate, ci = agreement_rate(triplet_correct(np.asarray(d), indexed))
            print(f"{name:<16}{rate:>12.3f} ±{ci:.3f}")
            metrics[f"{name}_triplet_agreement"] = rate

    run_id = tracking.log_run(
        f"bakeoff-n{len(shared)}",
        {"n_clips": len(shared), "dims": dims, "k": k, "seed": SEED}
        | {f"weight_{n}": round(w, 4) for n, w in weights.items()},
        metrics,
    )
    print(f"\nlogged run={run_id[:8]}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dims", type=int, default=64)
    args = parser.parse_args()
    run(args.dims)


if __name__ == "__main__":
    main()
