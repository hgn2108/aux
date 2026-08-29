"""Per-axis acoustic evaluation on FMA.

Scores every embedding subspace against every ground-truth axis, producing a
matrix rather than a single number. That cross-table is the disentanglement
question made measurable: does the tonal subspace track key better than the
timbre subspace does, or does one subspace carry everything?

Ground truth comes from our own extraction, so the rhythm block is required —
FMA's precomputed features contain no rhythm information whatsoever.

Run with:  python -m aux.experiments.axis_check --n 1500 --workers 6
"""

import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances

from aux.config.settings import get_settings
from aux.eval import axes as axis_metrics
from aux.eval.embeddings import standardize_distances
from aux.experiments import tracking
from aux.experiments.extraction_check import ensure_audio_extracted
from aux.ingest import fma
from aux.models.acoustic import build_embeddings
from aux.models.features import AXES, axis_columns, extract_many, flatten_columns, restore_columns

CACHE = "fma_features_rhythm.parquet"


def _cached_features(n: int, workers: int, seed: int) -> pd.DataFrame:
    path = get_settings().data_dir / "interim" / CACHE
    if path.exists():
        frame = restore_columns(pd.read_parquet(path))
        print(f"loaded {len(frame)} cached tracks")
        return frame

    ensure_audio_extracted()
    ids = fma.subset(fma.load_tracks(), "small")
    rng = np.random.default_rng(seed)
    sample = rng.choice(ids, size=min(n, len(ids)), replace=False)

    print(f"extracting {len(sample)} tracks with {workers} workers ...")
    frame = extract_many(
        {int(t): fma.audio_path(int(t)) for t in sample}, workers=workers, rhythm=True
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    flatten_columns(frame).to_parquet(path)
    print(f"cached {len(frame)} tracks")
    return frame


def ground_truth(frame: pd.DataFrame) -> dict[str, tuple[np.ndarray, object]]:
    """Per-axis ground truth, each paired with the similarity rule that scores it."""
    chroma = frame["chroma_cens"]["mean"].to_numpy()
    keys = axis_metrics.estimate_keys(chroma)

    def scaled(name: str, family: str) -> tuple[np.ndarray, object]:
        values = frame[family]["mean"].to_numpy().ravel()
        # Tolerance of a quarter standard deviation: tight enough to mean
        # something, loose enough not to be measuring float noise.
        tolerance = 0.25 * values.std()
        return values, lambda a, b: axis_metrics.proximity_similarity(a, b, tolerance)

    return {
        "tempo": (
            frame["tempo"]["mean"].to_numpy().ravel(),
            lambda a, b: axis_metrics.tempo_similarity(a, b),
        ),
        "key": (keys, axis_metrics.key_similarity),
        "loudness": scaled("loudness", "rmse"),
        "brightness": scaled("brightness", "spectral_centroid"),
    }


def run(n: int = 1500, workers: int = 6, dims: int = 64, seed: int = 0, k: int = 5) -> None:
    tracking.setup()
    frame = _cached_features(n, workers, seed)
    truth = ground_truth(frame)

    distances = {}
    for name in AXES:
        columns = axis_columns(frame, name)
        embedded = build_embeddings(frame[columns], n_components=min(dims, len(columns) - 1))
        distances[name] = standardize_distances(pairwise_distances(embedded, metric="cosine"))

    everything = build_embeddings(frame, n_components=dims)
    distances["all (concat)"] = standardize_distances(
        pairwise_distances(everything, metric="cosine")
    )
    distances["fused"] = np.mean([distances[a] for a in AXES], axis=0)

    names = list(truth)
    header = "".join(f"{a:>13}" for a in names)
    print(f"\nlift over chance, k={k}, {len(frame)} tracks\n")
    print(f"{'subspace':<16}{header}")

    metrics: dict[str, float] = {}
    chances: dict[str, float] = {}
    for space, d in distances.items():
        row = ""
        for axis in names:
            values, similarity = truth[axis]
            observed, chance = axis_metrics.neighbourhood_coherence(d, values, similarity, k=k)
            chances[axis] = chance
            row += f"{observed / chance:>12.2f}x" if chance > 0 else f"{'n/a':>13}"
            metrics[f"{space.replace(' ', '_')}_{axis}_observed"] = observed
        print(f"{space:<16}{row}")

    print(f"\n{'chance rate':<16}" + "".join(f"{chances[a]:>13.3f}" for a in names))

    run_id = tracking.log_run(
        f"axis-check-n{len(frame)}",
        {"n_tracks": len(frame), "dims": dims, "k": k, "seed": seed},
        metrics | {f"chance_{a}": c for a, c in chances.items()},
    )
    print(f"\nlogged run={run_id[:8]}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=1500)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--dims", type=int, default=64)
    args = parser.parse_args()
    run(args.n, args.workers, args.dims)


if __name__ == "__main__":
    main()
