"""Probing what our embedding does and does not capture about perception.

Perceptual agreement sits at 0.397 against 0.333 chance while genre accuracy
reaches 0.630. Before spending effort on a larger model, three questions about
*why*:

1. **Does agreement rise with annotator confidence?** If we score higher on
   triplets listeners agreed strongly about, the embedding tracks perception
   noisily and a better representation should help. If agreement is flat across
   confidence, we are not tracking perception at all and no amount of feature
   engineering will fix it. This is the result that should gate Batch 2.

2. **Which feature families carry the perceptual signal?** Eleven families feed
   one space and we have never asked which of them the agreement comes from.

3. **Does fusing per-axis spaces help perception, or only metadata?** Fusion was
   validated on genre, artist and album — all metadata, which has misled us
   before.

Features are cached, since extraction dominates the runtime.

Run with:  python -m aux.experiments.perceptual_probe
"""

import argparse
import itertools

import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances

from aux.config.settings import get_settings
from aux.eval.embeddings import agreement_rate, standardize_distances, triplet_correct
from aux.experiments import tracking
from aux.ingest import mtat
from aux.models.acoustic import build_embeddings
from aux.models.features import AXES, axis_columns, extract_many, flatten_columns, restore_columns

CACHE = "mtat_features.parquet"


def _cached_features(n_fit: int, workers: int, seed: int) -> pd.DataFrame:
    """Extract MTAT features once and reuse them across probes."""
    path = get_settings().data_dir / "interim" / CACHE
    if path.exists():
        frame = restore_columns(pd.read_parquet(path))
        print(f"loaded {len(frame)} cached clips from {path.name}")
        return frame

    needed = np.unique(mtat.similarity_triplets())
    paths = mtat.audio_paths()
    rng = np.random.default_rng(seed)
    extra = rng.choice(
        [c for c in paths if c not in set(needed.tolist())],
        size=max(0, min(n_fit - len(needed), len(paths) - len(needed))),
        replace=False,
    )
    chosen = [*needed.tolist(), *extra.tolist()]

    print(f"extracting {len(chosen)} clips with {workers} workers ...")
    wanted = {int(c): paths[int(c)] for c in chosen if int(c) in paths}
    frame = extract_many(wanted, workers=workers)

    path.parent.mkdir(parents=True, exist_ok=True)
    flatten_columns(frame).to_parquet(path)
    print(f"cached {len(frame)} clips to {path.name}")
    return frame


def run(n_fit: int = 2500, workers: int = 6, dims: int = 128, seed: int = 0) -> None:
    tracking.setup()
    frame = _cached_features(n_fit, workers, seed)
    position = {int(t): i for i, t in enumerate(frame.index)}

    full = build_embeddings(frame, n_components=min(dims, len(frame) - 1))
    d_full = pairwise_distances(full, metric="cosine")

    comparisons = mtat.load_comparisons()
    metrics: dict[str, float] = {}

    # --- Probe 1: does agreement track annotator confidence? -----------------
    print("\n1. Agreement by annotator confidence (margin = lead of the outlier):")
    print(f"{'margin':>8}{'triplets':>10}{'agreement':>12}{'95% CI':>10}")
    votes = comparisons[mtat.VOTE_COLUMNS].to_numpy()
    ordered_all = np.sort(votes, axis=1)
    margins_all = ordered_all[:, 2] - ordered_all[:, 1]

    for lo, hi in ((1, 1), (2, 2), (3, 4), (5, 99)):
        keep = (margins_all >= lo) & (margins_all <= hi)
        subset = mtat.triplets_from(comparisons[keep], min_votes=0, min_margin=lo)

        usable = np.array([t for t in subset if all(int(x) in position for x in t)])
        if len(usable) < 10:
            continue
        indexed = np.vectorize(position.get)(usable)
        rate, ci = agreement_rate(triplet_correct(d_full, indexed))
        label = f"{lo}" if lo == hi else f"{lo}-{hi if hi < 99 else '+'}"
        print(f"{label:>8}{len(usable):>10}{rate:>12.3f}{ci:>10.3f}")
        metrics[f"margin_{lo}_agreement"] = rate
        metrics[f"margin_{lo}_n"] = float(len(usable))

    # --- Probe 2 and 3: per-axis signal, and fusion ---------------------------
    triplets = mtat.similarity_triplets()
    usable = np.array([t for t in triplets if all(int(x) in position for x in t)])
    indexed = np.vectorize(position.get)(usable)

    print(f"\n2. Per-axis agreement ({len(usable)} triplets, chance 0.333):")
    axis_distances = {}
    for axis in AXES:
        columns = axis_columns(frame, axis)
        if len(columns) == 0:
            continue  # cache predates this family (e.g. rhythm)
        embedded = build_embeddings(frame[columns], n_components=min(dims, len(columns) - 1))
        d = pairwise_distances(embedded, metric="cosine")
        axis_distances[axis] = standardize_distances(d)
        rate, ci = agreement_rate(triplet_correct(d, indexed))
        print(f"  {axis:<10} {len(columns):>4} cols   {rate:.3f} ±{ci:.3f}")
        metrics[f"axis_{axis}_agreement"] = rate

    rate, ci = agreement_rate(triplet_correct(d_full, indexed))
    print(f"  {'all':<10} {frame.shape[1]:>4} cols   {rate:.3f} ±{ci:.3f}")
    metrics["axis_all_agreement"] = rate

    print("\n3. Fusing axes (equal weight over each combination):")
    names = list(axis_distances)
    for size in (2, 3):
        for combo in itertools.combinations(names, size):
            fused = np.mean([axis_distances[a] for a in combo], axis=0)
            rate, ci = agreement_rate(triplet_correct(fused, indexed))
            print(f"  {'+'.join(combo):<28} {rate:.3f} ±{ci:.3f}")
            metrics[f"fused_{'_'.join(combo)}_agreement"] = rate

    run_id = tracking.log_run(
        f"perceptual-probe-n{len(usable)}",
        {"n_fit": len(frame), "dims": dims, "seed": seed},
        metrics,
    )
    print(f"\nlogged run={run_id[:8]}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-fit", type=int, default=2500)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    run(args.n_fit, args.workers)


if __name__ == "__main__":
    main()
