"""A10 — score the embedding against human similarity judgements.

Every other target we use is metadata: genre, artist, album. Those are proxies
for "sounds alike". MagnaTagATune's triplets are the only evaluation here that
measures what listeners actually heard, via an "odd one out" game.

Chance is 1/3. Report the confidence interval alongside the score — at a few
hundred triplets a difference of several points is not a difference.

Two caveats that must travel with any number this produces:

- MTAT audio is 16 kHz, 32 kbps mono. Our descriptors are computed from heavily
  compressed audio, which should understate agreement rather than inflate it.
- The embedding is fit on MTAT clips, not FMA, so this measures the pipeline
  rather than the exact FMA-fitted space used elsewhere.

Run with:  python -m aux.experiments.perceptual_check --n-fit 2500 --workers 8
"""

import argparse

import numpy as np

from aux.eval.embeddings import triplet_agreement
from aux.experiments import tracking
from aux.ingest import mtat
from aux.models.acoustic import build_embeddings
from aux.models.features import extract_many


def run(n_fit: int = 2500, workers: int = 8, dims: int = 128, seed: int = 0) -> None:
    tracking.setup()

    triplets = mtat.similarity_triplets()
    needed = np.unique(triplets)
    print(f"{len(triplets)} usable triplets referencing {len(needed)} clips")

    # Fit the embedding on a wider sample than the triplets alone: PCA estimated
    # from ~600 points would be dominated by sampling noise.
    all_paths = mtat.audio_paths()
    rng = np.random.default_rng(seed)
    extra = rng.choice(
        [c for c in all_paths if c not in set(needed.tolist())],
        size=max(0, min(n_fit - len(needed), len(all_paths) - len(needed))),
        replace=False,
    )
    chosen = [*needed.tolist(), *extra.tolist()]
    wanted = {int(c): all_paths[int(c)] for c in chosen if int(c) in all_paths}

    print(f"extracting {len(wanted)} clips with {workers} workers ...")
    frame = extract_many(wanted, workers=workers)
    print(f"extracted {len(frame)}")

    embeddings = build_embeddings(frame, n_components=min(dims, len(frame) - 1))
    position = {track_id: i for i, track_id in enumerate(frame.index)}

    # Keep only triplets whose three clips all extracted successfully.
    usable = np.array([t for t in triplets if all(int(c) in position for c in t)])
    dropped = len(triplets) - len(usable)
    if dropped:
        print(f"dropped {dropped} triplet(s) with unreadable audio")

    indexed = np.vectorize(position.get)(usable)

    print("\nagreement with human 'odd one out' judgements:")
    metrics = {}
    for metric in ("cosine", "euclidean"):
        rate, ci95 = triplet_agreement(embeddings, indexed, metric=metric)
        lift = rate - 1 / 3
        print(f"  {metric:<10} {rate:.3f} ±{ci95:.3f}   (chance 0.333, lift {lift:+.3f})")
        metrics[f"{metric}_agreement"] = rate
        metrics[f"{metric}_ci95"] = ci95

    run_id = tracking.log_run(
        f"a10-perceptual-n{len(usable)}",
        {"n_triplets": len(usable), "n_fit": len(frame), "dims": dims, "seed": seed},
        metrics,
    )
    print(f"\nlogged run={run_id[:8]}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-fit", type=int, default=2500)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    run(args.n_fit, args.workers)


if __name__ == "__main__":
    main()
