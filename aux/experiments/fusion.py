"""Fusing spaces that are strong on different axes.

Batch 1 found that PCA and LDA fail in opposite directions: PCA retrieves album
and artist far better, LDA retrieves genre far better, and each is poor where the
other is strong. This asks whether combining them beats either alone.

Fusion happens at the **distance** level, not the feature level. Concatenating a
7-dimensional space with a 128-dimensional one lets the larger space dominate by
sheer column count, and it discards the per-axis distances that Model C is
required to report. Combining after each space has produced its own distances
keeps both the weighting explicit and the axes separable.

Distances are standardised before combination: spaces of different dimensionality
produce distances on different scales, so a weighted sum of raw distances would be
governed by whichever space spreads widest rather than by the chosen weight.

**Protocol.** Projections are fit on the training split only and retrieval is
scored within the held-out split. Absolute numbers are therefore lower than
whole-corpus figures — fewer candidates are available to retrieve — and comparable
only to each other.

Run with:  python -m aux.experiments.fusion
"""

import argparse

import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import pairwise_distances
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from aux.eval.embeddings import precision_at_k_from_distances, standardize_distances
from aux.experiments import tracking
from aux.models.acoustic import load_subset

SEED = 0
TEST_SIZE = 0.2
K = 5
WEIGHTS = (0.0, 0.2, 0.35, 0.5, 0.65, 0.8, 1.0)


def run(subset: str = "small", dims: int = 128) -> None:
    tracking.setup()

    features, labels = load_subset(subset)
    encoded = {name: LabelEncoder().fit_transform(values) for name, values in labels.items()}
    x = features.to_numpy()

    train, test = train_test_split(
        np.arange(len(x)), test_size=TEST_SIZE, random_state=SEED, stratify=encoded["genre"]
    )

    pca = Pipeline(
        [("scale", StandardScaler()), ("pca", PCA(dims, whiten=True, random_state=SEED))]
    )
    lda = Pipeline(
        [("scale", StandardScaler()), ("lda", LinearDiscriminantAnalysis(n_components=7))]
    )
    pca.fit(x[train])
    lda.fit(x[train], encoded["genre"][train])

    d_pca = standardize_distances(pairwise_distances(pca.transform(x[test]), metric="cosine"))
    d_lda = standardize_distances(pairwise_distances(lda.transform(x[test]), metric="cosine"))

    print(f"{len(test):,} held-out tracks | retrieval P@{K}\n")
    print(f"{'w_lda':>7}{'genre':>9}{'artist':>9}{'album':>9}")

    metrics: dict[str, float] = {}
    for w in WEIGHTS:
        combined = (1 - w) * d_pca + w * d_lda
        scores = {
            name: precision_at_k_from_distances(combined, values[test], K)
            for name, values in encoded.items()
        }
        label = f"{w:.2f}" + ("  (pca)" if w == 0 else "  (lda)" if w == 1 else "")
        print(f"{label:>7}{scores['genre']:>9.3f}{scores['artist']:>9.3f}{scores['album']:>9.3f}")
        metrics |= {f"w{int(w * 100):03d}_{name}_p_at_k": v for name, v in scores.items()}

    run_id = tracking.log_run(
        f"fusion-pca-lda-{subset}",
        {"subset": subset, "dims": dims, "seed": SEED, "k": K, "n_test": len(test)},
        metrics,
    )
    print(f"\nlogged run={run_id[:8]}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset", default="small")
    parser.add_argument("--dims", type=int, default=128)
    args = parser.parse_args()
    run(args.subset, args.dims)


if __name__ == "__main__":
    main()
