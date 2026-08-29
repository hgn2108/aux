"""Batch 1 — what do the features contain, and does our projection keep it?

Every result so far is ambiguous between two explanations: hand-crafted features
are weak, or our embedding discards what they hold. Those call for opposite
responses and nothing we had run distinguished them.

Two questions:

1. **Ceiling.** What does a strong supervised classifier extract from the raw 518
   features? FMA's own paper reports 63% genre accuracy with an SVM (16 genres,
   ~10x chance). That is a supervised classifier on raw features, not an
   unsupervised projection — the comparison we never made.

2. **Loss.** Does our embedding retain it? kNN on raw features versus kNN on each
   projection isolates the projection's contribution, holding the classifier fixed.

Supervised projections are fit on the training split only. Fitting LDA on all
labels and splitting afterwards inflates its kNN accuracy by ~8.7 points — the
projection has already seen the test set's labels. PCA is unaffected, being
unsupervised, which is exactly how the leak was spotted.

Projections compared:

- **PCA** — maximises variance. Variance directions need not be similarity
  directions; this is the mismatch Batch 1 exists to test.
- **PCA + log** — heavy-tailed magnitude descriptors logged first.
- **LDA** — supervised, maximises between-class separation. Capped at
  ``n_classes - 1`` components, so 7 here.
- **NCA** — supervised, directly optimises kNN performance. The closest cheap
  approximation to "learn a metric for retrieval".

Run with:  python -m aux.experiments.ceiling
"""

import argparse
import time

import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, NeighborhoodComponentsAnalysis
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC, LinearSVC

from aux.eval.embeddings import evaluate
from aux.experiments import tracking
from aux.models.acoustic import load_subset, log_heavy_tails

SEED = 0
TEST_SIZE = 0.2
K = 5


def _classifier_ceiling(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """Accuracy of strong supervised classifiers on the raw feature matrix."""
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=TEST_SIZE, random_state=SEED, stratify=y
    )

    models = {
        "svm_rbf": SVC(C=10.0, gamma="scale", random_state=SEED),
        "svm_linear": LinearSVC(C=1.0, dual="auto", random_state=SEED, max_iter=5000),
        "hist_gradient_boosting": HistGradientBoostingClassifier(random_state=SEED),
        "knn_raw": KNeighborsClassifier(n_neighbors=K, metric="cosine"),
    }

    out = {}
    for name, model in models.items():
        started = time.time()
        model.fit(x_train, y_train)
        out[name] = float(model.score(x_test, y_test))
        print(f"  {name:<24} {out[name]:.3f}   ({time.time() - started:.0f}s)")
    return out


def _projections(
    x: np.ndarray, x_log: np.ndarray, y: np.ndarray, train: np.ndarray, dims: int
) -> dict:
    """Build each candidate embedding space, fitting only on the training split."""
    pca = Pipeline(
        [("scale", StandardScaler()), ("pca", PCA(dims, whiten=True, random_state=SEED))]
    )

    # LDA yields at most n_classes - 1 components; that ceiling is the point of
    # comparing it, not a limitation to work around.
    n_lda = min(dims, len(np.unique(y)) - 1)
    lda = Pipeline(
        [("scale", StandardScaler()), ("lda", LinearDiscriminantAnalysis(n_components=n_lda))]
    )

    # NCA is O(n^2) per iteration, so it runs on PCA-reduced input rather than the
    # raw 518 dims. Standard practice, and it keeps the run to minutes.
    nca = Pipeline(
        [
            ("scale", StandardScaler()),
            ("pca", PCA(dims, random_state=SEED)),
            (
                "nca",
                NeighborhoodComponentsAnalysis(n_components=64, max_iter=50, random_state=SEED),
            ),
        ]
    )

    spaces = {}
    for name, pipeline, source, supervised in (
        ("pca", pca, x, False),
        ("pca_log", pca, x_log, False),
        ("lda", lda, x, True),
        ("nca", nca, x, True),
    ):
        started = time.time()
        if supervised:
            pipeline.fit(source[train], y[train])
        else:
            pipeline.fit(source[train])
        spaces[name] = pipeline.transform(source)
        print(f"  built {name:<10} dims {spaces[name].shape[1]:>4}  ({time.time() - started:.0f}s)")
    return spaces


def run(subset: str = "small", dims: int = 128) -> None:
    tracking.setup()

    features, labels = load_subset(subset)
    y = LabelEncoder().fit_transform(labels["genre"])
    x = StandardScaler().fit_transform(features.to_numpy())
    x_log = StandardScaler().fit_transform(log_heavy_tails(features).to_numpy())

    print(
        f"{len(features):,} tracks | {features.shape[1]} raw features | {len(np.unique(y))} genres"
    )

    print("\n1. Supervised ceiling on raw features (genre accuracy):")
    ceiling = _classifier_ceiling(x, y)

    train, _ = train_test_split(
        np.arange(len(y)), test_size=TEST_SIZE, random_state=SEED, stratify=y
    )

    print("\n2. Building projections (fit on training split only):")
    spaces = {"raw": x, "raw_log": x_log, **_projections(x, x_log, y, train, dims)}

    print("\n3. Retrieval on each space (cosine):")
    metrics = dict(ceiling)
    for name, embedding in spaces.items():
        report = evaluate(embedding, labels, k=K, metric="cosine")
        genre = report.labels["genre"]
        print(
            f"  {name:<10} dims {report.n_dims:>4}  "
            f"genre P@5 {genre.precision_at_k:.3f}  kNN {genre.knn_accuracy:.3f}  "
            f"artist P@5 {report.labels['artist'].precision_at_k:.3f}  "
            f"album P@5 {report.labels['album'].precision_at_k:.3f}"
        )
        metrics |= {f"{name}_{k}": v for k, v in report.to_metrics().items()}

    run_id = tracking.log_run(
        f"batch1-ceiling-{subset}",
        {"subset": subset, "dims": dims, "seed": SEED, "k": K},
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
