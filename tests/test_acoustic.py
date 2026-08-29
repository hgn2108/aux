"""Model A and eval-harness tests on synthetic data — no FMA download required."""

import numpy as np
import pandas as pd

from aux.eval.embeddings import evaluate, triplet_agreement
from aux.models.acoustic import build_embeddings


def _clustered(n_per_class: int = 60, n_features: int = 30) -> tuple:
    """Three well-separated blobs: a sane embedding must score well above chance."""
    rng = np.random.default_rng(0)
    centres = rng.normal(scale=8.0, size=(3, n_features))
    x = np.vstack([c + rng.normal(size=(n_per_class, n_features)) for c in centres])
    y = np.repeat(["a", "b", "c"], n_per_class)
    return x, y


def test_build_embeddings_shape() -> None:
    x, _ = _clustered()
    assert build_embeddings(pd.DataFrame(x), n_components=8).shape == (len(x), 8)


def test_whitening_equalizes_component_variance() -> None:
    """The defect whitening fixes: unwhitened PCA variance is dominated by the first axes."""
    x, _ = _clustered()
    raw = build_embeddings(pd.DataFrame(x), n_components=8, whiten=False)
    whitened = build_embeddings(pd.DataFrame(x), n_components=8, whiten=True)

    spread = lambda m: m.std(axis=0).max() / m.std(axis=0).min()  # noqa: E731
    assert spread(raw) > 2.0
    assert spread(whitened) < 1.2


def test_beats_chance_on_separable_data() -> None:
    x, y = _clustered()
    report = evaluate(x, {"genre": y}, k=5)
    genre = report.labels["genre"]

    assert genre.n_classes == 3
    assert genre.precision_at_k > 0.9
    assert genre.knn_accuracy is not None and genre.knn_accuracy > genre.majority_baseline


def test_near_chance_on_noise() -> None:
    """Guards against a harness that flatters any input."""
    rng = np.random.default_rng(1)
    x = rng.normal(size=(180, 20))
    report = evaluate(x, {"genre": np.repeat(["a", "b", "c"], 60)}, k=5)
    assert report.labels["genre"].precision_at_k < 0.55


def test_high_cardinality_labels_skip_classification() -> None:
    """Album-like labels: retrieval still scored, classification correctly skipped."""
    x, _ = _clustered()
    many = np.arange(len(x)) // 2  # 90 classes, 2 tracks each
    report = evaluate(x, {"album": many.astype(str)}, k=5)

    album = report.labels["album"]
    assert album.n_classes == 90
    assert album.knn_accuracy is None
    assert album.silhouette is None
    assert album.precision_at_k >= 0.0


def test_cosine_metric_supported() -> None:
    x, y = _clustered()
    report = evaluate(x, {"genre": y}, k=5, metric="cosine")
    assert report.metric == "cosine"
    assert report.labels["genre"].precision_at_k > 0.9


def test_triplet_agreement_recovers_known_structure() -> None:
    """Triplets built so the outlier is genuinely far away must score near 1."""
    rng = np.random.default_rng(2)
    x = np.vstack([rng.normal(size=(2, 5)), rng.normal(size=(1, 5)) + 50])
    x = np.vstack([x] * 10)

    triplets = np.array([[i * 3, i * 3 + 1, i * 3 + 2] for i in range(10)])
    rate, ci95 = triplet_agreement(x, triplets)

    assert rate == 1.0
    assert ci95 == 0.0


def test_triplet_filtering_drops_ties_and_low_votes() -> None:
    """Ties carry no judgement; single-vote triplets carry almost none."""
    from aux.ingest.mtat import triplets_from

    df = pd.DataFrame(
        {
            "clip1_id": [1, 4, 7, 10],
            "clip2_id": [2, 5, 8, 11],
            "clip3_id": [3, 6, 9, 12],
            # clear outlier | tie | too few votes | clear, outlier first
            "clip1_numvotes": [1, 3, 1, 9],
            "clip2_numvotes": [1, 3, 0, 1],
            "clip3_numvotes": [6, 1, 0, 1],
        }
    )
    out = triplets_from(df, min_votes=3, min_margin=1)

    assert len(out) == 2
    assert list(out[0]) == [1, 2, 3]  # most-voted clip moved last
    assert out[1][2] == 10  # outlier moved from first position to last


def test_precision_at_k_from_distances_matches_embedding_path() -> None:
    """Distance-level and embedding-level retrieval must agree on the same space."""
    from sklearn.metrics import pairwise_distances

    from aux.eval.embeddings import precision_at_k_from_distances

    x, y = _clustered()
    direct = evaluate(x, {"genre": y}, k=5, metric="cosine").labels["genre"].precision_at_k
    from_distances = precision_at_k_from_distances(
        pairwise_distances(x, metric="cosine"), np.asarray(y), k=5
    )
    assert abs(direct - from_distances) < 1e-9


def test_standardize_distances_makes_scales_comparable() -> None:
    """Fusion weights must control the blend, not the spaces' arbitrary scales."""
    from aux.eval.embeddings import standardize_distances

    rng = np.random.default_rng(0)
    small = rng.random((50, 50)) * 0.01
    large = small * 1000
    np.fill_diagonal(small, 0.0)
    np.fill_diagonal(large, 0.0)

    a, b = standardize_distances(small), standardize_distances(large)
    off = ~np.eye(50, dtype=bool)
    np.testing.assert_allclose(a[off], b[off], atol=1e-9)
    assert abs(a[off].std() - 1.0) < 1e-9
