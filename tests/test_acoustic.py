"""Model A baseline tests using synthetic data — no FMA download required."""

import numpy as np

from aux.eval.embeddings import evaluate
from aux.models.acoustic import build_embeddings


def _clustered_features(n_per_class: int = 60, n_features: int = 30) -> tuple:
    """Three well-separated blobs: a sane embedding must score well above chance."""
    rng = np.random.default_rng(0)
    centres = rng.normal(scale=8.0, size=(3, n_features))
    x = np.vstack([c + rng.normal(size=(n_per_class, n_features)) for c in centres])
    y = np.repeat(["a", "b", "c"], n_per_class)
    return x, y


def test_build_embeddings_shape() -> None:
    import pandas as pd

    x, _ = _clustered_features()
    out = build_embeddings(pd.DataFrame(x), n_components=8)
    assert out.shape == (len(x), 8)


def test_evaluate_beats_baseline_on_separable_data() -> None:
    x, y = _clustered_features()
    report = evaluate(x, y, k=5)
    assert report.n_classes == 3
    assert report.knn_accuracy > report.majority_baseline
    assert report.retrieval_precision_at_k > 0.9


def test_evaluate_near_chance_on_noise() -> None:
    """Guards against a harness that flatters any input."""
    rng = np.random.default_rng(1)
    x = rng.normal(size=(180, 20))
    y = np.repeat(["a", "b", "c"], 60)
    report = evaluate(x, y, k=5)
    assert report.retrieval_precision_at_k < 0.55
