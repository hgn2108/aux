"""Generic evaluation for embedding spaces.

Embeddings are judged the way the product uses them — retrieval — not by
classification accuracy alone. Labels are a proxy for musical structure, never
the thing we are trying to predict.
"""

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.preprocessing import LabelEncoder


@dataclass
class EmbeddingReport:
    n_tracks: int
    n_dims: int
    n_classes: int
    majority_baseline: float
    knn_accuracy: float
    retrieval_precision_at_k: float
    silhouette: float

    def __str__(self) -> str:
        lift = self.knn_accuracy - self.majority_baseline
        return (
            f"tracks {self.n_tracks:,} | dims {self.n_dims} | classes {self.n_classes}\n"
            f"  majority baseline   {self.majority_baseline:.3f}\n"
            f"  kNN accuracy        {self.knn_accuracy:.3f}  (lift {lift:+.3f})\n"
            f"  retrieval P@k       {self.retrieval_precision_at_k:.3f}\n"
            f"  silhouette          {self.silhouette:.3f}"
        )


def _precision_at_k(x: np.ndarray, y: np.ndarray, k: int) -> float:
    """Fraction of a track's k nearest neighbours sharing its label.

    Self-matches are excluded: this asks whether the neighbourhood a retrieval
    query would return is musically coherent.
    """
    nn = NearestNeighbors(n_neighbors=k + 1).fit(x)
    neighbours = nn.kneighbors(x, return_distance=False)[:, 1:]
    return float((y[neighbours] == y[:, None]).mean())


def evaluate(
    x: np.ndarray,
    labels: np.ndarray,
    k: int = 5,
    test_size: float = 0.2,
    seed: int = 0,
) -> EmbeddingReport:
    """Score an embedding matrix against categorical labels."""
    y = LabelEncoder().fit_transform(labels)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=seed, stratify=y
    )
    knn = KNeighborsClassifier(n_neighbors=k).fit(x_train, y_train)

    counts = np.bincount(y_test)
    return EmbeddingReport(
        n_tracks=len(x),
        n_dims=x.shape[1],
        n_classes=len(np.unique(y)),
        majority_baseline=float(counts.max() / counts.sum()),
        knn_accuracy=float(knn.score(x_test, y_test)),
        retrieval_precision_at_k=_precision_at_k(x, y, k),
        silhouette=float(silhouette_score(x, y)),
    )
