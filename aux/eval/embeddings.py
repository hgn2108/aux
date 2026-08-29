"""Evaluation for embedding spaces.

Embeddings are judged the way the product uses them — retrieval — not by
classification accuracy alone. Labels are proxies for musical structure, never
the thing we are trying to predict.

Proxies get stronger down this list: genre is coarse, artist narrower, album
narrower still (shared production, instrumentation, session). Human similarity
triplets are the only target here that measures perception rather than metadata.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.preprocessing import LabelEncoder

# Above this many distinct labels, per-class metrics stop being meaningful and a
# stratified split stops being possible (albums with a single track). Retrieval
# precision still works, so that is all we report for high-cardinality labels.
MAX_CLASSES_FOR_CLASSIFICATION = 50


@dataclass
class LabelReport:
    """Scores for one label set."""

    name: str
    n_classes: int
    precision_at_k: float
    majority_baseline: float
    knn_accuracy: float | None = None
    silhouette: float | None = None

    def __str__(self) -> str:
        parts = [
            f"{self.name:<8} classes {self.n_classes:>5}",
            f"P@k {self.precision_at_k:.3f} (chance {self.majority_baseline:.3f})",
        ]
        if self.knn_accuracy is not None:
            parts.append(f"kNN {self.knn_accuracy:.3f}")
        if self.silhouette is not None:
            parts.append(f"sil {self.silhouette:+.3f}")
        return "  ".join(parts)


@dataclass
class EmbeddingReport:
    """Scores for one embedding space across every label set."""

    n_tracks: int
    n_dims: int
    metric: str
    labels: dict[str, LabelReport] = field(default_factory=dict)
    triplet_agreement: float | None = None
    triplet_ci95: float | None = None

    def __str__(self) -> str:
        lines = [f"tracks {self.n_tracks:,} | dims {self.n_dims} | metric {self.metric}"]
        lines += [f"  {report}" for report in self.labels.values()]
        if self.triplet_agreement is not None:
            ci = f" ±{self.triplet_ci95:.3f}" if self.triplet_ci95 is not None else ""
            lines.append(f"  human triplets  agreement {self.triplet_agreement:.3f}{ci}")
        return "\n".join(lines)

    def to_metrics(self) -> dict[str, float]:
        """Flatten to a name->value mapping for experiment tracking."""
        out: dict[str, float] = {}
        for name, report in self.labels.items():
            out[f"{name}_p_at_k"] = report.precision_at_k
            out[f"{name}_chance"] = report.majority_baseline
            if report.knn_accuracy is not None:
                out[f"{name}_knn_accuracy"] = report.knn_accuracy
            if report.silhouette is not None:
                out[f"{name}_silhouette"] = report.silhouette
        if self.triplet_agreement is not None:
            out["triplet_agreement"] = self.triplet_agreement
        return out


def _precision_at_k(x: np.ndarray, y: np.ndarray, k: int, metric: str) -> float:
    """Fraction of a track's k nearest neighbours sharing its label.

    Self-matches are excluded: this asks whether the neighbourhood a retrieval
    query would actually return is musically coherent.
    """
    nn = NearestNeighbors(n_neighbors=k + 1, metric=metric).fit(x)
    neighbours = nn.kneighbors(x, return_distance=False)[:, 1:]
    return float((y[neighbours] == y[:, None]).mean())


def precision_at_k_from_distances(distances: np.ndarray, labels: np.ndarray, k: int = 5) -> float:
    """Retrieval precision from a precomputed square distance matrix.

    Used when distances are combined across several spaces — fusing per-axis
    distances is the point of a multi-axis embedding, and that combination happens
    after each space has produced its own distances, not before.
    """
    distances = distances.copy()
    np.fill_diagonal(distances, np.inf)
    neighbours = np.argpartition(distances, k, axis=1)[:, :k]
    return float((labels[neighbours] == labels[:, None]).mean())


def standardize_distances(distances: np.ndarray) -> np.ndarray:
    """Centre and scale a distance matrix by its off-diagonal distribution.

    Spaces of different dimensionality produce distances on different scales, so a
    weighted sum of raw distances would be dominated by whichever space happens to
    spread widest rather than by the weight chosen.
    """
    off_diagonal = ~np.eye(len(distances), dtype=bool)
    values = distances[off_diagonal]
    return (distances - values.mean()) / values.std()


def _chance_precision(y: np.ndarray) -> float:
    """Probability a random other track shares a given track's label.

    The right chance baseline for retrieval: sampling by label frequency, not the
    majority-class rate used for classification.
    """
    counts = np.bincount(y)
    n = counts.sum()
    return float((counts * (counts - 1)).sum() / (n * (n - 1)))


def _score_labels(
    x: np.ndarray, name: str, raw_labels: np.ndarray, k: int, metric: str, seed: int
) -> LabelReport:
    y = LabelEncoder().fit_transform(raw_labels)
    n_classes = int(len(np.unique(y)))

    report = LabelReport(
        name=name,
        n_classes=n_classes,
        precision_at_k=_precision_at_k(x, y, k, metric),
        majority_baseline=_chance_precision(y),
    )
    if n_classes > MAX_CLASSES_FOR_CLASSIFICATION:
        return report

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=seed, stratify=y
    )
    knn = KNeighborsClassifier(n_neighbors=k, metric=metric).fit(x_train, y_train)
    report.knn_accuracy = float(knn.score(x_test, y_test))
    report.silhouette = float(silhouette_score(x, y, metric=metric))
    return report


def evaluate(
    x: np.ndarray,
    labels: Mapping[str, np.ndarray] | np.ndarray,
    k: int = 5,
    metric: str = "euclidean",
    seed: int = 0,
) -> EmbeddingReport:
    """Score an embedding matrix against one or more label sets."""
    if not isinstance(labels, Mapping):
        labels = {"label": np.asarray(labels)}

    return EmbeddingReport(
        n_tracks=len(x),
        n_dims=int(x.shape[1]),
        metric=metric,
        labels={
            name: _score_labels(x, name, np.asarray(values), k, metric, seed)
            for name, values in labels.items()
        },
    )


def paired_comparison(correct_a: np.ndarray, correct_b: np.ndarray) -> tuple[int, int, float]:
    """McNemar test for two models scored on the *same* items.

    Returns ``(b_only, a_only, p_value)`` where ``a_only`` counts items model A
    got right and B did not, and vice versa. One-sided: tests whether B beats A.

    Comparing two independent proportions throws away the pairing and cannot
    resolve realistic differences on small benchmarks — at 307 triplets the
    interval is +/-0.056, wider than most differences worth acting on. A paired
    test counts only the items where the models disagree, and resolved a 12-point
    difference at p=0.0003 on exactly that data.
    """
    from scipy import stats

    a_only = int((correct_a & ~correct_b).sum())
    b_only = int((~correct_a & correct_b).sum())
    discordant = a_only + b_only
    if discordant == 0:
        return a_only, b_only, 1.0

    p = stats.binomtest(b_only, discordant, 0.5, alternative="greater").pvalue
    return a_only, b_only, float(p)


def minimum_detectable_rate(
    n: int, baseline: float, power: float = 0.8, alpha: float = 0.05
) -> float:
    """Smallest rate a sample of ``n`` can reliably distinguish from ``baseline``.

    Reporting a score without this invites the mistake we already made once:
    treating a significant p-value on an underpowered test as a result. At n=307
    against chance 1/3 the detectable rate is 0.402, and we observed 0.397 —
    below our own threshold.
    """
    import math

    from scipy import stats

    z_alpha, z_beta = stats.norm.ppf(1 - alpha), stats.norm.ppf(power)
    h = (z_alpha + z_beta) / math.sqrt(n)
    return float(math.sin(h / 2 + math.asin(math.sqrt(baseline))) ** 2)


def triplet_correct(distances: np.ndarray, triplets: np.ndarray) -> np.ndarray:
    """Per-triplet agreement with an 'odd one out' verdict, from a distance matrix.

    Each row is ``(a, b, c)`` where listeners judged ``c`` the outlier; we agree
    when ``d(a, b)`` is the smallest of the three pairwise distances.
    """
    a, b, c = triplets[:, 0], triplets[:, 1], triplets[:, 2]
    return (distances[a, b] < distances[a, c]) & (distances[a, b] < distances[b, c])


def agreement_rate(correct: np.ndarray) -> tuple[float, float]:
    """Agreement rate and its 95% interval."""
    rate = float(correct.mean())
    return rate, float(1.96 * np.sqrt(rate * (1 - rate) / max(len(correct), 1)))


def triplet_agreement(
    x: np.ndarray, triplets: np.ndarray, metric: str = "euclidean"
) -> tuple[float, float]:
    """Agreement with human 'odd one out' judgements, and a 95% interval.

    Each row is ``(a, b, c)`` where listeners judged ``c`` the outlier. We agree
    when ``d(a, b)`` is the smallest of the three pairwise distances. Chance is
    1/3. The interval matters because these sets are small — MagnaTagATune has
    533 triplets, so a difference of a few points is not a real difference.
    """
    from sklearn.metrics import pairwise_distances

    return agreement_rate(triplet_correct(pairwise_distances(x, metric=metric), triplets))
