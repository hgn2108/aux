"""Model A — acoustic embeddings.

Baseline first: standardize precomputed spectral features, then reduce with PCA.
A learned encoder (roadmap A7) is only justified if it beats this on the same
eval harness.
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from aux.eval.embeddings import EmbeddingReport, evaluate
from aux.ingest import fma

LABEL_COLUMNS = {
    "genre": ("track", "genre_top"),
    "artist": ("artist", "id"),
    "album": ("album", "id"),
}


def build_embeddings(
    features: pd.DataFrame, n_components: int = 128, whiten: bool = True
) -> np.ndarray:
    """Standardize then PCA-reduce a feature matrix to fixed-length vectors.

    Defaults chosen by the sweep in ``aux.experiments.sweeps`` (see RESULTS.md).

    PCA components carry decreasing variance, so without whitening a distance is
    dominated by the first few — nominally 128 dimensions, effectively a handful.
    Whitening rescales each component to unit variance.

    Whitening only helps under cosine. Under Euclidean it actively hurts, because
    rescaling low-variance components to unit variance amplifies noise directions
    that then count equally with signal. Pair whitening with ``metric="cosine"``.
    """
    pipeline = Pipeline(
        [
            ("scale", StandardScaler()),
            ("pca", PCA(n_components=n_components, whiten=whiten, random_state=0)),
        ]
    )
    return np.asarray(pipeline.fit_transform(features.to_numpy()))


def load_subset(name: str = "small") -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Feature matrix and every available label set for one FMA subset."""
    tracks = fma.load_tracks()
    features = fma.load_features()

    ids = fma.subset(tracks, name).intersection(features.index)
    genres = tracks.loc[ids, LABEL_COLUMNS["genre"]]

    keep = genres.notna()
    ids = ids[keep]

    labels = {key: tracks.loc[ids, column].to_numpy() for key, column in LABEL_COLUMNS.items()}
    return features.loc[ids], labels


def run_baseline(
    subset: str = "small",
    n_components: int = 128,
    whiten: bool = True,
    metric: str = "cosine",
) -> EmbeddingReport:
    """End-to-end Model A baseline on FMA precomputed features."""
    features, labels = load_subset(subset)
    embeddings = build_embeddings(features, n_components, whiten)
    return evaluate(embeddings, labels, metric=metric)


if __name__ == "__main__":
    print(run_baseline())
