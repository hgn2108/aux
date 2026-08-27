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


def build_embeddings(features: pd.DataFrame, n_components: int = 64) -> np.ndarray:
    """Standardize then PCA-reduce a feature matrix to fixed-length vectors."""
    pipeline = Pipeline(
        [("scale", StandardScaler()), ("pca", PCA(n_components=n_components, random_state=0))]
    )
    return np.asarray(pipeline.fit_transform(features.to_numpy()))


def load_subset(name: str = "small") -> tuple[pd.DataFrame, pd.Series]:
    """Feature matrix and top-level genre labels for one FMA subset."""
    tracks = fma.load_tracks()
    features = fma.load_features()

    ids = fma.subset(tracks, name).intersection(features.index)
    genres = tracks.loc[ids, ("track", "genre_top")]

    keep = genres.notna()
    return features.loc[ids][keep], genres[keep]


def run_baseline(subset: str = "small", n_components: int = 64) -> EmbeddingReport:
    """End-to-end Model A baseline on FMA precomputed features."""
    features, genres = load_subset(subset)
    embeddings = build_embeddings(features, n_components)
    return evaluate(embeddings, genres.to_numpy())


if __name__ == "__main__":
    print(run_baseline())
