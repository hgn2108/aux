"""The encoder adapter contract.

Every joint music-text encoder aux benchmarks sits behind this interface, so that E0
(CLAP vs MuQ-MuLan) swaps one object and holds everything else fixed. Anything an encoder
needs that differs between models, input sample rate, window length, normalisation --
belongs inside its adapter, never in ingestion or in the evaluation harness.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from ..ingest import AudioAsset
from .resample import resample
from .segments import select_segments


def l2_normalise(vectors: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalisation, safe on zero rows."""
    vectors = np.atleast_2d(np.asarray(vectors, dtype=np.float32))
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, 1e-12)


class EncoderAdapter(ABC):
    """A frozen pretrained joint music-text encoder."""

    name: str
    version: str
    """Recorded alongside every embedding. DESIGN.md requires model/version on an
    impression for a past result to be reproducible; the same applies to a cached vector,
    which is invalid the moment the model behind it changes."""
    sample_rate: int
    """The rate this encoder requires. The adapter resamples to it; the library does not."""
    embedding_dim: int
    segment_seconds: float

    @abstractmethod
    def embed_audio(self, waveforms: list[np.ndarray]) -> np.ndarray:
        """Embed a batch of mono waveforms already at ``self.sample_rate``."""

    @abstractmethod
    def embed_text(self, texts: list[str]) -> np.ndarray:
        """Embed a batch of strings into the same space as ``embed_audio``."""

    def embed_track(
        self, asset: AudioAsset, *, n_segments: int = 1
    ) -> tuple[np.ndarray, np.ndarray]:
        """Turn an AudioAsset into one track vector.

        Returns ``(track_vector, segment_vectors)``. The segments come back rather than
        being discarded because within-track spread is what explains whether pooling helped.

        Resample -> select deterministic segments -> encode -> L2-normalise each -> mean
        pool -> L2-normalise. Normalising before pooling matters: without it a loud segment
        outweighs a quiet one through magnitude alone, which is a volume artefact.
        """
        audio = resample(asset.samples, asset.sample_rate, self.sample_rate)
        segments = select_segments(
            audio,
            self.sample_rate,
            n_segments=n_segments,
            segment_seconds=self.segment_seconds,
        )
        segment_vectors = l2_normalise(
            self.embed_audio([s.samples for s in segments])
        )
        track_vector = l2_normalise(segment_vectors.mean(axis=0))[0]
        return track_vector, segment_vectors
