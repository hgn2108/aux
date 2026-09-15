"""Recommendation from a reference track or a text query.

Three modes: audio (MuQ-MuLan), lyrics (Qwen3 over Whisper transcripts), and a weighted
blend of the two. See `blend` for the combination rule.

Rank fusion was tried first and dropped: it weights its inputs equally by construction, so
there is nothing to ablate. A weight is explicit and measurable.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


def _minmax(scores: np.ndarray) -> np.ndarray:
    """Normalise to [0, 1] per query row, leaving a constant row at zero."""
    scores = np.asarray(scores, dtype=float)
    lo = scores.min(axis=-1, keepdims=True)
    hi = scores.max(axis=-1, keepdims=True)
    span = hi - lo
    return np.where(span > 0, (scores - lo) / np.where(span > 0, span, 1.0), 0.0)


def _zscore(scores: np.ndarray) -> np.ndarray:
    """Standardise per query row. Scale comes from the whole distribution, not the two
    extreme candidates, so one outlier cannot rescale everything else."""
    scores = np.asarray(scores, dtype=float)
    mu = scores.mean(axis=-1, keepdims=True)
    sd = scores.std(axis=-1, keepdims=True)
    return np.where(sd > 0, (scores - mu) / np.where(sd > 0, sd, 1.0), 0.0)


NORMALISERS = {"zscore": _zscore, "minmax": _minmax}
DEFAULT_NORMALISER = "zscore"


def blend(audio_scores: np.ndarray, lyric_scores: np.ndarray | None, alpha: float,
          has_lyrics: np.ndarray | None = None,
          normaliser: str = DEFAULT_NORMALISER) -> np.ndarray:
    """Combine one row of audio scores with one row of lyric scores.

    `alpha` weights audio: 1.0 is audio only, 0.0 lyrics only. Normalised per query first,
    or alpha would favour whichever model spreads its scores wider.

    Candidates with no transcript keep their audio score instead of zero, which would sink
    every instrumental. `lyric_scores=None` does that for the whole row, for when the query
    itself has no lyrics.
    """
    norm = NORMALISERS[normaliser]
    audio = norm(np.asarray(audio_scores, dtype=float))
    if lyric_scores is None or alpha >= 1.0:
        return audio

    lyric_scores = np.asarray(lyric_scores, dtype=float)
    usable = np.isfinite(lyric_scores)
    if has_lyrics is not None:
        usable &= np.asarray(has_lyrics, dtype=bool)
    if not usable.any():
        return audio

    lyric = np.zeros_like(audio)
    lyric[usable] = norm(lyric_scores[usable])
    if alpha <= 0.0:
        # Lyrics only: a candidate without a transcript cannot be ranked at all.
        out = np.where(usable, lyric, -np.inf)
        return out

    out = alpha * audio + (1 - alpha) * lyric
    out[~usable] = audio[~usable]
    return out


def score_queries(query_vectors: np.ndarray, item_vectors: np.ndarray,
                  usable: np.ndarray | None = None) -> np.ndarray:
    """Cosine scores for every text query against every track.

    Items marked unusable score -inf so they never rank, which is how a track with no
    transcript is kept out of a lyric ranking without being confused for a bad match.
    """
    scores = np.asarray(query_vectors, np.float32) @ np.asarray(item_vectors, np.float32).T
    if usable is not None:
        scores[:, ~np.asarray(usable, dtype=bool)] = -np.inf
    return scores.astype(float)


def blend_rows(audio: np.ndarray, lyric: np.ndarray, alpha: float,
               has_lyrics: np.ndarray, normaliser: str = DEFAULT_NORMALISER) -> np.ndarray:
    """Apply `blend` to every row of a query-by-track score matrix."""
    out = np.empty_like(np.asarray(audio, dtype=float))
    for i in range(out.shape[0]):
        out[i] = blend(audio[i], lyric[i], alpha, has_lyrics, normaliser)
    return out


@dataclass(frozen=True, slots=True)
class Recommendation:
    index: int
    rank: int
    score: float
    audio_score: float
    lyric_score: float | None
    """None when the track has no transcript, so the UI can say so rather than show a
    fabricated zero."""

    def explain(self) -> str:
        """A deterministic one-line reason, for display alongside the result."""
        def band(v: float) -> str:
            return "high" if v >= 0.66 else ("moderate" if v >= 0.33 else "low")

        if self.lyric_score is None:
            return f"{band(self.audio_score)} acoustic similarity; no lyrics available"
        return (f"{band(self.audio_score)} acoustic similarity, "
                f"{band(self.lyric_score)} lyrical similarity")


class Recommender:
    """Recommends tracks from an indexed collection."""

    def __init__(self, audio_vectors: np.ndarray, *, lyric_vectors: np.ndarray | None = None,
                 has_lyrics: np.ndarray | None = None, paths: list[Path] | None = None,
                 normaliser: str = DEFAULT_NORMALISER) -> None:
        if normaliser not in NORMALISERS:
            raise ValueError(f"unknown normaliser {normaliser!r}; "
                             f"expected one of {sorted(NORMALISERS)}")
        self.normaliser = normaliser
        self._norm = NORMALISERS[normaliser]
        self.audio = np.asarray(audio_vectors, dtype=np.float32)
        self.lyrics = None if lyric_vectors is None else np.asarray(lyric_vectors, np.float32)
        n = self.audio.shape[0]
        if has_lyrics is None:
            has_lyrics = np.zeros(n, bool) if self.lyrics is None else np.ones(n, bool)
        self.has_lyrics = np.asarray(has_lyrics, dtype=bool)
        self.paths = list(paths) if paths is not None else None

    def __len__(self) -> int:
        return int(self.audio.shape[0])

    # --- scoring ---------------------------------------------------------------------

    def audio_scores(self, track_index: int) -> np.ndarray:
        return self.audio @ self.audio[track_index]

    def lyric_scores(self, track_index: int) -> np.ndarray:
        if self.lyrics is None:
            raise ValueError("no lyric vectors were provided")
        scores = self.lyrics @ self.lyrics[track_index]
        # A candidate without a transcript cannot be judged lyrically.
        scores[~self.has_lyrics] = -np.inf
        return scores

    def score(self, track_index: int, *, modality: str = "audio",
              alpha: float = 0.5) -> np.ndarray:
        """Score every track against a query track. See `blend` for what alpha does."""
        if modality == "audio":
            return self.audio_scores(track_index)
        if modality == "lyrics":
            return self.lyric_scores(track_index)
        if modality != "fused":
            raise ValueError(f"unknown modality {modality!r}")

        # No lyrics on the query means nothing to match lyrically.
        query_has_lyrics = self.lyrics is not None and self.has_lyrics[track_index]
        return blend(self.audio_scores(track_index),
                     self.lyric_scores(track_index) if query_has_lyrics else None,
                     alpha, normaliser=self.normaliser)

    # --- recommendation --------------------------------------------------------------

    def recommend(self, track_index: int, *, modality: str = "audio", top_k: int = 10,
                  alpha: float = 0.5) -> list[Recommendation]:
        """Top-k recommendations for a track, excluding the track itself."""
        scores = np.array(self.score(track_index, modality=modality, alpha=alpha), dtype=float)
        scores[track_index] = -np.inf

        # Display uses min-max whatever the ranking uses: explain() bands into
        # low/moderate/high and needs a bounded range.
        audio_n = _minmax(self.audio_scores(track_index))
        lyric_n = None
        if self.lyrics is not None and self.has_lyrics[track_index]:
            raw = self.lyric_scores(track_index)
            usable = np.isfinite(raw)
            lyric_n = np.full(len(self), np.nan)
            if usable.any():
                lyric_n[usable] = _minmax(raw[usable])

        order = [j for j in np.argsort(-scores) if np.isfinite(scores[j])][:top_k]
        return [Recommendation(
            index=int(j), rank=r + 1, score=float(scores[j]),
            audio_score=float(audio_n[j]),
            lyric_score=(None if lyric_n is None or not np.isfinite(lyric_n[j])
                         else float(lyric_n[j])),
        ) for r, j in enumerate(order)]

    def recommend_from_text(self, query_vector: np.ndarray, *, top_k: int = 10
                            ) -> list[Recommendation]:
        """Rank by a text query embedded in the audio space.

        On the same object as track-to-track so both share one index and one cache.
        """
        scores = self.audio @ np.asarray(query_vector, dtype=np.float32)
        normalised = _minmax(scores)
        order = np.argsort(-scores)[:top_k]
        return [Recommendation(index=int(j), rank=r + 1, score=float(scores[j]),
                               audio_score=float(normalised[j]), lyric_score=None)
                for r, j in enumerate(order)]

    def score_matrix(self, *, modality: str = "audio", alpha: float = 0.5) -> np.ndarray:
        """Every track against every track. What the evaluation consumes."""
        return np.stack([self.score(i, modality=modality, alpha=alpha) for i in range(len(self))])
