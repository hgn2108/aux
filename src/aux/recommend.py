"""Track-to-track and text-to-track recommendation over two modalities.

The retrieval surface the rest of the project is measured through. Three modes:

- **audio** — MuQ-MuLan embeddings of the recording itself;
- **lyrics** — Qwen3 embeddings of Whisper transcripts;
- **fused** — a weighted blend, with `alpha` controlling the balance.

**Why weighted-score fusion rather than rank fusion.** Reciprocal rank fusion was measured
first and is the wrong tool here: it weights its inputs equally by construction, and E3
showed that halves performance when one modality knows nothing about the query (lyric-line
retrieval, where audio sits at chance). A weight makes the balance explicit, tunable, and
measurable — which is what an ablation needs.

**Why per-query normalisation.** Audio and lyric cosines come from different models with
different score distributions, so `0.6 * audio + 0.4 * lyric` on raw values silently weights
whichever has more spread. Each modality's scores are min-max normalised across candidates
for that query first, so `alpha` means what it says.

**Missing modality.** A track with no reliable transcript — instrumental, or a failed
transcription — has no lyric vector. It falls back to its audio score rather than being
scored zero, since zero would push every instrumental track to the bottom of every fused
ranking and quietly turn fusion into a vocal-music filter.
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


@dataclass(frozen=True, slots=True)
class Recommendation:
    index: int
    rank: int
    score: float
    audio_score: float
    lyric_score: float | None
    """None when this track has no usable transcript — surfaced rather than faked, so the
    UI can say "no lyrics" instead of showing a fabricated zero."""

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
                 has_lyrics: np.ndarray | None = None, paths: list[Path] | None = None) -> None:
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
        """Score every track against a query track.

        `alpha` is the audio weight: 1.0 is audio-only, 0.0 is lyrics-only, so the fused
        mode collapses exactly onto each unimodal system at the ends of the sweep. That is
        what makes an alpha ablation interpretable rather than three unrelated systems.
        """
        if modality == "audio":
            return self.audio_scores(track_index)
        if modality == "lyrics":
            return self.lyric_scores(track_index)
        if modality != "fused":
            raise ValueError(f"unknown modality {modality!r}")

        audio = _minmax(self.audio_scores(track_index))
        if self.lyrics is None or not self.has_lyrics[track_index]:
            # The *query* has no lyrics, so there is nothing lyrical to match against.
            return audio

        raw = self.lyric_scores(track_index)
        usable = np.isfinite(raw)
        lyric = np.zeros_like(audio)
        if usable.any():
            lyric[usable] = _minmax(raw[usable])

        fused = alpha * audio + (1 - alpha) * lyric
        # Candidates without a transcript keep their audio score rather than being zeroed,
        # which would make fusion a vocal-music filter.
        fused[~usable] = audio[~usable]
        return fused

    # --- recommendation --------------------------------------------------------------

    def recommend(self, track_index: int, *, modality: str = "audio", top_k: int = 10,
                  alpha: float = 0.5) -> list[Recommendation]:
        """Top-k recommendations for a track, excluding the track itself."""
        scores = np.array(self.score(track_index, modality=modality, alpha=alpha), dtype=float)
        scores[track_index] = -np.inf

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
        """Text-to-track search: rank by a query embedding in the audio space.

        Kept on the same object as track-to-track so both retrieval modes share one index,
        one cache and one set of results — they are two entry points to the same system, not
        two systems.
        """
        scores = self.audio @ np.asarray(query_vector, dtype=np.float32)
        normalised = _minmax(scores)
        order = np.argsort(-scores)[:top_k]
        return [Recommendation(index=int(j), rank=r + 1, score=float(scores[j]),
                               audio_score=float(normalised[j]), lyric_score=None)
                for r, j in enumerate(order)]

    def score_matrix(self, *, modality: str = "audio", alpha: float = 0.5) -> np.ndarray:
        """Every track against every track — what the evaluation consumes."""
        return np.stack([self.score(i, modality=modality, alpha=alpha) for i in range(len(self))])
