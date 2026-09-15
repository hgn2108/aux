"""Track-to-track and text-to-track recommendation over two modalities.

The retrieval surface the rest of the project is measured through. Three modes:

- **audio**, MuQ-MuLan embeddings of the recording itself;
- **lyrics**, Qwen3 embeddings of Whisper transcripts;
- **fused**, a weighted blend, with `alpha` controlling the balance.

**Why weighted-score fusion rather than rank fusion.** Reciprocal rank fusion was measured
first and is the wrong tool here: it weights its inputs equally by construction, and E3
showed that halves performance when one modality knows nothing about the query (lyric-line
retrieval, where audio sits at chance). A weight makes the balance explicit, tunable, and
measurable, which is what an ablation needs.

**Why per-query normalisation.** Audio and lyric cosines come from different models with
different score distributions, so `0.6 * audio + 0.4 * lyric` on raw values silently weights
whichever has more spread. Each modality's scores are normalised across candidates for that
query first, so `alpha` means what it says.

**Why z-score rather than min-max.** Min-max was measured first and is worse: its range is
set by the two most extreme candidates, so a single outlier rescales every other score and
compresses the differences that matter. `scripts/diagnose_fusion.py` found z-score ahead at
every interior alpha (0.143 vs 0.106 at alpha=0.5 on the artist label, NDCG@10), so it is
the default. Min-max is kept selectable because that comparison is the evidence for the
choice, and because the displayed per-result scores still use it: a 0-1 range is meaningful
to a reader, and a standard deviation is not.

**Missing modality.** A track with no reliable transcript, instrumental, or a failed
transcription, has no lyric vector. It falls back to its audio score rather than being
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


def _zscore(scores: np.ndarray) -> np.ndarray:
    """Standardise per query row, leaving a constant row at zero.

    Unlike min-max, the scale comes from the whole distribution rather than its two extreme
    candidates, so one outlier cannot rescale everything else.
    """
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

    The single implementation of the rule. It was written out five times across the
    recommender, the evaluations and both pages of the app, which is four chances for the
    published numbers and the shipped behaviour to stop agreeing.

    `alpha` is the weight on audio: 1.0 is audio only and 0.0 is lyrics only, so a sweep
    collapses exactly onto each single-signal system at its ends. Scores are normalised per
    query first, because the two come from different models with different spreads, and
    without that `alpha` would silently favour whichever spreads wider.

    A candidate with no usable transcript keeps its audio score rather than being scored
    zero, which would push every instrumental to the bottom and quietly turn the blend into
    a vocal-music filter. Passing `lyric_scores=None` does the same for the whole row, which
    is the case when the *query* has no lyrics to match against.
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
    """None when this track has no usable transcript, surfaced rather than faked, so the
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

        # The query itself having no lyrics means there is nothing lyrical to match
        # against, which `blend` treats the same way as a row with no usable candidates.
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

        # Displayed scores are always min-max, whatever the ranking uses: `explain()`
        # bands them into low/moderate/high, which needs a bounded 0-1 range.
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
        one cache and one set of results, they are two entry points to the same system, not
        two systems.
        """
        scores = self.audio @ np.asarray(query_vector, dtype=np.float32)
        normalised = _minmax(scores)
        order = np.argsort(-scores)[:top_k]
        return [Recommendation(index=int(j), rank=r + 1, score=float(scores[j]),
                               audio_score=float(normalised[j]), lyric_score=None)
                for r, j in enumerate(order)]

    def score_matrix(self, *, modality: str = "audio", alpha: float = 0.5) -> np.ndarray:
        """Every track against every track, what the evaluation consumes."""
        return np.stack([self.score(i, modality=modality, alpha=alpha) for i in range(len(self))])
