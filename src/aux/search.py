"""The search entry point, everything Slice 0 to 2 established, in one place.

Every component here earned its place by measurement, and the ones that did not are absent.
What survived:

- **negation split** (DEC-014), always on. A contrastive encoder cannot represent "not X",
  so the query is split and the exclusion applied at score level. Largest single measured
  gain in the project: a rated query moved 1.00 to 5.00.
- **five-segment pooling** (DEC-011), in the index. Beat single-segment on two independent
  encoders.
- **MuQ-MuLan** (DEC-012). Beat CLAP at every pooling depth, paired p = 4.5e-08.

What was measured and rejected: a fixed context lexicon (DEC-016), five reranking methods
(DEC-015), rank fusion (DEC-017), and the LLM planner as a default (DEC-018).

**The planner is off by default** and available behind `use_planner`. It helps vague queries
(+1.07 on ones the baseline handled badly) and *harms* specific ones (-0.58), so applied
indiscriminately it nets to nothing, measured at +0.05, p = 1.000. It also costs ~1.7 s and
an internet connection, in a project whose first principle is local-first. DEC-021 records
the rule for when it is worth enabling; that rule is not yet validated on held-out queries.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .encode.base import l2_normalise
from .index import build_index
from .query import score_query
from .query.retrieve import score_plan


@dataclass(frozen=True, slots=True)
class Result:
    path: Path
    score: float
    rank: int


@dataclass(frozen=True, slots=True)
class SearchResponse:
    query: str
    results: list[Result]
    used_planner: bool
    rewritten: str | None
    top_score: float
    z_top: float
    low_coverage: bool
    """True when even the best match is a poor one, the library probably cannot answer.

    Phase B validated the *signal*: raw top score detected a removed genre in 20 of 22
    cases, where every distribution-shape measure sat at or below chance. The **threshold**
    is not validated, so this is surfaced as advisory rather than used to suppress results.
    """


class Library:
    """An indexed music library, ready to search."""

    def __init__(self, root: Path, encoder, *, n_segments: int = 5,
                 cache_path: Path | None = None, limit: int | None = None) -> None:
        self.root = Path(root)
        self.encoder = encoder
        self.n_segments = n_segments
        self.vectors, self.paths, _ = build_index(
            self.root, encoder, n_segments=n_segments,
            cache_path=cache_path, limit=limit, progress_every=0)

    def __len__(self) -> int:
        return len(self.paths)

    def search(self, query: str, *, k: int = 10, use_planner: bool = False,
               planner=None, low_coverage_score: float = 0.25) -> SearchResponse:
        """Search the library.

        `use_planner` defaults to False: see the module docstring. When enabled it rewrites
        the query before retrieval, and a planner failure degrades to exactly this path
        rather than to anything unknown.
        """
        rewritten = None
        if use_planner and planner is not None:
            plan, meta = planner.plan(query)
            if not meta.get("fallback"):
                rewritten = plan.rewritten
                scores = score_plan(self.encoder, plan, self.vectors)
            else:
                scores, _, _ = score_query(self.encoder, query, self.vectors)
                use_planner = False
        else:
            scores, _, _ = score_query(self.encoder, query, self.vectors)
            use_planner = False

        order = np.argsort(-scores)[:k]
        sd = float(scores.std())
        return SearchResponse(
            query=query,
            results=[Result(self.paths[j], float(scores[j]), i + 1)
                     for i, j in enumerate(order)],
            used_planner=use_planner,
            rewritten=rewritten,
            top_score=float(scores.max()),
            z_top=float((scores.max() - scores.mean()) / sd) if sd > 0 else 0.0,
            low_coverage=float(scores.max()) < low_coverage_score,
        )

    def similar_to(self, path: Path, *, k: int = 10) -> SearchResponse:
        """Reference-track search: find neighbours of a track already in the library."""
        try:
            i = self.paths.index(Path(path))
        except ValueError as exc:
            raise KeyError(f"not in this library: {path}") from exc
        scores = self.vectors @ self.vectors[i]
        # Drop the reference explicitly rather than relying on a sentinel score: with -inf
        # and k larger than the library, the slice still includes it at the tail.
        order = [j for j in np.argsort(-scores) if j != i][:k]
        others = np.delete(scores, i)
        sd = float(others.std())
        return SearchResponse(
            query=str(path), used_planner=False, rewritten=None,
            results=[Result(self.paths[j], float(scores[j]), n + 1)
                     for n, j in enumerate(order)],
            top_score=float(others.max()),
            z_top=float((others.max() - others.mean()) / sd) if sd > 0 else 0.0,
            low_coverage=False,
        )
