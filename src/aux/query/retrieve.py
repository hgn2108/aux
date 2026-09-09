"""Retrieval that handles negation at the score level.

A single embedding cannot represent "X but not Y" — see `negation`. So the query is split,
the parts are embedded separately, and the exclusion is applied where subtraction is
meaningful:

    score(track) = cos(positive, track) - lambda * max_j cos(negative_j, track)

`max` rather than a sum over negatives: two exclusions should not penalise a track twice as
hard as one, and a track needs to be pushed down if it matches *any* excluded concept.

`lambda` is the one free parameter. Its value is chosen by measurement in
`scripts/eval_negation.py`, against ground truth that costs no human rating: genre folders
label what should have been excluded.
"""

from __future__ import annotations

import numpy as np

from ..encode.base import l2_normalise
from .negation import ParsedQuery, parse

DEFAULT_NEGATION_WEIGHT = 0.5
"""Chosen by sweep in `scripts/eval_negation.py` (2026-09-09).

Selected on the *margin* between satisfying the request and honouring the exclusion, not on
leakage alone. Leakage alone is trivially minimised by returning things nobody asked for:
at weight 1.5 leakage reaches 0.00 while the share of correctly-targeted results falls from
0.90 to 0.75, and at 3.0 to 0.50.

0.5, 0.75 and 1.0 all tie on margin (0.82) across the 12 sweep cases, trading leakage
against on-target rate. 0.5 is the conservative end of that tie: it disturbs a
non-negated query's results least, which matters because the parser fires on surface
patterns and a false positive should stay cheap.
"""


def score_query(
    encoder,
    query: str,
    track_vectors: np.ndarray,
    *,
    negation_weight: float = DEFAULT_NEGATION_WEIGHT,
) -> tuple[np.ndarray, ParsedQuery]:
    """Score every track for one query, subtracting any negated concepts."""
    parsed = parse(query)
    positive = l2_normalise(encoder.embed_text([parsed.positive]))[0]
    scores = track_vectors @ positive

    if parsed.negatives and negation_weight:
        negatives = l2_normalise(encoder.embed_text(list(parsed.negatives)))
        penalty = (track_vectors @ negatives.T).max(axis=1)
        scores = scores - negation_weight * penalty

    return scores, parsed
