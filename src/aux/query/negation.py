"""Splitting a query into what it wants and what it excludes.

Eval 1 found negation broken in both directions. "solo piano" returns classical five times
over; "solo piano, no vocals" returns hip-hop and v-pop, sharing **none** of its top 5 with
the un-negated query. The opposite also happens: "acoustic guitar and soft vocals" and
"acoustic guitar, no vocals" return *identical* results.

The cause is structural, not a tuning problem. A contrastively trained encoder has no
negation operator — it learned to place text near matching audio, and "no vocals" is a
string containing "vocals", so it lands near vocal music. No amount of prompt phrasing fixes
that inside a single embedding.

The fix is to stop asking the encoder to represent negation: split the query, embed the
parts separately, and combine at the *score* level where subtraction actually means
something.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Ordered longest-first so "but not" wins over "not", and "without" is not split by "with".
NEGATION_MARKERS = (
    "but not",
    "but no",
    "without",
    "excluding",
    "no ",
    "not ",
)

_SPLIT_AFTER = re.compile(r"\s*(?:,|;|\band\b|\bbut\b)\s*")


@dataclass(frozen=True, slots=True)
class ParsedQuery:
    positive: str
    negatives: tuple[str, ...]

    @property
    def has_negation(self) -> bool:
        return bool(self.negatives)


def _clean(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text.strip(" ,;-")


def parse(query: str) -> ParsedQuery:
    """Split a query into its positive part and any negated phrases.

    Deliberately a small set of surface patterns rather than a parser. It is the cheap rung
    that an LLM rewrite has to beat before the LLM is worth its cost and its network
    dependency — if this handles the negations users actually write, the LLM is not earning
    its place.

    The negated phrase runs from the marker to the next clause boundary, so
    "jazz, no vocals and no drums" yields two negatives rather than one long one.
    """
    lowered = query.lower()
    cuts: list[tuple[int, int]] = []
    for marker in NEGATION_MARKERS:
        start = 0
        while (idx := lowered.find(marker, start)) != -1:
            # Require a word boundary before the marker so "piano" does not match "no ".
            if idx == 0 or not lowered[idx - 1].isalnum():
                cuts.append((idx, idx + len(marker)))
            start = idx + len(marker)
    if not cuts:
        return ParsedQuery(_clean(query), ())

    # Earliest first, and at the same position the *longest* marker wins: "but not" and
    # "but no" both start at "but", and taking the shorter one leaves a stray "t".
    cuts.sort(key=lambda c: (c[0], -c[1]))
    # Keep only non-overlapping markers, earliest first.
    kept: list[tuple[int, int]] = []
    for a, b in cuts:
        if not kept or a >= kept[-1][1]:
            kept.append((a, b))

    positive = _clean(query[: kept[0][0]])
    negatives: list[str] = []
    for i, (_, end) in enumerate(kept):
        stop = kept[i + 1][0] if i + 1 < len(kept) else len(query)
        tail = query[end:stop]
        # A negated phrase ends at the next clause boundary, not at the end of the query.
        piece = _SPLIT_AFTER.split(tail)[0]
        piece = _clean(piece)
        if piece:
            negatives.append(piece)

    # "no vocals" alone, with nothing positive, still needs something to retrieve against.
    if not positive:
        positive = "music"
    return ParsedQuery(positive, tuple(negatives))
