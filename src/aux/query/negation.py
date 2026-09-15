"""Splitting a query into what it wants and what it excludes.

A contrastively trained encoder has no negation operator. It learned to put text near
matching audio, and "no vocals" is a string containing "vocals", so it lands near vocal
music. Measured: "solo piano, no vocals" returned hip-hop and v-pop, sharing nothing with
the un-negated query; "acoustic guitar, no vocals" returned results identical to the
un-negated one. Prompt phrasing cannot fix that inside a single embedding.

So the query is split here and the parts are scored separately. See `retrieve`.
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

    Surface patterns rather than a parser: this is the cheap rung an LLM rewrite has to
    beat before it earns its latency.

    A negated phrase runs to the next clause boundary, so "jazz, no vocals and no drums"
    gives two negatives rather than one long one.
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

    # Earliest first, and at the same position the longest marker wins: "but not" and
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
