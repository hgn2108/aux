"""Context→acoustic translation by fixed lexicon — DEC-013's rung 1.

Irene's own queries are context-led: 7 of 12 name an activity or setting. Slice 1 found
context the weakest category (3.14), and specifically that context *without* a genre anchor
fails badly — "background music while reading" scored 1.6, "warming up before going out"
1.6. Meanwhile concrete instrumentation language works well ("solo piano" z = 4.97).

The gap is between the language people use and the language the encoder was trained on:
descriptive captions of *sound*. This module closes it the cheapest way available, by
rewriting situations into sounds.

**Why a hand-written table when an LLM would write a better one.** Because it costs nothing,
runs offline, is deterministic, and captures the genuinely universal half of the mapping —
running is fast, sleeping is slow. DEC-013 requires the LLM to beat this before its cost and
network dependency are justified, and a measured "the table was enough" is a real result.

**What it cannot do.** The other half of context is personal: "for studying" means lo-fi to
one listener and solo piano to another. No table and no LLM knows which. That is Slice 5's
intent profile, learned from behaviour, and this module is deliberately not an attempt at it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

CONTEXT_LEXICON: dict[str, str] = {
    # Activity — the dominant form in Irene's queries.
    "running": "fast tempo, driving percussion, high energy, steady beat",
    "workout": "fast tempo, driving percussion, high energy, aggressive",
    "gym": "fast tempo, heavy bass, high energy, aggressive",
    "studying": "steady quiet groove, minimal vocals, unobtrusive, mellow",
    "study": "steady quiet groove, minimal vocals, unobtrusive, mellow",
    "reading": "quiet, sparse instrumentation, no prominent vocals, calm",
    "focus": "repetitive, understated, few dynamic changes, calm",
    "sleeping": "very slow, quiet, soft texture, sparse, gentle",
    "falling asleep": "very slow, quiet, soft texture, sparse, gentle",
    "relax": "slow tempo, soft, warm, gentle, unhurried",
    "chilling": "slow tempo, soft, warm, laid back groove",
    "driving": "steady mid tempo, spacious production, sustained groove",
    "drive": "steady mid tempo, spacious production, sustained groove",
    "walk": "steady mid tempo, rhythmic, moderate energy",
    "cooking": "warm, upbeat, mid tempo, bright",
    "cleaning": "upbeat, energetic, rhythmic, bright",
    "party": "loud, fast, heavy bass, energetic, danceable",
    "club": "loud, heavy bass, danceable, four on the floor",
    "pre-game": "loud, energetic, heavy bass, hype",
    "getting ready": "upbeat, bright, energetic, catchy",
    "get ready": "upbeat, bright, energetic, catchy",
    "going out": "upbeat, energetic, danceable, bright",
    "dancing": "danceable, strong rhythm, steady beat, energetic",

    # Setting and time — weaker mappings, kept short on purpose.
    "late night": "slow, dark, spacious, subdued, intimate",
    "night drive": "slow to mid tempo, dark, spacious, sustained",
    "morning": "bright, gentle, warm, unhurried",
    "rainy": "slow, soft, melancholy, gentle",
    "summer": "bright, warm, upbeat, breezy",
    "winter": "sparse, cold, subdued, quiet",
    "background": "unobtrusive, quiet, few dynamic changes, no prominent vocals",
}
"""Situation to sound.

Terms are the ones that actually appeared in Irene's queries and in the Slice 1 set, plus
their near neighbours. Deliberately small: an unused entry cannot be validated, and a table
that guesses at situations nobody asks for is untested surface area.

Values name **acoustic properties** — tempo, density, dynamics, texture — not genres.
Mapping "studying" to "lo-fi hip hop" would bake one listener's taste into the system, which
is the personal half this module explicitly does not attempt.
"""

# Longest first, so "falling asleep" wins over "sleeping" and "night drive" over "drive".
_TERMS = sorted(CONTEXT_LEXICON, key=len, reverse=True)


@dataclass(frozen=True, slots=True)
class Expansion:
    original: str
    matched: tuple[str, ...]
    acoustic: str

    @property
    def expanded(self) -> bool:
        return bool(self.matched)


def expand(query: str) -> Expansion:
    """Find context terms in a query and return their acoustic translation.

    The original query is never discarded — the genre and mood words in it are doing real
    work, as Slice 1 showed when genre-anchored context queries (4.18) far outscored
    context-only ones. The expansion is additional evidence, combined at retrieval time.
    """
    lowered = query.lower()
    matched, parts, consumed = [], [], []
    for term in _TERMS:
        # Word-boundary match so "drive" does not fire inside "driven".
        if not re.search(rf"(?<![a-z]){re.escape(term)}(?![a-z])", lowered):
            continue
        # Skip a term already covered by a longer one ("drive" inside "night drive").
        if any(term in longer for longer in consumed):
            continue
        consumed.append(term)
        matched.append(term)
        parts.append(CONTEXT_LEXICON[term])
    return Expansion(query, tuple(matched), ", ".join(parts))
