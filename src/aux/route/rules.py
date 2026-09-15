"""The free baseline: decide from the words in the query.

Deliberately the first thing built. The task may simply be easy, "songs about X" is a
strong, almost unambiguous signal, and if a rule captures most of the oracle's +0.042, then
an LLM call per query buys latency and cost for nothing. Every other router in this package
has to beat this one to justify itself.

The acoustic vocabulary is borrowed from `aux.eval.intent`, which already lists terms with an
unambiguous physical direction, plus the genre and instrument words a listener actually
types. The semantic side keys on the constructions people use to ask about *meaning* rather
than on a list of topics, since topics are open-ended and phrasings are not.
"""

from __future__ import annotations

import re

from ..eval.intent import INTENT_TERMS
from .base import Route, Router

#: Phrases that ask what a song is *about*. Matched as whole phrases, so "about" alone  -
#: which appears in plenty of acoustic queries, is not enough on its own.
SEMANTIC_MARKERS = (
    "songs about", "song about", "tracks about", "music about", "lyrics about",
    "about being", "about a", "about the", "about someone", "about how",
    "talks about", "talking about", "deals with", "themes of", "lyrically",
    "story about", "stories about", "written about",
)

#: Sound words beyond the physical-direction vocabulary: genres, instruments, production.
ACOUSTIC_TERMS = frozenset({
    "jazz", "classical", "piano", "guitar", "drums", "bass", "808s", "808",
    "synth", "strings", "vocals", "instrumental", "acoustic", "electronic", "edm",
    "dnb", "breakbeat", "breakbeats", "techno", "house", "trap", "hip", "hop",
    "rnb", "r&b", "pop", "rock", "folk", "beat", "beats", "tempo", "bpm",
    "production", "produced", "mix", "sounds", "sounding", "sound", "groove",
    "melody", "harmony", "reverb", "distorted", "lo-fi", "sub",
})

WORD = re.compile(r"[a-z0-9&'-]+")

#: Weights for a query that names only one kind of thing. Not 1.0 and 0.0: a query is rarely
#: pure, and the measured curves are fairly flat near the ends, so conceding a little to the
#: other modality costs almost nothing and protects against a misread query.
ACOUSTIC_ALPHA = 0.9
SEMANTIC_ALPHA = 0.1
BALANCED_ALPHA = 0.5


class RuleRouter(Router):
    name = "rules"

    def route(self, query: str) -> Route:
        text = query.lower()
        words = set(WORD.findall(text))

        semantic_hits = sum(marker in text for marker in SEMANTIC_MARKERS)
        acoustic_hits = len(words & ACOUSTIC_TERMS) + len(words & set(INTENT_TERMS))

        if semantic_hits and acoustic_hits:
            return Route(BALANCED_ALPHA,
                         "asks about both how it sounds and what it is about", self.name)
        if semantic_hits:
            return Route(SEMANTIC_ALPHA, "asks what the songs are about", self.name)
        if acoustic_hits:
            return Route(ACOUSTIC_ALPHA, "describes how the music sounds", self.name)
        # Nothing recognised. Audio is the stronger channel on this corpus overall, so an
        # unreadable query defaults there rather than to an even split.
        return Route(ACOUSTIC_ALPHA, "no clear signal; matching on sound", self.name)
