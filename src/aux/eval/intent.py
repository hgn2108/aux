"""Does the retrieved audio have the properties the query asked for?

The closest objective proxy this project has to "did the system deliver what was asked",
and the one that needs neither an opposed pair nor a human.

A query — or a planner's rewrite of it — names properties: "quiet", "fast", "sparse". Those
map to measurable features of the waveform, which the encoder never sees. So the check is
direct: when a query says *quiet*, are the retrieved tracks actually quieter than the
library's average?

Deliberately a small, one-sided vocabulary. Every term here has an unambiguous physical
direction. Words like "warm", "dreamy" or "nostalgic" are excluded, not because they do not
matter but because no waveform feature honestly stands in for them, and a fabricated mapping
would produce numbers that look like evidence.
"""

from __future__ import annotations

import numpy as np

# term -> (feature, expected direction)
# features: "loudness" (RMS), "onset" (rhythmic density), "brightness" (zero-crossing rate)
INTENT_TERMS: dict[str, tuple[str, int]] = {
    # loudness
    "loud": ("loudness", +1), "quiet": ("loudness", -1),
    "soft": ("loudness", -1), "subdued": ("loudness", -1),
    "gentle": ("loudness", -1), "intense": ("loudness", +1),
    "powerful": ("loudness", +1), "hushed": ("loudness", -1),
    # rhythmic density and tempo
    "fast": ("onset", +1), "slow": ("onset", -1),
    "driving": ("onset", +1), "energetic": ("onset", +1),
    "uptempo": ("onset", +1), "upbeat": ("onset", +1),
    "relaxed": ("onset", -1), "mellow": ("onset", -1),
    "calm": ("onset", -1), "sparse": ("onset", -1),
    "minimal": ("onset", -1), "dense": ("onset", +1),
    "busy": ("onset", +1), "percussive": ("onset", +1),
    "rhythmic": ("onset", +1), "steady": ("onset", -1),
    "unhurried": ("onset", -1), "propulsive": ("onset", +1),
    # brightness
    "bright": ("brightness", +1), "dark": ("brightness", -1),
    "crisp": ("brightness", +1), "warm": ("brightness", -1),
    "muffled": ("brightness", -1), "shimmering": ("brightness", +1),
}

FEATURES = ("loudness", "onset", "brightness")


def terms_in(text: str) -> list[tuple[str, str, int]]:
    """Find intent terms in a string. Returns `(term, feature, direction)`."""
    lowered = f" {text.lower()} "
    found = []
    for term, (feature, direction) in INTENT_TERMS.items():
        if f" {term} " in lowered or f" {term}," in lowered or f" {term}." in lowered:
            found.append((term, feature, direction))
    return found


def intent_match(text: str, retrieved_features: np.ndarray,
                 feature_names: tuple[str, ...] = FEATURES) -> float | None:
    """How well retrieved audio matches the properties named in `text`.

    `retrieved_features` are z-scores against the library, one row per retrieved track.
    Returns the mean z-score in each named term's expected direction, so:

    - **positive** means the audio moved the way the words asked;
    - **zero** means the words had no effect;
    - **negative** means it moved the wrong way.

    Returns None when the text names no measurable property — reported as such rather than
    scored as zero, because "no opinion" and "no effect" are different.
    """
    found = terms_in(text)
    if not found:
        return None
    index = {name: i for i, name in enumerate(feature_names)}
    scores = [direction * float(retrieved_features[:, index[feature]].mean())
              for _, feature, direction in found if feature in index]
    return float(np.mean(scores)) if scores else None
