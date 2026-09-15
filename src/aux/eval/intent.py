"""Does the retrieved audio have the properties the query asked for?

A query naming "quiet" or "fast" names something measurable in the waveform, which the
encoder never sees. So: when a query says quiet, are the results actually quieter than the
library average? No human or opposed pair needed.

Small vocabulary on purpose. Every term here has an unambiguous physical direction. "Warm",
"dreamy" and "nostalgic" are left out because no waveform feature honestly stands in for
them, and a made-up mapping would produce numbers that look like evidence.
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

    `retrieved_features` are z-scores against the library. Returns the mean z-score in each
    named term's expected direction: positive means the audio moved the way the words asked,
    negative the wrong way.

    None when the text names nothing measurable. "No opinion" and "no effect" are different.
    """
    found = terms_in(text)
    if not found:
        return None
    index = {name: i for i, name in enumerate(feature_names)}
    scores = [direction * float(retrieved_features[:, index[feature]].mean())
              for _, feature, direction in found if feature in index]
    return float(np.mean(scores)) if scores else None
