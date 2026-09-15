"""Example queries for the demo, and their precomputed embeddings.

A cold container takes minutes to load the encoder. These queries are fixed, so their
embeddings are computed at export time and clicking one costs a dot product instead.
Typing something new still loads the model.

They live here, not in app.py, so the exporter and the app read the same list. If those
drifted, a stored vector would answer the wrong question.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

DEMO = Path(__file__).resolve().parents[3] / "demo"
EXAMPLES_PATH = DEMO / "example_queries.npz"

#: Descriptions of how music sounds. Offered on every corpus.
SOUND_EXAMPLES = (
    "fast aggressive drums with distorted guitars",
    "sparse piano, quiet and unhurried",
    "warm analogue soul with live instruments",
)

#: Questions about what songs are about. Offered only where lyrics exist, since a corpus
#: without them would demonstrate the limitation rather than the system.
LYRIC_EXAMPLES = (
    "songs about missing someone",
    "songs about money and ambition",
)

ALL_EXAMPLES = SOUND_EXAMPLES + LYRIC_EXAMPLES


def load_example_vectors() -> dict[str, dict[str, np.ndarray]] | None:
    """Precomputed embeddings per example query, or None if none were exported.

    Returns `{query: {"audio": vector, "lyric": vector or absent}}`. The lyric vector is
    absent when the export ran without a lyric encoder available.
    """
    if not EXAMPLES_PATH.exists():
        return None
    blob = np.load(EXAMPLES_PATH, allow_pickle=False)
    queries = json.loads(str(blob["queries"]))
    out: dict[str, dict[str, np.ndarray]] = {}
    for i, query in enumerate(queries):
        entry = {"audio": blob["audio"][i]}
        if "lyric" in blob.files:
            entry["lyric"] = blob["lyric"][i]
        out[query] = entry
    return out
