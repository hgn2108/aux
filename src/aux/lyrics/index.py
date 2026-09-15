"""Turning a corpus of transcripts into a matrix of lyric vectors.

Lived in `scripts/eval_recommendation.py` and was imported from there by four other
scripts, each first inserting the scripts directory onto `sys.path`. That works until two
of them disagree about what a reliable transcript is, at which point two published numbers
quietly stop being comparable. It belongs in the library.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..ingest.asset import content_hash

DEFAULT_CACHE = Path(".cache")


def transcripts_by_hash(cache: Path, whisper_model: str) -> dict[str, dict] | None:
    """Reliable transcripts only, keyed by the content hash of the file they came from.

    Keying on content rather than path means a track that moves or is renamed keeps its
    transcript, and two copies of one recording share it.
    """
    path = Path(cache) / f"transcripts_{whisper_model}.json"
    if not path.exists():
        return None
    records = json.loads(path.read_text())
    return {key.split("|")[0]: value for key, value in records.items()}


def load_lyric_vectors(tracks, whisper_model: str = "small", cache_tag: str = "corpus",
                       cache: Path = DEFAULT_CACHE) -> tuple[np.ndarray | None,
                                                             np.ndarray | None]:
    """Embed each track's transcript, caching the resulting matrix.

    Returns `(vectors, has_lyrics)`, or `(None, None)` when no track in the corpus has a
    usable transcript. A track without one gets a zero vector and a False flag; every
    caller reads the flag and none reads the zero, so an instrumental is never ranked as
    though its lyrics matched nothing.
    """
    by_hash = transcripts_by_hash(cache, whisper_model)
    if by_hash is None:
        return None, None

    texts: list[str] = []
    usable: list[bool] = []
    for track in tracks:
        record = by_hash.get(content_hash(track.path))
        if record and record.get("reliable"):
            texts.append(record["text"])
            usable.append(True)
        else:
            texts.append("")
            usable.append(False)

    has_lyrics = np.array(usable, dtype=bool)
    if not has_lyrics.any():
        return None, None

    vector_path = Path(cache) / f"lyrics_{cache_tag}.npy"
    if vector_path.exists():
        vectors = np.load(vector_path)
        if vectors.shape[0] == len(texts):
            return vectors, has_lyrics

    from .embed import LyricEmbedder

    vectors = LyricEmbedder().embed_documents(texts)
    vectors[~has_lyrics] = 0.0
    np.save(vector_path, vectors)
    return vectors, has_lyrics
