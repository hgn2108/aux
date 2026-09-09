"""Index a directory of media into track vectors, reusing cached embeddings."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from ..ingest import IngestError, decode, discover
from .cache import EmbeddingCache


def build_index(
    root: Path,
    encoder,
    *,
    n_segments: int = 5,
    cache_path: Path | None = None,
    limit: int | None = None,
    want_segments: bool = False,
    progress_every: int = 200,
) -> tuple[np.ndarray, list[Path], list[np.ndarray]]:
    """Return `(track_vectors, paths, segment_vectors)` for everything under `root`.

    Decoding still happens on a cache hit only when the hash is unknown — the hash is read
    from the file's bytes, which is far cheaper than decoding and encoding it.
    """
    root = Path(root)
    cache = EmbeddingCache(cache_path) if cache_path else None
    paths = [f.path for f in discover(root)]
    if limit:
        paths = paths[:limit]

    from ..ingest.asset import content_hash

    vectors, kept, segments = [], [], []
    for i, path in enumerate(paths, 1):
        key = None
        if cache is not None:
            key = cache.key(content_hash(path), encoder.version, n_segments)
            if (found := cache.get(key)) is not None:
                vec, segs = found
                vectors.append(vec)
                kept.append(path)
                segments.append(segs if want_segments else np.empty((0, 0), np.float32))
                continue
        try:
            vec, segs = encoder.embed_track(decode(path), n_segments=n_segments)
        except (IngestError, Exception):  # noqa: BLE001
            continue
        if cache is not None and key is not None:
            cache.put(key, vec, segs)
        vectors.append(vec)
        kept.append(path)
        segments.append(segs if want_segments else np.empty((0, 0), np.float32))
        if progress_every and i % progress_every == 0:
            print(f"  indexed {i}/{len(paths)}", file=sys.stderr)

    if cache is not None:
        cache.save()
        print(f"  cache: {cache.hits} hits, {cache.misses} misses, {len(cache)} stored",
              file=sys.stderr)
    return np.stack(vectors), kept, segments
