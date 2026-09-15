"""Content-hash-keyed embedding cache.

DESIGN.md specifies this and it went unbuilt until it started costing real time: every
experiment so far re-embedded its whole corpus, which made 20 minutes the price of asking a
question and pushed experiments toward being underpowered.

The cache key is `(content_hash, encoder version, n_segments)`. All three matter:

- **content hash**, not path, so a moved or renamed file is a hit, that was the stated
  reason for hashing bytes in the first place;
- **encoder version**, because a vector produced by CLAP is meaningless to MuQ-MuLan, and a
  cache that ignored this would silently mix two spaces;
- **n_segments**, because E1 established that pooling depth changes the vector.

Stored as one `.npz` per corpus. Nothing here is authoritative, deleting the cache costs
time and never correctness.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


class EmbeddingCache:
    """Persistent store of track vectors keyed by content and model identity."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._vectors: dict[str, np.ndarray] = {}
        self._segments: dict[str, np.ndarray] = {}
        self.hits = 0
        self.misses = 0
        if self.path.exists():
            with np.load(self.path, allow_pickle=False) as data:
                for key in data.files:
                    if key.startswith("v/"):
                        self._vectors[key[2:]] = data[key]
                    elif key.startswith("s/"):
                        self._segments[key[2:]] = data[key]

    @staticmethod
    def key(content_hash: str, encoder_version: str, n_segments: int) -> str:
        # Slashes appear in model ids and would collide with the v/ and s/ prefixes.
        return f"{content_hash}|{encoder_version.replace('/', '_')}|{n_segments}"

    def get(self, key: str) -> tuple[np.ndarray, np.ndarray] | None:
        if key not in self._vectors:
            self.misses += 1
            return None
        self.hits += 1
        return self._vectors[key], self._segments.get(key, np.empty((0, 0), dtype=np.float32))

    def put(self, key: str, vector: np.ndarray, segments: np.ndarray) -> None:
        self._vectors[key] = np.asarray(vector, dtype=np.float32)
        self._segments[key] = np.asarray(segments, dtype=np.float32)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {f"v/{k}": v for k, v in self._vectors.items()}
        payload.update({f"s/{k}": v for k, v in self._segments.items()})
        # Write then move, so an interrupted save cannot leave a corrupt cache behind.
        # Written through an open handle because np.savez appends ".npz" to any path that
        # lacks it, which would silently produce "cache.npz.tmp.npz" and break the rename.
        tmp = self.path.with_name(self.path.name + ".tmp")
        with tmp.open("wb") as handle:
            np.savez(handle, **payload)
        tmp.replace(self.path)

    def __len__(self) -> int:
        return len(self._vectors)
