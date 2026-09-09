"""Index building with a content-hash-keyed embedding cache (DESIGN.md)."""

from .build import build_index
from .cache import EmbeddingCache

__all__ = ["EmbeddingCache", "build_index"]
