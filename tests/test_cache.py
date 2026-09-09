"""Embedding cache tests."""

from __future__ import annotations

import numpy as np
import pytest

from aux.index import EmbeddingCache


def test_roundtrip(tmp_path):
    c = EmbeddingCache(tmp_path / "e.npz")
    k = c.key("abc123", "muq/large", 5)
    c.put(k, np.ones(4, np.float32), np.ones((5, 4), np.float32))
    c.save()

    reopened = EmbeddingCache(tmp_path / "e.npz")
    vec, segs = reopened.get(k)
    assert np.array_equal(vec, np.ones(4, np.float32))
    assert segs.shape == (5, 4)


def test_miss_returns_none_and_counts(tmp_path):
    c = EmbeddingCache(tmp_path / "e.npz")
    assert c.get(c.key("nope", "m", 5)) is None
    assert c.misses == 1 and c.hits == 0


@pytest.mark.parametrize(("a", "b"), [
    (("h1", "muq", 5), ("h2", "muq", 5)),          # different audio
    (("h1", "muq", 5), ("h1", "clap", 5)),         # different encoder
    (("h1", "muq", 5), ("h1", "muq", 1)),          # different pooling depth
])
def test_every_key_component_separates_entries(tmp_path, a, b):
    """A cache ignoring any of these would silently mix incompatible vectors."""
    c = EmbeddingCache(tmp_path / "e.npz")
    assert c.key(*a) != c.key(*b)


def test_model_ids_containing_slashes_are_safe(tmp_path):
    """Model ids look like "OpenMuQ/MuQ-MuLan-large" and must not collide with prefixes."""
    c = EmbeddingCache(tmp_path / "e.npz")
    k = c.key("h1", "OpenMuQ/MuQ-MuLan-large", 5)
    c.put(k, np.ones(3, np.float32), np.zeros((1, 3), np.float32))
    c.save()
    assert EmbeddingCache(tmp_path / "e.npz").get(k) is not None


def test_saving_twice_preserves_everything(tmp_path):
    c = EmbeddingCache(tmp_path / "e.npz")
    c.put(c.key("h1", "m", 5), np.ones(2, np.float32), np.ones((1, 2), np.float32))
    c.save()
    c2 = EmbeddingCache(tmp_path / "e.npz")
    c2.put(c2.key("h2", "m", 5), np.zeros(2, np.float32), np.zeros((1, 2), np.float32))
    c2.save()
    assert len(EmbeddingCache(tmp_path / "e.npz")) == 2


def test_an_absent_cache_file_is_simply_empty(tmp_path):
    assert len(EmbeddingCache(tmp_path / "missing.npz")) == 0
