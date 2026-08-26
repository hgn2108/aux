"""Smoke test: the package imports and the layout is intact (roadmap 0.1.5)."""

import importlib


def test_package_imports() -> None:
    assert importlib.import_module("aux") is not None


def test_subpackages_import() -> None:
    for name in ("aux.config", "aux.ingest", "aux.models", "aux.eval"):
        assert importlib.import_module(name) is not None
