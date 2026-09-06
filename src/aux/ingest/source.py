"""SourceAdapter -- the first stage of the ingestion contract.

Enumerates candidate media under a root. Deliberately dumb: it decides only what is worth
opening, never whether a file is valid. Validity is MediaProbe's job, because deciding it
here would mean opening every file twice.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from .asset import SUPPORTED_EXTENSIONS


@dataclass(frozen=True, slots=True)
class DiscoveredFile:
    path: Path
    size_bytes: int
    extension: str
    supported: bool
    """False files are reported, not silently dropped -- Eval 0A needs the denominator."""


def discover(
    root: Path,
    *,
    recursive: bool = True,
    include_unsupported: bool = False,
) -> Iterator[DiscoveredFile]:
    """Yield candidate media files under ``root`` in deterministic order.

    Sorted because Eval 0A measures deterministic repeatability: two runs over an unchanged
    directory must produce identical output, and filesystem walk order is not guaranteed.

    Hidden files and macOS ``._`` AppleDouble resource forks are skipped; the latter are
    not media but do carry media extensions.
    """
    root = Path(root)
    if not root.is_dir():
        raise NotADirectoryError(f"not a directory: {root}")

    pattern = "**/*" if recursive else "*"
    for path in sorted(root.glob(pattern)):
        if not path.is_file():
            continue
        if path.name.startswith("."):
            continue

        extension = path.suffix.lower()
        supported = extension in SUPPORTED_EXTENSIONS
        if not supported and not include_unsupported:
            continue

        yield DiscoveredFile(
            path=path,
            size_bytes=path.stat().st_size,
            extension=extension,
            supported=supported,
        )
