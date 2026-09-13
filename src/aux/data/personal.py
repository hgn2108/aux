"""Labels for the personal library, the only corpus here with a real lyric channel.

FMA is the right corpus for scale, but it cannot support a multimodal evaluation: 56% of a
sampled 75 clips were instrumental and the median transcript ran 11 words. Creative Commons
catalogues skew heavily instrumental, so the lyric modality has almost nothing to encode.
This library is the opposite — 160 commercially released tracks, 79% with a reliable
transcript at a median of 376 words — and is therefore used for the audio-vs-lyrics-vs-fused
comparison and the fusion-weight sweep.

**The audio is never published or redistributed**, and no track title or artist appears in
committed results; tracks are identified by a stable pseudonym derived from the content hash.

Labels come from the layout rather than a metadata database:

- **genre** — the containing folder. Files sitting at the library root predate the genre
  folders and are a hip-hop / R&B collection; they are labelled as such.
- **artist** — the filename prefix before the first `" - "`, case-folded. This library was
  assembled for listening rather than evaluation, so most artists appear exactly once; the
  artist label covers far fewer queries here than on FMA and is reported with its query
  count attached.

There is no album label: the filenames do not carry one.
"""

from __future__ import annotations

from pathlib import Path

from .fma import TrackMeta

DEFAULT_ROOT = Path("data/music")

#: Genre for tracks at the library root, which predate the per-genre folders.
ROOT_GENRE = "hiphop_rnb"


def _artist(path: Path) -> str:
    """Filenames are `Artist - Title - Channel (youtube).mp3`; take the leading field.

    Case-folded because the same artist is written inconsistently across downloads
    ("A Boogie Wit Da Hoodie" and "... Wit da Hoodie" are one artist).
    """
    return path.stem.split(" - ")[0].strip().casefold()


def load_tracks(root: Path = DEFAULT_ROOT) -> list[TrackMeta]:
    """Load the personal library, labelling genre by folder and artist by filename.

    `track_id` is the index in path order, and `title` / `album` are left empty: neither is
    recoverable from the filename, and titles are not published.
    """
    root = Path(root)
    out: list[TrackMeta] = []
    for i, path in enumerate(sorted(root.rglob("*.mp3"))):
        rel = path.relative_to(root)
        genre = rel.parts[0] if len(rel.parts) > 1 else ROOT_GENRE
        out.append(TrackMeta(track_id=i, path=path, title="",
                             artist=_artist(path), album="", genre=genre))
    return out
