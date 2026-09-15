"""Labels for the local library, the only corpus here with usable lyrics.

FMA has the scale but not the words: 56% instrumental, median transcript 11 words. This one
is 160 tracks, 127 with real lyrics, so the lyric comparisons run here. The audio is never
redistributed.

Labels come from the layout, not a metadata database. Genre is the containing folder; files
at the root predate the folders and are hip-hop / R&B. Artist is the filename prefix before
the first " - ", case-folded, since downloads capitalise inconsistently. Most artists appear
once, so the artist label covers far fewer queries here than on FMA. No album label, the
filenames do not carry one.
"""

from __future__ import annotations

import re
from pathlib import Path

from .fma import TrackMeta

DEFAULT_ROOT = Path("data/music")

#: Genre for tracks at the library root, which predate the per-genre folders.
ROOT_GENRE = "hiphop_rnb"


#: Junk that downloaded filenames carry: the uploader tag, and the bracketed noise labels
#: that video titles append. Stripped for display only, `artist` stays case-folded and
#: unstripped, because it is a matching key and must not depend on cosmetic choices.
_NOISE = re.compile(
    r"\s*[\(\[](?:"
    r"official\s*(?:music\s*)?(?:video|audio|visualizer|lyric\s*video)?|"
    r"lyrics?|audio|visualizer|hd|hq|4k|explicit|clean|remastered\s*\d*|"
    r"youtube|live|m/?v|mv"
    r")[\)\]]",
    re.IGNORECASE,
)
_TRAILING_CHANNEL = re.compile(r"\s*-\s*[^-]*\(youtube\)\s*$", re.IGNORECASE)


def clean_title(path: Path) -> str:
    """A readable song title from a downloaded filename.

    Filenames arrive as `Artist - Title - Channel (youtube).mp3` with assorted video-title
    noise attached. The uploader field and that noise are dropped; what remains is the
    title as a person would write it.
    """
    stem = _TRAILING_CHANNEL.sub("", path.stem)
    parts = [p.strip() for p in stem.split(" - ")]
    title = parts[1] if len(parts) > 1 else parts[0]
    title = _NOISE.sub("", title)
    title = re.sub(r"\s*[\(\[]\s*[\)\]]", "", title)
    title = re.sub(r"\s{2,}", " ", title).strip(" -·")
    # Some titles arrive wrapped in the quotes the uploader typed. Curly quotes are written
    # as escapes so that a sweep over source punctuation cannot alter what this matches.
    opening = "\"'\u2018\u201c"
    closing = "\"'\u2019\u201d"
    if len(title) > 1 and title[0] in opening and title[-1] in closing:
        title = title[1:-1].strip()
    return title or path.stem


def display_artist(path: Path) -> str:
    """The artist as written, for display. `_artist` is the case-folded matching key."""
    return _TRAILING_CHANNEL.sub("", path.stem).split(" - ")[0].strip()


def _artist(path: Path) -> str:
    """Filenames are `Artist - Title - Channel (youtube).mp3`; take the leading field.

    Case-folded because the same artist is written inconsistently across downloads
    ("A Boogie Wit Da Hoodie" and "... Wit da Hoodie" are one artist).
    """
    return path.stem.split(" - ")[0].strip().casefold()


def load_tracks(root: Path = DEFAULT_ROOT) -> list[TrackMeta]:
    """Load the personal library, labelling genre by folder and artist by filename.

    `track_id` is the index in path order. `title` is recovered from the filename and
    cleaned for display; `album` stays empty, since filenames do not carry one.
    """
    root = Path(root)
    out: list[TrackMeta] = []
    for i, path in enumerate(sorted(root.rglob("*.mp3"))):
        rel = path.relative_to(root)
        genre = rel.parts[0] if len(rel.parts) > 1 else ROOT_GENRE
        out.append(TrackMeta(track_id=i, path=path, title=clean_title(path),
                             artist=_artist(path), album="", genre=genre))
    return out
