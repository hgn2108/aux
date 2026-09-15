"""Export the personal library as a deployable bundle: no audio, no lyric text.

The public corpus cannot demonstrate the lyric modality, 56% of FMA is instrumental, so a
deployed demo that only loads FMA ships a lyrics feature it can never show working. This
exports what the personal library needs to be browsable and searchable on a public instance
while leaving behind everything that would be redistribution.

**What ships.** Track title, artist, genre, and the derived vectors: a 512-d audio embedding
and a lyric embedding per track. Titles and artist names are facts about recordings, and
embeddings are lossy derived representations from which no audio can be reconstructed.
Distributing features rather than audio is the standard arrangement in music information
retrieval, which is why public music datasets ship exactly this.

**What does not ship.** The audio, and the transcripts. A transcript is the lyrics, which are
a copyrighted text; the embedding of one is not. The app therefore has no playback and no
lyric display for this corpus, only ranking.

    python scripts/export_demo_corpus.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aux.data import load_personal_tracks  # noqa: E402
from aux.data.personal import display_artist  # noqa: E402
from aux.index import build_index  # noqa: E402
from aux.lyrics import load_lyric_vectors  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / ".cache"
DEMO = ROOT / "demo"


def export_fma(n_tracks: int, n_segments: int) -> int:
    """Export a browsable slice of FMA: metadata, vectors, and the audio itself.

    The full corpus cannot be deployed, 7.4 GB of audio and a 248 MB metadata CSV, none
    of it in the repository. This copies a genre-balanced slice small enough to ship, which
    is what lets a deployed demo play anything at all. It is Creative Commons, so unlike
    the personal library the audio travels with it.

    The evaluation always uses the full corpus; this is only what the demo browses.
    """
    from aux.data import balanced_subset, load_tracks

    per_genre = max(1, n_tracks // 8)
    tracks = balanced_subset(load_tracks(), per_genre=per_genre)
    from aux.encode.muq import MuQMuLanAdapter

    encoder = MuQMuLanAdapter()
    vectors, kept, _ = build_index(None, encoder, paths=[t.path for t in tracks],
                                   n_segments=n_segments,
                                   cache_path=CACHE / "fma_audio_250.npz",
                                   progress_every=0)
    kept_set = {str(p) for p in kept}
    tracks = [t for t in tracks if str(t.path) in kept_set]

    audio_dir = DEMO / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for t in tracks:
        target = audio_dir / f"{t.track_id:06d}.mp3"
        if not target.exists():
            target.write_bytes(t.path.read_bytes())
        manifest.append({"track_id": t.track_id, "title": t.title, "artist": t.artist,
                         "genre": t.genre, "file": target.name})

    (DEMO / "fma_manifest.json").write_text(json.dumps({
        "name": "FMA small",
        "note": ("Creative Commons, so it plays here. 56% instrumental in a sampled 75 "
                 "clips, median transcript 11 words, no lyric channel to search."),
        "tracks": manifest,
    }, indent=2))
    np.savez_compressed(DEMO / "fma_vectors.npz", audio=vectors.astype(np.float32))

    size = sum(f.stat().st_size for f in audio_dir.iterdir()) / 1e6
    print(f"{len(tracks)} FMA tracks, {size:.0f} MB of audio in {audio_dir}")
    return 0


def export_examples() -> int:
    """Embed the demo's fixed example queries, so clicking one needs no model.

    A hosted container spends minutes loading 2.5GB of weights on its first search. The
    example queries never change, so answering them can be a dot product against vectors
    that shipped with the bundle. Typing something new still loads the encoder; there is no
    way around that for an arbitrary string.
    """
    from aux.app.examples import ALL_EXAMPLES, EXAMPLES_PATH, LYRIC_EXAMPLES
    from aux.encode.muq import MuQMuLanAdapter
    from aux.lyrics import LyricEmbedder

    queries = list(ALL_EXAMPLES)
    audio = np.asarray(MuQMuLanAdapter().embed_text(queries), dtype=np.float32)

    # Only the lyric examples are ever searched against transcripts, but a vector is stored
    # for every query: the mode is the visitor's to change after picking one, and a missing
    # vector would silently fall back to loading the model.
    lyric = np.asarray(LyricEmbedder().embed_query(queries), dtype=np.float32)

    DEMO.mkdir(exist_ok=True)
    np.savez_compressed(EXAMPLES_PATH, queries=json.dumps(queries),
                        audio=audio, lyric=lyric)
    size = EXAMPLES_PATH.stat().st_size / 1e3
    print(f"{len(queries)} example queries ({len(LYRIC_EXAMPLES)} about lyrics), "
          f"{size:.0f} KB")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Export a deployable corpus bundle")
    ap.add_argument("--corpus", choices=("personal", "fma", "examples"), default="personal")
    ap.add_argument("--tracks", type=int, default=80, help="fma only, genre-balanced")
    ap.add_argument("--n-segments", type=int, default=5)
    args = ap.parse_args()

    if args.corpus == "examples":
        return export_examples()
    if args.corpus == "fma":
        return export_fma(args.tracks, args.n_segments)

    tracks = load_personal_tracks()
    from aux.encode.muq import MuQMuLanAdapter

    encoder = MuQMuLanAdapter()
    audio, kept, _ = build_index(None, encoder, paths=[t.path for t in tracks],
                                 n_segments=args.n_segments,
                                 cache_path=CACHE / "personal.npz", progress_every=0)
    kept_set = {str(p) for p in kept}
    tracks = [t for t in tracks if str(t.path) in kept_set]

    lyrics, has_lyrics = load_lyric_vectors(tracks, "small", "personal")

    DEMO.mkdir(exist_ok=True)
    manifest = [{"track_id": t.track_id, "title": t.title,
                 "artist": display_artist(t.path), "genre": t.genre,
                 "has_lyrics": bool(has_lyrics[i])}
                for i, t in enumerate(tracks)]
    (DEMO / "personal_manifest.json").write_text(json.dumps({
        "name": "personal library",
        "note": ("Commercially released music. Audio and transcripts are not "
                 "redistributed, so this corpus cannot be played here, only searched and "
                 "ranked, from embeddings computed locally."),
        "tracks": manifest,
    }, indent=2))
    np.savez_compressed(DEMO / "personal_vectors.npz", audio=audio.astype(np.float32),
                        lyrics=lyrics.astype(np.float32), has_lyrics=has_lyrics)

    size = sum(f.stat().st_size for f in DEMO.iterdir()) / 1e6
    print(f"{len(tracks)} tracks, {int(has_lyrics.sum())} with lyrics, {size:.1f} MB")
    print(f"wrote {(DEMO / 'personal_manifest.json').relative_to(ROOT)} and "
          f"{(DEMO / 'personal_vectors.npz').relative_to(ROOT)}")

    # Guard rather than trust: a transcript or a local path reaching the bundle would be a
    # problem no amount of intent prevents. Checked over the track records only, the
    # explanatory note legitimately talks *about* transcripts.
    allowed = {"track_id", "title", "artist", "genre", "has_lyrics"}
    for record in manifest:
        extra = set(record) - allowed
        if extra:
            print(f"REFUSING: manifest carries unexpected fields {sorted(extra)}",
                  file=sys.stderr)
            return 1
    blob = json.dumps(manifest)
    for pattern in (".mp3", "/Users/", str(ROOT)):
        if pattern in blob:
            print(f"REFUSING: manifest contains {pattern!r}", file=sys.stderr)
            return 1
    print("checked: titles, artists, genres and vectors only, no audio, no transcripts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
