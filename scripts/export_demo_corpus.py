"""Export the personal library as a deployable bundle: no audio, no lyric text.

The public corpus cannot demonstrate the lyric modality — 56% of FMA is instrumental — so a
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

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / ".cache"
DEMO = ROOT / "demo"


def main() -> int:
    ap = argparse.ArgumentParser(description="Export a deployable, audio-free corpus")
    ap.add_argument("--n-segments", type=int, default=5)
    args = ap.parse_args()

    tracks = load_personal_tracks()
    from aux.encode.muq import MuQMuLanAdapter

    encoder = MuQMuLanAdapter()
    audio, kept, _ = build_index(None, encoder, paths=[t.path for t in tracks],
                                 n_segments=args.n_segments,
                                 cache_path=CACHE / "personal.npz", progress_every=0)
    kept_set = {str(p) for p in kept}
    tracks = [t for t in tracks if str(t.path) in kept_set]

    sys.path.insert(0, str(ROOT / "scripts"))
    from eval_recommendation import load_lyric_vectors

    lyrics, has_lyrics = load_lyric_vectors(tracks, "small", "personal")

    DEMO.mkdir(exist_ok=True)
    manifest = [{"track_id": t.track_id, "title": t.title,
                 "artist": display_artist(t.path), "genre": t.genre,
                 "has_lyrics": bool(has_lyrics[i])}
                for i, t in enumerate(tracks)]
    (DEMO / "personal_manifest.json").write_text(json.dumps({
        "name": "personal library",
        "note": ("Commercially released music. Audio and transcripts are not "
                 "redistributed, so this corpus cannot be played here — only searched and "
                 "ranked, from embeddings computed locally."),
        "tracks": manifest,
    }, indent=2))
    np.savez_compressed(DEMO / "personal_vectors.npz", audio=audio.astype(np.float32),
                        lyrics=lyrics.astype(np.float32), has_lyrics=has_lyrics)

    size = sum(f.stat().st_size for f in DEMO.iterdir()) / 1e6
    print(f"{len(tracks)} tracks, {int(has_lyrics.sum())} with lyrics — {size:.1f} MB")
    print(f"wrote {(DEMO / 'personal_manifest.json').relative_to(ROOT)} and "
          f"{(DEMO / 'personal_vectors.npz').relative_to(ROOT)}")

    # Guard rather than trust: a transcript or a local path reaching the bundle would be a
    # problem no amount of intent prevents. Checked over the track records only -- the
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
    print("checked: titles, artists, genres and vectors only — no audio, no transcripts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
