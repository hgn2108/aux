"""Index the FMA evaluation subset: audio embeddings and transcribed lyrics.

Both modalities are cached by content hash, so re-running is free and an interrupted run
resumes where it stopped.

    python scripts/build_fma_index.py --per-genre 250 --stage audio
    python scripts/build_fma_index.py --per-genre 250 --stage lyrics
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aux.data import balanced_subset, load_tracks  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / ".cache"


def main() -> int:
    ap = argparse.ArgumentParser(description="Index the FMA evaluation subset")
    ap.add_argument("--per-genre", type=int, default=250)
    ap.add_argument("--stage", choices=["audio", "lyrics", "both"], default="both")
    ap.add_argument("--n-segments", type=int, default=5)
    ap.add_argument("--whisper", default="small")
    args = ap.parse_args()

    tracks = balanced_subset(load_tracks(), per_genre=args.per_genre)
    paths = [t.path for t in tracks]
    print(f"{len(tracks)} tracks, {len({t.genre for t in tracks})} genres", file=sys.stderr)

    manifest = ROOT / ".cache" / f"fma_subset_{args.per_genre}.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps([{
        "track_id": t.track_id, "path": str(t.path), "title": t.title,
        "artist": t.artist, "album": t.album, "genre": t.genre} for t in tracks], indent=2))

    if args.stage in ("audio", "both"):
        from aux.encode.muq import MuQMuLanAdapter
        from aux.index import build_index

        encoder = MuQMuLanAdapter()
        V, kept, _ = build_index(None, encoder, paths=paths, n_segments=args.n_segments,
                                 cache_path=CACHE / f"fma_audio_{args.per_genre}.npz",
                                 progress_every=100)
        print(f"audio: {V.shape[0]}/{len(paths)} embedded, dim {V.shape[1]}", file=sys.stderr)

    if args.stage in ("lyrics", "both"):
        from aux.ingest.asset import content_hash
        from aux.lyrics import Transcriber

        cache_path = CACHE / f"transcripts_{args.whisper}.json"
        cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
        transcriber = Transcriber(args.whisper)
        done = 0
        for i, path in enumerate(paths, 1):
            key = f"{content_hash(path)}|{transcriber.version}"
            if key in cache:
                done += 1
                continue
            try:
                t = transcriber.transcribe(path)
            except Exception as exc:  # noqa: BLE001
                print(f"  skip {path.name}: {exc}"[:120], file=sys.stderr)
                continue
            cache[key] = {"path": str(path), "content_hash": t.content_hash, "text": t.text,
                          "language": t.language, "language_probability": t.language_probability,
                          "mean_no_speech": t.mean_no_speech, "mean_logprob": t.mean_logprob,
                          "duration_seconds": t.duration_seconds,
                          "transcribe_seconds": t.transcribe_seconds,
                          "n_segments": t.n_segments,
                          "likely_instrumental": t.likely_instrumental,
                          "reliable": t.reliable, "n_words": len(t.text.split())}
            done += 1
            if i % 25 == 0:
                cache_path.write_text(json.dumps(cache))
                print(f"  transcribed {i}/{len(paths)}", file=sys.stderr)
        cache_path.write_text(json.dumps(cache))
        print(f"lyrics: {done}/{len(paths)} transcribed", file=sys.stderr)

    print(f"manifest: {manifest.relative_to(ROOT)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
