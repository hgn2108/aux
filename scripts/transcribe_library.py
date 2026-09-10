"""Transcribe a library, with caching, and report what came out.

Produces three things Slice 3 needs, from one pass:

- **lyrics text** for lyrical search;
- **detected language**, which addresses a measured Slice 1 failure ("sung in Vietnamese"
  returned jazz);
- **an instrumental flag**, which addresses another ("solo piano, no vocals" was the
  worst-scoring query in Slice 1).

Transcripts of personal music are personal data — arguably more identifying than filenames,
since they are the words — so the full output is written to a gitignored `.private.json` and
only aggregates are committed.

    python scripts/transcribe_library.py data/music --model small
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aux.ingest import IngestError, discover  # noqa: E402
from aux.ingest.asset import content_hash  # noqa: E402
from aux.lyrics import Transcriber  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
EVALS = ROOT / "evals"


def main() -> int:
    ap = argparse.ArgumentParser(description="Transcribe a library")
    ap.add_argument("root", type=Path)
    ap.add_argument("--model", default="small")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    cache_path = ROOT / ".cache" / f"transcripts_{args.model}.json"
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}

    transcriber = Transcriber(args.model)
    paths = [f.path for f in discover(args.root)]
    if args.limit:
        paths = paths[: args.limit]
    print(f"{len(paths)} files, model {transcriber.version}, device {transcriber.device}",
          file=sys.stderr)

    records, failures = [], []
    for i, path in enumerate(paths, 1):
        key = f"{content_hash(path)}|{transcriber.version}"
        if key in cache:
            records.append(cache[key])
            continue
        try:
            t = transcriber.transcribe(path)
        except (IngestError, Exception) as exc:  # noqa: BLE001
            failures.append({"path": str(path), "error": f"{type(exc).__name__}: {exc}"[:200]})
            continue
        rec = {"path": str(path), "content_hash": t.content_hash, "text": t.text,
               "language": t.language, "language_probability": t.language_probability,
               "mean_no_speech": t.mean_no_speech, "mean_logprob": t.mean_logprob,
               "duration_seconds": t.duration_seconds,
               "transcribe_seconds": t.transcribe_seconds, "n_segments": t.n_segments,
               "likely_instrumental": t.likely_instrumental, "reliable": t.reliable,
               "n_words": len(t.text.split())}
        cache[key] = rec
        records.append(rec)
        if i % 10 == 0:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(cache))
            print(f"  {i}/{len(paths)}", file=sys.stderr)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(cache))

    langs = Counter(r["language"] for r in records)
    instrumental = [r for r in records if r["likely_instrumental"]]
    reliable = [r for r in records if r["reliable"]]
    words = np.array([r["n_words"] for r in records])
    total_audio = sum(r["duration_seconds"] for r in records)
    total_time = sum(r["transcribe_seconds"] for r in records if r["transcribe_seconds"])

    summary = {
        "run_at": datetime.now(timezone.utc).isoformat(),
        "model": transcriber.version, "root": str(args.root),
        "files": len(paths), "transcribed": len(records), "failed": len(failures),
        "languages": dict(langs.most_common()),
        "likely_instrumental": len(instrumental),
        "reliable": len(reliable),
        "median_words": float(np.median(words)) if words.size else 0.0,
        "audio_hours": total_audio / 3600,
        "realtime_factor": (total_audio / total_time) if total_time else None,
    }

    print(f"\ntranscribed {len(records)}/{len(paths)}  ({len(failures)} failed)")
    print(f"languages detected: {dict(langs.most_common(8))}")
    print(f"likely instrumental: {len(instrumental)}   usable for lyric search: {len(reliable)}")
    print(f"median words per track: {summary['median_words']:.0f}")
    if summary["realtime_factor"]:
        print(f"speed: {summary['realtime_factor']:.0f}x realtime "
              f"over {summary['audio_hours']:.1f} h of audio")

    EVALS.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    (EVALS / f"transcription_{stamp}.private.json").write_text(
        json.dumps({"summary": summary, "records": records, "failures": failures}, indent=2))
    (EVALS / f"transcription_{stamp}.json").write_text(json.dumps({
        "summary": summary,
        "note": "Transcripts are the lyrics of a personal library and stay in the "
                ".private.json alongside this file, which is gitignored.",
    }, indent=2))
    print(f"\nwrote evals/transcription_{stamp}.json", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
