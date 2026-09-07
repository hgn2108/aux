"""Audit a personal music folder before it is used as an out-of-domain test set.

Slice 0's remaining pass condition is "plausible personal-library neighbours". That
judgement is worthless if the corpus is quietly broken -- a silent file, a truncated
download, or the same track present three times all distort a neighbour listing while
looking fine in a directory listing.

Checks, in order of how badly each would corrupt the result:

1. **decodes at all**, through the same ingestion path as the rest of the project;
2. **contains audible audio** -- decoding to digital silence is a real failure mode for
   scraped media and produces a vector that is nearest-neighbour to nothing meaningful;
3. **exact duplicates** by content hash -- the same bytes under two names;
4. **audio duplicates** by decoded-waveform fingerprint -- the same recording re-encoded,
   which a byte hash cannot see;
5. **title near-duplicates** -- reported only, never deleted, since two uploads of a track
   may be genuinely different recordings.

Deletion is opt-in and only ever removes files 3 and 4, keeping one copy of each.

    python scripts/audit_library.py data/music
    python scripts/audit_library.py data/music --delete-duplicates
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aux.ingest import IngestError, decode, discover  # noqa: E402

EVALS_DIR = Path(__file__).resolve().parents[1] / "evals"

SILENCE_RMS = 1e-4
SHORT_SECONDS = 30.0
FINGERPRINT_RATE = 1000
"""Waveform is reduced to ~1 kHz envelope before hashing.

Coarse on purpose: two encodes of one recording differ sample by sample but share their
envelope, so an exact hash of raw samples would miss them while a perceptual envelope
catches them.
"""


def audio_fingerprint(samples: np.ndarray, sample_rate: int) -> str:
    """Stable coarse fingerprint of the decoded audio, robust to re-encoding."""
    step = max(1, sample_rate // FINGERPRINT_RATE)
    n = (samples.size // step) * step
    if n == 0:
        return "empty"
    envelope = np.abs(samples[:n].reshape(-1, step)).max(axis=1)
    peak = float(envelope.max()) or 1.0
    quantised = np.round(envelope / peak * 15).astype(np.uint8)  # 16 levels
    import hashlib

    return hashlib.blake2b(quantised.tobytes(), digest_size=12).hexdigest()


def normalise_title(name: str) -> str:
    """Reduce a filename to a comparable title.

    Scraped filenames carry uploader suffixes and parenthetical noise -- "(Official
    Audio)", "- ArtistVEVO (youtube)" -- that differ between two uploads of the same track.
    """
    stem = Path(name).stem.lower()
    stem = re.sub(r"\((official|audio|video|visualizer|lyric[s]?|music video|hd|4k)[^)]*\)", " ", stem)
    stem = re.sub(r"\[[^\]]*\]", " ", stem)
    stem = re.sub(r"\((youtube|official)\)", " ", stem)
    stem = re.sub(r"\bvevo\b|\bofficial\b|\baudio\b|\bvideo\b|\bfeat\b|\bft\b", " ", stem)
    stem = re.sub(r"[^a-z0-9]+", " ", stem)
    return " ".join(sorted(set(stem.split())))


def main() -> int:
    ap = argparse.ArgumentParser(description="Audit a personal music library")
    ap.add_argument("root", type=Path)
    ap.add_argument("--delete-duplicates", action="store_true",
                    help="delete exact and audio duplicates, keeping one copy of each")
    args = ap.parse_args()

    files = [f.path for f in discover(args.root)]
    print(f"auditing {len(files)} files under {args.root}", file=sys.stderr)

    ok, failed = [], []
    by_content: dict[str, list[Path]] = defaultdict(list)
    by_audio: dict[str, list[Path]] = defaultdict(list)
    by_title: dict[str, list[Path]] = defaultdict(list)

    for i, path in enumerate(files, 1):
        try:
            asset = decode(path)
        except IngestError as exc:
            failed.append({"path": str(path), "category": exc.category.value, "message": exc.message})
            continue
        except Exception as exc:  # noqa: BLE001
            failed.append({"path": str(path), "category": "uncategorised",
                           "message": f"{type(exc).__name__}: {exc}"[:200]})
            continue

        rms = float(np.sqrt(np.mean(asset.samples**2)))
        record = {
            "path": str(path), "content_hash": asset.content_hash,
            "sample_rate": asset.sample_rate, "channels": asset.native_channels,
            "duration_seconds": round(asset.duration_seconds, 1), "codec": asset.codec,
            "rms": rms, "peak": float(np.abs(asset.samples).max()),
            "silent": rms < SILENCE_RMS, "short": asset.duration_seconds < SHORT_SECONDS,
        }
        ok.append(record)
        by_content[asset.content_hash].append(path)
        by_audio[audio_fingerprint(asset.samples, asset.sample_rate)].append(path)
        by_title[normalise_title(path.name)].append(path)
        if i % 20 == 0:
            print(f"  {i}/{len(files)}", file=sys.stderr)

    exact_dupes = {h: [str(p) for p in ps] for h, ps in by_content.items() if len(ps) > 1}
    audio_dupes = {h: [str(p) for p in ps] for h, ps in by_audio.items()
                   if len(ps) > 1 and h not in {"empty"}}
    # Title collisions that are not already caught by content or audio identity.
    caught = {p for ps in by_content.values() if len(ps) > 1 for p in ps}
    caught |= {p for ps in by_audio.values() if len(ps) > 1 for p in ps}
    title_dupes = {t: [str(p) for p in ps] for t, ps in by_title.items()
                   if len(ps) > 1 and not set(ps) <= caught}

    deleted: list[str] = []
    if args.delete_duplicates:
        for group in list(exact_dupes.values()) + list(audio_dupes.values()):
            for extra in sorted(group)[1:]:                 # keep the first, drop the rest
                p = Path(extra)
                if p.exists():
                    p.unlink()
                    deleted.append(extra)
        deleted = sorted(set(deleted))

    summary = {
        "root": str(args.root),
        "run_at": datetime.now(timezone.utc).isoformat(),
        "files_seen": len(files),
        "decoded_ok": len(ok),
        "failed": len(failed),
        "silent": sum(1 for r in ok if r["silent"]),
        "short": sum(1 for r in ok if r["short"]),
        "exact_duplicate_groups": len(exact_dupes),
        "audio_duplicate_groups": len(audio_dupes),
        "title_near_duplicate_groups": len(title_dupes),
        "deleted": deleted,
        "sample_rates": dict(Counter(r["sample_rate"] for r in ok)),
        "channels": dict(Counter(r["channels"] for r in ok)),
        "duration_total_hours": round(sum(r["duration_seconds"] for r in ok) / 3600, 2),
        "duration_median_seconds": float(np.median([r["duration_seconds"] for r in ok])) if ok else None,
    }

    EVALS_DIR.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d")

    # Two reports. The personal library is an out-of-domain test set that is never
    # published or redistributed (PROJECT.md), and track titles are as identifying as the
    # audio -- they reveal what someone listens to. So the committed report carries only
    # aggregates, and anything naming a file is written to a `.private.json` that
    # .gitignore excludes.
    private = EVALS_DIR / f"library_audit_{stamp}.private.json"
    private.write_text(json.dumps({
        "summary": summary, "failures": failed, "exact_duplicates": exact_dupes,
        "audio_duplicates": audio_dupes, "title_near_duplicates": title_dupes, "files": ok,
    }, indent=2))

    out = EVALS_DIR / f"library_audit_{stamp}.json"
    out.write_text(json.dumps({
        "summary": summary,
        # Failure categories are the evidence Eval 0A cares about; the filenames are not.
        "failure_categories": dict(Counter(f["category"] for f in failed)),
        "note": "Per-file detail is in the .private.json alongside this file, "
                "which is gitignored: the personal library is never published.",
    }, indent=2))

    print(json.dumps(summary, indent=2))
    for name, groups in (("EXACT DUPLICATES", exact_dupes), ("AUDIO DUPLICATES", audio_dupes),
                         ("TITLE NEAR-DUPLICATES (reported only)", title_dupes)):
        if groups:
            print(f"\n{name}:")
            for key, paths in groups.items():
                print(f"  [{key[:12]}]")
                for p in paths:
                    print(f"    {Path(p).name}")
    if failed:
        print("\nFAILED TO DECODE:")
        for f in failed:
            print(f"  {f['category']:24} {Path(f['path']).name}")
    print(f"\nreport: {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
