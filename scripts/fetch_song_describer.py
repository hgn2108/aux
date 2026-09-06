"""Fetch the Song Describer Dataset from Zenodo.

Eval 0B's benchmark. 706 tracks / 1,106 captions, CC BY-SA 4.0,
DOI 10.5281/zenodo.10072001.

A script rather than a shell one-liner so the exact provenance of the benchmark corpus is
recorded in the repository: which record, which files, and a checksum verifying that what
was measured is what was published.

    python scripts/fetch_song_describer.py            # metadata only (~0.5 MB)
    python scripts/fetch_song_describer.py --audio    # adds audio.zip (3.3 GB)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

RECORD_URL = "https://zenodo.org/api/records/10072001"
DEST = Path(__file__).resolve().parents[1] / "data" / "raw" / "song_describer"

METADATA_FILES = ["song_describer.csv", "audio_metadata.tsv", "audio_licenses.txt"]
AUDIO_FILE = "audio.zip"


def _md5(path: Path) -> str:
    digest = hashlib.md5()  # noqa: S324 -- matching Zenodo's published checksum, not security
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _download(url: str, dest: Path, expected_md5: str | None, size: float) -> None:
    if dest.exists() and expected_md5 and _md5(dest) == expected_md5:
        print(f"  {dest.name}: already present and verified")
        return
    print(f"  {dest.name}: downloading {size / 1e6:.1f} MB ...")
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url, timeout=120) as response, tmp.open("wb") as out:
        shutil.copyfileobj(response, out, length=1 << 20)
    if expected_md5:
        actual = _md5(tmp)
        if actual != expected_md5:
            tmp.unlink(missing_ok=True)
            raise RuntimeError(f"checksum mismatch for {dest.name}: {actual} != {expected_md5}")
    tmp.replace(dest)
    print(f"  {dest.name}: done, checksum verified")


def main() -> int:
    ap = argparse.ArgumentParser(description="Fetch Song Describer (Eval 0B benchmark)")
    ap.add_argument("--audio", action="store_true", help="also fetch audio.zip (3.3 GB)")
    ap.add_argument("--extract", action="store_true", help="unzip audio.zip after download")
    args = ap.parse_args()

    DEST.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(RECORD_URL, timeout=60) as response:
        record = json.load(response)

    by_key = {f["key"]: f for f in record["files"]}
    (DEST / "PROVENANCE.json").write_text(json.dumps({
        "title": record["metadata"]["title"],
        "doi": record.get("doi"),
        "license": record["metadata"].get("license"),
        "record_url": RECORD_URL,
        "files": {k: {"size": v["size"], "checksum": v.get("checksum")} for k, v in by_key.items()},
    }, indent=2))

    wanted = METADATA_FILES + ([AUDIO_FILE] if args.audio else [])
    for key in wanted:
        entry = by_key.get(key)
        if entry is None:
            print(f"  {key}: not in record, skipping", file=sys.stderr)
            continue
        _download(entry["links"]["self"], DEST / key,
                  (entry.get("checksum") or "").removeprefix("md5:") or None,
                  entry["size"])

    if args.extract and (DEST / AUDIO_FILE).exists():
        target = DEST / "audio"
        if target.exists():
            print(f"  {target} already extracted")
        else:
            print(f"  extracting {AUDIO_FILE} ...")
            with zipfile.ZipFile(DEST / AUDIO_FILE) as zf:
                zf.extractall(DEST)
            print("  extracted")

    print(f"\nready in {DEST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
