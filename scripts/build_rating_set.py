"""Build the blind rating set for Slice 1.

Runs every query against the library, pools the top-K, and writes the items in an order
that hides everything a rater could use instead of listening:

- **rank is not shown and not stored in display order** -- items are shuffled within each
  query, so a rater cannot infer confidence from position;
- **the score is not shown**, for the same reason;
- **filenames are not shown** -- a title tells you the artist and often the genre, which is
  precisely what we are asking the rater to judge from the audio.

Two files come out. `rating_set.private.json` holds the audio paths the local player needs
and stays out of the repo; `rating_set.json` is the committable record of which query drew
which pseudonymous item.

    python scripts/build_rating_set.py data/music --k 5
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aux.encode.base import l2_normalise  # noqa: E402
from aux.ingest import IngestError, decode, discover  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
EVALS = ROOT / "evals"

SHUFFLE_SEED = 20260908
"""Fixed so the rating order is reproducible: a re-run presents the same sequence, and a
disagreement between two rating sessions is about judgement rather than ordering."""


def load_queries(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) != 3:
            continue
        rows.append({"category": parts[0], "source": parts[1], "query": parts[2]})
    return rows


def genre_of(path: Path, root: Path) -> str:
    rel = path.relative_to(root)
    return rel.parts[0] if len(rel.parts) > 1 else "hiphop_rnb"


def main() -> int:
    ap = argparse.ArgumentParser(description="Build the Slice 1 blind rating set")
    ap.add_argument("root", type=Path)
    ap.add_argument("--queries", type=Path, default=ROOT / "queries" / "slice1.tsv")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--encoder", default="muq")
    ap.add_argument("--n-segments", type=int, default=5)
    args = ap.parse_args()

    from aux.encode.muq import MuQMuLanAdapter

    encoder = MuQMuLanAdapter()
    queries = load_queries(args.queries)
    paths = [f.path for f in discover(args.root)]
    print(f"{len(queries)} queries, {len(paths)} tracks", file=sys.stderr)

    vectors, kept = [], []
    for i, path in enumerate(paths, 1):
        try:
            vec, _ = encoder.embed_track(decode(path), n_segments=args.n_segments)
        except (IngestError, Exception):  # noqa: BLE001
            continue
        vectors.append(vec)
        kept.append(path)
        if i % 40 == 0:
            print(f"  indexed {i}/{len(paths)}", file=sys.stderr)
    V = np.stack(vectors)

    genres = [genre_of(p, args.root) for p in kept]
    counts: dict[str, int] = {}
    alias = []
    for g in genres:
        counts[g] = counts.get(g, 0) + 1
        alias.append(f"{g}/t{counts[g]:03d}")

    T = l2_normalise(encoder.embed_text([q["query"] for q in queries]))
    scores = T @ V.T

    rng = random.Random(SHUFFLE_SEED)
    items, public_items = [], []
    for qi, q in enumerate(queries):
        top = np.argsort(-scores[qi])[: args.k]
        block = [{"query_id": qi, "track_index": int(j), "rank": int(r + 1),
                  "score": float(scores[qi, j])} for r, j in enumerate(top)]
        rng.shuffle(block)
        for pos, item in enumerate(block):
            j = item["track_index"]
            items.append({**item, "position": pos, "alias": alias[j],
                          "audio": str(Path(kept[j]).resolve().relative_to(ROOT))})
            public_items.append({**{k: v for k, v in item.items() if k != "track_index"},
                                 "position": pos, "alias": alias[j], "genre": genres[j]})

    EVALS.mkdir(exist_ok=True)
    (EVALS / "rating_set.private.json").write_text(json.dumps(
        {"queries": queries, "items": items, "k": args.k,
         "encoder": encoder.version, "n_segments": args.n_segments,
         "shuffle_seed": SHUFFLE_SEED}, indent=2))
    (EVALS / "rating_set.json").write_text(json.dumps(
        {"queries": queries, "items": public_items, "k": args.k,
         "encoder": encoder.version, "n_segments": args.n_segments,
         "shuffle_seed": SHUFFLE_SEED,
         "note": "Audio paths are in the .private.json alongside this file, which is "
                 "gitignored: the personal library is never published."}, indent=2))

    print(f"\n{len(items)} items across {len(queries)} queries "
          f"({args.k} each) -> evals/rating_set.json", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
