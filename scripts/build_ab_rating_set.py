"""Build a blind A/B rating set: baseline vs planner, same queries.

Pools the top-K from each system and hides which produced what. Pooling rather than
side-by-side because a rater shown two labelled lists compares lists; a rater shown shuffled
clips judges tracks, which is what the metric needs.

A track returned by both systems appears **once**, and its rating counts for both. That is
what makes the comparison paired: the two systems are scored on the same judgements of the
same audio, so a difference cannot come from rating drift between sessions.

    python scripts/build_ab_rating_set.py data/music --k 3
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aux.encode.base import l2_normalise  # noqa: E402
from aux.index import build_index  # noqa: E402
from aux.plan import build_planner, plan_all  # noqa: E402
from aux.query import score_plan  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
EVALS = ROOT / "evals"
SHUFFLE_SEED = 20260909


def load_queries(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) == 3:
            rows.append({"category": parts[0], "source": parts[1], "query": parts[2]})
    return rows


def genre_of(path: Path, root: Path) -> str:
    rel = path.relative_to(root)
    return rel.parts[0] if len(rel.parts) > 1 else "hiphop_rnb"


def main() -> int:
    ap = argparse.ArgumentParser(description="Build the A/B rating set")
    ap.add_argument("root", type=Path)
    ap.add_argument("--queries", type=Path, default=ROOT / "queries" / "slice2_rating.tsv")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--n-segments", type=int, default=5)
    args = ap.parse_args()

    from aux.encode.muq import MuQMuLanAdapter

    encoder = MuQMuLanAdapter()
    V, paths, _ = build_index(args.root, encoder, n_segments=args.n_segments,
                              cache_path=ROOT / ".cache" / "personal.npz")
    genres = [genre_of(p, args.root) for p in paths]
    counts: dict[str, int] = {}
    alias = []
    for g in genres:
        counts[g] = counts.get(g, 0) + 1
        alias.append(f"{g}/t{counts[g]:03d}")

    queries = load_queries(args.queries)
    planner = build_planner("claude")
    plans = plan_all(planner, [q["query"] for q in queries], ROOT / ".cache" / "plans.json")

    rng = random.Random(SHUFFLE_SEED)
    items, public, systems = [], [], {}
    for qi, q in enumerate(queries):
        text = q["query"]
        base = V @ l2_normalise(encoder.embed_text([text]))[0]
        plan = score_plan(encoder, plans[text], V)
        top_base = np.argsort(-base)[: args.k].tolist()
        top_plan = np.argsort(-plan)[: args.k].tolist()

        # Which systems returned each track, and at what rank — kept out of the rater's view.
        systems[str(qi)] = {
            "query": text, "rewritten": plans[text].rewritten,
            "baseline": top_base, "planner": top_plan,
            "overlap": len(set(top_base) & set(top_plan)),
        }
        block = []
        for j in dict.fromkeys(top_base + top_plan):
            block.append({"query_id": qi, "track_index": j, "alias": alias[j]})
        rng.shuffle(block)
        for pos, item in enumerate(block):
            j = item["track_index"]
            items.append({**item, "position": pos,
                          "audio": str(Path(paths[j]).resolve().relative_to(ROOT))})
            public.append({"query_id": qi, "alias": alias[j], "position": pos,
                           "genre": genres[j]})

    payload = {"queries": queries, "items": items, "k": args.k,
               "encoder": encoder.version, "n_segments": args.n_segments,
               "planner": planner.version, "systems": systems,
               "shuffle_seed": SHUFFLE_SEED}
    EVALS.mkdir(exist_ok=True)
    (EVALS / "rating_set.private.json").write_text(json.dumps(payload, indent=2))
    public_payload = dict(payload)
    public_payload["items"] = public
    public_payload["note"] = ("Track identities are pseudonymised; the personal library is "
                              "never published.")
    (EVALS / "rating_set_ab.json").write_text(json.dumps(public_payload, indent=2))

    overlaps = [s["overlap"] for s in systems.values()]
    print(f"{len(queries)} queries, {len(items)} clips "
          f"({len(items) / len(queries):.1f} per query)")
    print(f"mean overlap between systems: {np.mean(overlaps):.1f} of {args.k}")
    print("\nrun: .venv/bin/python scripts/rate.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
