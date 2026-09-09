"""Eval 1 — human relevance on natural-language queries.

Slice 1's question: does whole-query embedding return results a listener judges relevant,
and do its weaknesses fall along category lines?

Reports three things, kept apart because they answer different questions:

- **mean relevance** -- were the returned tracks any good;
- **NDCG** -- given what was returned, was it *ordered* well. Its baseline is the same
  items shuffled, so it speaks only to ordering;
- **success@5** -- did at least one clearly-relevant track appear, which is closest to the
  product question of whether the user found something usable.

Results are broken out by category, and separately over Irene's own queries, because her
phrasing is the distribution the product actually faces.

    python scripts/eval_1_relevance.py
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aux.eval import bootstrap_ci, ndcg, random_ordering_ndcg, success_at_k  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
EVALS = ROOT / "evals"


def per_query(rating_set: dict, ratings: dict) -> list[dict]:
    """Assemble each query's ratings back into the system's own ranking order."""
    by_query: dict[int, list] = defaultdict(list)
    for item in rating_set["items"]:
        key = f"{item['query_id']}:{item['alias']}"
        if key in ratings:
            by_query[item["query_id"]].append((item["rank"], ratings[key], item["alias"]))

    out = []
    for qi, q in enumerate(rating_set["queries"]):
        rows = sorted(by_query.get(qi, []))
        if not rows:
            continue
        r = np.array([x[1] for x in rows])
        out.append({
            "query_id": qi, "query": q["query"], "category": q["category"],
            "source": q["source"], "n_rated": int(r.size),
            "ratings_in_rank_order": r.tolist(),
            "mean_relevance": float(r.mean()),
            "top1_relevance": float(r[0]),
            "ndcg": ndcg(r),
            "ndcg_random_order": random_ordering_ndcg(r),
            "success_at_5": success_at_k(r, k=5),
            "success_at_1": success_at_k(r, k=1),
            "n_irrelevant": int((r == 1).sum()),
        })
    return out


def summarise(rows: list[dict], label: str) -> dict:
    means = [r["mean_relevance"] for r in rows]
    ndcgs = [r["ndcg"] for r in rows]
    rand = [r["ndcg_random_order"] for r in rows]
    lo, hi = bootstrap_ci(means)
    nlo, nhi = bootstrap_ci(ndcgs)
    return {
        "label": label, "n_queries": len(rows),
        "mean_relevance": float(np.mean(means)), "mean_relevance_ci": [lo, hi],
        "ndcg": float(np.mean(ndcgs)), "ndcg_ci": [nlo, nhi],
        "ndcg_random_order": float(np.mean(rand)),
        "success_at_5": float(np.mean([r["success_at_5"] for r in rows])),
        "success_at_1": float(np.mean([r["success_at_1"] for r in rows])),
        "irrelevant_rate": float(np.sum([r["n_irrelevant"] for r in rows])
                                 / sum(r["n_rated"] for r in rows)),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Eval 1 — human relevance")
    ap.add_argument("--ratings", type=Path, default=EVALS / "ratings.private.json")
    ap.add_argument("--rating-set", type=Path, default=EVALS / "rating_set.private.json")
    args = ap.parse_args()

    rating_set = json.loads(args.rating_set.read_text())
    ratings = json.loads(args.ratings.read_text())["ratings"]
    rows = per_query(rating_set, ratings)

    overall = summarise(rows, "all queries")
    by_category = {c: summarise([r for r in rows if r["category"] == c], c)
                   for c in sorted({r["category"] for r in rows})}
    by_source = {s: summarise([r for r in rows if r["source"] == s], s)
                 for s in sorted({r["source"] for r in rows})}

    def show(s: dict) -> str:
        return (f"{s['label']:26}{s['n_queries']:>4}"
                f"{s['mean_relevance']:>7.2f}"
                f"  [{s['mean_relevance_ci'][0]:.2f},{s['mean_relevance_ci'][1]:.2f}]"
                f"{s['ndcg']:>8.3f}{s['ndcg_random_order']:>9.3f}"
                f"{s['success_at_5']:>9.0%}{s['success_at_1']:>9.0%}"
                f"{s['irrelevant_rate']:>8.0%}")

    header = (f"{'group':26}{'n':>4}{'mean':>7}{'  95% CI':>14}"
              f"{'NDCG':>8}{'NDCG-rnd':>9}{'succ@5':>9}{'succ@1':>9}{'irrel':>8}")
    print(header); print("-" * len(header))
    print(show(overall)); print("-" * len(header))
    for c in ("acoustic", "mood", "context", "compound"):
        if c in by_category:
            print(show(by_category[c]))
    print("-" * len(header))
    for s in by_source.values():
        print(show(s))

    print("\n=== weakest queries by mean relevance ===")
    for r in sorted(rows, key=lambda r: r["mean_relevance"])[:8]:
        print(f"  {r['mean_relevance']:.1f}  [{r['category']:8}] {r['query'][:56]}  "
              f"ratings {r['ratings_in_rank_order']}")
    print("\n=== strongest ===")
    for r in sorted(rows, key=lambda r: -r["mean_relevance"])[:5]:
        print(f"  {r['mean_relevance']:.1f}  [{r['category']:8}] {r['query'][:56]}")

    print("\n=== context control ===")
    for r in rows:
        if r["query"] in {"hip hop", "hip hop for running", "hip hop for studying",
                          "hip hop for falling asleep"}:
            print(f"  {r['mean_relevance']:.1f}  {r['query']:30} "
                  f"ratings {r['ratings_in_rank_order']}")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    payload = {"eval": "1", "run_at": datetime.now(timezone.utc).isoformat(),
               "encoder": rating_set["encoder"], "n_segments": rating_set["n_segments"],
               "overall": overall, "by_category": by_category, "by_source": by_source,
               "per_query": rows}
    out = EVALS / f"eval_1_relevance_{stamp}.json"
    out.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {out.relative_to(ROOT)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
