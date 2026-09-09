"""E2 human half — baseline against planner on identical queries and ratings.

The pooled design does the work here: a track returned by both systems was rated once, and
that single judgement scores both. So the comparison is paired at the level of the *audio*,
not just the query, and no difference can come from the rater grading differently on two
occasions.

Reports mean relevance and success@1 per system, with a paired sign test over queries.
Eighteen or fourteen queries is few, so the test is exact rather than normal-approximate.

    python scripts/eval_2_ab.py
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from math import comb
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aux.eval import bootstrap_ci, mcnemar_exact, ndcg, success_at_k  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
EVALS = ROOT / "evals"


def sign_test(better: int, worse: int) -> float:
    """Exact two-sided sign test on queries where the systems differed."""
    n = better + worse
    if n == 0:
        return 1.0
    k = min(better, worse)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / (2**n))


def main() -> int:
    ap = argparse.ArgumentParser(description="E2 human comparison")
    ap.add_argument("--rating-set", type=Path, default=EVALS / "rating_set.private.json")
    ap.add_argument("--ratings", type=Path, default=EVALS / "ratings.private.json")
    args = ap.parse_args()

    rs = json.loads(args.rating_set.read_text())
    ratings = json.loads(args.ratings.read_text())["ratings"]

    # track_index -> alias, per query, so a system's picks can be scored.
    alias_of: dict[tuple[int, int], str] = {
        (it["query_id"], it["track_index"]): it["alias"] for it in rs["items"]
    }

    rows = []
    for qid_str, sysrec in rs["systems"].items():
        qid = int(qid_str)
        meta = rs["queries"][qid]

        def rated(indices):
            out = []
            for j in indices:
                key = f"{qid}:{alias_of[(qid, j)]}"
                if key in ratings:
                    out.append(ratings[key])
            return np.array(out)

        base = rated(sysrec["baseline"])
        plan = rated(sysrec["planner"])
        if base.size == 0 or plan.size == 0:
            continue
        rows.append({
            "query": sysrec["query"], "category": meta["category"],
            "rewritten": sysrec["rewritten"], "overlap": sysrec["overlap"],
            "baseline_mean": float(base.mean()), "planner_mean": float(plan.mean()),
            "baseline_ratings": base.tolist(), "planner_ratings": plan.tolist(),
            "baseline_ndcg": ndcg(base), "planner_ndcg": ndcg(plan),
            "baseline_s1": bool(success_at_k(base, k=1)),
            "planner_s1": bool(success_at_k(plan, k=1)),
        })

    b = np.array([r["baseline_mean"] for r in rows])
    p = np.array([r["planner_mean"] for r in rows])
    d = p - b
    better, worse = int((d > 0).sum()), int((d < 0).sum())
    p_sign = sign_test(better, worse)

    bs1 = np.array([r["baseline_s1"] for r in rows])
    ps1 = np.array([r["planner_s1"] for r in rows])
    gained, lost = int((~bs1 & ps1).sum()), int((bs1 & ~ps1).sum())

    blo, bhi = bootstrap_ci(b.tolist())
    plo, phi = bootstrap_ci(p.tolist())
    dlo, dhi = bootstrap_ci(d.tolist())

    print(f"queries: {len(rows)}   mean overlap between systems: "
          f"{np.mean([r['overlap'] for r in rows]):.1f} of 3\n")
    print(f"{'system':14}{'mean rel':>10}{'  95% CI':>16}{'NDCG':>8}{'succ@1':>9}")
    print("-" * 57)
    print(f"{'baseline':14}{b.mean():>10.2f}  [{blo:.2f},{bhi:.2f}]"
          f"{np.mean([r['baseline_ndcg'] for r in rows]):>8.3f}{bs1.mean():>9.0%}")
    print(f"{'planner':14}{p.mean():>10.2f}  [{plo:.2f},{phi:.2f}]"
          f"{np.mean([r['planner_ndcg'] for r in rows]):>8.3f}{ps1.mean():>9.0%}")
    print("-" * 57)
    print(f"{'difference':14}{d.mean():>+10.2f}  [{dlo:+.2f},{dhi:+.2f}]")
    print(f"\nplanner better on {better} queries, worse on {worse}, "
          f"tied on {len(rows) - better - worse}")
    print(f"sign test p = {p_sign:.3f}")
    print(f"success@1: gained {gained}, lost {lost}, "
          f"McNemar p = {mcnemar_exact(lost, gained):.3f}")

    print(f"\n{'query':44}{'base':>7}{'plan':>7}{'delta':>8}")
    print("-" * 66)
    for r in sorted(rows, key=lambda r: r["planner_mean"] - r["baseline_mean"]):
        delta = r["planner_mean"] - r["baseline_mean"]
        print(f"{r['query'][:42]:44}{r['baseline_mean']:>7.2f}{r['planner_mean']:>7.2f}"
              f"{delta:>+8.2f}")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    out = EVALS / f"eval_2_ab_{stamp}.json"
    out.write_text(json.dumps({
        "run_at": datetime.now(timezone.utc).isoformat(),
        "encoder": rs["encoder"], "planner": rs["planner"], "k": rs["k"],
        "n_queries": len(rows),
        "baseline": {"mean_relevance": float(b.mean()), "ci": [blo, bhi],
                     "ndcg": float(np.mean([r["baseline_ndcg"] for r in rows])),
                     "success_at_1": float(bs1.mean())},
        "planner": {"mean_relevance": float(p.mean()), "ci": [plo, phi],
                    "ndcg": float(np.mean([r["planner_ndcg"] for r in rows])),
                    "success_at_1": float(ps1.mean())},
        "difference": {"mean": float(d.mean()), "ci": [dlo, dhi],
                       "better": better, "worse": worse, "sign_test_p": p_sign,
                       "s1_gained": gained, "s1_lost": lost,
                       "s1_mcnemar_p": mcnemar_exact(lost, gained)},
        "per_query": rows,
    }, indent=2))
    print(f"\nwrote {out.relative_to(ROOT)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
