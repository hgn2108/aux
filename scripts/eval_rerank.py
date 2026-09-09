"""Can reranking close the gap Eval 1 found?

Eval 1: NDCG over the returned top 5 was 0.866 against 0.863 for the same items shuffled,
and success@1 (67%) trailed success@5 (89%). Ordering was arbitrary, and a clearly-relevant
track was often retrieved without being placed first.

This reorders **exactly the five already-rated tracks per query**, so the existing ratings
score it and no new rating effort is needed. That scope is also the honest limit: it
measures ordering, not retrieval, and cannot credit a method for finding better candidates.

Comparisons are paired -- every method sees the same queries -- and the ceiling is reported
alongside, because a method should be judged against what was actually achievable on these
candidates, not against 1.0.

    python scripts/eval_rerank.py data/music
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aux.encode.base import l2_normalise  # noqa: E402
from aux.eval import bootstrap_ci, mcnemar_exact, ndcg, success_at_k  # noqa: E402
from aux.ingest import IngestError, decode, discover  # noqa: E402
from aux.rank import cosine, csls, max_segment, mean_plus_max, query_z  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
EVALS = ROOT / "evals"


def main() -> int:
    ap = argparse.ArgumentParser(description="Rerank the rated candidates")
    ap.add_argument("root", type=Path)
    ap.add_argument("--n-segments", type=int, default=5)
    args = ap.parse_args()

    rating_set = json.loads((EVALS / "rating_set.private.json").read_text())
    ratings = json.loads((EVALS / "ratings.private.json").read_text())["ratings"]

    from aux.encode.muq import MuQMuLanAdapter

    encoder = MuQMuLanAdapter()
    by_path: dict[str, int] = {}
    vectors, segments = [], []
    for path in [f.path for f in discover(args.root)]:
        try:
            vec, segs = encoder.embed_track(decode(path), n_segments=args.n_segments)
        except (IngestError, Exception):  # noqa: BLE001
            continue
        by_path[str(Path(path).resolve().relative_to(ROOT))] = len(vectors)
        vectors.append(vec)
        segments.append(segs)
    V = np.stack(vectors)
    print(f"{V.shape[0]} tracks indexed", file=sys.stderr)

    queries = rating_set["queries"]
    Q = l2_normalise(encoder.embed_text([q["query"] for q in queries]))

    # Gather each query's rated candidates.
    cand: dict[int, list[tuple[int, int]]] = {}
    for item in rating_set["items"]:
        key = f"{item['query_id']}:{item['alias']}"
        if key in ratings and item["audio"] in by_path:
            cand.setdefault(item["query_id"], []).append(
                (by_path[item["audio"]], ratings[key]))

    methods = {
        "cosine (current)": lambda qi, idx: cosine(Q[qi], V[idx]),
        "max-segment": lambda qi, idx: max_segment(Q[qi], [segments[i] for i in idx]),
        "mean+max": lambda qi, idx: mean_plus_max(
            Q[qi], V[idx], [segments[i] for i in idx], alpha=0.5),
        "CSLS (hubness)": lambda qi, idx: cosine(Q[qi], V[idx]) - _density(idx, V),
        "query-z": lambda qi, idx: query_z(Q[qi], V[idx], Q),
    }

    results = {}
    for name, fn in methods.items():
        ndcgs, s1, per_query = [], [], {}
        for qi, rows in cand.items():
            idx = np.array([r[0] for r in rows])
            rel = np.array([r[1] for r in rows])
            order = np.argsort(-np.asarray(fn(qi, idx)))
            ordered = rel[order]
            ndcgs.append(ndcg(ordered))
            s1.append(success_at_k(ordered, k=1))
            per_query[qi] = {"ndcg": ndcg(ordered), "success_at_1": bool(s1[-1])}
        lo, hi = bootstrap_ci(ndcgs)
        results[name] = {"ndcg": float(np.mean(ndcgs)), "ndcg_ci": [lo, hi],
                         "success_at_1": float(np.mean(s1)), "per_query": per_query}

    # Ceiling: the best any reordering of these same candidates could achieve.
    ceiling_s1 = float(np.mean([max(r[1] for r in rows) >= 4 for rows in cand.values()]))

    base = results["cosine (current)"]
    header = f"{'method':22}{'NDCG':>8}{'  95% CI':>16}{'succ@1':>9}{'vs base':>10}{'p':>10}"
    print(header); print("-" * len(header))
    for name, r in results.items():
        b = np.array([base["per_query"][q]["success_at_1"] for q in sorted(cand)])
        t = np.array([r["per_query"][q]["success_at_1"] for q in sorted(cand)])
        gained, lost = int((~b & t).sum()), int((b & ~t).sum())
        p = mcnemar_exact(lost, gained)
        delta = r["success_at_1"] - base["success_at_1"]
        print(f"{name:22}{r['ndcg']:>8.3f}  [{r['ndcg_ci'][0]:.3f},{r['ndcg_ci'][1]:.3f}]"
              f"{r['success_at_1']:>9.0%}{delta:>+10.0%}{p:>10.3f}")
    print("-" * len(header))
    print(f"{'ceiling (perfect)':22}{1.0:>8.3f}{'':>16}{ceiling_s1:>9.0%}")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    out = EVALS / f"eval_rerank_{stamp}.json"
    out.write_text(json.dumps(
        {"run_at": datetime.now(timezone.utc).isoformat(), "encoder": encoder.version,
         "n_queries": len(cand), "ceiling_success_at_1": ceiling_s1,
         "methods": {k: {kk: vv for kk, vv in v.items() if kk != "per_query"}
                     for k, v in results.items()}}, indent=2))
    print(f"\nwrote {out.relative_to(ROOT)}", file=sys.stderr)
    return 0


def _density(idx: np.ndarray, V: np.ndarray, k: int = 10) -> np.ndarray:
    """Local density of each candidate against the whole index, not just the candidates.

    Computed over the full library on purpose: a hub is a hub because of where it sits in
    the collection, and measuring density inside a five-item shortlist would miss that.
    """
    sims = V[idx] @ V.T
    for row, i in enumerate(idx):
        sims[row, i] = -np.inf
    kk = min(k, V.shape[0] - 1)
    return np.sort(sims, axis=1)[:, -kk:].mean(axis=1)


if __name__ == "__main__":
    raise SystemExit(main())
