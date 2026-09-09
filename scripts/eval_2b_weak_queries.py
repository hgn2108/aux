"""E2b — the planner on the queries where the baseline is weak.

Pre-registered in `docs/PREREG_E2B.md`, committed before this ran. Query selection is
mechanical: every Slice 1 query below 3.5 mean relevance.

Three objective metrics, none needing human time, reported before any rating round:

- **intent match** — do retrieved tracks have the properties the query or rewrite named,
  measured on waveform features the encoder never sees;
- **z-top** — how far the best match stands above the corpus for that query;
- **top-K overlap** — how much the planner changed at all.

    python scripts/eval_2b_weak_queries.py data/music
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
from aux.eval import bootstrap_ci, intent_match  # noqa: E402
from aux.index import build_index  # noqa: E402
from aux.ingest import IngestError, decode  # noqa: E402
from aux.plan import build_planner, plan_all  # noqa: E402
from aux.query import score_plan  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
EVALS = ROOT / "evals"
WEAK_THRESHOLD = 3.5

# The acoustic direction each query implies, from PREREG_E2B.md Amendment 1. Written from
# the plain meaning of each query, applied identically to both systems, and used only for
# scoring -- never for retrieval. Two queries name instrumentation with no honest physical
# direction and are deliberately left out rather than given an invented mapping.
IMPLIED_INTENT = {
    "solo piano, no vocals": "quiet sparse",
    "warming up before going out": "loud fast bright",
    "distorted electric guitar": "loud bright",
    "background music while reading": "quiet sparse",
    "jersey club and dancey vibes": "fast loud",
    "girly pop songs to get ready to": "fast bright loud",
    "something for a rainy morning": "quiet slow",
    "playful and light-hearted": "bright fast",
    "something to fall asleep to": "quiet slow sparse",
}

# Fixed in the pre-registration, before the run.
EXPECTED_HELP = {
    "warming up before going out": True,
    "background music while reading": True,
    "something for a rainy morning": True,
    "something to fall asleep to": True,
    "playful and light-hearted": True,
    "distorted electric guitar": False,
    "orchestral strings": False,
    "jersey club and dancey vibes": False,
    "saxophone over an upright bass": False,
    "girly pop songs to get ready to": None,
    "solo piano, no vocals": None,
}


def waveform_features(samples: np.ndarray, sr: int) -> tuple[float, float, float]:
    rms = float(np.sqrt(np.mean(samples**2)))
    hop = max(1, sr // 100)
    n = (samples.size // hop) * hop
    if n == 0:
        return rms, 0.0, 0.0
    env = np.abs(samples[:n].reshape(-1, hop)).max(axis=1)
    d = np.diff(env)
    onset = float((d > d.std() * 1.5).mean()) if d.size else 0.0
    zcr = float(np.mean(np.abs(np.diff(np.sign(samples))) > 0))
    return rms, onset, zcr


def main() -> int:
    ap = argparse.ArgumentParser(description="E2b — planner on weak queries")
    ap.add_argument("root", type=Path)
    ap.add_argument("--slice1", type=Path,
                    default=EVALS / "eval_1_relevance_20260909.json")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--n-segments", type=int, default=5)
    args = ap.parse_args()

    slice1 = json.loads(args.slice1.read_text())
    weak = [r for r in slice1["per_query"] if r["mean_relevance"] < WEAK_THRESHOLD]
    weak.sort(key=lambda r: r["mean_relevance"])
    print(f"{len(weak)} weak queries (Slice 1 mean < {WEAK_THRESHOLD})", file=sys.stderr)

    from aux.encode.muq import MuQMuLanAdapter

    encoder = MuQMuLanAdapter()
    V, paths, _ = build_index(args.root, encoder, n_segments=args.n_segments,
                              cache_path=ROOT / ".cache" / "personal.npz")
    feats = []
    for path in paths:
        try:
            asset = decode(path)
            feats.append(waveform_features(asset.samples, asset.sample_rate))
        except (IngestError, Exception):  # noqa: BLE001
            feats.append((0.0, 0.0, 0.0))
    F = np.array(feats)
    Fz = (F - F.mean(0)) / F.std(0)

    planner = build_planner("claude")
    queries = [r["query"] for r in weak]
    plans = plan_all(planner, queries, ROOT / ".cache" / "plans.json")

    rows = []
    for r in weak:
        q = r["query"]
        base = V @ l2_normalise(encoder.embed_text([q]))[0]
        plan_scores = score_plan(encoder, plans[q], V)
        tb = np.argsort(-base)[: args.k]
        tp = np.argsort(-plan_scores)[: args.k]

        # Intent is judged against the direction the query *implies*, identically for both
        # systems. Scoring the planner against its own rewrite would grade it on its own
        # homework; scoring against the raw query is impossible, since these queries name no
        # measurable property -- which is why they are weak.
        implied = IMPLIED_INTENT.get(q)
        rows.append({
            "query": q, "slice1_mean": r["mean_relevance"], "category": r["category"],
            "expected_help": EXPECTED_HELP.get(q),
            "rewritten": plans[q].rewritten,
            "implied_intent": implied,
            "intent_base": intent_match(implied, Fz[tb]) if implied else None,
            "intent_plan": intent_match(implied, Fz[tp]) if implied else None,
            "ztop_base": float((base.max() - base.mean()) / base.std()),
            "ztop_plan": float((plan_scores.max() - plan_scores.mean()) / plan_scores.std()),
            "overlap": len(set(tb.tolist()) & set(tp.tolist())) / len(set(tb.tolist()) | set(tp.tolist())),
        })

    print(f"\n{'query':34}{'S1':>5}{'help?':>7}{'int-b':>7}{'int-p':>7}"
          f"{'delta':>8}{'z-b':>6}{'z-p':>6}{'ovl':>6}")
    print("-" * 86)
    for r in rows:
        ib = "  n/a" if r["intent_base"] is None else f"{r['intent_base']:>5.2f}"
        ip = "  n/a" if r["intent_plan"] is None else f"{r['intent_plan']:>5.2f}"
        dl = ("   n/a" if r["intent_base"] is None or r["intent_plan"] is None
              else f"{r['intent_plan'] - r['intent_base']:>+6.2f}")
        exp = {True: "yes", False: "no", None: "?"}[r["expected_help"]]
        print(f"{r['query'][:32]:34}{r['slice1_mean']:>5.1f}{exp:>7}{ib:>7}{ip:>7}{dl:>8}"
              f"{r['ztop_base']:>6.1f}{r['ztop_plan']:>6.1f}{r['overlap']:>6.2f}")

    def summarise(subset, label):
        pairs = [(r["intent_base"], r["intent_plan"]) for r in subset
                 if r["intent_base"] is not None and r["intent_plan"] is not None]
        if not pairs:
            print(f"{label:34}  no query names a measurable property")
            return None
        b = np.array([x[0] for x in pairs])
        p = np.array([x[1] for x in pairs])
        d = p - b
        lo, hi = bootstrap_ci(d.tolist()) if d.size > 1 else (float("nan"), float("nan"))
        print(f"{label:34}{len(pairs):>4}{b.mean():>8.2f}{p.mean():>8.2f}"
              f"{d.mean():>+8.2f}  [{lo:+.2f},{hi:+.2f}]  {int((d > 0).sum())}/{len(pairs)}")
        return {"n": len(pairs), "baseline": float(b.mean()), "planner": float(p.mean()),
                "delta": float(d.mean()), "ci": [lo, hi], "wins": int((d > 0).sum())}

    print(f"\n{'group':34}{'n':>4}{'base':>8}{'plan':>8}{'delta':>8}{'  95% CI':>16}{'wins':>7}")
    print("-" * 86)
    groups = {
        "predicted to help": summarise([r for r in rows if r["expected_help"] is True],
                                       "predicted to help"),
        "predicted NOT to help": summarise([r for r in rows if r["expected_help"] is False],
                                           "predicted NOT to help"),
        "all weak queries": summarise(rows, "all weak queries"),
    }

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    out = EVALS / f"eval_2b_weak_{stamp}.json"
    out.write_text(json.dumps({"run_at": datetime.now(timezone.utc).isoformat(),
                               "prereg": "docs/PREREG_E2B.md",
                               "encoder": encoder.version, "planner": planner.version,
                               "k": args.k, "groups": groups, "per_query": rows}, indent=2))
    print(f"\nwrote {out.relative_to(ROOT)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
