"""Is the library deep enough to measure the planner, or is depth the limiter?

E2's objective metrics compared the top 10 of 160 tracks — 6% of the library. If a query has
only a handful of genuinely good matches, its top 10 must include tracks that do not match,
which caps any measurable separation no matter how good the planner is.

Two checks:

1. **K sweep.** If the planner's separation improves as K shrinks, the limiter is library
   depth rather than the planner: the good matches exist but run out.
2. **Per-query depth.** How many tracks score near the top for each query. A query whose
   scores fall away immediately has few real matches, and any metric over its top 10 is
   mostly measuring filler.

    python scripts/diagnose_depth.py data/music
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aux.encode.base import l2_normalise  # noqa: E402
from aux.ingest import IngestError, decode, discover  # noqa: E402
from aux.query import score_plan  # noqa: E402
from aux.plan.schema import validate  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

PAIRS = [
    ("hip hop for running", "hip hop for falling asleep"),
    ("hip hop for a workout", "hip hop to relax to"),
    ("music for a party", "music for reading"),
    ("music for the gym", "music for sleeping"),
]


def onset(samples: np.ndarray, sr: int) -> float:
    hop = max(1, sr // 100)
    n = (samples.size // hop) * hop
    if n == 0:
        return 0.0
    env = np.abs(samples[:n].reshape(-1, hop)).max(axis=1)
    d = np.diff(env)
    return float((d > d.std() * 1.5).mean()) if d.size else 0.0


def main() -> int:
    ap = argparse.ArgumentParser(description="Library depth diagnostic")
    ap.add_argument("root", type=Path)
    ap.add_argument("--plans", type=Path,
                    default=ROOT / "evals" / "eval_2_planner_claude_20260909.json")
    ap.add_argument("--n-segments", type=int, default=5)
    args = ap.parse_args()

    from aux.encode.muq import MuQMuLanAdapter

    encoder = MuQMuLanAdapter()
    vectors, onsets = [], []
    for path in [f.path for f in discover(args.root)]:
        try:
            asset = decode(path)
            vec, _ = encoder.embed_track(asset, n_segments=args.n_segments)
        except (IngestError, Exception):  # noqa: BLE001
            continue
        vectors.append(vec)
        onsets.append(onset(asset.samples, asset.sample_rate))
    V = np.stack(vectors)
    oz = (np.array(onsets) - np.mean(onsets)) / np.std(onsets)
    plans = json.loads(args.plans.read_text())["plans"]
    print(f"{V.shape[0]} tracks", file=sys.stderr)

    def base_scores(q):
        return V @ l2_normalise(encoder.embed_text([q]))[0]

    def plan_scores(q):
        return score_plan(encoder, validate(plans[q], q), V)

    print(f"\n{'K':>4}{'K as % of lib':>15}{'baseline sep':>15}{'planner sep':>14}{'delta':>9}")
    print("-" * 57)
    rows = {}
    for k in (1, 3, 5, 10, 20, 40):
        b, p = [], []
        for hi, lo in PAIRS:
            b.append(float(oz[np.argsort(-base_scores(hi))[:k]].mean()
                           - oz[np.argsort(-base_scores(lo))[:k]].mean()))
            p.append(float(oz[np.argsort(-plan_scores(hi))[:k]].mean()
                           - oz[np.argsort(-plan_scores(lo))[:k]].mean()))
        rows[k] = {"baseline": float(np.mean(b)), "planner": float(np.mean(p))}
        rows[k]["delta"] = rows[k]["planner"] - rows[k]["baseline"]
        print(f"{k:>4}{100 * k / V.shape[0]:>14.1f}%{rows[k]['baseline']:>15.2f}"
              f"{rows[k]['planner']:>14.2f}{rows[k]['delta']:>+9.2f}")

    print(f"\n=== how many real matches does each query have? ===")
    print(f"{'query':40}{'top1':>7}{'>=90% of top1':>15}{'>=80%':>8}")
    print("-" * 70)
    depth = {}
    for hi, lo in PAIRS:
        for q in (hi, lo):
            s = plan_scores(q)
            top = s.max()
            n90 = int((s >= top * 0.9).sum()) if top > 0 else 0
            n80 = int((s >= top * 0.8).sum()) if top > 0 else 0
            depth[q] = {"top": float(top), "n90": n90, "n80": n80}
            print(f"{q[:38]:40}{top:>7.3f}{n90:>15}{n80:>8}")

    out = ROOT / "evals" / "depth_diagnostic.json"
    out.write_text(json.dumps({"tracks": int(V.shape[0]), "k_sweep": rows,
                               "per_query_depth": depth}, indent=2))
    print(f"\nwrote {out.relative_to(ROOT)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
