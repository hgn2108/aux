"""Re-run E2's context separation on a corpus deep enough to measure it.

`scripts/diagnose_depth.py` found that most context queries have one or two genuinely good
matches in the 160-track personal library. A top-K metric over that is comparing one real
match against filler, which is why the baseline's separation swung between -0.04 and 1.01
depending on K.

**The product question and the mechanism question need different corpora.** Whether aux
works on Irene's library was Slice 0's Eval 0C, answered on her library, and it passed.
Whether *query rewriting improves retrieval* is a claim about a mechanism, and measuring it
needs enough candidates that the top-K is not exhausting the collection.

FMA small is already on disk, already validated by Eval 0A at 99.92% decode, and is
genre-diverse public audio — so this costs indexing time and nothing else. Personal audio
stays what PROJECT.md says it is: an out-of-domain test, never a development corpus.

Only the onset-based separation metric runs here; genre fidelity needs labels the folder
layout does not provide for FMA.

    python scripts/eval_2_depth_corpus.py --limit 1500
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
from aux.ingest import IngestError, decode, discover  # noqa: E402
from aux.plan.schema import validate  # noqa: E402
from aux.query import score_plan  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

PAIRS = [
    ("hip hop for running", "hip hop for falling asleep"),
    ("hip hop for a workout", "hip hop to relax to"),
    ("music for a party", "music for reading"),
    ("music for the gym", "music for sleeping"),
    ("energetic music", "calm music"),
    ("music for dancing", "music for meditation"),
]
"""Two extra pairs beyond the personal-library set: with a deep corpus the metric can
support more pairs, and four was too few to distinguish signal from noise."""


def onset(samples: np.ndarray, sr: int) -> float:
    hop = max(1, sr // 100)
    n = (samples.size // hop) * hop
    if n == 0:
        return 0.0
    env = np.abs(samples[:n].reshape(-1, hop)).max(axis=1)
    d = np.diff(env)
    return float((d > d.std() * 1.5).mean()) if d.size else 0.0


def main() -> int:
    ap = argparse.ArgumentParser(description="E2 separation on a deep corpus")
    ap.add_argument("--root", type=Path, default=ROOT / "data" / "raw" / "fma" / "fma_small")
    ap.add_argument("--limit", type=int, default=1500)
    ap.add_argument("--n-segments", type=int, default=5)
    ap.add_argument("--planner", default="claude")
    args = ap.parse_args()

    from aux.encode.muq import MuQMuLanAdapter
    from aux.plan import build_planner

    encoder = MuQMuLanAdapter()
    paths = [f.path for f in discover(args.root)][: args.limit]
    vectors, onsets = [], []
    for i, path in enumerate(paths, 1):
        try:
            asset = decode(path)
            vec, _ = encoder.embed_track(asset, n_segments=args.n_segments)
        except (IngestError, Exception):  # noqa: BLE001
            continue
        vectors.append(vec)
        onsets.append(onset(asset.samples, asset.sample_rate))
        if i % 200 == 0:
            print(f"  {i}/{len(paths)}", file=sys.stderr)
    V = np.stack(vectors)
    oz = (np.array(onsets) - np.mean(onsets)) / np.std(onsets)
    print(f"{V.shape[0]} tracks indexed", file=sys.stderr)

    planner = build_planner(args.planner)
    queries = [q for pair in PAIRS for q in pair]
    plans = {q: planner.plan(q)[0] for q in queries}

    def base(q):
        return V @ l2_normalise(encoder.embed_text([q]))[0]

    def plan_(q):
        return score_plan(encoder, plans[q], V)

    print(f"\n{'K':>5}{'K as % of lib':>15}{'baseline':>11}{'planner':>10}{'delta':>9}")
    print("-" * 50)
    sweep = {}
    for k in (5, 10, 25, 50, 100):
        b = [float(oz[np.argsort(-base(hi))[:k]].mean() - oz[np.argsort(-base(lo))[:k]].mean())
             for hi, lo in PAIRS]
        p = [float(oz[np.argsort(-plan_(hi))[:k]].mean() - oz[np.argsort(-plan_(lo))[:k]].mean())
             for hi, lo in PAIRS]
        sweep[k] = {"baseline": float(np.mean(b)), "planner": float(np.mean(p)),
                    "baseline_per_pair": b, "planner_per_pair": p}
        sweep[k]["delta"] = sweep[k]["planner"] - sweep[k]["baseline"]
        print(f"{k:>5}{100 * k / V.shape[0]:>14.1f}%{sweep[k]['baseline']:>11.2f}"
              f"{sweep[k]['planner']:>10.2f}{sweep[k]['delta']:>+9.2f}")

    k = 25
    print(f"\n=== per pair at K={k} ===")
    print(f"{'pair':50}{'base':>8}{'planner':>10}{'delta':>9}")
    print("-" * 77)
    wins = 0
    for i, (hi, lo) in enumerate(PAIRS):
        b, p = sweep[k]["baseline_per_pair"][i], sweep[k]["planner_per_pair"][i]
        wins += p > b
        print(f"{hi[:22]} vs {lo[:22]:26}{b:>8.2f}{p:>10.2f}{p - b:>+9.2f}")
    print(f"\nplanner wins {wins}/{len(PAIRS)} pairs at K={k}")

    out = ROOT / "evals" / "eval_2_deep_corpus.json"
    out.write_text(json.dumps({"run_at": datetime.now(timezone.utc).isoformat(),
                               "corpus": str(args.root), "tracks": int(V.shape[0]),
                               "planner": planner.version, "pairs": PAIRS,
                               "sweep": {str(k): v for k, v in sweep.items()}}, indent=2))
    print(f"\nwrote {out.relative_to(ROOT)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
