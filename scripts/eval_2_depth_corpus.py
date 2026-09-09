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
from aux.index import build_index  # noqa: E402
from aux.ingest import IngestError, decode  # noqa: E402
from aux.plan.schema import validate  # noqa: E402
from aux.query import score_plan  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

PAIRS = [
    # Genre-anchored context, the form Irene actually writes.
    ("hip hop for running", "hip hop for falling asleep"),
    ("hip hop for a workout", "hip hop to relax to"),
    ("rock for a workout", "rock for winding down"),
    ("electronic music for running", "electronic music for sleeping"),
    ("jazz for a party", "jazz for studying"),
    ("pop for dancing", "pop for reading"),
    # Context with no genre anchor — Slice 1's weakest category.
    ("music for a party", "music for reading"),
    ("music for the gym", "music for sleeping"),
    ("music for running", "music for meditation"),
    ("music for a night out", "music for a quiet evening"),
    ("music for cleaning the house", "music for falling asleep"),
    ("music to wake up to", "music to fall asleep to"),
    # Pure mood and energy, no context at all — the control.
    ("energetic music", "calm music"),
    ("aggressive music", "gentle music"),
    ("fast music", "slow music"),
    ("loud intense music", "quiet subdued music"),
    ("upbeat cheerful music", "sombre reflective music"),
    ("dense busy production", "sparse minimal production"),
]
"""Eighteen pairs, up from six.

The first run could not adjudicate: per-pair deltas swung between -0.52 and +0.37 with a
mean of -0.01, so the standard error over six pairs swamped the difference. Deepening the
corpus fixed a different problem than the binding one.

Grouped in three, because the six-pair run hinted that the two losses were both the
genre-less "music for X" form. Eighteen pairs can test that rather than hint at it."""

PAIR_GROUPS = {"genre-anchored context": slice(0, 6),
               "context, no genre": slice(6, 12),
               "mood only (control)": slice(12, 18)}


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
    V, paths, _ = build_index(args.root, encoder, n_segments=args.n_segments,
                              cache_path=ROOT / ".cache" / "fma_small.npz",
                              limit=args.limit)
    # Waveform features are cheap relative to encoding, but still worth not recomputing.
    onsets = []
    for path in paths:
        try:
            asset = decode(path)
            onsets.append(onset(asset.samples, asset.sample_rate))
        except (IngestError, Exception):  # noqa: BLE001
            onsets.append(0.0)
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
    print(f"\n=== by pair group at K={k} ===")
    print(f"{'group':26}{'n':>4}{'base':>8}{'planner':>10}{'delta':>9}{'se':>7}{'wins':>7}")
    print("-" * 71)
    groups = {}
    for name, sl in PAIR_GROUPS.items():
        b = np.array(sweep[k]["baseline_per_pair"][sl])
        p = np.array(sweep[k]["planner_per_pair"][sl])
        d = p - b
        se = float(d.std(ddof=1) / np.sqrt(d.size)) if d.size > 1 else float("nan")
        groups[name] = {"n": int(d.size), "baseline": float(b.mean()),
                        "planner": float(p.mean()), "delta": float(d.mean()),
                        "stderr": se, "wins": int((d > 0).sum())}
        print(f"{name:26}{d.size:>4}{b.mean():>8.2f}{p.mean():>10.2f}"
              f"{d.mean():>+9.2f}{se:>7.2f}{groups[name]['wins']:>4}/{d.size}")
    allb = np.array(sweep[k]["baseline_per_pair"])
    allp = np.array(sweep[k]["planner_per_pair"])
    alld = allp - allb
    se = float(alld.std(ddof=1) / np.sqrt(alld.size))
    print("-" * 71)
    print(f"{'all pairs':26}{alld.size:>4}{allb.mean():>8.2f}{allp.mean():>10.2f}"
          f"{alld.mean():>+9.2f}{se:>7.2f}{int((alld > 0).sum()):>4}/{alld.size}")
    print(f"\n95% CI on the overall delta: "
          f"[{alld.mean() - 1.96 * se:+.2f}, {alld.mean() + 1.96 * se:+.2f}]")

    out = ROOT / "evals" / "eval_2_deep_corpus.json"
    out.write_text(json.dumps({"run_at": datetime.now(timezone.utc).isoformat(),
                               "corpus": str(args.root), "tracks": int(V.shape[0]),
                               "planner": planner.version, "pairs": PAIRS,
                               "groups_at_k25": groups,
                               "sweep": {str(k): v for k, v in sweep.items()}}, indent=2))
    print(f"\nwrote {out.relative_to(ROOT)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
