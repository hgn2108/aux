"""Eval 2A + E2 — does the LLM planner earn its place?

Two evaluations, run in this order on purpose. The first is cheap and can disqualify the
planner before any human rating time is spent on the second.

**Eval 2A — planner output correctness.** Schema conformance, fallback rate, latency, tokens
and cost. A planner that produces good rewrites but falls back on a fifth of queries is not
shippable regardless of how the surviving rewrites score.

**E2 (objective half) — does retrieval change, and in the right direction?** The same two
metrics that rejected the fixed lexicon, so the two rungs are directly comparable:

- **genre fidelity** — does a named genre still dominate results? The lexicon failed here,
  falling 0.80 to 0.46 as it diluted the genre away.
- **context separation** — how far apart opposed contexts pull retrieval, measured on
  waveform features the encoder never sees.

Neither needs ratings. Human comparison follows only for a planner that survives both.

    python scripts/eval_2_planner.py data/music --planner claude
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
from aux.plan import build_planner  # noqa: E402
from aux.query import score_plan  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
EVALS = ROOT / "evals"

# USD per million tokens. Recorded so cost is a measured line rather than an impression;
# update if pricing changes.
PRICING = {
    "claude-haiku-4-5-20251001": (1.00, 5.00),
    "claude-sonnet-5": (3.00, 15.00),
}

OPPOSED_PAIRS = [
    ("hip hop for running", "hip hop for falling asleep"),
    ("hip hop for a workout", "hip hop to relax to"),
    ("music for a party", "music for reading"),
    ("music for the gym", "music for sleeping"),
]
GENRE_ANCHORED = [
    ("hip hop for running", "hiphop_rnb"),
    ("hip hop for studying", "hiphop_rnb"),
    ("jazz for reading", "jazz"),
    ("classical for sleeping", "classical"),
    ("edm for a party", "edm"),
]
K = 10


def load_queries(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) == 3:
            rows.append({"category": parts[0], "source": parts[1], "query": parts[2]})
    return rows


def waveform_onset(samples: np.ndarray, sr: int) -> float:
    hop = max(1, sr // 100)
    n = (samples.size // hop) * hop
    if n == 0:
        return 0.0
    env = np.abs(samples[:n].reshape(-1, hop)).max(axis=1)
    d = np.diff(env)
    return float((d > d.std() * 1.5).mean()) if d.size else 0.0


def genre_of(path: Path, root: Path) -> str:
    rel = path.relative_to(root)
    return rel.parts[0] if len(rel.parts) > 1 else "hiphop_rnb"


def main() -> int:
    ap = argparse.ArgumentParser(description="Eval 2A + E2 objective half")
    ap.add_argument("root", type=Path)
    ap.add_argument("--planner", default="claude")
    ap.add_argument("--model", default=None)
    ap.add_argument("--queries", type=Path, default=ROOT / "queries" / "slice1.tsv")
    ap.add_argument("--n-segments", type=int, default=5)
    args = ap.parse_args()

    from aux.encode.muq import MuQMuLanAdapter

    encoder = MuQMuLanAdapter()
    paths, vectors, onsets = [], [], []
    for path in [f.path for f in discover(args.root)]:
        try:
            asset = decode(path)
            vec, _ = encoder.embed_track(asset, n_segments=args.n_segments)
        except (IngestError, Exception):  # noqa: BLE001
            continue
        paths.append(path)
        vectors.append(vec)
        onsets.append(waveform_onset(asset.samples, asset.sample_rate))
    V = np.stack(vectors)
    onset_z = (np.array(onsets) - np.mean(onsets)) / np.std(onsets)
    genres = np.array([genre_of(p, args.root) for p in paths])
    print(f"{V.shape[0]} tracks indexed", file=sys.stderr)

    planner = build_planner(args.planner, args.model)
    queries = load_queries(args.queries)
    extra = [q for pair in OPPOSED_PAIRS for q in pair] + [q for q, _ in GENRE_ANCHORED]
    all_queries = list(dict.fromkeys([q["query"] for q in queries] + extra))

    plans, metas = {}, {}
    for i, q in enumerate(all_queries, 1):
        plans[q], metas[q] = planner.plan(q)
        if i % 10 == 0:
            print(f"  planned {i}/{len(all_queries)}", file=sys.stderr)

    # --- Eval 2A ---
    lat = np.array([m["latency_ms"] for m in metas.values() if "latency_ms" in m])
    tin = sum(m.get("input_tokens") or 0 for m in metas.values())
    tout = sum(m.get("output_tokens") or 0 for m in metas.values())
    fallbacks = sum(m["fallback"] for m in metas.values())
    retried = sum(m["attempts"] > 1 and not m["fallback"] for m in metas.values())
    price_in, price_out = PRICING.get(planner.version, (0.0, 0.0))
    cost = tin / 1e6 * price_in + tout / 1e6 * price_out

    print(f"\n=== Eval 2A — planner output ({planner.name} / {planner.version}) ===")
    print(f"  queries planned      {len(all_queries)}")
    print(f"  fallback rate        {fallbacks / len(all_queries):.1%} ({fallbacks})")
    print(f"  recovered on retry   {retried}")
    print(f"  latency p50 / p95    {np.percentile(lat, 50):.0f} / {np.percentile(lat, 95):.0f} ms")
    print(f"  tokens in / out      {tin} / {tout}")
    print(f"  cost for this run    ${cost:.4f}  (${cost / len(all_queries) * 1000:.2f} per 1k queries)")
    got = lambda f: sum(1 for p in plans.values() if getattr(p, f))  # noqa: E731
    print("  facets populated     " + ", ".join(
        f"{f} {got(f)}/{len(plans)}" for f in ("acoustic", "mood", "genre", "context", "exclude")))

    # --- E2 objective half ---
    def scores_for(q, use_plan: bool) -> np.ndarray:
        if use_plan:
            return score_plan(encoder, plans[q], V)
        return V @ l2_normalise(encoder.embed_text([q]))[0]

    rows = {}
    for label, use_plan in (("baseline (Slice 1)", False), ("planner", True)):
        seps = []
        for hi, lo in OPPOSED_PAIRS:
            t_hi = np.argsort(-scores_for(hi, use_plan))[:K]
            t_lo = np.argsort(-scores_for(lo, use_plan))[:K]
            seps.append(float(onset_z[t_hi].mean() - onset_z[t_lo].mean()))
        fid = [float((genres[np.argsort(-scores_for(q, use_plan))[:K]] == g).mean())
               for q, g in GENRE_ANCHORED]
        rows[label] = {"separation": float(np.mean(seps)),
                       "genre_fidelity": float(np.mean(fid)), "per_pair": seps}

    # How much does the planner change results at all?
    jac = []
    for q in [x["query"] for x in queries]:
        a = set(np.argsort(-scores_for(q, False))[:K].tolist())
        b = set(np.argsort(-scores_for(q, True))[:K].tolist())
        jac.append(len(a & b) / len(a | b))

    print("\n=== E2 objective half ===")
    print(f"{'system':22}{'separation':>12}{'genre fidelity':>16}")
    print("-" * 50)
    for label, r in rows.items():
        print(f"{label:22}{r['separation']:>12.2f}{r['genre_fidelity']:>16.2f}")
    print(f"{'rung 1 lexicon (w=.45)':22}{0.66:>12.2f}{0.68:>16.2f}   (DEC-016, rejected)")
    print(f"\n  top-{K} overlap with baseline: {np.mean(jac):.2f} "
          f"(1.00 would mean the planner changed nothing)")

    print("\n=== sample rewrites ===")
    for q in ["background music while reading", "hip hop for falling asleep",
              "girly pop songs to get ready to", "solo piano, no vocals"]:
        if q in plans:
            p = plans[q]
            print(f"  {q}\n    -> {p.rewritten}")
            if p.exclude:
                print(f"       exclude: {list(p.exclude)}")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    out = EVALS / f"eval_2_planner_{planner.name}_{stamp}.json"
    out.write_text(json.dumps({
        "run_at": datetime.now(timezone.utc).isoformat(),
        "planner": {"name": planner.name, "version": planner.version},
        "encoder": encoder.version,
        "eval_2a": {"queries": len(all_queries), "fallbacks": fallbacks,
                    "fallback_rate": fallbacks / len(all_queries), "retried": retried,
                    "latency_p50": float(np.percentile(lat, 50)),
                    "latency_p95": float(np.percentile(lat, 95)),
                    "tokens_in": tin, "tokens_out": tout, "cost_usd": cost},
        "e2_objective": rows,
        "overlap_with_baseline": float(np.mean(jac)),
        "plans": {q: plans[q].as_dict() for q in all_queries},
    }, indent=2))
    print(f"\nwrote {out.relative_to(ROOT)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
