"""Does the context phrase in a query do anything?

The Slice 1 query set contains four queries that differ only in context: "hip hop for
running", "hip hop for studying", "hip hop for falling asleep", and bare "hip hop". If they
retrieve substantially the same tracks, the context phrase is decoration and the genre word
is doing all the work -- which is the precondition DEC-013 is gated on.

Three measurements, from most to least direct:

1. **query-embedding cosine** -- do the *query vectors* even differ? If "for running" barely
   moves the text embedding, nothing downstream can recover it. This is upstream of
   retrieval entirely and is the cleanest signal.
2. **top-K Jaccard** -- do the returned sets differ, which is what a user sees.
3. **Spearman correlation over the full ranking** -- do the orderings differ, including
   below the cut.

All three need a calibration reference, because "0.9 cosine" means nothing in isolation.
Pairs that *should* be far apart ("solo piano" vs "hip hop") give the scale.

    python scripts/overlap_control.py data/music
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aux.encode.base import l2_normalise  # noqa: E402
from aux.ingest import IngestError, decode, discover  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

CONTEXT_SET = [
    "hip hop",
    "hip hop for running",
    "hip hop for studying",
    "hip hop for falling asleep",
]
IRENE_CONTEXT_SET = [
    "fast, upbeat rap and hip hop for running",
    "chill r&b and hip hop for studying",
    "pre-game and club dance hip hop",
    "west coast hip hop drive",
]
CALIBRATION_SET = [
    "hip hop",
    "solo piano",
    "orchestral strings",
    "fast breakbeat drums",
]


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = np.argsort(np.argsort(-a)).astype(float)
    rb = np.argsort(np.argsort(-b)).astype(float)
    ra -= ra.mean()
    rb -= rb.mean()
    return float((ra @ rb) / (np.linalg.norm(ra) * np.linalg.norm(rb)))


def report(name: str, queries: list[str], T: np.ndarray, scores: np.ndarray, k: int) -> dict:
    print(f"\n=== {name} ===")
    print(f"{'pair':62}{'q-cos':>8}{'J@'+str(k):>7}{'rho':>7}")
    print("-" * 84)
    rows = []
    for i, j in combinations(range(len(queries)), 2):
        top_i = set(np.argsort(-scores[i])[:k].tolist())
        top_j = set(np.argsort(-scores[j])[:k].tolist())
        jac = len(top_i & top_j) / len(top_i | top_j)
        rho = spearman(scores[i], scores[j])
        qcos = float(T[i] @ T[j])
        label = f"{queries[i][:28]} | {queries[j][:28]}"
        print(f"{label:62}{qcos:>8.3f}{jac:>7.2f}{rho:>7.3f}")
        rows.append({"a": queries[i], "b": queries[j], "query_cosine": qcos,
                     "jaccard": jac, "spearman": rho})
    return {"name": name, "pairs": rows,
            "mean_query_cosine": float(np.mean([r["query_cosine"] for r in rows])),
            "mean_jaccard": float(np.mean([r["jaccard"] for r in rows])),
            "mean_spearman": float(np.mean([r["spearman"] for r in rows]))}


def main() -> int:
    ap = argparse.ArgumentParser(description="Context-phrase contribution control")
    ap.add_argument("root", type=Path)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--n-segments", type=int, default=5)
    args = ap.parse_args()

    from aux.encode.muq import MuQMuLanAdapter

    encoder = MuQMuLanAdapter()
    paths = [f.path for f in discover(args.root)]
    vectors = []
    for path in paths:
        try:
            vec, _ = encoder.embed_track(decode(path), n_segments=args.n_segments)
        except (IngestError, Exception):  # noqa: BLE001
            continue
        vectors.append(vec)
    V = np.stack(vectors)
    print(f"{V.shape[0]} tracks indexed", file=sys.stderr)

    out = {}
    for name, qs in (("CONTEXT (minimal pairs)", CONTEXT_SET),
                     ("CONTEXT (Irene's phrasing)", IRENE_CONTEXT_SET),
                     ("CALIBRATION (should differ)", CALIBRATION_SET)):
        T = l2_normalise(encoder.embed_text(qs))
        out[name] = report(name, qs, T, T @ V.T, args.k)

    ctx = out["CONTEXT (minimal pairs)"]
    cal = out["CALIBRATION (should differ)"]
    print("\n=== VERDICT ===")
    print(f"context pairs : query-cos {ctx['mean_query_cosine']:.3f}  "
          f"J@{args.k} {ctx['mean_jaccard']:.2f}  rho {ctx['mean_spearman']:.3f}")
    print(f"calibration   : query-cos {cal['mean_query_cosine']:.3f}  "
          f"J@{args.k} {cal['mean_jaccard']:.2f}  rho {cal['mean_spearman']:.3f}")
    verdict = ("context contributes little — DEC-013 precondition holds"
               if ctx["mean_jaccard"] > 0.5 and ctx["mean_spearman"] > 0.7
               else "context already changes retrieval — revisit DEC-013")
    print(f"verdict: {verdict}")

    path = ROOT / "evals" / "overlap_control.json"
    path.write_text(json.dumps({"encoder": encoder.version, "k": args.k,
                                "tracks": int(V.shape[0]), "groups": out,
                                "verdict": verdict}, indent=2))
    print(f"\nwrote {path.relative_to(ROOT)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# --- acoustic-direction check -------------------------------------------------------
# Appended after the first run: low top-K overlap with high rank correlation has two very
# different explanations. Either the context phrase genuinely redirects retrieval, or the
# score distribution is flat near the top and a small shift reshuffles a noisy head.
#
# They separate on a question that needs no human ratings: do the retrieved tracks differ
# in the direction the context word implies? "For running" should return louder, denser,
# faster material than "for falling asleep". If the two sets are acoustically
# indistinguishable, the reshuffling is noise however sensible the rank gradient looks.
