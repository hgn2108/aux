"""Does split-and-penalise fix negation, and at what weight?

Ground truth for free: the library's genre folders label exactly what a query like
"music, not hip hop" should have excluded. No human rating is needed to measure whether an
exclusion worked, so this experiment is cheap enough to sweep.

Reports, for each negation weight:

- **leak rate** -- share of the top 5 belonging to the excluded genre. Lower is better;
  the weight-0 column is the current whole-query baseline.
- **base rate** -- the same share for the *positive part alone*, so a low leak rate is not
  credited when the genre was never going to appear anyway.
- **drift** -- overlap between the negated query's results and the positive-only results.
  A weight high enough to eliminate leakage but which also throws away everything the user
  asked *for* is not a fix, and this is what catches that.

    python scripts/eval_negation.py data/music
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
from aux.query import parse, score_query  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

# (query, genre that must NOT appear, genre that SHOULD appear or None)
#
# The positive target is what stops the weight being tuned into nonsense. Driving leakage to
# zero is trivial if you are willing to return anything at all -- the exclusion is satisfied
# and the user's actual request is discarded. Cases carrying an expected genre measure both
# halves at once.
CASES = [
    ("music, not hip hop", "hiphop_rnb", None),
    ("vocal music, not hip hop", "hiphop_rnb", None),
    ("music, no classical", "classical", None),
    ("calm music, no classical", "classical", None),
    ("energetic music, not jazz", "jazz", None),
    ("instrumental music, not jazz", "jazz", None),
    ("electronic music, not drum and bass", "dnb", None),
    ("dance music, no edm", "edm", None),
    # Both halves checked: exclude one genre, still return the right one.
    ("piano music, no vocals", "hiphop_rnb", "classical"),
    ("orchestral music, not electronic", "edm", "classical"),
    ("saxophone and upright bass, no rapping", "hiphop_rnb", "jazz"),
    ("fast breakbeat drums, not hip hop", "hiphop_rnb", "dnb"),
]
WEIGHTS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
K = 5


def genre_of(path: Path, root: Path) -> str:
    rel = path.relative_to(root)
    return rel.parts[0] if len(rel.parts) > 1 else "hiphop_rnb"


def main() -> int:
    ap = argparse.ArgumentParser(description="Negation weight sweep")
    ap.add_argument("root", type=Path)
    ap.add_argument("--n-segments", type=int, default=5)
    args = ap.parse_args()

    from aux.encode.muq import MuQMuLanAdapter

    encoder = MuQMuLanAdapter()
    paths, vectors = [], []
    for path in [f.path for f in discover(args.root)]:
        try:
            vec, _ = encoder.embed_track(decode(path), n_segments=args.n_segments)
        except (IngestError, Exception):  # noqa: BLE001
            continue
        paths.append(path)
        vectors.append(vec)
    V = np.stack(vectors)
    genres = np.array([genre_of(p, args.root) for p in paths])
    print(f"{V.shape[0]} tracks indexed", file=sys.stderr)

    results = {}
    print(f"{'weight':>8}{'leak@5':>9}{'base':>7}{'hit@5':>8}{'margin':>9}")
    print("-" * 41)
    for w in WEIGHTS:
        leaks, bases, hits = [], [], []
        for query, excluded, expected in CASES:
            scores, parsed = score_query(encoder, query, V, negation_weight=w)
            top = np.argsort(-scores)[:K]
            leaks.append(float((genres[top] == excluded).mean()))

            pos_only = V @ l2_normalise(encoder.embed_text([parsed.positive]))[0]
            bases.append(float((genres[np.argsort(-pos_only)[:K]] == excluded).mean()))
            if expected is not None:
                hits.append(float((genres[top] == expected).mean()))
        leak = float(np.mean(leaks))
        hit = float(np.mean(hits)) if hits else float("nan")
        results[w] = {"leak": leak, "base": float(np.mean(bases)), "hit": hit,
                      "margin": hit - leak}
        print(f"{w:>8.2f}{leak:>9.2f}{results[w]['base']:>7.2f}{hit:>8.2f}"
              f"{results[w]['margin']:>9.2f}")

    print("\n=== per case at the chosen weight ===")
    # Chosen on the margin between satisfying the request and honouring the exclusion, not
    # on leakage alone: a weight that returns nothing relevant also leaks nothing.
    best = max((w for w in WEIGHTS if w > 0), key=lambda w: results[w]["margin"])
    per_case = []
    for query, excluded, expected in CASES:
        s0, _ = score_query(encoder, query, V, negation_weight=0.0)
        sb, parsed = score_query(encoder, query, V, negation_weight=best)
        t0 = np.argsort(-s0)[:K]
        tb = np.argsort(-sb)[:K]
        row = {"query": query, "excluded": excluded, "expected": expected,
               "leak_baseline": float((genres[t0] == excluded).mean()),
               "leak_fixed": float((genres[tb] == excluded).mean()),
               "positive": parsed.positive, "negatives": list(parsed.negatives)}
        per_case.append(row)
        print(f"  {row['leak_baseline']:.1f} -> {row['leak_fixed']:.1f}   {query}")

    print("\n=== minimal pair check ===")
    for q in ["solo piano", "solo piano, no vocals", "acoustic guitar and soft vocals",
              "acoustic guitar, no vocals"]:
        s, _ = score_query(encoder, q, V, negation_weight=best)
        top = np.argsort(-s)[:K]
        print(f"  {q:34}{', '.join(genres[top])}")

    out = ROOT / "evals" / "negation_sweep.json"
    out.write_text(json.dumps({"encoder": encoder.version, "k": K,
                               "chosen_weight": best,
                               "sweep": {str(k): v for k, v in results.items()},
                               "per_case": per_case}, indent=2))
    print(f"\nchosen weight: {best}\nwrote {out.relative_to(ROOT)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
