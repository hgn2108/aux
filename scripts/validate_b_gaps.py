"""Phase B — can the system tell when the library cannot answer?

Step 4 of DEC-020's design — say "you probably don't own anything like this" instead of
returning the least-bad five — has no evidence at all. This creates the evidence by
**making gaps on purpose**.

Remove every classical track from the index, then ask for orchestral strings. The library
provably cannot answer, and that label is certain rather than judged. Repeat across genres
and it becomes a detection problem with known ground truth: does confidence collapse when
the material is gone?

No human rating, no proxy for relevance — only a fact about what is in the index.

    python scripts/validate_b_gaps.py
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
from aux.eval import HIGHER_MEANS_MORE_CONFIDENT, MEASURES, all_measures  # noqa: E402
from aux.index import build_index  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PERSONAL = ROOT / "data" / "music"

# Queries that target one genre. Each is answerable with that genre present and not without.
GENRE_QUERIES = {
    "classical": ["orchestral strings", "solo piano", "a string quartet",
                  "classical orchestral music"],
    "jazz": ["saxophone over an upright bass", "acoustic jazz with upright bass",
             "a jazz trio", "swinging jazz drums"],
    "dnb": ["fast breakbeat drums", "drum and bass", "amen break jungle rhythm",
            "175 bpm breakbeats"],
    "edm": ["four on the floor dance music", "big room electronic dance music",
            "festival edm drop", "synth heavy dance track"],
    "hiphop_rnb": ["hip hop with heavy 808s", "rap over a trap beat",
                   "smooth r&b vocals", "boom bap drums"],
    # v-pop is included knowing the encoder is blind to language (Slice 1). If detection
    # fails only here, that is the known limitation reappearing, not a failure of the method.
    "vpop": ["vietnamese pop ballad", "pop sung in vietnamese"],
}


def genre_of(path: Path, root: Path) -> str:
    rel = path.relative_to(root)
    return rel.parts[0] if len(rel.parts) > 1 else "hiphop_rnb"


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase B — synthetic library gaps")
    ap.add_argument("--n-segments", type=int, default=5)
    args = ap.parse_args()

    from aux.encode.muq import MuQMuLanAdapter

    encoder = MuQMuLanAdapter()
    V, paths, _ = build_index(PERSONAL, encoder, n_segments=args.n_segments,
                              cache_path=ROOT / ".cache" / "personal.npz", progress_every=0)
    genres = np.array([genre_of(p, PERSONAL) for p in paths])

    queries = [q for qs in GENRE_QUERIES.values() for q in qs]
    T = l2_normalise(encoder.embed_text(queries))
    index_of = {q: i for i, q in enumerate(queries)}

    rows = []
    for genre, qs in GENRE_QUERIES.items():
        present = V
        removed = V[genres != genre]
        for q in qs:
            t = T[index_of[q]]
            with_ = all_measures(present @ t)
            without = all_measures(removed @ t)
            rows.append({"genre": genre, "query": q,
                         "n_removed": int((genres == genre).sum()),
                         "with": with_, "without": without,
                         "top_with": float((present @ t).max()),
                         "top_without": float((removed @ t).max())})

    print(f"removing each genre in turn from a {V.shape[0]}-track index\n")
    print(f"{'genre':11}{'query':34}{'top score':>20}{'z_top':>14}")
    print(f"{'':11}{'':34}{'present  absent':>20}{'present absent':>14}")
    print("-" * 80)
    for r in rows:
        print(f"{r['genre']:11}{r['query'][:32]:34}"
              f"{r['top_with']:>10.3f}{r['top_without']:>10.3f}"
              f"{r['with']['z_top']:>8.1f}{r['without']['z_top']:>7.1f}")

    print(f"\n{'measure':18}{'present':>10}{'absent':>10}{'delta':>10}"
          f"{'detects gap':>14}")
    print("-" * 62)
    summary = {}
    for m in MEASURES:
        a = np.array([r["with"][m] for r in rows])
        b = np.array([r["without"][m] for r in rows])
        higher_confident = HIGHER_MEANS_MORE_CONFIDENT[m]
        # Confidence should FALL when the material is removed.
        correct = int(((b < a) if higher_confident else (b > a)).sum())
        summary[m] = {"present": float(a.mean()), "absent": float(b.mean()),
                      "delta": float((b - a).mean()), "correct": correct, "n": len(rows)}
        print(f"{m:18}{a.mean():>10.2f}{b.mean():>10.2f}{(b - a).mean():>+10.2f}"
              f"{correct:>10}/{len(rows)}")

    tw = np.array([r["top_with"] for r in rows])
    tb = np.array([r["top_without"] for r in rows])
    print(f"{'raw top score':18}{tw.mean():>10.2f}{tb.mean():>10.2f}{(tb - tw).mean():>+10.2f}"
          f"{int((tb < tw).sum()):>10}/{len(rows)}")
    summary["raw_top_score"] = {"present": float(tw.mean()), "absent": float(tb.mean()),
                                "delta": float((tb - tw).mean()),
                                "correct": int((tb < tw).sum()), "n": len(rows)}

    out = ROOT / "evals" / "validate_b_gaps.json"
    out.write_text(json.dumps({"run_at": datetime.now(timezone.utc).isoformat(),
                               "encoder": encoder.version, "tracks": int(V.shape[0]),
                               "summary": summary, "per_query": rows}, indent=2))
    print(f"\nwrote {out.relative_to(ROOT)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
