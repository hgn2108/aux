"""Probe how the index responds to a set of queries.

Diagnostic, not evaluation: it reports *how the score distribution behaves* per query, not
whether results are good. Used to tell two very different failures apart before Slice 1
spends human rating effort on them:

- **library gap** -- the query is answerable in principle but nothing here matches, so
  scores are uniformly low and the top result is barely above the rest;
- **encoder weakness** -- the encoder has no strong representation for that kind of
  language, so scores are low *and* flat, and the ranking barely separates from the corpus
  mean.

The distinguishing signal is spread relative to the corpus, not absolute score. A query
that finds nothing still ranks something first; a query the encoder cannot represent ranks
almost everything equally.

    python scripts/probe_queries.py data/music --queries queries/probe.txt
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


def genre_of(path: Path, root: Path) -> str:
    rel = path.relative_to(root)
    return rel.parts[0] if len(rel.parts) > 1 else "hiphop_rnb"


def build_encoder(name: str, checkpoint: str | None):
    if name in {"muq", "muq-mulan"}:
        from aux.encode.muq import DEFAULT_CHECKPOINT, MuQMuLanAdapter

        return MuQMuLanAdapter(checkpoint or DEFAULT_CHECKPOINT)
    from aux.encode.clap import DEFAULT_CHECKPOINT, ClapAdapter

    return ClapAdapter(checkpoint or DEFAULT_CHECKPOINT)


def main() -> int:
    ap = argparse.ArgumentParser(description="Probe query response over a library")
    ap.add_argument("root", type=Path)
    ap.add_argument("--queries", type=Path, required=True)
    ap.add_argument("--encoder", default="muq")
    ap.add_argument("--n-segments", type=int, default=5)
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()

    queries = [q.strip() for q in args.queries.read_text().splitlines()
               if q.strip() and not q.startswith("#")]
    encoder = build_encoder(args.encoder, None)
    paths = [f.path for f in discover(args.root)]

    vectors, kept = [], []
    for path in paths:
        try:
            vec, _ = encoder.embed_track(decode(path), n_segments=args.n_segments)
        except (IngestError, Exception):  # noqa: BLE001
            continue
        vectors.append(vec)
        kept.append(path)
    V = np.stack(vectors)
    genres = np.array([genre_of(p, args.root) for p in kept])

    T = l2_normalise(encoder.embed_text(queries))
    scores = T @ V.T

    rows = []
    print(f"{'query':52}{'top':>7}{'mean':>7}{'sd':>6}{'z-top':>7}{'top genres'}")
    print("-" * 108)
    for i, q in enumerate(queries):
        s = scores[i]
        top_idx = np.argsort(-s)[: args.k]
        # z of the best hit against this query's own score distribution: how far the winner
        # stands out from the corpus, which is scale-free across queries.
        z = (s.max() - s.mean()) / (s.std() or 1e-9)
        top_genres = ", ".join(f"{g}" for g in genres[top_idx][:3])
        rows.append({"query": q, "top": float(s.max()), "mean": float(s.mean()),
                     "sd": float(s.std()), "z_top": float(z),
                     "top_genres": genres[top_idx].tolist()})
        print(f"{q[:50]:52}{s.max():>7.3f}{s.mean():>7.3f}{s.std():>6.3f}{z:>7.2f}  {top_genres}")

    out = Path("evals") / "probe_query_response.json"
    out.write_text(json.dumps({"encoder": encoder.version, "n_segments": args.n_segments,
                               "tracks": len(kept), "queries": rows}, indent=2))
    print(f"\nwrote {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
