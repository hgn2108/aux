"""Does context→acoustic expansion help, and at what weight? (DEC-013 rung 1)

Two metrics, because optimising either alone gives a wrong answer — the lesson from the
negation sweep, where minimising leakage picked a weight that satisfied the exclusion by
returning things nobody asked for.

- **separation** — how far apart two opposed contexts pull retrieval, measured on
  waveform features the encoder never sees. `scripts/acoustic_direction.py` established the
  baseline: running − sleeping = +0.48 onset rate, workout − relax = +0.77. If expansion
  works, that gap widens.
- **genre fidelity** — for genre-anchored queries, does the named genre still dominate the
  results? An expansion strong enough to separate contexts but which discards "hip hop" has
  broken the query, and Slice 1 showed the genre word carries real signal (4.18 for
  genre-anchored context against 3.14 for context alone).

Neither needs human ratings, so the sweep is cheap enough to run before spending any.

    python scripts/eval_expansion.py data/music
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aux.ingest import IngestError, decode, discover  # noqa: E402
from aux.query import score_query  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

OPPOSED_PAIRS = [
    ("hip hop for running", "hip hop for falling asleep"),
    ("hip hop for a workout", "hip hop to relax to"),
    ("music for a party", "music for reading"),
    ("music for the gym", "music for sleeping"),
]
"""Pairs sharing everything but their context, so any difference is the context's doing."""

GENRE_ANCHORED = [
    ("hip hop for running", "hiphop_rnb"),
    ("hip hop for studying", "hiphop_rnb"),
    ("jazz for reading", "jazz"),
    ("classical for sleeping", "classical"),
    ("edm for a party", "edm"),
]
WEIGHTS = [0.0, 0.15, 0.3, 0.45, 0.6, 0.8]
K = 10


def waveform_features(samples: np.ndarray, sr: int) -> tuple[float, float]:
    """Loudness and a crude onset-rate proxy, computed independently of the encoder."""
    rms = float(np.sqrt(np.mean(samples**2)))
    hop = max(1, sr // 100)
    n = (samples.size // hop) * hop
    if n == 0:
        return rms, 0.0
    env = np.abs(samples[:n].reshape(-1, hop)).max(axis=1)
    d = np.diff(env)
    onset = float((d > d.std() * 1.5).mean()) if d.size else 0.0
    return rms, onset


def genre_of(path: Path, root: Path) -> str:
    rel = path.relative_to(root)
    return rel.parts[0] if len(rel.parts) > 1 else "hiphop_rnb"


def main() -> int:
    ap = argparse.ArgumentParser(description="Context expansion weight sweep")
    ap.add_argument("root", type=Path)
    ap.add_argument("--n-segments", type=int, default=5)
    args = ap.parse_args()

    from aux.encode.muq import MuQMuLanAdapter

    encoder = MuQMuLanAdapter()
    paths, vectors, feats = [], [], []
    for path in [f.path for f in discover(args.root)]:
        try:
            asset = decode(path)
            vec, _ = encoder.embed_track(asset, n_segments=args.n_segments)
        except (IngestError, Exception):  # noqa: BLE001
            continue
        paths.append(path)
        vectors.append(vec)
        feats.append(waveform_features(asset.samples, asset.sample_rate))
    V = np.stack(vectors)
    F = np.array(feats)
    Fz = (F - F.mean(0)) / F.std(0)
    genres = np.array([genre_of(p, args.root) for p in paths])
    print(f"{V.shape[0]} tracks indexed", file=sys.stderr)

    results = {}
    print(f"{'weight':>8}{'separation':>12}{'genre fidelity':>16}{'margin':>9}")
    print("-" * 45)
    for w in WEIGHTS:
        seps = []
        for hi, lo in OPPOSED_PAIRS:
            s_hi, _, _ = score_query(encoder, hi, V, expansion_weight=w)
            s_lo, _, _ = score_query(encoder, lo, V, expansion_weight=w)
            top_hi = np.argsort(-s_hi)[:K]
            top_lo = np.argsort(-s_lo)[:K]
            seps.append(float(Fz[top_hi, 1].mean() - Fz[top_lo, 1].mean()))

        fid = []
        for query, expected in GENRE_ANCHORED:
            s, _, _ = score_query(encoder, query, V, expansion_weight=w)
            fid.append(float((genres[np.argsort(-s)[:K]] == expected).mean()))

        sep, fidelity = float(np.mean(seps)), float(np.mean(fid))
        results[w] = {"separation": sep, "genre_fidelity": fidelity,
                      "margin": sep + fidelity, "per_pair": seps}
        print(f"{w:>8.2f}{sep:>12.2f}{fidelity:>16.2f}{sep + fidelity:>9.2f}")

    best = max(WEIGHTS, key=lambda w: results[w]["margin"])
    print(f"\nbest margin at weight {best}")
    print(f"\n{'pair':52}{'w=0':>8}{'w=' + str(best):>8}")
    print("-" * 68)
    for i, (hi, lo) in enumerate(OPPOSED_PAIRS):
        print(f"{hi[:24]} vs {lo[:24]:26}"
              f"{results[0.0]['per_pair'][i]:>8.2f}{results[best]['per_pair'][i]:>8.2f}")

    out = ROOT / "evals" / "expansion_sweep.json"
    out.write_text(json.dumps({"run_at": datetime.now(timezone.utc).isoformat(),
                               "encoder": encoder.version, "k": K, "chosen_weight": best,
                               "sweep": {str(k): v for k, v in results.items()}}, indent=2))
    print(f"\nwrote {out.relative_to(ROOT)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
