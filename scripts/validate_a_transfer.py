"""Phase A — does a confidence measure mean the same thing in different libraries?

DEC-020's gate compares a confidence score against a fixed threshold. That only works if the
score's distribution is stable across collections. `z_top` is suspect: it divides by the
library's spread, but the maximum of 8,000 samples sits further into the tail than the
maximum of 26, so the same query can score differently purely because of collection size.

Runs the same queries against six collections of very different size and composition, and
reports how much each measure's distribution moves. The measure whose distribution moves
least is the one a single threshold can be set on.

    python scripts/validate_a_transfer.py
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
from aux.ingest import discover  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PERSONAL = ROOT / "data" / "music"

QUERIES = [
    # deliberately spanning kinds, since a measure must be stable across query types too
    "solo piano", "orchestral strings", "distorted electric guitar",
    "saxophone over an upright bass", "fast breakbeat drums",
    "something to fall asleep to", "background music while reading",
    "warming up before going out", "something for a rainy morning",
    "hip hop for studying", "jazz for a party", "energetic music", "calm music",
    "melancholy and heartbroken", "playful and light-hearted",
]


def genre_of(path: Path, root: Path) -> str:
    rel = path.relative_to(root)
    return rel.parts[0] if len(rel.parts) > 1 else "hiphop_rnb"


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase A — confidence transfer")
    ap.add_argument("--fma-limit", type=int, default=1500)
    ap.add_argument("--n-segments", type=int, default=5)
    args = ap.parse_args()

    from aux.encode.muq import MuQMuLanAdapter

    encoder = MuQMuLanAdapter()

    Vp, paths_p, _ = build_index(PERSONAL, encoder, n_segments=args.n_segments,
                                 cache_path=ROOT / ".cache" / "personal.npz",
                                 progress_every=0)
    genres = np.array([genre_of(p, PERSONAL) for p in paths_p])

    collections: dict[str, np.ndarray] = {
        "personal — all": Vp,
        "personal — hip-hop only": Vp[genres == "hiphop_rnb"],
        "personal — no hip-hop": Vp[genres != "hiphop_rnb"],
        "personal — acoustic": Vp[np.isin(genres, ["jazz", "classical", "vpop"])],
        "personal — electronic": Vp[np.isin(genres, ["edm", "dnb"])],
    }

    fma = ROOT / "data" / "raw" / "fma" / "fma_small"
    if fma.is_dir():
        Vf, _, _ = build_index(fma, encoder, n_segments=args.n_segments,
                               cache_path=ROOT / ".cache" / "fma_small.npz",
                               limit=args.fma_limit, progress_every=0)
        collections["FMA (public)"] = Vf

    sdd = ROOT / "data" / "raw" / "song_describer" / "audio"
    if sdd.is_dir():
        Vs, _, _ = build_index(sdd, encoder, n_segments=args.n_segments,
                               cache_path=ROOT / ".cache" / "sdd.npz", progress_every=0)
        collections["Song Describer"] = Vs

    T = l2_normalise(encoder.embed_text(QUERIES))

    results: dict[str, dict[str, list[float]]] = {m: {} for m in MEASURES}
    for name, V in collections.items():
        for qi in range(len(QUERIES)):
            vals = all_measures(V @ T[qi])
            for m, v in vals.items():
                results[m].setdefault(name, []).append(v)

    print(f"\n{'collection':28}{'n':>7}" + "".join(f"{m:>16}" for m in MEASURES))
    print("-" * (35 + 16 * len(MEASURES)))
    for name, V in collections.items():
        row = "".join(f"{np.median(results[m][name]):>16.2f}" for m in MEASURES)
        print(f"{name:28}{V.shape[0]:>7}{row}")

    print(f"\n{'measure':18}{'median range':>14}{'spread ratio':>14}   verdict")
    print("-" * 76)
    summary = {}
    for m in MEASURES:
        medians = np.array([np.median(results[m][n]) for n in collections])
        rng_ = float(medians.max() - medians.min())
        # Range relative to the typical value: a measure whose median doubles between
        # collections cannot carry a single threshold, however tidy its units.
        ratio = float(medians.max() / medians.min()) if medians.min() > 0 else float("inf")
        summary[m] = {"medians": {n: float(np.median(results[m][n])) for n in collections},
                      "range": rng_, "ratio": ratio,
                      "higher_is_confident": HIGHER_MEANS_MORE_CONFIDENT[m]}
        verdict = ("stable — a single threshold is plausible" if ratio < 1.5
                   else "drifts — threshold would need per-library calibration")
        print(f"{m:18}{rng_:>14.2f}{ratio:>14.2f}   {verdict}")

    best = min(MEASURES, key=lambda m: summary[m]["ratio"])
    print(f"\nmost stable across collections: {best} (ratio {summary[best]['ratio']:.2f})")

    out = ROOT / "evals" / "validate_a_transfer.json"
    out.write_text(json.dumps({"run_at": datetime.now(timezone.utc).isoformat(),
                               "encoder": encoder.version, "queries": QUERIES,
                               "collections": {n: int(V.shape[0]) for n, V in collections.items()},
                               "summary": summary,
                               "per_query": {m: results[m] for m in MEASURES}}, indent=2))
    print(f"\nwrote {out.relative_to(ROOT)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
