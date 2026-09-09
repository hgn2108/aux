"""Phase C — the premise check, against real ground truth.

Every other test of DEC-020 uses a proxy metric or human ratings. This one has a right
answer.

Song Describer's captions are **already written in sound-describing language** — the
encoder's native distribution. DEC-020's premise is that rewriting helps by closing a gap
between the user's language and the encoder's. If that premise is right, then here:

- confidence should already be high, so the gate should decline to rewrite;
- forcing a rewrite should not help, and may hurt.

If forcing rewrites *improves* Recall@K on captions that are already sound-language, the
premise is wrong at the root and no threshold tuning rescues it.

Measured against known correct tracks. No proxy, no rating.

    python scripts/validate_c_ground_truth.py --limit 150
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aux.encode.base import l2_normalise  # noqa: E402
from aux.eval import all_measures, mcnemar_exact, ranks_of_truth, retrieval_metrics  # noqa: E402
from aux.index import build_index  # noqa: E402
from aux.plan import build_planner, plan_all  # noqa: E402
from aux.query import score_plan  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SDD = ROOT / "data" / "raw" / "song_describer"


def resolve_audio(audio_root: Path, rel: str) -> Path | None:
    direct = audio_root / rel
    if direct.exists():
        return direct
    excerpt = direct.with_suffix("").with_suffix(".2min.mp3")
    if excerpt.exists():
        return excerpt
    matches = sorted(direct.parent.glob(f"{direct.stem}.*.mp3"))
    return matches[0] if matches else None


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase C — ground-truth premise check")
    ap.add_argument("--limit", type=int, default=150, help="captions to test")
    ap.add_argument("--n-segments", type=int, default=5)
    args = ap.parse_args()

    from aux.encode.muq import MuQMuLanAdapter

    encoder = MuQMuLanAdapter()
    V, paths, _ = build_index(SDD / "audio", encoder, n_segments=args.n_segments,
                              cache_path=ROOT / ".cache" / "sdd.npz", progress_every=0)
    by_name = {p.name: i for i, p in enumerate(paths)}

    rows = list(csv.DictReader((SDD / "song_describer.csv").open()))
    rows = [r for r in rows if r["is_valid_subset"] == "True"]
    captions, truth = [], []
    for r in rows:
        p = resolve_audio(SDD / "audio", r["path"])
        if p is not None and p.name in by_name:
            captions.append(r["caption"])
            truth.append(by_name[p.name])
        if len(captions) >= args.limit:
            break
    truth = np.array(truth)
    print(f"{len(captions)} captions over {V.shape[0]} candidate tracks", file=sys.stderr)

    planner = build_planner("claude")
    plans = plan_all(planner, captions, ROOT / ".cache" / "plans_sdd.json")

    base_scores = np.stack([V @ l2_normalise(encoder.embed_text([c]))[0] for c in captions])
    plan_scores = np.stack([score_plan(encoder, plans[c], V) for c in captions])

    base_ranks = ranks_of_truth(base_scores, truth)
    plan_ranks = ranks_of_truth(plan_scores, truth)
    mb = retrieval_metrics(base_ranks, V.shape[0])
    mp = retrieval_metrics(plan_ranks, V.shape[0])

    print(f"\n{'system':22}{'R@1':>8}{'R@5':>8}{'R@10':>8}{'median':>9}{'MRR':>8}")
    print("-" * 63)
    print(f"{'caption as written':22}{mb['recall@1']:>8.3f}{mb['recall@5']:>8.3f}"
          f"{mb['recall@10']:>8.3f}{mb['median_rank']:>9.0f}{mb['mrr']:>8.3f}")
    print(f"{'forced rewrite':22}{mp['recall@1']:>8.3f}{mp['recall@5']:>8.3f}"
          f"{mp['recall@10']:>8.3f}{mp['median_rank']:>9.0f}{mp['mrr']:>8.3f}")

    for k in (1, 5, 10):
        bh, ph = base_ranks <= k, plan_ranks <= k
        gained, lost = int((~bh & ph).sum()), int((bh & ~ph).sum())
        print(f"  R@{k:<2} gained {gained:>3}, lost {lost:>3}, "
              f"McNemar p = {mcnemar_exact(lost, gained):.2e}")

    # Does the gate correctly decline to rewrite here?
    conf = [all_measures(base_scores[i]) for i in range(len(captions))]
    z = np.array([c["z_top"] for c in conf])
    print(f"\nconfidence on captions (already sound-language):")
    print(f"  z_top median {np.median(z):.2f}   share below the 3.0 gate: "
          f"{float((z < 3.0).mean()):.0%}")
    print("  DEC-020 predicts high confidence here, so the gate should mostly decline "
          "to rewrite.")

    verdict = ("premise HOLDS — forcing rewrites does not help already-sound-language queries"
               if mp["recall@10"] <= mb["recall@10"]
               else "premise CONTRADICTED — rewriting helps even in-distribution queries")
    print(f"\nverdict: {verdict}")

    out = ROOT / "evals" / "validate_c_ground_truth.json"
    out.write_text(json.dumps({"run_at": datetime.now(timezone.utc).isoformat(),
                               "encoder": encoder.version, "planner": planner.version,
                               "n_captions": len(captions), "n_tracks": int(V.shape[0]),
                               "baseline": mb, "forced_rewrite": mp,
                               "z_top_median": float(np.median(z)),
                               "share_below_gate": float((z < 3.0).mean()),
                               "verdict": verdict}, indent=2))
    print(f"\nwrote {out.relative_to(ROOT)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
