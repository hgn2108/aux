"""The crossover: does the best fusion weight depend on the query?

Two evaluations in this repo reach opposite conclusions with the same two systems.
Track-to-track similarity against genre labels wants pure audio; "songs about X" wants pure
lyrics. Neither is a failure of the other, they are different questions, and this script
puts them side by side to show that the fusion weight is not a constant to be tuned once.

It reads the JSON both evaluations already wrote rather than recomputing anything, so the
numbers here cannot drift from the numbers they were taken from.

What the averaged column is and is not. The two families are scored against different
labels on different query sets, so their mean is a presentational device for comparing
policies, not a metric anyone should quote on its own. It answers one question only: if a
single fusion weight had to serve both families, what would it cost? The per-family columns
are the real evidence.

    python scripts/eval_routing.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)
METRIC = "ndcg@10"


def latest(prefix: str) -> Path:
    """Most recent run of one evaluation.

    Anchored on the date stamp rather than a bare wildcard: a tagged variant such as
    `semantic_highagreement_20260913.json` sorts after the run it is a variant of, so a
    loose glob would silently compare the sensitivity check instead of the headline.
    """
    matches = sorted(RESULTS.glob(f"{prefix}_2*.json"))
    if not matches:
        raise SystemExit(f"no results file matching {prefix}_2*.json, run the evaluation")
    return matches[-1]


def system_for(alpha: float) -> str:
    """Alpha is the audio share, so the ends of the sweep are the unimodal systems."""
    if alpha == 1.0:
        return "audio"
    if alpha == 0.0:
        return "lyrics"
    return f"fused a={alpha}"


def main() -> int:
    ap = argparse.ArgumentParser(description="Compare a fixed fusion weight against routing")
    ap.add_argument("--label", default="genre",
                    help="which track-to-track label to use as the acoustic family")
    ap.add_argument("--semantic", default="semantic",
                    help="results prefix for the semantic family, e.g. "
                         "semantic_highagreement for the label-sensitivity variant")
    args = ap.parse_args()

    similarity = json.loads(latest("recommendation_personal").read_text())["results"][args.label]
    semantic = json.loads(latest(args.semantic).read_text())["results"]

    rows = []
    for alpha in ALPHAS:
        key = system_for(alpha)
        sim = similarity[key][METRIC]
        sem = semantic[key][METRIC]
        rows.append((alpha, sim, sem, (sim + sem) / 2))

    print(f"\n=== fixed fusion weight across both query families ({METRIC}) ===")
    print(f"{'alpha':>7}{'track->track':>15}{'semantic':>12}{'mean':>9}")
    print("-" * 43)
    for alpha, sim, sem, mean in rows:
        print(f"{alpha:>7.2f}{sim:>15.3f}{sem:>12.3f}{mean:>9.3f}")

    best_alpha, b_sim, b_sem, best_fixed = max(rows, key=lambda r: r[3])
    routed_sim = max(r[1] for r in rows)
    routed_sem = max(r[2] for r in rows)
    routed = (routed_sim + routed_sem) / 2

    print(f"\n{'best single weight':26}alpha={best_alpha:.2f}  "
          f"{b_sim:.3f} / {b_sem:.3f}  mean {best_fixed:.3f}")
    print(f"{'best weight per family':26}{'':10}"
          f"{routed_sim:.3f} / {routed_sem:.3f}  mean {routed:.3f}")
    print(f"\nin-sample gap {routed - best_fixed:+.3f}. Both weights are picked on the same")
    print("aggregates they are scored on, so this is descriptive, not a held-out result.")
    print("The generalising version is in scripts/eval_router.py, cross-validated.")

    # The cost of guessing wrong: applying each family's own best weight to the other.
    worst_sim = min(r[1] for r in rows)
    worst_sem = min(r[2] for r in rows)
    print(f"\ncost of the wrong weight: track->track {routed_sim - worst_sim:.3f}, "
          f"semantic {routed_sem - worst_sem:.3f}")

    suffix = "" if args.semantic == "semantic" else f"_{args.semantic}"
    out = RESULTS / f"routing_crossover{suffix}.json"
    out.write_text(json.dumps({
        "metric": METRIC, "similarity_label": args.label,
        "semantic_source": latest(args.semantic).name,
        "by_alpha": [{"alpha": a, "similarity": s, "semantic": m, "mean": av}
                     for a, s, m, av in rows],
        "best_fixed_alpha": best_alpha, "best_fixed_mean": best_fixed,
        "routed_mean": routed,
        "in_sample_gap": routed - best_fixed,
        "note": ("Descriptive and in-sample: both weights are selected on the aggregates "
                 "they are scored on. See results/router_*.json family_routing_cv for the "
                 "held-out comparison."),
    }, indent=2))
    print(f"\nwrote {out.relative_to(ROOT)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
