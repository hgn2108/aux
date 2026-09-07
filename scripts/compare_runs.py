"""Paired comparison of two Eval 0B runs.

Recall@K differences between two configurations look meaningful long before they are.
Because both runs score the *same* queries, the honest test is paired: McNemar's exact
test on the queries whose hit/miss status changed, which ignores the (large) majority that
behaved identically and so is far more sensitive than comparing two proportions.

    python scripts/compare_runs.py evals/eval_0b_clap_1seg_*.json evals/eval_0b_clap_5seg_*.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aux.eval.paired import (  # noqa: E402
    bonferroni_threshold,
    compare_at_k,
    mcnemar_exact,
    sign_test_on_ranks,
)


def load(path: Path) -> tuple[str, dict[str, int]]:
    d = json.loads(Path(path).read_text())
    return d["label"], {q["caption_id"]: q["rank"] for q in d["per_query"]}


def main() -> int:
    ap = argparse.ArgumentParser(description="Paired comparison of two Eval 0B runs")
    ap.add_argument("baseline", type=Path)
    ap.add_argument("treatment", type=Path)
    ap.add_argument("--k", type=int, nargs="+", default=[1, 5, 10])
    args = ap.parse_args()

    base_label, base = load(args.baseline)
    treat_label, treat = load(args.treatment)
    shared = sorted(set(base) & set(treat))
    if not shared:
        raise SystemExit("runs share no queries")

    br = np.array([base[q] for q in shared])
    tr = np.array([treat[q] for q in shared])

    print(f"baseline : {base_label}")
    print(f"treatment: {treat_label}")
    print(f"paired queries: {len(shared)}\n")
    print(f"{'K':>4}{'base':>9}{'treat':>9}{'delta':>9}{'gained':>8}{'lost':>7}{'p':>10}")
    print("-" * 56)
    for k in args.k:
        r = compare_at_k(br, tr, k)
        print(f"{k:>4}{r['baseline']:>9.3f}{r['treatment']:>9.3f}"
              f"{r['delta']:>+9.3f}{r['gained']:>8}{r['lost']:>7}{r['p_value']:>10.2e}")

    print(f"\nmedian rank: {np.median(br):.0f} -> {np.median(tr):.0f}")
    print(f"MRR:         {(1/br).mean():.4f} -> {(1/tr).mean():.4f}")
    st = sign_test_on_ranks(br, tr)
    print(f"rank improved on {st['improved']} queries, worsened on {st['worsened']}, "
          f"unchanged on {st['unchanged']}")
    print(f"sign test on rank change: p = {st['p_value']:.2e}")
    thresh = bonferroni_threshold(len(args.k))
    print(f"\nBonferroni threshold for {len(args.k)} tests in this run: p < {thresh:.4f}")
    print("Comparing several configurations multiplies that count — correct across the "
          "whole comparison set, not per run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
