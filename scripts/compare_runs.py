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
from math import comb
from pathlib import Path

import numpy as np


def load(path: Path) -> tuple[str, dict[str, int]]:
    d = json.loads(Path(path).read_text())
    return d["label"], {q["caption_id"]: q["rank"] for q in d["per_query"]}


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact binomial p-value on the discordant pairs."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(comb(n, i) for i in range(k + 1)) / (2**n)
    return min(1.0, 2 * tail)


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
        bh, th = br <= k, tr <= k
        gained = int((~bh & th).sum())
        lost = int((bh & ~th).sum())
        p = mcnemar_exact(lost, gained)
        print(f"{k:>4}{bh.mean():>9.3f}{th.mean():>9.3f}"
              f"{th.mean() - bh.mean():>+9.3f}{gained:>8}{lost:>7}{p:>10.2e}")

    print(f"\nmedian rank: {np.median(br):.0f} -> {np.median(tr):.0f}")
    print(f"MRR:         {(1/br).mean():.4f} -> {(1/tr).mean():.4f}")
    improved = int((tr < br).sum())
    worsened = int((tr > br).sum())
    print(f"rank improved on {improved} queries, worsened on {worsened}, "
          f"unchanged on {len(shared) - improved - worsened}")
    print(f"sign test on rank change: p = {mcnemar_exact(worsened, improved):.2e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
