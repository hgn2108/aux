"""How confidently does a library answer a query?

DEC-020 gates query rewriting on this: rewrite only when nothing already stands out. The
gate is only useful if the measure means the same thing in different libraries, which is not
obvious, `z_top` divides by the library's spread, but the maximum of 8,000 samples sits
further into the tail than the maximum of 26, so a fixed threshold can drift with collection
size even though the statistic is nominally normalised.

Four measures are provided so that question can be settled by measurement rather than
assumed. Three are scale-free by construction.
"""

from __future__ import annotations

import numpy as np


def z_top(scores: np.ndarray) -> float:
    """Standard deviations between the best match and the library mean.

    The measure DEC-020 was fitted with. Included as the incumbent, not the favourite.
    """
    scores = np.asarray(scores, dtype=float)
    sd = scores.std()
    return float((scores.max() - scores.mean()) / sd) if sd > 0 else 0.0


def relative_gap(scores: np.ndarray, k: int = 10) -> float:
    """How far the winner beats the chasing pack, as a fraction of its own margin.

    `(top1 - topk) / (top1 - mean)`. Scale-free: both numerator and denominator are
    differences in the same units, so the units cancel. Near 0 means the top k are
    interchangeable; near 1 means the winner stands alone.
    """
    scores = np.asarray(scores, dtype=float)
    if scores.size <= k:
        return 0.0
    top = np.sort(scores)[::-1]
    margin = top[0] - scores.mean()
    return float((top[0] - top[k - 1]) / margin) if margin > 0 else 0.0


def crowding(scores: np.ndarray, fraction: float = 0.9) -> float:
    """Share of the library scoring within `fraction` of the top.

    Scale-free because it is a proportion. **High crowding means low confidence**, many
    tracks are nearly as good as the best one, so it runs opposite to the others and is
    inverted where a single direction is wanted.
    """
    scores = np.asarray(scores, dtype=float)
    top = scores.max()
    if top <= 0:
        # Cosine scores can be negative; shift so the threshold stays meaningful.
        shifted = scores - scores.min()
        top = shifted.max()
        if top <= 0:
            return 1.0
        return float((shifted >= top * fraction).mean())
    return float((scores >= top * fraction).mean())


def top_k_entropy(scores: np.ndarray, k: int = 10) -> float:
    """Evenness of the top-k scores, normalised to [0, 1].

    1 means the top k are indistinguishable, 0 means one dominates. Scale-free after the
    softmax-like normalisation, and unlike `crowding` it uses the whole shape of the head
    rather than a single cutoff.
    """
    scores = np.asarray(scores, dtype=float)
    k = min(k, scores.size)
    if k < 2:
        return 0.0
    top = np.sort(scores)[::-1][:k]
    weights = top - top.min()
    total = weights.sum()
    if total <= 0:
        return 1.0
    p = weights / total
    p = p[p > 0]
    return float(-(p * np.log(p)).sum() / np.log(k))


MEASURES = {
    "z_top": z_top,
    "relative_gap": relative_gap,
    "crowding": crowding,
    "top_k_entropy": top_k_entropy,
}

HIGHER_MEANS_MORE_CONFIDENT = {
    "z_top": True, "relative_gap": True, "crowding": False, "top_k_entropy": False,
}


def all_measures(scores: np.ndarray) -> dict[str, float]:
    return {name: fn(scores) for name, fn in MEASURES.items()}
