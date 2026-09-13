"""Why does fusing audio and lyrics lose to audio alone?

The recommendation evaluation found that every fusion weight scores below audio-only, on
both label definitions, despite the lyric channel scoring well above random. That has three
possible explanations, and they call for completely different fixes:

1. **Combination.** The blend is miscalibrated. Scores are min-max normalised per query, and
   min-max is set by the two most extreme candidates, so one outlier rescales everything. If
   this is the cause, a better normaliser fixes it.
2. **Routing.** Lyrics win on some queries and lose on others, and a single global alpha
   averages the winner with the loser. If this is the cause, per-query routing fixes it.
3. **Redundancy.** The labels available — same genre, same artist — are acoustic constructs
   that lyrics predict only weakly and never independently. If this is the cause, nothing
   fixes it, because there is no complementary signal to recover.

The decisive test is an **oracle**: a cheating router allowed to see the labels and pick the
better modality per query. It is an upper bound on every routing strategy that could ever be
written. If the oracle barely beats audio alone, explanations 1 and 2 are ruled out no matter
what normaliser or router is tried, and the answer is 3.

    python scripts/diagnose_fusion.py
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aux.data import load_personal_tracks, relevance_matrix  # noqa: E402
from aux.eval import ndcg_at_k  # noqa: E402
from aux.index import build_index  # noqa: E402
from aux.recommend import Recommender, _minmax  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / ".cache"
RESULTS = ROOT / "results"

ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)
K = 10


def per_query_ndcg(scores: np.ndarray, relevant: np.ndarray, k: int = K) -> np.ndarray:
    """NDCG@k for every query, rather than the mean — the mean hides routing headroom.

    Queries with no relevant candidate get NaN, so they can be dropped consistently across
    every system being compared rather than scored as zero.
    """
    scores = np.array(scores, dtype=float, copy=True)
    np.fill_diagonal(scores, -np.inf)
    order = np.argsort(-scores, axis=1)
    totals = relevant.sum(axis=1)
    out = np.full(scores.shape[0], np.nan)
    for i in range(scores.shape[0]):
        if totals[i] > 0:
            out[i] = ndcg_at_k(relevant[i, order[i]], k, int(totals[i]))
    return out


def _zscore(scores: np.ndarray) -> np.ndarray:
    """Standardise per query. Unlike min-max, not set by the two extreme candidates."""
    scores = np.asarray(scores, dtype=float)
    mu = scores.mean(axis=-1, keepdims=True)
    sd = scores.std(axis=-1, keepdims=True)
    return np.where(sd > 0, (scores - mu) / np.where(sd > 0, sd, 1.0), 0.0)


def fused_matrix(rec: Recommender, alpha: float, norm) -> np.ndarray:
    """Fuse with a chosen normaliser, so calibration can be varied independently of alpha.

    Mirrors `Recommender.score` exactly apart from the normaliser: a candidate with no
    transcript keeps its audio score, and a query with no transcript falls back to audio.
    """
    n = len(rec)
    out = np.empty((n, n))
    for i in range(n):
        audio = norm(rec.audio_scores(i))
        if not rec.has_lyrics[i]:
            out[i] = audio
            continue
        raw = rec.lyric_scores(i)
        usable = np.isfinite(raw)
        lyric = np.zeros_like(audio)
        lyric[usable] = norm(raw[usable])
        row = alpha * audio + (1 - alpha) * lyric
        row[~usable] = audio[~usable]
        out[i] = row
    return out


def rrf_matrix(rec: Recommender, k_const: int = 60) -> np.ndarray:
    """Reciprocal rank fusion, as a calibration-free reference point.

    RRF ignores score magnitudes entirely, so it cannot be hurt by a miscalibrated blend.
    If RRF also loses to audio, the problem is not calibration.
    """
    n = len(rec)
    out = np.zeros((n, n))
    for i in range(n):
        contributions = []
        for scores in (rec.audio_scores(i), rec.lyric_scores(i)):
            finite = np.isfinite(scores)
            ranks = np.full(n, np.inf)
            order = np.argsort(-np.where(finite, scores, -np.inf))[: int(finite.sum())]
            ranks[order] = np.arange(len(order))
            contributions.append(1.0 / (k_const + ranks))
        out[i] = sum(contributions)
    return out


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Rank correlation between two score vectors over the same candidates."""
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean()
    rb -= rb.mean()
    denom = np.sqrt((ra**2).sum() * (rb**2).sum())
    return float((ra * rb).sum() / denom) if denom > 0 else 0.0


def main() -> int:
    ap = argparse.ArgumentParser(description="Diagnose why audio+lyric fusion underperforms")
    ap.add_argument("--label", default="artist", choices=("genre", "artist"))
    ap.add_argument("--n-segments", type=int, default=5)
    args = ap.parse_args()

    tracks = load_personal_tracks()
    from aux.encode.muq import MuQMuLanAdapter

    encoder = MuQMuLanAdapter()
    V, kept, _ = build_index(None, encoder, paths=[t.path for t in tracks],
                             n_segments=args.n_segments, cache_path=CACHE / "personal.npz",
                             progress_every=0)
    kept_set = {str(p) for p in kept}
    tracks = [t for t in tracks if str(t.path) in kept_set]

    sys.path.insert(0, str(ROOT / "scripts"))
    from eval_recommendation import load_lyric_vectors

    L, has_lyrics = load_lyric_vectors(tracks, "small", "personal")
    rec = Recommender(V, lyric_vectors=L, has_lyrics=has_lyrics)
    R = relevance_matrix(tracks, args.label)
    print(f"{len(tracks)} tracks, {int(has_lyrics.sum())} with lyrics, label={args.label}",
          file=sys.stderr)

    audio_m = rec.score_matrix(modality="audio")
    lyric_m = rec.score_matrix(modality="lyrics")
    nd_audio = per_query_ndcg(audio_m, R)
    nd_lyric = per_query_ndcg(lyric_m, R)
    scored = ~np.isnan(nd_audio)

    report: dict = {"label": args.label, "n_queries": int(scored.sum())}

    # --- 1. calibration: does a different normaliser change the verdict? ----------------
    print(f"\n=== calibration: fusion under three combination rules (NDCG@{K}) ===")
    print(f"{'alpha':>8}{'min-max':>12}{'z-score':>12}")
    print("-" * 32)
    calib = {}
    for a in ALPHAS:
        mm = float(np.nanmean(per_query_ndcg(fused_matrix(rec, a, _minmax), R)))
        zz = float(np.nanmean(per_query_ndcg(fused_matrix(rec, a, _zscore), R)))
        calib[a] = {"minmax": mm, "zscore": zz}
        print(f"{a:>8.2f}{mm:>12.3f}{zz:>12.3f}")
    rrf = float(np.nanmean(per_query_ndcg(rrf_matrix(rec), R)))
    print(f"{'RRF':>8}{rrf:>12.3f}{'':>12}   (rank-based, immune to calibration)")
    report["calibration"] = {"by_alpha": {str(k): v for k, v in calib.items()}, "rrf": rrf}

    # --- 2. routing: how much headroom does an oracle router have? ---------------------
    best_single = np.fmax(nd_audio, nd_lyric)
    # NaN, not -inf: unscorable queries must stay NaN so `fmax` leaves them out rather
    # than propagating a sentinel into the mean.
    oracle_alpha = np.full(len(tracks), np.nan)
    for a in ALPHAS:
        oracle_alpha = np.fmax(oracle_alpha, per_query_ndcg(fused_matrix(rec, a, _minmax), R))

    mean_audio = float(np.nanmean(nd_audio))
    lyric_wins = float(np.nanmean((nd_lyric > nd_audio)[scored]))
    print(f"\n=== routing headroom (NDCG@{K}, {int(scored.sum())} queries) ===")
    print(f"{'audio only':28}{mean_audio:.3f}")
    print(f"{'lyrics only':28}{float(np.nanmean(nd_lyric)):.3f}")
    print(f"{'ORACLE modality per query':28}{float(np.nanmean(best_single)):.3f}"
          f"   (+{float(np.nanmean(best_single)) - mean_audio:.3f})")
    print(f"{'ORACLE alpha per query':28}{float(np.nanmean(oracle_alpha)):.3f}"
          f"   (+{float(np.nanmean(oracle_alpha)) - mean_audio:.3f})")
    print(f"\nlyrics beat audio on {lyric_wins:.0%} of queries")
    report["routing"] = {
        "audio": mean_audio, "lyrics": float(np.nanmean(nd_lyric)),
        "oracle_modality": float(np.nanmean(best_single)),
        "oracle_alpha": float(np.nanmean(oracle_alpha)),
        "lyric_win_rate": lyric_wins,
    }

    # --- 3. redundancy: are the two rankings independent? ------------------------------
    both = np.flatnonzero(has_lyrics)
    rhos = [spearman(audio_m[i][both], lyric_m[i][both]) for i in both]
    print(f"\n=== redundancy ===")
    print(f"mean Spearman(audio rank, lyric rank) = {float(np.mean(rhos)):+.3f}")
    report["redundancy"] = {"mean_spearman": float(np.mean(rhos))}

    # --- 4. coverage: does the missing-transcript fallback explain it? ------------------
    # 33 of 160 tracks have no transcript and keep their audio score inside a fused
    # ranking, so a fused list mixes two scoring functions. Restricting to tracks that all
    # have lyrics removes that confound entirely.
    sub = Recommender(V[both], lyric_vectors=L[both], has_lyrics=has_lyrics[both])
    sub_tracks = [tracks[i] for i in both]
    R_sub = relevance_matrix(sub_tracks, args.label)
    print(f"\n=== lyric-complete subset ({len(both)} tracks, no fallback) ===")
    print(f"{'alpha':>8}{'NDCG@10':>12}")
    print("-" * 20)
    cov = {}
    for a in ALPHAS:
        v = float(np.nanmean(per_query_ndcg(fused_matrix(sub, a, _minmax), R_sub)))
        cov[a] = v
        print(f"{a:>8.2f}{v:>12.3f}")
    report["coverage_clean"] = {str(k): v for k, v in cov.items()}

    RESULTS.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    out = RESULTS / f"fusion_diagnosis_{args.label}_{stamp}.json"
    report["run_at"] = datetime.now(timezone.utc).isoformat()
    out.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {out.relative_to(ROOT)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
