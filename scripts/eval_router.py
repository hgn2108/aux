"""Does routing the fusion weight beat fixing it, and does the LLM beat the free baselines?

The crossover established that no single fusion weight serves both query families, and that
an oracle picking per query gains +0.042 NDCG@10 over the best fixed weight. That is the
ceiling. This measures how much of it each router actually captures, and what each costs.

Three routers, in increasing order of price:

- rules, keyword and phrase matching. Free, instant.
- prototype, the query embedded once and compared to two small prototype sets. Free,
  about a millisecond, no hand-written vocabulary.
- claude, a model call per query. Roughly half a second and a fraction of a cent.

Measured end to end, not as classification accuracy: each router's weight is fed to the
actual retrieval and scored against objective relevance. A router that classifies well but
picks unhelpful weights should lose, and accuracy alone would hide that.

Relevance. Queries name a genre, a theme, or both. Genre comes from the folder layout,
theme from the validated labels (Cohen's kappa 0.60 against blind human judgement). A
compound query is relevant only where both hold, which is what makes it need both channels.

    python scripts/eval_router.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aux.data import load_personal_tracks  # noqa: E402
from aux.eval import (  # noqa: E402
    bonferroni_threshold,
    bootstrap_ci,
    cross_validate_routing,
    ndcg_at_k,
    permutation_test,
)
from aux.index import build_index  # noqa: E402
from aux.ingest.asset import content_hash  # noqa: E402
from aux.lyrics import load_lyric_vectors  # noqa: E402
from aux.recommend import NORMALISERS, blend_rows, score_queries  # noqa: E402
from aux.route import PrototypeRouter, Route, RuleRouter  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / ".cache"
QUERIES = ROOT / "queries"
RESULTS = ROOT / "results"

ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)
K = 10


def relevance(spec: dict, tracks, hashes, theme_labels) -> np.ndarray:
    """A track is relevant when it matches every constraint the query names."""
    mask = np.ones(len(tracks), dtype=bool)
    if spec.get("genre"):
        mask &= np.array([t.genre == spec["genre"] for t in tracks])
    if spec.get("theme"):
        mask &= np.array([spec["theme"] in theme_labels.get(h, {}).get("themes", [])
                          for h in hashes])
    return mask


def main() -> int:
    ap = argparse.ArgumentParser(description="Compare fusion-weight routers")
    ap.add_argument("--normaliser", default="zscore", choices=sorted(NORMALISERS))
    ap.add_argument("--no-claude", action="store_true", help="skip the paid arm")
    ap.add_argument("--folds", type=int, default=5,
                    help="cross-validation folds for the routing comparison")
    ap.add_argument("--seed", type=int, default=0, help="fold assignment seed")
    ap.add_argument("--stability-seeds", type=int, default=20,
                    help="how many fold assignments to repeat for the stability check")
    ap.add_argument("--queries", default="routing_queries_expanded.json",
                    help="query set under queries/; the expanded set has the statistical "
                         "power the 24-query original lacked")
    args = ap.parse_args()

    specs = json.loads((QUERIES / args.queries).read_text())["queries"]
    theme_labels = json.loads((CACHE / "themes.json").read_text())

    tracks = load_personal_tracks()
    from aux.encode.muq import MuQMuLanAdapter

    encoder = MuQMuLanAdapter()
    V, kept, _ = build_index(None, encoder, paths=[t.path for t in tracks],
                             n_segments=5, cache_path=CACHE / "personal.npz",
                             progress_every=0)
    kept_set = {str(p) for p in kept}
    tracks = [t for t in tracks if str(t.path) in kept_set]
    hashes = [content_hash(t.path) for t in tracks]

    L, has_lyrics = load_lyric_vectors(tracks, "small", "personal")
    from aux.lyrics import LyricEmbedder

    queries = [s["query"] for s in specs]
    R = np.stack([relevance(s, tracks, hashes, theme_labels) for s in specs])
    families = np.array([s["family"] for s in specs])
    print(f"{len(tracks)} tracks, {len(queries)} queries, "
          f"{R.sum(1).mean():.1f} relevant each", file=sys.stderr)

    audio_scores = score_queries(encoder.embed_text(queries), V)
    lyric_scores = score_queries(LyricEmbedder().embed_query(queries), L, has_lyrics)
    norm = NORMALISERS[args.normaliser]

    def ndcg_at_alpha(i: int, alpha: float) -> float:
        """NDCG@K for one query answered at one fusion weight."""
        row = blend_rows(audio_scores[i : i + 1], lyric_scores[i : i + 1], alpha,
                         has_lyrics, args.normaliser)[0]
        order = np.argsort(-row)
        return ndcg_at_k(R[i][order], K, int(R[i].sum()))

    # Dense grid so a router's weight is scored at what it actually asked for, and so the
    # oracle is a genuine per-query best rather than the best of five coarse options.
    grid = np.round(np.arange(0.0, 1.01, 0.05), 2)
    table = np.array([[ndcg_at_alpha(i, a) for a in grid] for i in range(len(queries))])

    def score_for(i: int, alpha: float) -> float:
        return float(table[i, int(np.argmin(np.abs(grid - alpha)))])

    routers: dict[str, list[Route]] = {}
    timings: dict[str, float] = {}
    for router in (RuleRouter(), PrototypeRouter(encoder)):
        start = time.perf_counter()
        routers[router.name] = router.route_all(queries)
        timings[router.name] = (time.perf_counter() - start) / len(queries) * 1000
    if not args.no_claude:
        from aux.route.claude import ClaudeRouter

        cache_path = CACHE / "router_claude.json"
        cache = {k: tuple(v) for k, v in json.loads(cache_path.read_text()).items()} \
            if cache_path.exists() else {}
        # Carry the last measured latency forward when the cache answers everything, so a
        # rerun does not overwrite a real measurement with zero.
        previous = sorted(RESULTS.glob("router_2*.json"))
        cached_latency = None
        if previous:
            prior = json.loads(previous[-1].read_text())["routers"].get("claude")
            cached_latency = prior["ms_per_query"] if prior else None
        router = ClaudeRouter(cache=cache)
        start = time.perf_counter()
        routes = router.route_all(queries)
        timings[router.name] = (time.perf_counter() - start) / len(queries) * 1000
        if router.failures:
            # Every fallback returns the same weight, so a fully failed arm looks like a
            # confident constant. Refuse to report it as a measurement.
            raise SystemExit(f"the claude router failed on {router.failures}/{len(queries)} "
                             f"queries; fix the cause or rerun with --no-claude")
        if router.api_calls:
            timings[router.name] = router.api_seconds / router.api_calls * 1000
        elif cached_latency is not None:
            timings[router.name] = cached_latency   # every query served from cache
        routers[router.name] = routes
        cache_path.write_text(json.dumps({k: list(v) for k, v in cache.items()}, indent=2))

    # --- baselines ---------------------------------------------------------------------
    fixed = {a: float(np.mean([score_for(i, a) for i in range(len(queries))]))
             for a in ALPHAS}
    best_alpha = max(fixed, key=fixed.get)
    oracle = float(table.max(axis=1).mean())

    print(f"\n=== fixed weights (NDCG@{K}, {len(queries)} queries) ===")
    for a, v in fixed.items():
        print(f"  alpha={a:.2f}   {v:.3f}" + ("   <- best fixed" if a == best_alpha else ""))
    print(f"  ORACLE     {oracle:.3f}   <- per-query ceiling")

    print(f"\n=== routers ===")
    print(f"{'router':12}{'NDCG@10':>10}{'vs best fixed':>15}{'% of oracle gap':>18}"
          f"{'ms/query':>11}")
    print("-" * 66)
    rows = {}
    gap = oracle - fixed[best_alpha]
    for name, routes in routers.items():
        value = float(np.mean([score_for(i, r.alpha) for i, r in enumerate(routes)]))
        captured = (value - fixed[best_alpha]) / gap if gap > 0 else 0.0
        rows[name] = {"ndcg": value, "gain": value - fixed[best_alpha],
                      "oracle_fraction": captured, "ms_per_query": timings[name],
                      "alphas": [r.alpha for r in routes]}
        print(f"{name:12}{value:>10.3f}{value - fixed[best_alpha]:>+15.3f}"
              f"{captured:>17.0%}{timings[name]:>11.1f}")

    # --- significance -------------------------------------------------------------------
    # The margins here are small and the query set is 24 items written by the same person
    # who wrote the rules, so a difference that is not significant must not be reported as a
    # win. Each router is tested against the best fixed weight, and the two leading routers
    # against each other, with the threshold corrected for the number of comparisons.
    per_query = {n: np.array([score_for(i, r.alpha) for i, r in enumerate(routes)])
                 for n, routes in routers.items()}
    fixed_per_query = np.array([score_for(i, best_alpha) for i in range(len(queries))])

    comparisons = [(n, "best fixed", per_query[n], fixed_per_query) for n in routers]
    ordered = sorted(routers, key=lambda n: -float(per_query[n].mean()))
    if len(ordered) >= 2:
        top, second = ordered[0], ordered[1]
        comparisons.append((top, second, per_query[top], per_query[second]))
    threshold = bonferroni_threshold(len(comparisons))

    print(f"\n=== significance (paired permutation, threshold {threshold:.4f}) ===")
    print(f"{'comparison':32}{'difference':>12}{'p':>10}{'verdict':>16}")
    print("-" * 70)
    significance = {}
    for name, against, left, right in comparisons:
        test = permutation_test(left, right)
        verdict = "significant" if test["p_value"] < threshold else "not significant"
        significance[f"{name} vs {against}"] = {**test, "significant": verdict == "significant"}
        print(f"{name + ' vs ' + against:32}{test['observed']:>+12.3f}"
              f"{test['p_value']:>10.4f}{verdict:>16}")

    print(f"\n=== by query family (NDCG@{K}) ===")
    print(f"{'family':12}{'n':>4}" + "".join(f"{n:>12}" for n in routers)
          + f"{'best fixed':>12}{'oracle':>9}")
    print("-" * (16 + 12 * len(routers) + 21))
    per_family = {}
    for family in ("acoustic", "semantic", "compound"):
        idx = np.flatnonzero(families == family)
        cells = {n: float(np.mean([score_for(i, routers[n][i].alpha) for i in idx]))
                 for n in routers}
        bf = float(np.mean([score_for(i, best_alpha) for i in idx]))
        orc = float(table[idx].max(axis=1).mean())
        per_family[family] = {**cells, "best_fixed": bf, "oracle": orc}
        print(f"{family:12}{len(idx):>4}" + "".join(f"{cells[n]:>12.3f}" for n in routers)
              + f"{bf:>12.3f}{orc:>9.3f}")

    # --- family-level routing, cross-validated ------------------------------------------
    # The routers above choose per query and mostly fail. This asks the prior question: is
    # the decision worth making at all?
    #
    # Weights are chosen on training folds and applied unchanged to held-out ones, because
    # the earlier version of this picked both the global weight and each family's weight on
    # the same queries it then scored. Routing fits one weight per family against the
    # baseline's one overall, so on shared data it wins partly by having more freedom.
    cv = cross_validate_routing(table, grid, families, n_splits=args.folds, seed=args.seed)
    cv_test = permutation_test(cv.routed, cv.fixed)
    lo, hi = bootstrap_ci(list(cv.delta), seed=args.seed)
    rel = cv_test["observed"] / cv.fixed.mean() if cv.fixed.mean() else float("nan")

    print(f"\n=== family-conditioned routing, {args.folds}-fold cross-validated ===")
    print(f"  held-out queries          {cv.fixed.size}")
    print(f"  fixed global weight       {cv.fixed.mean():.3f}")
    print(f"  family-conditioned        {cv.routed.mean():.3f}")
    print(f"  difference                {cv_test['observed']:+.3f}  ({rel:+.1%})")
    print(f"  95% CI on the difference  [{lo:+.3f}, {hi:+.3f}]")
    print(f"  paired permutation p      {cv_test['p_value']:.4f}")
    verdict = "significant" if cv_test["p_value"] < 0.05 else "not significant"
    print(f"  verdict                   {verdict} at 0.05")
    print("  weights chosen per fold:")
    for i, (g, fam) in enumerate(zip(cv.fixed_alpha, cv.family_alpha)):
        picks = " ".join(f"{k}={v:.2f}" for k, v in sorted(fam.items()))
        print(f"    fold {i}: global={g:.2f}  {picks}")

    # One split could be lucky. Seed 0 stays the headline because it was fixed in advance;
    # the spread across seeds says whether that headline is stable, and is reported whatever
    # it shows rather than used to pick a seed.
    spread = []
    for s_ in range(args.stability_seeds):
        c = cross_validate_routing(table, grid, families, n_splits=args.folds, seed=s_)
        spread.append((float(c.delta.mean()), permutation_test(c.routed, c.fixed)["p_value"]))
    deltas = np.array([d for d, _ in spread])
    pvals = np.array([p_ for _, p_ in spread])
    print(f"\n  across {args.stability_seeds} fold assignments: difference "
          f"{deltas.min():+.3f} to {deltas.max():+.3f} (median {np.median(deltas):+.3f}), "
          f"p<0.05 in {(pvals < 0.05).sum()}/{len(pvals)}")

    # Kept for contrast, and labelled: this is the number the old analysis reported, with
    # selection and scoring on the same queries. The gap between it and the held-out figure
    # above is the size of the optimism that procedure introduced.
    in_sample_global = int(np.argmax(table.mean(axis=0)))
    in_sample_fixed = table[:, in_sample_global]
    in_sample_routed = np.zeros(len(queries))
    for family in sorted(set(families)):
        idx = np.flatnonzero(families == family)
        in_sample_routed[idx] = table[idx, int(np.argmax(table[idx].mean(axis=0)))]
    in_sample_delta = float(in_sample_routed.mean() - in_sample_fixed.mean())
    print(f"\n  for contrast, selecting and scoring on the same queries: "
          f"{in_sample_delta:+.3f}")

    print(f"\n=== chosen weights ===")
    print(f"{'query':52}" + "".join(f"{n:>11}" for n in routers) + f"{'best':>7}")
    print("-" * (52 + 11 * len(routers) + 7))
    for i, spec in enumerate(specs):
        best_for_query = float(grid[int(np.argmax(table[i]))])
        print(f"{spec['query'][:50]:52}"
              + "".join(f"{routers[n][i].alpha:>11.2f}" for n in routers)
              + f"{best_for_query:>7.2f}")

    RESULTS.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    out = RESULTS / f"router_{stamp}.json"
    out.write_text(json.dumps({
        "run_at": datetime.now(timezone.utc).isoformat(),
        "k": K, "normaliser": args.normaliser, "n_queries": len(queries),
        "query_set": args.queries,
        "fixed": {str(k): v for k, v in fixed.items()},
        "best_fixed_alpha": best_alpha, "oracle": oracle,
        "routers": rows, "per_family": per_family,
        "significance": significance, "significance_threshold": threshold,
        "family_routing_cv": {
            "folds": args.folds, "seed": args.seed,
            "held_out_queries": int(cv.fixed.size),
            "fixed": float(cv.fixed.mean()),
            "routed": float(cv.routed.mean()),
            "difference": cv_test["observed"],
            "relative": float(rel),
            "ci95": [lo, hi],
            "p_value": cv_test["p_value"],
            "significant": bool(cv_test["p_value"] < 0.05),
            "alpha_by_fold": [{"global": g, "family": f}
                              for g, f in zip(cv.fixed_alpha, cv.family_alpha)],
            "in_sample_difference": in_sample_delta,
            "stability": {
                "seeds": args.stability_seeds,
                "delta_min": float(deltas.min()), "delta_max": float(deltas.max()),
                "delta_median": float(np.median(deltas)),
                "significant_at_05": int((pvals < 0.05).sum()),
            },
        },
    }, indent=2))
    print(f"\nwrote {out.relative_to(ROOT)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
