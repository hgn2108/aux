"""Does routing the fusion weight beat fixing it — and does the LLM beat the free baselines?

The crossover established that no single fusion weight serves both query families, and that
an oracle picking per query gains +0.042 NDCG@10 over the best fixed weight. That is the
ceiling. This measures how much of it each router actually captures, and what each costs.

Three routers, in increasing order of price:

- **rules** — keyword and phrase matching. Free, instant.
- **prototype** — the query embedded once and compared to two small prototype sets. Free,
  about a millisecond, no hand-written vocabulary.
- **claude** — a model call per query. Roughly half a second and a fraction of a cent.

Measured end to end, not as classification accuracy: each router's weight is fed to the
actual retrieval and scored against objective relevance. A router that classifies well but
picks unhelpful weights should lose, and accuracy alone would hide that.

**Relevance.** Queries name a genre, a theme, or both. Genre comes from the folder layout,
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
from aux.eval import bonferroni_threshold, ndcg_at_k, permutation_test  # noqa: E402
from aux.index import build_index  # noqa: E402
from aux.ingest.asset import content_hash  # noqa: E402
from aux.recommend import NORMALISERS  # noqa: E402
from aux.route import PrototypeRouter, Route, RuleRouter  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / ".cache"
EVALS = ROOT / "evals"
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
    ap.add_argument("--queries", default="routing_queries_expanded.json",
                    help="query set under evals/; the expanded set has the statistical "
                         "power the 24-query original lacked")
    args = ap.parse_args()

    specs = json.loads((EVALS / args.queries).read_text())["queries"]
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

    sys.path.insert(0, str(ROOT / "scripts"))
    from eval_recommendation import load_lyric_vectors
    from eval_semantic import fuse, score_queries

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
        row = fuse(audio_scores[i : i + 1], lyric_scores[i : i + 1], alpha, has_lyrics, norm)[0]
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
        router = ClaudeRouter(cache=cache)
        start = time.perf_counter()
        routes = router.route_all(queries)
        timings[router.name] = (time.perf_counter() - start) / len(queries) * 1000
        if router.failures:
            # Every fallback returns the same weight, so a fully failed arm looks like a
            # confident constant. Refuse to report it as a measurement.
            raise SystemExit(f"the claude router failed on {router.failures}/{len(queries)} "
                             f"queries; fix the cause or rerun with --no-claude")
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

    # --- family-level routing -----------------------------------------------------------
    # The routers above choose per query and mostly fail. This asks the prior question:
    # is the *decision* worth making at all? One weight per family, fitted on that family,
    # is what a perfect family classifier would achieve — above any per-query router, below
    # the per-query oracle. If this is significant while the routers are not, the gap is an
    # implementation problem rather than a missing effect.
    global_col = int(np.argmax(table.mean(axis=0)))
    baseline_per_query = table[:, global_col]
    routed_per_query = np.zeros(len(queries))
    family_alpha = {}
    for family in sorted(set(families)):
        idx = np.flatnonzero(families == family)
        col = int(np.argmax(table[idx].mean(axis=0)))
        routed_per_query[idx] = table[idx, col]
        family_alpha[family] = float(grid[col])
    family_test = permutation_test(routed_per_query, baseline_per_query)

    print(f"\n=== family-level routing (one weight per family) ===")
    for family, alpha in family_alpha.items():
        print(f"  {family:12} alpha={alpha:.2f}")
    print(f"  routed {routed_per_query.mean():.3f} vs fixed alpha="
          f"{grid[global_col]:.2f} {baseline_per_query.mean():.3f}   "
          f"{family_test['observed']:+.3f}  p={family_test['p_value']:.4f}")

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
        "family_routing": {"alpha_by_family": family_alpha,
                           "routed": float(routed_per_query.mean()),
                           "fixed": float(baseline_per_query.mean()),
                           **family_test},
    }, indent=2))
    print(f"\nwrote {out.relative_to(ROOT)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
