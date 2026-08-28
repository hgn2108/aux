"""Reproducible Track A sweeps. Every published number comes from here.

Run with:  python -m aux.experiments.sweeps
"""

import argparse
from itertools import product

from aux.eval.embeddings import EmbeddingReport, evaluate
from aux.experiments import tracking
from aux.models.acoustic import build_embeddings, load_subset

DIMS = (16, 32, 64, 128, 256)
METRICS = ("euclidean", "cosine")
WHITEN = (False, True)


def _headline(report: EmbeddingReport) -> str:
    cells = [f"{name} {r.precision_at_k:.3f}" for name, r in report.labels.items()]
    return "  ".join(cells)


def sweep(subset: str = "small", k: int = 5) -> list[tuple[dict, EmbeddingReport]]:
    """Grid over dimensionality, whitening, and distance metric."""
    tracking.setup()
    features, labels = load_subset(subset)

    results = []
    for n_components, whiten, metric in product(DIMS, WHITEN, METRICS):
        params = {
            "source": "fma",
            "subset": subset,
            "n_components": n_components,
            "whiten": whiten,
            "metric": metric,
            "k": k,
        }
        embeddings = build_embeddings(features, n_components, whiten)
        report = evaluate(embeddings, labels, k=k, metric=metric)
        name = f"pca{n_components}-{'whiten' if whiten else 'raw'}-{metric}"
        run_id = tracking.log_report(name, params, report)

        print(f"{name:<28} {_headline(report)}  run={run_id[:8]}")
        results.append((params, report))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset", default="small")
    parser.add_argument("-k", type=int, default=5)
    args = parser.parse_args()
    sweep(args.subset, args.k)


if __name__ == "__main__":
    main()
