"""A4b — validate our extraction against FMA's precomputed reference.

Two questions, in order:

1. **Is the extractor correct?** Per-family rank correlation between our features
   and FMA's for the same tracks. This is the check A4a existed to make possible.
2. **Does it carry the same information?** Both feature sets scored on the same
   harness. Correlation can be high while retrieval quality differs, so agreement
   on the metric that matters is a separate question.

Exact equality is not expected: librosa versions, decoder behaviour, and framing
all differ. Strong rank agreement plus comparable retrieval is the bar.

Run with:  python -m aux.experiments.extraction_check --n 300 --workers 8
"""

import argparse
import zipfile

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from aux.config.settings import get_settings
from aux.eval.embeddings import evaluate
from aux.experiments import tracking
from aux.ingest import fma
from aux.models.acoustic import LABEL_COLUMNS, build_embeddings
from aux.models.features import FAMILY_SIZES, extract_many

AUDIO_ARCHIVE = "fma/fma_small.zip"


def ensure_audio_extracted() -> None:
    """Unpack the audio archive once; skip if already present."""
    raw = get_settings().raw_dir
    target = raw / "fma/fma_small"
    if target.exists():
        return

    archive = raw / AUDIO_ARCHIVE
    if not archive.exists():
        raise FileNotFoundError(f"{archive} not found — download fma_small.zip first")

    print(f"extracting {archive.name} ...")
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(raw / "fma")


def family_correlations(ours: pd.DataFrame, reference: pd.DataFrame) -> pd.Series:
    """Median Spearman correlation per feature family, across tracks."""
    out = {}
    for family in FAMILY_SIZES:
        a = ours.xs(family, axis=1, level=0)
        b = reference.xs(family, axis=1, level=0)
        # Correlate each column across tracks, then take the family median.
        cols = [
            scipy_stats.spearmanr(a[c], b[c]).statistic
            for c in a.columns
            if a[c].std() > 0 and b[c].std() > 0
        ]
        out[family] = float(np.nanmedian(cols)) if cols else float("nan")
    return pd.Series(out).sort_values()


def run(n: int = 300, workers: int = 8, seed: int = 0) -> None:
    ensure_audio_extracted()
    tracking.setup()

    tracks = fma.load_tracks()
    reference = fma.load_features()
    ids = fma.subset(tracks, "small").intersection(reference.index)

    rng = np.random.default_rng(seed)
    sample = pd.Index(rng.choice(ids, size=min(n, len(ids)), replace=False))

    print(f"extracting {len(sample)} tracks with {workers} workers ...")
    ours = extract_many({int(t): fma.audio_path(int(t)) for t in sample}, workers=workers)

    common = ours.index.intersection(reference.index)
    ours, ref = ours.loc[common], reference.loc[common]
    print(f"extracted {len(common)} of {len(sample)}")

    # 1. Correctness
    corr = family_correlations(ours, ref)
    print("\nper-family Spearman correlation with FMA reference:")
    for family, value in corr.items():
        print(f"  {family:<20} {value:+.3f}")
    print(f"  {'OVERALL median':<20} {corr.median():+.3f}")

    # 2. Equivalent information on the harness
    labels = {k: tracks.loc[common, c].to_numpy() for k, c in LABEL_COLUMNS.items()}
    keep = pd.notna(labels["genre"])
    labels = {k: v[keep] for k, v in labels.items()}

    print("\nharness comparison (same tracks, same settings):")
    metrics: dict[str, float] = {}
    for name, frame in (("ours", ours), ("fma_reference", ref)):
        dims = min(128, len(common) - 1)
        report = evaluate(build_embeddings(frame[keep], n_components=dims), labels, metric="cosine")
        print(f"\n{name}:\n{report}")
        metrics |= {f"{name}_{k}": v for k, v in report.to_metrics().items()}

    metrics |= {f"corr_{k}": v for k, v in corr.items()}
    run_id = tracking.log_run(
        f"a4b-extraction-check-n{len(common)}",
        {"n_tracks": len(common), "workers": workers, "seed": seed},
        metrics,
    )
    print(f"\nlogged run={run_id[:8]}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=300)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    run(args.n, args.workers)


if __name__ == "__main__":
    main()
