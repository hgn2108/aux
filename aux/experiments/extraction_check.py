"""A4b — validate our extraction against FMA's precomputed reference.

Two questions, in order:

1. **How closely do we track FMA's reference?** Per-family rank correlation.
2. **Does our feature set carry the same information?** Both scored on the same
   harness. This is the question that decides whether the extractor is usable.

**Correlation is not a correctness test here, and low correlation is expected.**
FMA's reference features were computed from 44.1 kHz audio while letting librosa
default to ``sr=22050`` in the feature calls, so their frequency axis is
mislabelled by 2x — verified by extracting a track three ways and comparing to
their stored values (their numbers match the native-rate, default-sr variant, and
the correctly-labelled variant is exactly double).

We deliberately do not reproduce that. Every corpus is resampled to a fixed rate
(``features.SAMPLE_RATE``) so FMA, MagnaTagATune, and later AcousticBrainz share
one frequency axis. Matching FMA's numbers would mean matching their bug and
giving up cross-corpus comparability, which the whole architecture depends on.

So the bar is the harness comparison, not the correlation.

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


def _column_correlations(ours: pd.DataFrame, reference: pd.DataFrame) -> pd.Series:
    """Spearman correlation per column, across tracks, indexed by (family, statistic)."""
    values = {}
    for col in ours.columns:
        a, b = ours[col], reference[col]
        if a.std() > 0 and b.std() > 0:
            values[(col[0], col[1])] = scipy_stats.spearmanr(a, b).statistic
    return pd.Series(values).groupby(level=[0, 1]).median()


def family_correlations(ours: pd.DataFrame, reference: pd.DataFrame) -> pd.Series:
    """Median Spearman correlation per feature family, across tracks."""
    per_column = _column_correlations(ours, reference)
    return per_column.groupby(level=0).median().reindex(FAMILY_SIZES).sort_values()


def statistic_correlations(ours: pd.DataFrame, reference: pd.DataFrame) -> pd.Series:
    """Median correlation per summary statistic, pooled across families.

    Separates two very different failure modes: a descriptor computed wrongly
    (a whole family disagrees) versus higher moments being inherently unstable
    across decoders and framing (skew and kurtosis disagree everywhere, while
    mean and median hold).
    """
    return _column_correlations(ours, reference).groupby(level=1).median().sort_values()


def run(n: int = 300, workers: int = 8, seed: int = 0) -> None:
    ensure_audio_extracted()
    tracking.setup()

    tracks = fma.load_tracks()
    reference = fma.load_features()
    ids = fma.subset(tracks, "small").intersection(reference.index)

    rng = np.random.default_rng(seed)
    sample = pd.Index(rng.choice(ids, size=min(n, len(ids)), replace=False))

    print(f"extracting {len(sample)} tracks with {workers} workers ...")
    # rhythm=False: FMA's reference has no rhythm block, so the frames must match.
    ours = extract_many(
        {int(t): fma.audio_path(int(t)) for t in sample}, workers=workers, rhythm=False
    )

    common = ours.index.intersection(reference.index)
    ours, ref = ours.loc[common], reference.loc[common]
    print(f"extracted {len(common)} of {len(sample)}")

    # 1. Correctness
    corr = family_correlations(ours, ref)
    print("\nper-family Spearman correlation with FMA reference:")
    for family, value in corr.items():
        print(f"  {family:<20} {value:+.3f}")
    print(f"  {'OVERALL median':<20} {corr.median():+.3f}")

    by_stat = statistic_correlations(ours, ref)
    print("\nper-statistic (pooled across families):")
    for stat, value in by_stat.items():
        print(f"  {stat:<20} {value:+.3f}")

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
