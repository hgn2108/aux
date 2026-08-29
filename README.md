# aux

A personal music intelligence platform. It learns one listener's musical world from three
independent signals — how music *sounds*, how it is actually *listened to*, and what it is
*about* — and answers conversationally.

It describes relationships; it never scores whether taste is good.

## What it does

- **Bridge Finder** — given two tracks, find what sits between them, measured per axis.
- **Vibe Match** — natural language to a region of acoustic and behavioral space.
- **Acoustic placement** — where a track sits beyond its genre label.
- **Listening recall** — what was played, when.

## Setup

```bash
conda env create -f environment.yml
conda activate aux
pip install -e .
pytest -q
```

Copy `.env.example` to `.env` and add credentials for the sources you need. `.env` is
gitignored and never committed.

## Layout

| Path | Contents |
|---|---|
| `aux/` | Library code — ingestion, models, evaluation, experiments |
| `aux/experiments/` | Reproducible experiments; every published number comes from here |
| `tests/` | Runs without any dataset download |
| `notebooks/` | Exploration and interpretation, committed with figures |
| `docs/` | All written documentation (see below) |
| `data/` | Datasets. Structure tracked, contents never committed |

## Documentation

Everything written lives in `docs/`. Nothing else belongs at the repository root.

| Document | Purpose |
|---|---|
| [project-context.md](docs/project-context.md) | Product rationale, principles, decision log. Start here. |
| [roadmap.md](docs/roadmap.md) | Execution plan with checkable tasks |
| [results.md](docs/results.md) | Experiment index and what each result meant |
| [data-sources.md](docs/data-sources.md) | Licensing register — what we verified and when |
| [track-a-acoustic.md](docs/track-a-acoustic.md) | Model A reasoning and approach ladder |
| [lessons.md](docs/lessons.md) | Approaches abandoned, with the evidence |

`AGENTS.md` (and its `CLAUDE.md` symlink) sits at the root by convention — it is the
working protocol for coding sessions, not project documentation.

## Experiments

```bash
python -m aux.experiments.baseline_pca   # reruns the Tier 0 baseline grid
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

## Status

Model A (acoustic) has a measured baseline on FMA. Model B (behavioral) is blocked on a
pending Spotify data export. See [roadmap.md](docs/roadmap.md).

## Attribution

Uses the [FMA dataset](https://github.com/mdeff/fma) (metadata CC BY 4.0),
[AcousticBrainz](https://acousticbrainz.org) (CC0), and MagnaTagATune. Per-source terms
are recorded in [data-sources.md](docs/data-sources.md).
