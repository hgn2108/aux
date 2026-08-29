# Project instructions

This file is `CLAUDE.md` via symlink — one file, two names. Edit `AGENTS.md`.

## Where things live

Check here before searching. If a file is not listed, it probably should not exist.

| Path | Holds | Go here when |
|---|---|---|
| `docs/project-context.md` | Product rationale, principles, decision log | Starting any work; recording a decision |
| `docs/roadmap.md` | Numbered tasks, what is done and next | Choosing what to work on |
| `docs/results.md` | Every experiment's numbers and interpretation | Looking up a result, or writing one up |
| `docs/lessons.md` | Approaches abandoned, with evidence | Before retrying something; after a dead end |
| `docs/track-a-acoustic.md` | Model A reasoning and approach ladder | Choosing an acoustic method |
| `docs/data-sources.md` | Licensing register, verified dates | Before using any new data or weights |
| `aux/config/` | Typed settings, paths | Adding config or a credential |
| `aux/ingest/` | Dataset loaders (`fma`, `mtat`, `lastfm`) | Reading a corpus |
| `aux/models/` | `features` (audio→descriptors), `acoustic` (descriptors→embedding) | Changing a representation |
| `aux/eval/` | `embeddings` — the harness, metrics, triplets | Changing how anything is scored |
| `aux/experiments/` | One module per experiment; `tracking` wraps MLflow | Running or adding an experiment |
| `tests/` | Runs with no dataset present | Adding a test |
| `notebooks/` | Exploration and interpretation, committed with figures | Explaining a result visually |
| `data/` | Corpora. Structure tracked, contents gitignored | Never edited by hand |

Results go in `docs/results.md`, never in code comments or `data-sources.md`.
Reasoning goes in `docs/`, never duplicated into docstrings.

Before planning or implementing project work, read `docs/project-context.md`.

The canonical product source is the linked Google Doc PRD recorded there. Treat Sections 2–5 of that PRD as stable product rationale. Implementation details in Sections 6+ may be superseded by the repository.

## Research the literature before modelling

Before choosing or building any modelling approach, research the current literature and practice for that specific problem. Do not rely on training-data recall, and do not default to the first reasonable-sounding method.

This applies whenever the question is *which approach should we use* — representations, embeddings, similarity metrics, evaluation design, training objectives, model selection.

1. Search for how the problem is currently solved, what the published baselines are, and what the reported numbers look like on comparable data. Know the benchmark before running the experiment.
2. Confirm technical details from primary sources — model cards, config files, official repositories — not blog summaries. Sample rates, embedding dimensions, licences, and known defects have all differed from secondary accounts on this project.
3. Verify licensing of model weights as carefully as data licensing. Weights and code often carry different licences.
4. Search for known failure modes of a candidate before adopting it.
5. Prefer breadth over depth early: three configurations across five method families beats twenty configurations of one. Tuning inside a single family without checking the family is the wrong one is the most expensive mistake available.
6. Establish the ceiling before optimising toward it. Know what a strong supervised method extracts from the same inputs, so a weak result can be attributed to the representation rather than guessed at.
7. Record what was rejected and why in `docs/lessons.md`, with the evidence. Dead ends are results.

## Recording decisions

When a meaningful product or architecture decision is made during implementation:

1. Update `docs/project-context.md` with the decision, date, rationale, and affected PRD section.
2. If the decision changes the canonical product specification, propose or make the matching Google Doc update when the user asks or confirms that the PRD should be synchronized.
3. Never silently weaken the non-judgmental product principle, evaluation requirements, data-source constraints, or lyric embed-then-discard rule.

## Repository standards

This repository is shown to employers. Keep it legible.

- Code states what runs; `docs/` states why. Neither duplicates the other.
- Superseded approaches are removed from the code and summarised in `docs/lessons.md`. Do not leave dead paths in place for reference.
- Every published number is reproducible by one command and logged to MLflow.
- Nothing written belongs at the repository root except `README.md` and this file.
