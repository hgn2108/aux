# Project Workflow

This repository follows my shared Project Workflow.

## Startup

Before substantial project work:

1. Read `STATUS.md`.
2. Read `PROJECT.md`.
3. Read `DESIGN.md` when relevant.
4. Use the current state in `STATUS.md` to determine what should happen next.
5. Use the relevant Project Workflow skill when available.

## Sources of Truth

- `PROJECT.md` — project scope, intent, and MVP
- `DESIGN.md` — technical design and material decisions
- `STATUS.md` — current workflow state and exactly one next action
- `evals/` — raw technical evaluation evidence
- `README.md` — public-facing documentation

Project-specific supplements:

- `DECISIONS.md` — append-only history of accepted/rejected material decisions
- `docs/INIT_RESEARCH.md` — one-off research snapshot and evidence archive.
  **Do not load every session.** Read only when a design/research question requires it.
- `EVALS.md` — the layered evaluation plan; `evals/` holds raw evidence

Shared Project Workflow rules and gates are maintained outside this repository.

## Ownership

The user owns material decisions involving:

- product/problem framing
- architecture
- ML / AI approach
- evaluation design
- system boundaries
- meaningful technical tradeoffs

Claude may aggressively handle routine implementation.

Do not silently make a material decision on the user's behalf.

## Build Behavior

Build meaningful vertical slices.

For each meaningful slice:

SPECIFY → IMPLEMENT → VERIFY → UNDERSTANDING GATE → NEXT

Before implementation, define:

- input
- output
- expected behavior

Do not begin the next meaningful slice before the current one passes its gate.

## Project-specific rules

**Train/inference parity is non-negotiable.** Every representation the production system
needs must be reproducibly computable from arbitrary local audio files. Do not introduce
dataset-specific or proprietary features that cannot be generated at inference time.

**Do not fabricate a similarity target.** Musical similarity is subjective; similarity is
an emergent property of representations and ranking, never a supervised scalar label.

**Personal audio is an out-of-domain test, never training data**, and is never published
or redistributed.

**Evidence classification.** When recording a research finding, state whether it is a
verified fact, a reasonable inference, or a project recommendation. Cite primary sources —
model cards, config files, official repositories — not blog summaries.

**Experiments must change a decision.** Do not run experiments for volume. Every added
component must beat the simpler baseline on a stated metric or be removed.

## State

Keep `STATUS.md` current and concise.

Do not turn it into a work log.

If the user asks to continue or resume, read `STATUS.md` and continue from its recorded
next action.

## External Notes

Interview preparation and reusable personal knowledge live in Obsidian.

Do not create duplicate interview or general knowledge notes in this repository.
