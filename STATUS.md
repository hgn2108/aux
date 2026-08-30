# Workflow Status

stage: DESIGN
gate: 1

## Current Slice

name: Slice 0 — Local media → searchable music
status: specifying

## Slice Contract

### Input

- ~200–500 public tracks (MTG-Jamendo subset), plus Song Describer where appropriate
- 10–30 private local files spanning MP3 / WAV / FLAC / M4A / MP4

Both go through the identical pipeline.

### Output

- Decoded, normalised `AudioAsset` per file
- One L2-normalised track embedding per file
- Reference-track nearest-neighbour results
- Song Describer text→music retrieval scores
- Latency and storage measurements

### Expected Behavior

**Hypothesis:** at least one pretrained joint music-text encoder can produce useful
retrieval over both public audio and arbitrary user-owned local media using the same
raw-audio pipeline.

Pass the slice if at least one representation:

- processes ≥95% of supported media
- performs meaningfully above weak/random retrieval on Song Describer
- gives plausible personal-library neighbours
- has practical compute/storage cost

Experiments: **E0** CLAP vs MuQ-MuLan · **E1** one segment vs 3–5 segment mean pooling.

## Next Action

Run Gate 1 (Design Ownership) on the proposed architecture before any implementation.

Irene should be able to answer, without being given the answers first:

1. Walk through the system from input to output.
2. Why this architecture?
3. Where does the ML/AI intelligence live?
4. How will you know whether the system works?
5. What assumption is most likely to fail?

## Blockers

none

## Pending Decisions

1. **Repository identity** — repo and directory are still named `aux` from the previous
   project; remote is `hgn2108/aux`. Rename to `vibesearch`?
2. **Retained data** — 21 GB from the previous project is still on disk (FMA 15 GB,
   MagnaTagATune 5.7 GB, dim-sim 4.4 MB). FMA is a named dataset candidate; the others
   are probably not. Keep, prune, or delete?
3. **Artifact schema** — `EVALS.md`, `DECISIONS.md` and `INIT_RESEARCH.md` are outside the
   canonical workflow set. Retained as supplementary; confirm this is wanted.

> Machine-maintained workflow state.
> Claude should keep this current and concise.
> Do not use this file as a project diary.
