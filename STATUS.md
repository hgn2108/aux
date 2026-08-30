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

## Upcoming

**Slice 1B — Search → play → observe** was added to the roadmap (DEC-005). It sits after
Slice 1 and is *not* active. Documentation and architecture only; no player implementation
until Slice 0 and Slice 1 have passed their gates.

**Slice 5 behavioural strategy** was refined by research (DEC-006): mechanisms will be
developed offline on Music4All-Onion behaviour with item representations from aux's own
encoder, then adapted and validated on first-party playback. Research and documentation
only — no behavioural implementation, and no dataset download, until Slice 5 is active.

## Blockers

none

## Pending Decisions

1. **Retained data** — 21 GB from the previous project is still on disk (FMA 15 GB,
   MagnaTagATune 5.7 GB, dim-sim 4.4 MB). FMA is a named dataset candidate; the others
   are probably not. Keep, prune, or delete?
2. **Artifact schema** — `EVALS.md`, `DECISIONS.md` and `docs/INIT_RESEARCH.md` are outside
   the canonical workflow set. Retained as supplementary; confirm this is wanted.
3. **Playback technology** — deliberately unresolved. Research when Slice 1B activates,
   not now.
4. **Music4All base-audio access** — open download or request-gated? Unverified and it
   gates the Slice 5 transfer path. Worth confirming early since access requests take
   time, even though the slice is far off.

> Machine-maintained workflow state.
> Claude should keep this current and concise.
> Do not use this file as a project diary.
