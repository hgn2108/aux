# Workflow Status

stage: BUILD
gate: 1 — waived, not passed (2026-09-06)

## Current Slice

name: Slice 0 — Local media → searchable music
status: implementing

## Slice Contract

### Input

- ~200–500 public tracks (MTG-Jamendo subset), plus Song Describer where appropriate
- 10–30 private local files, MP3 for now (DEC-008: other formats supported in code,
  validated later)

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

**Eval 0B — acquire Song Describer and measure text->music retrieval.** The encoder layer
is built and CLAP produces a working joint space, but retrieval quality is still unmeasured
against any benchmark; the only evidence so far is a six-track qualitative smoke test.

Then E0 (add the MuQ-MuLan adapter and compare) and E1 (single vs 3-5 segment pooling).

Instrument during indexing, since both are cheap and both are named Slice 0 risks:
**embedding distribution statistics** (mean pairwise cosine, to detect a collapsed space on
out-of-domain audio) and **within-track segment variance** (which is what makes an E1
result interpretable rather than a guess). `embed_track` already returns segment vectors
for this reason.

### Implemented

- `src/aux/ingest/` — `SourceAdapter -> MediaProbe -> AudioDecoder -> AudioAsset`,
  decoding via PyAV (bundles its own FFmpeg; no system dependency).
- `src/aux/encode/` — encoder adapter contract, deterministic segment selection,
  per-encoder resampling, normalise-pool-normalise track vectors, and a LAION-CLAP adapter
  running on MPS.
- `scripts/eval_0a_ingestion.py` — Eval 0A harness; writes JSON evidence and a Markdown
  report to `evals/`.
- `tests/` — 45 tests. Ingestion contract over generated fixtures in all five formats;
  encoder-layer tests are model-free apart from the checkpoint guard.

Runtime requires a **native arm64 interpreter** — torch has no macOS x86_64 wheels for
Python 3.13, so an x86_64 Python under Rosetta cannot install it at all (DEC-010).

### Eval 0A — first evidence

Corpus: FMA small, 8,000 files, MP3 only. Full results in
`evals/eval_0a_fma_small_full_20260906.{md,json}`.

- Decode success **99.92%** (7,994/8,000); gate is >=95%. **PASS**, with the format caveat
  below.
- Deterministic on repeat decode, so the content-hash cache is sound.
- probe p50 0.9 ms, decode p50 27.6 ms, ~1098x realtime over 66.6 h of audio. Ingestion is
  not a bottleneck at personal-library scale.
- Numbers of record are the native arm64 run (`eval_0a_fma_small_arm64_20260906.*`). An
  earlier x86_64-under-Rosetta run gave identical success rate and identical failures at
  ~40% worse latency, which confirms ingestion is deterministic across architectures and
  not merely across runs (DEC-010).

**The six failures separate into two distinct causes, which is the point of categorising
them:**

- 3 x `container_unparseable`, all 1.3-1.8 KB — too small to be valid MP3s at all. The
  demuxer cannot open them.
- 3 x `decode_failed`, 140-300 KB — plausible files that open fine and fail mid-stream on
  `avcodec_send_packet`. Corrupt payload, not a truncated download.

Both are corpus damage rather than pipeline defects, and they are cheap to distinguish
only because the failure carries a category. A single "6 files failed" number would not
have separated them.

**Two findings worth carrying forward:**

1. **Sample rate genuinely varies in the wild** — 44.1 kHz (7,569), 48 kHz (411) and
   22.05 kHz (14) in a single corpus, with 85 mono files among 7,909 stereo. Direct
   evidence for the DESIGN.md decision to preserve source rate and resample per encoder
   rather than normalising at ingest.
2. **Demuxer selection is extension-driven**, so a corrupt file carrying a media extension
   can open successfully and present an audio stream declaring a sample rate of 0. Probe
   now rejects this as `container_unparseable`. Left unguarded it would have produced
   assets with nonsense durations and silently corrupted downstream measurements.

**Format scope (DEC-008).** This corpus is MP3 only, and the gate is claimed for MP3
only. WAV / FLAC / M4A / MP4 remain supported in code and pass on generated fixtures, but
real-world pathologies in those formats (DRM-protected M4A, unusual MP4 stream layouts)
are unobserved. Revalidated when a real need arises; the harness is corpus-agnostic, so
that is one command per corpus.

### Gate 1 — waived

Design Ownership was **not** self-answered. Under timeline pressure the five answers were
written by Claude and read by Irene (2026-09-06). The design is understood but not
rehearsed.

Outstanding: Irene should re-derive Q2 (architecture choices and their rejected
alternatives) and Q5 (most-likely-to-fail assumption and its observable symptoms) unaided
before the Slice 0 write-up is treated as portfolio-ready.

## Upcoming

**Slice 1B — Search → play → observe** was added to the roadmap (DEC-005). It sits after
Slice 1 and is *not* active. Documentation and architecture only; no player implementation
until Slice 0 and Slice 1 have passed their gates.

**Slice 2 planner** is now specified as an LLM emitting schema-constrained structured
output, with planner-output metrics, a validated LLM judge, and a distillation experiment
(DEC-007). Position in the roadmap is unchanged — it still comes after Slice 1 and still
has to beat the whole-query baseline. Documentation only; no planner implementation.

**Slice 5 behavioural strategy** was refined by research (DEC-006): mechanisms will be
developed offline on Music4All-Onion behaviour with item representations from aux's own
encoder, then adapted and validated on first-party playback. Research and documentation
only — no behavioural implementation, and no dataset download, until Slice 5 is active.

## Blockers

none

## Pending Decisions

1. **Artifact schema** — `EVALS.md`, `DECISIONS.md` and `docs/INIT_RESEARCH.md` are outside
   the canonical workflow set. Retained as supplementary; confirm this is wanted.
2. **Playback technology** — deliberately unresolved. Research when Slice 1B activates,
   not now.
3. **Music4All base-audio access** — open download or request-gated? Unverified and it
   gates the Slice 5 transfer path. Worth confirming early since access requests take
   time, even though the slice is far off.
4. **`data/interim/fma_features_rhythm.parquet`** (5.9 MB) — a leftover derived artifact
   from the previous project, with no role in any aux slice. Keep or delete?

### Resolved

- Retained data (DEC-009): MagnaTagATune and dim-sim deleted; FMA retained as the Eval 0A
  ingestion corpus.
- Format validation scope (DEC-008): MP3 now, other formats later.

> Machine-maintained workflow state.
> Claude should keep this current and concise.
> Do not use this file as a project diary.
