# Workflow Status

stage: BUILD
gate: 1 — waived, not passed (2026-09-06)

## Current Slice

name: Slice 0 — Local media → searchable music
status: implementing

## Slice Contract

### Input

- Song Describer (706 tracks / 1,106 captions) as the public retrieval benchmark; FMA
  small (8,000 MP3s) as the ingestion corpus
- **50–150 private local tracks** for the out-of-domain neighbour test. The original
  10–30 was set before the pipeline existed and is too thin: with 30 tracks each
  neighbour is drawn from 29 candidates, which cannot distinguish a good space from a
  lucky one. Genre spread and personal familiarity matter more than raw count — the
  judgement is "are these plausible neighbours", which only Irene can make.

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

**Slice 0 is substantively complete.** All four pass conditions are met; what remains is
the encoder licensing call (below) and then the Slice 0 gate.

After that, Slice 1 — natural-language retrieval with per-query-category human relevance,
which is the first evidence about *aux's actual task* rather than about exact-track
identification from a caption.

### Encoder: MuQ-MuLan (DEC-012, accepted)

CC-BY-NC weights accepted for a portfolio/research project. CLAP stays as the shippable
fallback at a measured cost of -0.099 R@10; swapping is one config change and a re-index.

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

### Slice 0 pass conditions

| Condition | Status |
|---|---|
| ≥95% decode/encode success, failures categorised | **met** — 99.92% on 8,000 MP3s; MP3 only (DEC-008) |
| Meaningfully above weak/random retrieval | **met** — R@10 0.407 vs 0.014 chance, 29× |
| Plausible personal-library neighbours | **met** — Eval 0C, 160 tracks / 6 genres; genre purity 5.1× chance, same-artist 6.0× |
| Practical compute/storage cost | **met** — 0.63 s/track, 8.7 ms/query, 2 KB/track |
| At least one meaningful model comparison, recorded | **met** — E0 (DEC-012) and E1 (DEC-011) |

**All five conditions are met.** Slice 0's hypothesis holds: a pretrained joint music-text
encoder produces useful retrieval over both public audio and arbitrary user media through
one raw-audio pipeline.

### Eval 0C — out-of-domain library (2026-09-07)

160 personal tracks across six genres (94 hip-hop/R&B, 20 jazz, 14 EDM, 13 v-pop, 12 DnB,
7 classical), MuQ-MuLan, 5 segments. Audited first: 160/162 decode; the two failures were
zero-byte downloads and were deleted. No silent, truncated or duplicate files.

**The homogeneity hypothesis was tested and confirmed.** The earlier single-genre run gave
a mean track cosine of 0.724 — 0.32 above the benchmark, crossing a naive collapse
threshold. The prediction was that this reflected a narrow library rather than a failing
encoder, and that diversifying genres would pull it down. It did:

| | 94 tracks, 1 genre | 160 tracks, 6 genres |
|---|---:|---:|
| mean track cosine | 0.724 | **0.518** |
| delta vs benchmark | +0.318 | **+0.113** |
| same-artist lift | 4.0× | **6.0×** |
| hubness (top 1% share) | 3.8% | **2.2%** |

**Genre structure — the sharper test:**

| genre | n | within | between | top-5 purity | chance |
|---|---:|---:|---:|---:|---:|
| hip-hop/R&B | 94 | 0.724 | 0.424 | 92.3% | 58.5% |
| jazz | 20 | 0.405 | 0.311 | 69.0% | 11.9% |
| EDM | 14 | 0.702 | 0.465 | 78.6% | 8.2% |
| v-pop | 13 | 0.723 | 0.500 | 80.0% | 7.5% |
| DnB | 12 | 0.732 | 0.433 | 85.0% | 6.9% |
| classical | 7 | 0.606 | **0.157** | 88.6% | 3.8% |

Within-genre 0.710 vs between-genre 0.401 (separation +0.309); overall top-5 genre purity
82.2% against 16.1% chance, a 5.1× lift. **No collapse**, on a test that does not depend on
where the global mean happens to land.

Three details worth keeping:

- **Classical behaves exactly as a control should** — between-genre cosine 0.157, far below
  every other genre, and 23× chance purity. It was included precisely because a space that
  cannot separate orchestral music from trap is broken; it separates it cleanly.
- **Jazz has the lowest within-genre cosine (0.405)** and the lowest purity (69%). That is
  the vocal/instrumental split doing its job: jazz was deliberately assembled as ~8
  instrumental and ~5 vocal, so internal spread is the expected result, not a defect. It
  also shows the separation is not merely vocal-presence detection.
- **EDM sits closest to hip-hop** (between 0.465, highest of the new genres), consistent
  with sharing programmed drums, synth bass and heavy compression.

**Text queries improved where the library gained material**, which is the behaviour a
working system should show: "upbeat energetic party track" moved from hip-hop to Calvin
Harris and J-Lo (0.348 → 0.424); "warm and nostalgic" now returns quiet piano and acoustic
pieces (0.291 → 0.350); "dark and menacing" surfaces a DnB remix at rank 1.

**Cross-lingual generalisation, unplanned and notable:** Vietnamese ballads take the top
two slots for "melodic and melancholy, sung rather than rapped" (0.523, 0.504) — an English
query retrieving Vietnamese-language music. MuQ-MuLan documents English and Chinese text
support; this suggests the *audio* tower generalises past its text languages.

**One query did not improve: "dreamy atmospheric production with reverb"** stayed at 0.189
despite classical and ambient material arriving. Either the library still lacks it or the
encoder handles that descriptor poorly. Carried into Slice 1 as a question rather than
resolved here.

### E0 — encoder comparison (2026-09-07)

| Encoder | seg | R@1 | R@5 | R@10 | median | MRR | s/track | hubness |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| CLAP | 5 | 0.063 | 0.193 | 0.307 | 27 | 0.141 | 0.26 | 4.9% |
| **MuQ-MuLan** | 5 | **0.090** | **0.270** | **0.407** | **15** | **0.189** | 0.63 | 8.0% |

Paired: 256 queries gained the top 10 against 146 lost, p = 4.5e-08. Holds at matched 1
segment (p = 2.6e-10), so it is the encoder and not the pooling. **E1 replicates on
MuQ-MuLan** — multi-segment wins (p = 1.05e-04), 3 vs 5 stays unseparated.

Recorded for Slice 6: MuQ-MuLan retrieves better but covers less catalogue (55 of 706
tracks never retrieved, vs 36 for CLAP).

### Eval 0B — text→music retrieval (2026-09-06)

CLAP `laion/larger_clap_music_and_speech` on Song Describer (1,106 captions, 706 candidate
tracks), single centre segment:

| Metric | Measured | Random | Lift |
|---|---:|---:|---:|
| Recall@1 | 0.049 | 0.0014 | 34× |
| Recall@5 | 0.165 | 0.0071 | 23× |
| Recall@10 | 0.252 | 0.0142 | 18× |
| Median rank | 33 / 706 | 353.5 | — |

**Slice 0's "meaningfully above weak/random retrieval" condition is met.** Cost is a
non-issue: 0.18 s/track indexing, 4.3 ms/query, 2 KB/track — a 5,000-track library indexes
in ~15 minutes for 10 MB, which retires any argument for ANN indexing at this scale.

**The space is healthy**, which matters more than the recall figure. Track–track cosine
mean 0.296 with sd 0.190 — no collapse. Hubness mild: top 1% of tracks take 5.1% of top-10
slots, 40 of 706 never surface. So the number is a genuine ranking result, not an artefact
of a degenerate space; that distinction is what stops later effort going into the wrong
layer.

**What this does not say.** The benchmark scores exact-track identification from a
descriptive caption — narrower than aux's use case, where many tracks may legitimately
satisfy a query, and where returning ten similar tracks scores zero unless the specific one
is among them. Treat R@1 as a floor on usefulness, not a ceiling.

**E1: multi-segment pooling wins** (DEC-011). R@10 0.252 → 0.307, median 33 → 27, paired
p = 5.2e-7. Within-track segment cosine ~0.82 explains why: real intra-track variation
exists for pooling to capture. 3 vs 5 segments is unseparated — see Awaiting Irene above.

**Benchmark caveat.** Song Describer ships 2-minute excerpts, not full tracks. Five 10 s
windows cover ~42% of a 2-minute clip but only ~17% of a 5-minute song, so E1's conclusion
is established on excerpts. The direction should hold or strengthen on full tracks, but
that is an inference.

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
