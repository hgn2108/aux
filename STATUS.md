# Workflow Status

stage: BUILD
gate: 1 — waived, not passed (2026-09-06)

## Current Slice

name: Slice 3 — Audio + lyrical semantics
status: implementing

## Slice Contract

### Input

- The 160-track library, already indexed by audio
- Transcribed lyrics for each track, from its own audio

### Output

- Lyric-based retrieval alongside audio retrieval
- A measured comparison: audio only, lyrics only, both (E3)

### Expected Behavior

**Hypothesis:** two Slice 1 failures are lyrical and unfixable by any audio method —
"r&b songs about yearning" had the *highest* mean score of any query (0.398), matching
everything moderately because the encoder cannot isolate what a song is about; and
"sung in Vietnamese" returns jazz despite 13 v-pop tracks.

### Transcription — done (DEC-022)

160/160 transcribed, 0 failures, 18x realtime. 127 usable for lyric search; 32 flagged
instrumental (7/7 classical, 11/20 jazz); **13/13 Vietnamese detected**, exactly the v-pop
folder.

Whisper's language detection needed E1's fix — it detects from one 30-second window and got
3/5 on v-pop, then *translated* a Vietnamese song into English. Voting over five windows
gets 13/13.

The two signals cross-validate unplanned: 12 tracks reported as Javanese and 3 as Norwegian
Nynorsk are **all 15 flagged instrumental**. Whisper was guessing at audio with nothing sung
in it.

### E3 — lyrics work, and naive fusion is harmful (DEC-023)

**Test 1 — find a track from a line of its own lyrics** (126 queries, 160 candidates):

| modality | R@1 | R@10 | median rank |
|---|---:|---:|---:|
| audio | 0.008 | 0.063 | 64 |
| **lyrics** | **0.460** | **0.706** | **2** |
| fused | 0.056 | 0.349 | 14 |
| *chance* | *0.006* | *0.062* | *80* |

Audio is **exactly at chance**, as it must be — no audio encoder identifies a song from its
words. Lyrics put the right track at median rank 2 of 160. **Fusion halves that**, because
rank fusion weights both inputs equally and one of them knows nothing.

**Test 2 — language** (13 Vietnamese tracks in the top 13): audio 5.7, lyrics 6.3,
**fused 8.7**. Here fusion is best, because both modalities carry partial signal.

**This is the routing evidence Slice 2 could not produce.** The same algorithm helps in one
test and halves the other; the difference is whether both modalities know anything about the
query. The rule is narrow and testable: do not consult a modality that cannot answer.

Also visible: audio scores 0/13 on "sung in Vietnamese" and 9/13 on "a song with Vietnamese
lyrics". The phrasing sensitivity from Slice 1 has not gone away.

## Slice 2 outcome## Slice 2 outcome## Slice 2 outcome

**Shipped:** negation handling. A contrastive encoder cannot represent "not X", so the query
is split and the exclusion applied at score level. Excluded-genre leakage fell 0.22 to 0.08,
and a rated query moved **1.00 to 5.00** — the largest single measured gain in the project.

**Built, measured, off by default:** the LLM query planner. It helps vague queries (+1.07 on
ones the baseline handled badly) and *harms* specific ones (-0.58), so applied
indiscriminately it nets to nothing (+0.05, p = 1.000). It also costs ~1.7 s and an internet
connection. Available behind `use_planner`; DEC-021 records the rule for when it is worth
enabling, and that rule is not validated on held-out queries.

**Measured and rejected:** fixed context lexicon (DEC-016), five reranking methods
(DEC-015), rank fusion (DEC-017).

**Established about *why* rewriting works**, tested against ground truth: it closes a gap
between the user's language and the encoder's. Where no gap exists — Song Describer
captions, already written in sound-describing language — it has no effect (Phase C).

**A design principle that outlives the component** (DEC-021): gate on how confidently *this
library* answers, not on what kind of query it is. Two signals doing two jobs — an absolute
top score to detect "the library cannot answer" (20/22 on synthetic gaps), and a
library-relative percentile to decide whether rewriting is worth it. A fixed threshold does
not transfer; z_top rises monotonically with collection size, 2.23 at 26 tracks to 3.54 at
706.

`src/aux/search.py` is now the single entry point carrying everything that survived.

## Next Action

**Irene's call: Slice 3 (lyrics) or Slice 1B (player).** Her instinct is lyrics first.

The case for lyrics is measured, not speculative — two Slice 1 failures are lyrical:
"r&b songs about yearning" had the *highest* mean score of any query (0.398), meaning it
matched everything moderately because the audio encoder cannot isolate lyrical content; and
"sung in Vietnamese" returns jazz despite 13 v-pop tracks, because language identity is not
represented.

**The open question is sourcing, and PROJECT.md already constrains it.** Lyrics APIs match a
track against a database, which is exactly the failure DEC-001 avoided — it does not work
for arbitrary local files. Transcribing the audio does, and is the only source satisfying
train/inference parity. That needs deciding before any of Slice 3 is built.

**The case for the player:** nothing in this project can currently be shown to anyone.
There is no interface. It also unblocks Slice 5, which needs behavioural data that only
in-product playback produces.

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

### Gates — both waived, answers read not derived

**Gate 1 (Design Ownership), 2026-09-06** and the **Slice 0 gate, 2026-09-07** were both
waived under timeline pressure. Claude wrote the answers and Irene read them. The design and
the results are understood but not rehearsed.

Slice 0 gate questions, kept for later self-testing:

1. Walk the pipeline as built — a file to a ranked result. Where does the channel downmix
   happen, where does the sample-rate conversion happen, and why are they in different
   places?
2. E0 — the evidence for MuQ-MuLan, why it is trustworthy, what it cost, and the four things
   it does *not* establish.
3. E1 — what it established, what it did not, and the honest reason 5 segments was chosen.
4. The 0.724 cosine — what it actually was, the two pieces of evidence that settled it, and
   what would have convinced you the encoder was broken.
5. What does Slice 0 not tell you? Name the biggest remaining unknown.

Follow-ups: why R@1 = 0.090 is not damning; why E0 holding at both 1 and 5 segments matters
more than the headline; why same-artist retrieval is valid evidence with no labelling; why
E0 must hold the caption set fixed; what the CLAP checkpoint incident generalises to.

Still outstanding from Gate 1: Q2 (architecture choices and their rejected alternatives) and
Q5 (most-likely-to-fail assumption). Q5 is now partly answered by evidence.

**Question 4 is the one worth rehearsing.** The naive reading said failure; the correct
reading required a second, relative measurement and a prediction that could have falsified
it. It is the strongest evidence in the project that decisions here are made from evidence
rather than assumption — which is what PROJECT.md claims as the hiring signal.

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
