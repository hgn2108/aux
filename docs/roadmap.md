# aux — Plan of Attack

Execution view. `project-context.md` carries motivation and the skills this project
should showcase; details here get refined as implementation teaches us things.

**How this plan works**

- **Vertical slices, not phases.** Ship a thin working path first, then deepen it.
- **Each model track owns its own data.** Source checks, ingestion, and processing
  happen inside the track that needs them, when it needs them — not as one upfront audit.
- **Baseline before sophistication.** Every track lands a crude measurable version
  before anything learned. Escalate only when evals justify it.
- **Tick boxes as you go.** Tasks are numbered (`B3`) for reference in conversation.

**Never cut, regardless of schedule:** the eval harness; descriptive (non-scoring)
Model C output; data-terms compliance; lyric embed-then-discard if Model D exists.

---

## M0 — Spine

One thin end-to-end path: real data → model → measured answer → served → agent tool.
Everything after this deepens a link in a chain that already works.

- [x] **M0.1** Confirm Last.fm API terms cover personal ML use; one line in `data-sources.md`
- [x] **M0.2** Secrets config — `.env.example`, typed settings loader, no keys in code
- [ ] **M0.3** Pull full scrobble history; archive raw responses before parsing
      > **BLOCKED 2026-08-27:** the Last.fm account (`irenehng`, created 2026-08-26)
      > has 0 scrobbles. Credentials verified working — there is simply no history.
      > Needs a backfill source before M0.4-M0.9 can proceed.
- [ ] **M0.4** Normalize to an events table: `(ts, artist, track, album)`
- [ ] **M0.5** Profile it — date range, unique tracks, play distribution, gaps
- [ ] **M0.6** Baseline recommender: popularity + item-item co-occurrence, no training
- [ ] **M0.7** Eval v0 — time-based holdout, recall@k, beat the popularity baseline
- [ ] **M0.8** FastAPI `/similar` endpoint over the baseline
- [ ] **M0.9** One agent tool calling that endpoint

**Done when:** you can ask for tracks similar to an anchor and get an answer whose
quality is a number you measured. Time-ordered split from the start — a random split
leaks the future and flatters every model you build after.

---

## Track B — Behavioral

**BLOCKED pending Spotify export.** The Last.fm account has no history (see M0.3).
Spotify's extended streaming history is the backfill source; Last.fm becomes the
live-refresh source once scrobbling is connected.

- [ ] **B0** Request Spotify extended streaming history — *do first, up to 30 days lead time*
- [ ] **B0.1** Connect Last.fm scrobbling so live data accumulates from now on
- [ ] **B0.2** Parse the export into the M0.4 events schema; reconcile with Last.fm going forward
- [ ] **B1** Sessionize scrobbles — session boundaries, repeat handling
- [ ] **B2** Decide refresh strategy: batch vs. live *(open decision)*
- [ ] **B3** Model approach *(open decision)* — implement matrix factorization / item2vec;
      move to a sequence model only if it beats this on B5
- [ ] **B4** MLflow tracking — introduced here, where there are versions worth comparing
- [ ] **B5** Eval — held-out next-track and co-occurrence retrieval vs. M0.6 baseline
- [ ] **B6** Cold-start + long-tail behavior; write down the failure modes
- [ ] **B7** Swap into the serving path behind M0.8

**MLE note:** the n-of-1 setting means a small, dense matrix. Favor methods that
work at that scale over methods that impress on paper.

---

## Track A — Acoustic

**Leads while the export is pending.** Needs no personal history: FMA is public
CC-licensed audio with its own labels, so embeddings can be genuinely evaluated
before any listening data exists.

Reasoning in `track-a-acoustic.md`; results in `results.md`; dead ends in `lessons.md`.

### Done

- [x] **A1** Licensing verified — FMA, AcousticBrainz, MagnaTagATune; audio sources
      investigated and closed (see `data-sources.md`)
- [x] **A2** Baseline on FMA precomputed features; harness built first
- [x] **A3** Baseline embeddings — `StandardScaler → PCA(128, whiten) → cosine`,
      chosen from a 20-configuration sweep
- [x] **A4** Harness targets — genre, artist, album retrieval with per-label chance rates
- [x] **A5** Own feature extraction from audio, validated on the harness
      (genre kNN 0.400 vs FMA reference 0.325)
- [x] **A6** Perceptual eval — MagnaTagATune triplets, cosine 0.397 vs 0.333 chance,
      stable across two feature pipelines

### Batch 1 — establish the ceiling, replace the objective

The gap that makes every other result ambiguous: we do not know whether weak retrieval
means weak features or a lossy projection.

- [x] **A7.1** Supervised ceiling — SVM (RBF) reaches 0.630 on 8 genres, matching FMA's
      published 0.63 on 16. The features are not the problem.
- [x] **A7.2** Log-transform heavy-tailed families — no measurable effect (`lessons.md`)
- [x] **A7.3** Supervised projections — LDA in 7 dims reaches kNN 0.629, matching the
      SVM ceiling; NCA 0.562; PCA 0.503
- [x] **A7.4** Decided: most of the gap was the classifier, not the projection. But LDA's
      artist/album retrieval collapses (0.038 / 0.021), so no single projection serves
      every axis — Batch 3 is now required by evidence, not preference.

### Batch 2 — pretrained embeddings

- [ ] **A8.1** Add torch + transformers; record weight licences in `data-sources.md`
- [ ] **A8.2** CLAP audio embeddings — 48 kHz input, 10 s window, 512-dim output.
      Own loading path; the 22050 Hz constant does not apply. Chunk 30 s clips and pool.
- [ ] **A8.3** Checkpoint choice — `larger_clap_music` for audio only; its text tower is
      degenerate. `clap-htsat-unfused` where text→audio is needed.
- [ ] **A8.4** Score on the full harness including perceptual triplets; compare to 0.397
- [ ] **A8.5** PANNs as a second candidate if CLAP underperforms

### Batch 3 — multi-axis subspaces

The published best design, and what Model C requires.

- [x] **A9.1** Partition features into axes — `features.AXES`, following Essentia's
      taxonomy (rhythm / tonal / timbre / dynamics)
- [x] **A9.2** Embed each axis separately; per-axis scores reported. Diagonal dominance
      confirmed: rhythm wins tempo 2.69x, tonal wins key 2.84x, dynamics wins loudness 2.32x
- [ ] **A9.3** Learn axis weights per query — equal weight dilutes tempo to 1.50x against
      rhythm-alone's 2.69x, so weighting must depend on what is being asked
- [ ] **A9.4** Expose per-axis distances in the output schema — this is Model C's
      non-scalar requirement arriving as a modelling decision, not a presentation layer

### Evaluation upgrade — done

- [x] **A10.1** Per-axis targets — tempo, key, loudness, brightness, each scored by its
      own MIREX-conventional rule
- [x] **A10.2** Chance and lift reported per axis. Result: clean diagonal dominance —
      rhythm wins tempo 2.69x, tonal wins key 2.84x, dynamics wins loudness 2.32x, all
      off-diagonals near chance. Concatenation scores 1.08x on tempo (chance) where the
      rhythm subspace scores 2.69x.
- [x] **A10.3** Rhythm descriptors added — tempo, onset rate, pulse clarity, onset
      envelope. The core 518 had no temporal information at all.
- [ ] **A10.4** Per-query axis weighting — equal-weight fusion dilutes tempo to 1.50x
      against rhythm-alone's 2.69x. Weights must depend on what is being asked.

### Later / blocked

- [ ] **A11** Coverage check — overlap between listening history and available audio.
      *Blocked on the Spotify export.*
- [ ] **A12** AcousticBrainz — CC0 features keyed by MBID; the only representation that
      reaches the user's own library. *Best done once A11 gives a library to join to.*
- [ ] **A13** Contrastive CNN (InfoNCE) at FMA Small scale. Positives must span
      *different* tracks sharing album/artist — same-song chunks let the network learn
      production signature instead of similarity. *Only if Batches 1–3 leave a gap.*
- [ ] **A14** State coverage honestly wherever acoustic signal is used

---

## Track C — Compatibility

The product surface, and where the non-judgmental principle is easiest to break.
Needs B and A.

- [ ] **C1** Define axes explicitly: tempo, energy, key, spectral, behavioral proximity
- [ ] **C2** Output schema — per-axis relationships plus confidence/coverage.
      **No scalar quality score.**
- [ ] **C3** Golden set — hand-labeled Bridge Finder anchor pairs and Vibe Match prompts,
      with a blind slice never inspected during development
- [ ] **C4** Bridge Finder — retrieve tracks that measurably reduce anchor gaps
- [ ] **C5** Vibe Match — natural language → target region; "closest available" fallback
- [ ] **C6** Confidence/coverage representation *(open decision)*
- [ ] **C7** Combine A + B signals; document the weighting and why
- [ ] **C8** Eval against C3; regression check wired into the harness
- [ ] **C9** Language audit — sweep user-facing strings for evaluative phrasing

**MLE note:** label C3 before looking at model output, or the labels drift toward
whatever the model already does and the eval stops meaning anything.

---

## Track D — Lyrical *(optional, gated)*

- [ ] **D1** **GATE** — Musixmatch terms permit this use. If not, omit Model D entirely.
- [ ] **D2** Embed-then-discard pipeline; raw lyric text never stored, logged, or displayed
- [ ] **D3** Automated test asserting no lyric text reaches disk or logs
- [ ] **D4** Embedding approach + minimum viable coverage *(open decision)*
- [ ] **D5** Thematic distance in Bridge Finder or Vibe Match only *(open decision)*

---

## Cross-cutting

Pulled in when a track needs them, not built speculatively.

- [ ] **X1** Structured recommendation-event logging — request, results, model versions
- [ ] **X2** Feedback capture: accept / reject / refine
- [ ] **X3** Preference pairs → contrastive reweighting of Model B *(open decision: method + cadence)*
- [ ] **X4** Preference-loop proxy metrics *(open decision)*; framed as small-scale, not RLHF
- [ ] **X5** Docker compose — API + MLflow + vector store
- [ ] **X6** Vector DB *(open decision)* — only when linear scan actually hurts
- [ ] **X7** Drift + latency monitoring; thresholds written down
- [ ] **X8** MPD co-occurrence pretraining *(open decision: sample size/value)*; 2017 catalog caveat
- [ ] **X9** README, results writeup incl. what didn't work, limitations, demo

---

## Open decisions

Resolved in: B2 refresh · B3 model approach · A7 encoder · C6 confidence repr ·
D4 embedding · D5 thematic placement · X3 reweighting · X4 proxy metrics ·
X6 vector DB · X8 MPD value

Log each resolution in the `project-context.md` decision log per `AGENTS.md`.

---

## Order

`M0.1-M0.2` (done) → `A` on FMA → `B` when the export lands → `M0` spine completed
on real history → `C` → `X` as needed → `D` if the gate passes.

A1's coverage check moves to *after* the export arrives — it needs a library to
overlap against. If that overlap turns out thin, Track A covers a small slice and
Track C has to lean on behavioral signal, which changes what C is.
