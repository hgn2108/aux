# aux — Plan of Attack

Working checklist derived from `PROJECT_CONTEXT.md`. The Google Doc PRD remains
canonical for product intent; this file is the execution view.

**How to use:** tick boxes as you go. Tasks are numbered (`3.2`) so they can be
referenced in conversation and commits. Phases are ordered by dependency, not by
date — no deadlines are invented here.

**Phase gate:** each phase has a Done-when line. Do not start the next phase
until it holds. Two rules override convenience at every phase:

- The eval harness is never cut (principle 3).
- Model C output stays descriptive, never a quality score (principle 1).

---

## Phase 0 — Foundations

Nothing here is ML. It exists so later phases are reproducible and legally clean.

- [ ] **0.1 Repo scaffolding**
  - [ ] 0.1.1 Package layout (`aux/` source, `tests/`, `notebooks/`, `data/`, `configs/`)
  - [ ] 0.1.2 Dependency + env management (choose: uv / poetry / conda) and pin Python version
  - [ ] 0.1.3 `.gitignore` for `data/`, model artifacts, `.env`, MLflow local store
  - [ ] 0.1.4 Lint/format/typecheck (ruff + mypy) and pre-commit hooks
  - [ ] 0.1.5 `pytest` wired up with one trivial passing test
- [ ] **0.2 Secrets and config**
  - [ ] 0.2.1 `.env.example` listing every credential (Last.fm API key, etc.)
  - [ ] 0.2.2 Config loader with typed settings; no credentials in code or notebooks
- [ ] **0.3 Data-source terms verification — BLOCKING GATE**
  - [ ] 0.3.1 Re-verify Last.fm API terms permit personal ML use; record date checked
  - [ ] 0.3.2 Re-verify MusicBrainz licensing + rate-limit/user-agent requirements
  - [ ] 0.3.3 Confirm FMA and/or Jamendo CC license tiers usable for audio features
  - [ ] 0.3.4 Confirm Spotify Web API audio-features status; if unverified, mark unusable
  - [ ] 0.3.5 Record Genius as excluded; record Musixmatch decision (permitted or dropped)
  - [ ] 0.3.6 Write `DATA_SOURCES.md` with per-source verdict, date, and quoted terms link
- [ ] **0.4 MLflow tracking server** running locally, logging a dummy run end to end

**Done when:** a clean clone can install, lint, test, and log an MLflow run; and
`DATA_SOURCES.md` gives a dated yes/no for every source in the PRD.

---

## Phase 1 — Data ingestion

- [ ] **1.1 Last.fm listening history (behavioral spine)**
  - [ ] 1.1.1 Full scrobble backfill for the user account, paginated + resumable
  - [ ] 1.1.2 Raw response archival before any parsing (replayable without refetch)
  - [ ] 1.1.3 Normalize to canonical event schema: `(ts, artist, track, album, source)`
  - [ ] 1.1.4 Decide + implement refresh strategy — resolves open decision *batch vs. live scrobble refresh*
  - [ ] 1.1.5 Profile the history: date range, unique tracks/artists, play distribution, gaps
- [ ] **1.2 MusicBrainz enrichment**
  - [ ] 1.2.1 Resolve scrobbles to MBIDs; measure match rate
  - [ ] 1.2.2 Manual/fuzzy fallback for unmatched high-play tracks
  - [ ] 1.2.3 Attach metadata (release date, artist, tags); respect rate limits
- [ ] **1.3 CC-licensed audio (FMA / Jamendo)**
  - [ ] 1.3.1 Determine overlap between listening history and CC-available audio — *reality check on Model A coverage*
  - [ ] 1.3.2 Download audio for the covered subset; store checksums + license per track
  - [ ] 1.3.3 Document the coverage ceiling honestly (principle 2)
- [ ] **1.4 Unified track table** joining behavioral, metadata, and audio-availability keys

**Done when:** one reproducible pipeline command rebuilds the track table from raw
archives, and coverage numbers for each signal are written down.

---

## Phase 2 — Evaluation harness (BUILD BEFORE MODELS)

Deliberately ahead of the models. Built after them, it gets cut under time pressure —
and the PRD forbids that.

- [ ] **2.1 Golden set construction**
  - [ ] 2.1.1 Hand-label Vibe Match cases: natural-language prompt → acceptable acoustic/thematic ranges
  - [ ] 2.1.2 Hand-label Bridge Finder cases: anchor pairs + what "between" should mean per axis
  - [ ] 2.1.3 Hold out a blind slice never inspected during development
- [ ] **2.2 Metrics**
  - [ ] 2.2.1 Bridge Finder: measurable gap reduction between anchors across axes
  - [ ] 2.2.2 Vibe Match: fraction of results inside golden-set ranges
  - [ ] 2.2.3 Model B: held-out next-track / co-occurrence retrieval metrics
  - [ ] 2.2.4 Coverage + confidence reporting as a first-class metric (principle 2)
- [ ] **2.3 Harness mechanics**
  - [ ] 2.3.1 One command runs all evals against a named model version
  - [ ] 2.3.2 Results logged to MLflow and diffable across versions
  - [ ] 2.3.3 Regression check that fails loudly when a metric drops
  - [ ] 2.3.4 Baselines: random, popularity, raw-metadata nearest-neighbor

**Done when:** the harness runs green against stub models and produces a comparison
report. Every later model phase must move a number in this harness.

---

## Phase 3 — Model A: acoustic / content embeddings

- [ ] **3.1 Feature extraction** — tempo, key, spectral, MFCC/mel from CC audio
- [ ] **3.2 Architecture decision** — resolves open decision *Model A architecture*
  - [ ] 3.2.1 Start with classical features + dimensionality reduction as the baseline
  - [ ] 3.2.2 Only escalate to a learned encoder if the baseline underperforms in Phase 2
  - [ ] 3.2.3 Log the decision + rationale to the decision log
- [ ] **3.3 Fixed-length embeddings** for every audio-covered track, versioned in MLflow
- [ ] **3.4 Sanity checks** — known-similar tracks land near each other; genre clusters emerge without genre labels
- [ ] **3.5 Coverage statement** — which fraction of the library has acoustic vectors

**Done when:** embeddings exist for the covered subset, beat the Phase 2 baselines,
and the coverage gap is documented rather than hidden.

---

## Phase 4 — Model B: behavioral / taste embeddings

- [ ] **4.1 Approach decision** — resolves open decision *sequence vs. co-occurrence/factorization*
  - [ ] 4.1.1 Implement the simpler co-occurrence/factorization baseline first
  - [ ] 4.1.2 Evaluate a sequence model only if the baseline is beaten meaningfully
  - [ ] 4.1.3 Log the decision + rationale
- [ ] **4.2 Training data** — sessionize scrobbles; define session boundaries; handle repeats
- [ ] **4.3 Train/held-out split that respects time** (no future leakage into past)
- [ ] **4.4 Train + evaluate** against Phase 2 metrics
- [ ] **4.5 Cold-start behavior** for tracks with few plays — state it, don't paper over it
- [ ] **4.6 Version in MLflow** with data snapshot reference

**Done when:** Model B beats popularity baseline on held-out history, and its
failure modes are written down.

---

## Phase 5 — Model C: Bridge Finder + Vibe Match

The product surface. Where the non-judgmental principle is easiest to violate.

- [ ] **5.1 Multi-axis compatibility output**
  - [ ] 5.1.1 Define axes explicitly (tempo, energy, key, spectral, behavioral proximity)
  - [ ] 5.1.2 Return per-axis relationships — **no scalar quality score** (principle 1)
  - [ ] 5.1.3 Output schema carries confidence + coverage per axis (principle 2)
- [ ] **5.2 Bridge Finder** — given two anchors, retrieve tracks that reduce measured gaps
  - [ ] 5.2.1 Define "between" per axis
  - [ ] 5.2.2 Decide thematic-distance inclusion — resolves open decision *thematic distance in Bridge Finder vs. Vibe Match-only*
- [ ] **5.3 Vibe Match** — natural language → acoustic/behavioral target region
  - [ ] 5.3.1 Prompt-to-target-region mapping
  - [ ] 5.3.2 Confidence/coverage representation — resolves open decision *Vibe Match confidence/coverage representation*
  - [ ] 5.3.3 "Closest available" fallback wording when the library is thin
- [ ] **5.4 Combine Model A + B signals**; document the weighting and why
- [ ] **5.5 Evaluate against the Phase 2 golden set**
- [ ] **5.6 Language audit** — sweep every user-facing string for evaluative phrasing

**Done when:** both features beat baselines on the golden set, and 5.6 passes.

---

## Phase 6 — Serving

- [ ] **6.1 Vector database** — resolves open decision *vector DB choice*
  - [ ] 6.1.1 Pick based on library size (likely small — favor simplicity over scale)
  - [ ] 6.1.2 Index Model A + B embeddings; benchmark query latency
  - [ ] 6.1.3 Log the decision + rationale
- [ ] **6.2 FastAPI service** — endpoints for similar / bridge / vibe-match / track-profile
  - [ ] 6.2.1 Typed request/response schemas carrying confidence + coverage
  - [ ] 6.2.2 Model version pinned and reported in every response
- [ ] **6.3 Docker** — reproducible image; compose file for API + MLflow + vector DB
- [ ] **6.4 Load model artifacts from the MLflow registry**, not local paths
- [ ] **6.5 API tests** against a fixture library

**Done when:** `docker compose up` serves working endpoints with versioned models.

---

## Phase 7 — Agent layer

- [ ] **7.1 Tool definitions** — models and retrieval exposed as real callable tools
  - [ ] 7.1.1 `find_similar`, `bridge`, `vibe_match`, `track_profile`, `listening_recall`
  - [ ] 7.1.2 Tool schemas surface confidence + coverage so the agent can hedge honestly
- [ ] **7.2 Conversation layer** handling the five PRD user intents
- [ ] **7.3 System prompt encoding principles 1 and 2** — descriptive, coverage-honest
- [ ] **7.4 Agent-level eval** — scripted conversations asserting correct tool selection
- [ ] **7.5 Refusal//hedge behavior** when the library genuinely lacks a match

**Done when:** scripted conversations pass, and the agent never volunteers a taste judgment.

---

## Phase 8 — Event logging + preference loop

- [ ] **8.1 Structured recommendation-event logging** — request, results, model versions, timestamps
- [ ] **8.2 Feedback capture** — accept / reject / refine
- [ ] **8.3 Preference-pair construction** from feedback events
- [ ] **8.4 Contrastive reweighting of Model B** — resolves open decision *reweighting method and cadence*
- [ ] **8.5 Proxy metrics** — resolves open decision *preference-loop proxy metrics*
- [ ] **8.6 Version reweighted models in MLflow**; regression-check against Phase 2
- [ ] **8.7 Framing check** — documented as small-scale preference learning, explicitly **not** Spotify-scale DPO/RLHF

**Done when:** feedback measurably shifts recommendations and the shift is tracked
across model versions.

---

## Phase 9 — Monitoring

- [ ] **9.1 Drift detection** on input distributions (listening habits shift over time)
- [ ] **9.2 Embedding distribution monitoring** across model versions
- [ ] **9.3 Latency monitoring** at useful percentiles
- [ ] **9.4 Dashboard** — functional first; polish is explicitly cuttable
- [ ] **9.5 Alert thresholds** written down, even if manual

**Done when:** a rebuild that degrades quality is visible without reading code.

---

## Phase 10 — Optional breadth (strictly in cut order)

Only after Phases 1–9 form a coherent system. Cut from the bottom up.

- [ ] **10.1 MPD co-occurrence pretraining** — resolves open decision *MPD sample size/value*
  - [ ] 10.1.1 Sample size experiment
  - [ ] 10.1.2 Measure whether it beats Model B trained on history alone
  - [ ] 10.1.3 Document the 2017 catalog cutoff as a known limitation
- [ ] **10.2 Model D — lyrical/thematic embeddings** *(build only if 10.2.1 passes)*
  - [ ] 10.2.1 **GATE:** Musixmatch terms permit this use — if not, omit Model D entirely
  - [ ] 10.2.2 Embed-then-discard pipeline: raw lyric text never stored, displayed, logged, or redistributed
  - [ ] 10.2.3 Automated test asserting no lyric text reaches disk or logs
  - [ ] 10.2.4 Embedding approach + minimum viable coverage — resolves that open decision
  - [ ] 10.2.5 Integrate thematic axis per the 5.2.2 decision

**Done when:** each optional item either ships with evals or is explicitly recorded as cut.

---

## Phase 11 — Portfolio surface

The PRD frames this as portfolio-quality work; that framing needs its own artifact.

- [ ] **11.1 README** — problem, architecture diagram, model boundaries, results
- [ ] **11.2 Results writeup** — eval numbers vs. baselines, including what didn't work
- [ ] **11.3 Architecture decision records** for each resolved open decision
- [ ] **11.4 Demo** — recorded walkthrough or reproducible notebook
- [ ] **11.5 Limitations section** — coverage gaps, n-of-1 caveats, cold start (principle 2 applied to the project itself)

---

## Open decisions → where they get resolved

| Open decision | Resolved in |
|---|---|
| Vector DB choice | 6.1 |
| Model A architecture | 3.2 |
| Model B sequence vs. co-occurrence | 4.1 |
| Batch vs. live scrobble refresh | 1.1.4 |
| Preference reweighting method + cadence | 8.4 |
| Preference-loop proxy metrics | 8.5 |
| MPD sample size / value | 10.1 |
| Vibe Match confidence/coverage representation | 5.3.2 |
| Model D embedding approach + coverage | 10.2.4 |
| Thematic distance in Bridge Finder vs. Vibe Match-only | 5.2.2 |

Per `AGENTS.md`, log each resolution in the `PROJECT_CONTEXT.md` decision log with
date, rationale, and affected PRD section.

---

## Critical path

`0.3` (terms verification) → `1.1` (history) → `2.x` (eval harness) → `4.x` (Model B)
→ `5.x` (Model C) → `6.x` (serving) → `7.x` (agent)

Model A (Phase 3) parallelizes with Model B but is **coverage-limited by 1.3.1** —
run that overlap check early, since a thin CC-audio overlap reshapes how much of the
product can lean on acoustic signal.
