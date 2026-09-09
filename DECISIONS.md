# DECISIONS.md

> **Audience:** Shared, but Claude is expected to maintain it.
> **Purpose:** Append-only history of material architecture/product decisions.
> **Update frequency:** Only when a meaningful decision is accepted/rejected.
> **Rule:** Do not duplicate transient status here.

# aux — Decision Log

## DEC-001 — Local/user-provided media is the canonical product input

**Status:** Accepted

**Decision:**  
Do not depend on Spotify or another streaming platform for the core system.

**Why:**  
The previous design risked train/inference mismatch because public datasets and streaming APIs exposed different features. Raw user-provided media lets the same preprocessing and encoding pipeline run in development and inference.

**Implication:**  
Support local audio/video files and treat decoded audio as the canonical acoustic input.

**Revisit if:**  
A future platform integration exposes legally permitted raw/compatible inputs without breaking train/inference parity.

---

## DEC-002 — Do not supervise a fabricated scalar acoustic-similarity target

**Status:** Accepted

**Decision:**  
Use pretrained representations and retrieval/ranking evaluation instead of inventing pairwise similarity labels.

**Why:**  
Musical similarity/vibe is subjective and multi-dimensional.

**Implication:**  
Similarity becomes a retrieval/ranking construct, not a fake universal target.

---

## DEC-003 — Establish a simple joint text/music baseline before routed multimodal retrieval

**Status:** Accepted

**Decision:**  
Build whole-query text->music retrieval first.

**Why:**  
The more complex query-planner architecture must prove incremental value against a strong simple baseline.

**Implication:**  
Slice 1 exists before routed retrieval.

---

## DEC-004 — Personalization must be subordinate to explicit query intent

**Status:** Accepted

**Decision:**  
Behavioral history may refine ambiguity but may not override explicit current-query constraints.

**Default preference hierarchy:**

1. current query
2. current session
3. intent-specific profile
4. global preference

**Why:**  
A user may explicitly request music outside their normal taste.

---

## DEC-005 — Include a lightweight local playback layer

**Status:** Accepted

**Decision:**
aux will include a minimal in-product music player: play/pause, seek, next/previous,
queue, play directly from a search or recommendation result, basic metadata, and explicit
feedback (save, like/dislike, good match / bad match).

Added to the roadmap as **Slice 1B — Search → play → observe**, positioned after simple
natural-language search works.

**Why:**
Local media is already the system's canonical input, so playback is cheap to add and
requires no new data source. More importantly it closes the feedback loop:

```text
query -> recommendation impression -> user selects track -> playback
      -> behavioural event -> later preference model
```

Without it, Slice 5 personalization would have to assume access to listening history from
another platform, or rely on synthetic behaviour — which can test software but cannot
support any claim about effectiveness.

**Tradeoffs accepted:**
Some engineering effort goes to a component that is not the intellectual centre. Bounded
by an explicit non-goal list: no library-management UI, streaming, social features,
crossfade, equalizer, elaborate playlists, synchronized lyrics, or general player polish.
The player must not displace retrieval, ranking and recommendation as the project's
technical story.

**Evidence:**
- Direct evidence: none. The research ledger does not establish this.
- Transfer evidence: intent-based personalization from repeated relevance feedback
  (INIT_RESEARCH §7) assumes a channel through which feedback arrives.
- Engineering inference: **primary basis.** This is a product/system-design judgement
  about closing the feedback loop, not a finding from the research ledger.

**Implication:**
DESIGN.md gains a playback boundary, playback/queue state, a behavioural event model
(SearchRequest, RecommendationImpression, PlaybackEvent, ExplicitFeedback), an append-only
behaviour store, and impression→play attribution. Event types are not equivalent
preference evidence; the strong/medium/weak weighting is a hypothesis to evaluate.

Personalization authority is unchanged: explicit query > current session > intent profile
> global preference. Behavioural history never overrides an explicit query constraint.

**Removal/revisit condition:**
Revisit if the player starts consuming time that belongs to retrieval/ranking work, or if
impression→play attribution proves unreliable enough that behavioural data cannot support
personalization evaluation.

**Files updated:**
- PROJECT.md: product framing, third capability, Slice 1B in roadmap, player non-goals
- DESIGN.md: playback layer section, updated target architecture, open question 10
- EVALS.md: Slice 1B evaluation, Slice 5 real-behaviour data source, new stop condition
- STATUS.md: Slice 1B recorded as upcoming; current slice unchanged

---

## DEC-006 — Develop personalization on public behaviour; validate on first-party

**Status:** Accepted

**Decision:**
aux will not wait for first-party behaviour to accumulate before developing the
behavioural layer. Public behavioural datasets may be used to develop and pretrain general
preference, session and intent mechanisms.

Two constraints bind that permission:

1. Any behavioural model intended for production must have a feasible path to arbitrary
   user-uploaded tracks through **content representations computable at inference time**.
2. Final behavioural models must be validated and adapted using **first-party aux
   behaviour**. Public results are offline benchmarks, never product claims.

**Why:**
Waiting for first-party data would stall the behavioural layer for months. It is also
unnecessary: a dataset exists that satisfies the transfer constraint.

**The transfer path:**

```text
Music4All 30s audio -> aux's own encoder -> item representation
                                       +
Music4All-Onion behaviour (same 109,269 track ids)
                                       |
                                       v
             preference / session / intent mechanism
                                       |
                                       v
                  applies to arbitrary local audio
```

Because item representations come from **our** encoder over raw audio rather than from
dataset-specific IDs or dataset-specific precomputed features, a mechanism learned on
public data applies to a file the model has never seen. A model learning
`user_id -> external_song_id` would not.

**Evidence:**
- Direct evidence: Music4All-Onion is CC BY 4.0, openly downloadable, 109,269 tracks,
  119,140 users, 252,984,396 listening records, with timestamps. Music4All provides
  30-second 44.1 kHz clips for the same tracks. A RecSys replicability study found
  BERT4Rec's published results not reproducible under default configuration, and
  nearest-neighbour methods beating BERT4Rec/GRU4Rec/SASRec on some datasets — supporting
  simple baselines first. Playlist-title research shows short natural-language intent
  labels carry usable semantics, especially in cold start.
- Transfer evidence: playlist-title-conditioned recommendation is structurally similar to
  aux's free-text queries, but not the same task.
- Engineering inference: that a shared encoder makes the mechanism transfer. Reasonable,
  and **untested**.
- Open hypothesis: that external pretraining actually helps at small first-party history
  sizes. This is E10c and could fail.

**Tradeoffs accepted:**
External listeners are not aux users. Last.fm scrobbling is passive; aux behaviour is
deliberate action on an explicit query. The two may not carry the same preference
semantics — reported separately, never pooled.

Music4All ships 30-second centre clips only, so behavioural research is inherently
single-segment. If E1 chooses multi-segment pooling, training and production
representations diverge and the gap must be measured.

**Licensing/data implications:**
Music4All-Onion CC BY 4.0 (permissive). Music4All A+A is CC BY-NC-SA 4.0 — prefer Onion.
Music4All base-audio access terms unverified and gate the transfer path. Spotify MPD is no
longer directly downloadable and must not be critical path. LFM-1b appears unavailable.
MSD Taste Profile has no timestamps and no audio path — CF benchmark only.

**Removal/revisit condition:**
If E10c shows external pretraining does not beat a simple first-party content centroid at
realistic history sizes, external pretraining does not ship — it stays documented research
and the product uses the simple profile. If Music4All base audio proves inaccessible, the
transfer path collapses and the strategy must be reconsidered.

**Files updated:**
- PROJECT.md: Slice 5 refined; personalization-data section; open questions 7-8
- DESIGN.md: behavioural personalization architecture; open questions 11-13
- EVALS.md: E10a/E10b/E10c; two separated data sources; new stop conditions
- STATUS.md: unchanged slice; recorded as future work
- docs/INIT_RESEARCH.md: Appendix A, dated 2026-08-30

---

## DEC-007 — Slice 2's query planner is an LLM producing structured output

**Status:** Accepted

**Current slice:** Slice 0 (this decision governs Slice 2; nothing changes now)

**Question:**
Open Design Question 3 left "exact query-planner model" unresolved. What produces the
structured interpretation of a free-form query?

**Decision:**
Slice 2's planner is an **LLM emitting a schema-constrained structured interpretation** of
the query — facets, constraints, negation, operator/relation, reference transform — rather
than a classifier, rule set, or fine-tuned tagger.

Slice 2 additionally gains:

1. **Structured-output correctness as a first-class metric**, not just downstream retrieval
   quality. Schema-conformance rate and field-level precision/recall/F1 on the labelled
   query set.
2. **LLM-as-judge, validated against the human protocol.** The judge is calibrated on
   human ratings from the existing protocol and reported with its agreement against them.
   It is a scaling mechanism for evaluation, never a replacement for human judgment.
3. **A distillation path (E2a).** Once the LLM planner has produced labelled structured
   outputs, train a small local model to reproduce them and compare quality, latency and
   cost. This is the project's tuning/training story, earned by data rather than added for
   its own sake.

Slice 2's position in the roadmap is unchanged. DEC-003 still binds: the planner must beat
whole-query embedding on stated metrics or it does not ship.

**Why:**
The planner already existed in the roadmap with an unnamed model, and `EVALS.md` already
planned the labelled query set that evaluates it. Naming it an LLM answers an open question
rather than widening scope, and it does so with a built-in counterfactual — per-category
ablation against a working baseline — which is the part most LLM work omits.

Query interpretation is also the only place in aux where an LLM is genuinely the right
tool: it is text→structure over open-vocabulary expressive language, where a fixed label
set would fail on unseen phrasing.

**Alternatives considered:**
- *Classifier / rule-based extractor* — cannot generalise over open-vocabulary vibe
  language; would need a closed facet taxonomy the product does not have.
- *Fine-tuned tagger as the first move* — no labelled data exists yet to fine-tune on.
  E2a reaches the same place afterwards, with data the LLM produced.
- *An agent loop (multi-step plan → tool call → observe → revise)* — **rejected.** A search
  query needs one or two interpretation calls. There is no long-horizon task, no tool
  surface, and no failure-recovery requirement to justify a loop. Building one would
  violate non-negotiable 10: complexity without an experiment that changes a decision.

### Evidence
- Direct evidence: none in the research ledger. This is a component-selection judgement.
- Transfer evidence: playlist-title research (INIT_RESEARCH §7, cited in DEC-006) shows
  short natural-language intent labels carry usable structured semantics.
- Engineering inference: **primary basis.** That open-vocabulary query interpretation is
  better served by a general language model than a closed taxonomy.
- Open hypothesis: that decomposition beats whole-query embedding at all. Published
  retrieval results on query decomposition are mixed. This is E2 and it may fail.

**Experiment:** E2 (whole-query vs decomposition), E2a (distilled planner vs LLM planner).

**Result:** not yet run.

**Tradeoffs accepted:**
An LLM planner introduces a hosted-API dependency into a project framed as local-first.
Accepted on the grounds that only the *query text* leaves the machine — never user audio,
never the library, never behavioural events. E2a's distilled local planner is the intended
resolution of that tension, not an afterthought; if it matches the LLM's quality, aux
returns to fully local operation.

Second tradeoff: a negative E2 result is a real possibility. That is acceptable. A measured
"decomposition wins only on negation and contrast queries, so it routes only those" is a
valid and reportable outcome under the existing removal conditions.

**Licensing/data implications:**
Query text sent to a hosted model must not include file paths, library contents, or any
personal-library metadata — query string only. Personal audio remains local and
unredistributed, unchanged.

**Removal/revisit condition:**
If E2 shows no per-category improvement over whole-query embedding, the planner does not
ship and Slice 2 stays documented research. If the local distilled planner (E2a) matches
the hosted model, the hosted dependency is removed from the product path.

**Files updated:**
- PROJECT.md: hiring signal (LLM/eval skills), Slice 2 roadmap row, open questions 3 and 9
- DESIGN.md: query interpretation layer section; open design questions 3 and 14
- EVALS.md: Slice 2 structured-output metrics, judge validation, E2a
- STATUS.md: recorded under Upcoming; current slice unchanged

**Understanding check:**
Why is this not an agent? Why must the planner beat a baseline that already works?

---

## DEC-008 — Validate ingestion on MP3 now; defer the other four formats

**Status:** Accepted

**Current slice:** Slice 0

**Question:**
Eval 0A passed at 99.92% on 8,000 MP3s, but WAV / FLAC / M4A / MP4 were exercised only by
generated test fixtures. Does Slice 0 block on assembling real files in those formats?

**Decision:**
No. Proceed on MP3.

- The **code** continues to support all five formats; nothing is removed, and the decoder
  path is format-agnostic by construction.
- The **validated** set is MP3 only. Eval 0A's gate is claimed for MP3 and explicitly not
  claimed for the rest.
- The other four are revalidated when a real need arises — a user library containing them,
  or the private out-of-domain test set being assembled.

**Why:**
The ingestion contract is one code path: probe, decode, downmix, preserve rate. MP4 is not
a separate pipeline but a container whose audio stream enters that same path, and the
generated fixtures already show every format reaching an AudioAsset. The residual risk is
therefore about *real-world file pathologies* (DRM-protected M4A, unusual MP4 stream
layouts), not about untested code.

Weighed against that: Slice 0's actual hypothesis is about whether a pretrained joint
encoder produces useful retrieval at all. Blocking that on format coverage would delay the
question the slice exists to answer, in exchange for reducing a risk that is cheap to
retire later and does not compound.

**Alternatives considered:**
- *Block Slice 0 until multi-format files are assembled* — rejected as sequencing that
  serves completeness over the slice's hypothesis.
- *Narrow the product to MP3* — rejected. This is a validation-scope decision, not a
  product-scope one. PROJECT.md's supported-format claim is unchanged.

### Evidence
- Direct evidence: Eval 0A over 8,000 MP3s at 99.92%, deterministic, ~767x realtime
  (`evals/eval_0a_fma_small_full_20260906.json`). Generated-fixture tests cover all five
  formats.
- Engineering inference: **primary basis.** That a shared, format-agnostic decode path
  makes untested-format risk low and non-compounding.
- Open risk: real M4A and MP4 pathologies (DRM, multi-stream layouts) are unobserved.

**Tradeoffs accepted:**
Any claim about format coverage must say "MP3, validated; four others supported but
unvalidated" until the harness has been run on real files. Reporting a bare Eval 0A pass
as covering all five formats would be a misstatement, and STATUS.md carries that caveat.

**Removal/revisit condition:**
Revisit when the private out-of-domain test library is assembled, or the moment a real
non-MP3 file fails. The harness (`scripts/eval_0a_ingestion.py`) is corpus-agnostic, so
revalidation is one command per corpus, not new work.

**Files updated:**
- PROJECT.md: unchanged (product format claim stands)
- DESIGN.md: unchanged (contract is format-agnostic)
- EVALS.md: Eval 0A pass gate qualified per-format
- STATUS.md: pending decision 1 resolved; slice contract and next action updated

---

## DEC-009 — Prune MagnaTagATune and dim-sim

**Status:** Accepted

**Decision:**
Deleted `data/raw/mtat` (5.7 GB) and `data/raw/dimsim` (4.4 MB), retained from the previous
project. FMA is retained: it is the Eval 0A ingestion corpus.

**Why:**
Neither is named as a dataset in PROJECT.md, DESIGN.md or EVALS.md, and neither has a role
in any planned slice. dim-sim in particular is a pairwise musical-similarity judgement set,
which DEC-002 rules out as a target for this project by design.

**Removal/revisit condition:**
Both are publicly redownloadable if a future slice justifies them. MagnaTagATune would
return only as a tagging benchmark; dim-sim would require DEC-002 to be revisited first.

---

## DEC-010 — Native arm64 runtime, and a verified CLAP checkpoint

**Status:** Accepted

**Current slice:** Slice 0

**Question:**
Two blockers surfaced when standing up the encoder layer, both silent.

### Part A — the interpreter was running under Rosetta

The development machine is an Apple M4 Pro, but the active Python was miniconda's
**x86_64** build running under Rosetta 2. PyTorch publishes no macOS x86_64 wheels past
2.2.2 (which caps at Python 3.12), so `pip install torch` failed with "no matching
distribution" rather than anything that named the real cause.

**Decision:** the project runs on a native arm64 interpreter. `pyproject.toml` records why.

**Consequences beyond unblocking torch:**
- MPS acceleration becomes available, which is the difference between a usable and an
  unusable local E0.
- Ingestion got materially faster with no code change: decode p50 **39.6 ms -> 27.6 ms**,
  realtime factor **767x -> 1098x**. Eval 0A was re-run natively and is the number of
  record (`evals/eval_0a_fma_small_arm64_20260906.*`). Success rate and the six failures
  were identical across architectures, which is itself a useful confirmation that
  ingestion is deterministic across platforms and not just across runs.
- PROJECT.md's compute constraint ("normal local development hardware, with occasional
  consumer/cloud GPU") is more favourable than assumed. E0 is likely to be feasible
  entirely locally.

### Part B — the intended CLAP checkpoint is broken

The adapter initially defaulted to `laion/larger_clap_music`, chosen because a
music-specialised checkpoint gives CLAP its strongest showing in E0. That checkpoint loads
with **no missing-key warning** but arrives with its joint-space head untrained:
`text_projection`/`audio_projection` biases exactly zero, and `logit_scale_t = -0.005`
where a trained value is ~2.5-4.0.

**Why this was dangerous rather than merely broken.** Nothing crashed and nothing looked
obviously wrong. Embeddings were unit-norm. Audio-audio similarity showed real structure
(0.51-0.98), because the audio backbone *is* trained and a random linear map preserves some
geometry. Only the joint space was destroyed: every text embedding collapsed to mutual
cosine **0.999**, so all queries ranked tracks identically and retrieval was driven
entirely by the audio side. Every Eval 0B number computed on it would have been
meaningless while looking plausible.

Verified against the model's own `forward()` logits and against an explicit
tower-plus-projection path, and reproduced under both transformers 4.57 and 5.16 -- so the
fault is the published checkpoint, not the library version or the adapter.

**Decision:**
1. Default checkpoint is **`laion/larger_clap_music_and_speech`** -- music-relevant and
   verified to load with trained weights (bias std 0.022, `logit_scale_t` 2.659).
2. `ClapAdapter` runs `_assert_projection_trained` at load and **refuses** a checkpoint
   whose projection biases are zero or whose logit scale is near zero.

**Why the guard and not just the checkpoint swap:** the checkpoint swap fixes today's
instance; the guard catches the class. A silently-untrained projection head is invisible in
exactly the numbers it corrupts, so it has to be caught at load rather than in evaluation.

**Verified working checkpoints:** `laion/larger_clap_music_and_speech`,
`laion/larger_clap_general`, `laion/clap-htsat-unfused`. **Broken:**
`laion/larger_clap_music`.

### Evidence
- Direct evidence: weight statistics and logit scales at load; agreement across three
  extraction paths; reproduction under two transformers major versions; Eval 0A re-run
  under both architectures.
- Engineering inference: that checkpoint-integrity checks belong at load time generally,
  not only for this model.

**Tradeoffs accepted:**
The chosen checkpoint is music-*and-speech* rather than music-only, so CLAP is not being
given its theoretically strongest music-specialised variant in E0 -- because that variant
is not usable. Recorded so E0's result is not over-read as "music-specialised CLAP lost".

**Removal/revisit condition:**
If `laion/larger_clap_music` is republished with complete weights, re-test it as the E0
baseline. The guard stays regardless.

**Files updated:**
- PROJECT.md: unchanged
- DESIGN.md: unchanged (checkpoint selection is adapter-level, not architectural)
- EVALS.md: unchanged
- STATUS.md: Eval 0A numbers re-recorded natively; encoder layer progress
- pyproject.toml: torch/transformers deps and the arm64 note

---

## DEC-011 — E1 result: multi-segment pooling beats a single segment

**Status:** Accepted (2026-09-07) — **5 segments**. The multi-segment finding is settled by
evidence; the 3-vs-5 choice was Irene's, taken on cost grounds rather than on a claimed
significant difference.

**Current slice:** Slice 0

**Question:**
Should a track be represented by one deterministic centre segment (baseline) or by 3-5
deterministic windows, mean-pooled?

**Experiment:** E1, run through `scripts/eval_0b_retrieval.py` on Song Describer
(1,106 captions, 706 candidate tracks), with `--n-segments` as the only variable.
Compared with a **paired** McNemar exact test on discordant queries
(`scripts/compare_runs.py`), because both configurations score the identical query set and
comparing two independent proportions would discard most of the power.

**Result:**

| Comparison | R@1 | R@5 | R@10 | median | MRR | p (R@10) | sign test |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 seg (baseline) | 0.049 | 0.165 | 0.252 | 33 | 0.119 | — | — |
| 3 seg | 0.062 | 0.190 | 0.286 | 28 | 0.139 | 7.2e-04 | 1.3e-04 |
| 5 seg | 0.063 | 0.193 | 0.307 | 27 | 0.142 | 5.2e-07 | 2.3e-08 |
| 5 seg vs 3 seg | +0.001 | +0.004 | +0.022 | 28->27 | +0.003 | 1.8e-02 | 5.7e-02 |

**Decision (the settled part):**
Adopt multi-segment mean pooling. Single-segment is beaten at every K by both variants,
and the effect is large relative to noise: against 5 segments, 104 queries entered the
top 10 while 43 left it, p = 5.2e-7.

**What is NOT settled:**
5 segments over 3. R@1 and R@5 are flat (p = 1.00 and 0.73). Only R@10 favours 5, at
p = 0.018 — which does **not** survive correction across the nine tests run here
(Bonferroni threshold 0.0056). Treat 3 and 5 as tied on this evidence.

**Chosen: 5 segments.** Not because the R@10 gap
is proven, but because the cost that would argue against it does not exist -- indexing
moves 0.18 -> 0.26 s/track and storage is unchanged, since pooling still yields exactly one
vector. On a 5,000-track library that is 22 minutes instead of 15, once.

**Mechanism, not just outcome:**
Within-track segment cosine is ~0.82 -- a track's own segments are similar but genuinely
not identical, so there is real intra-track variation for pooling to capture. Meanwhile
track-track cosine *rises* with more segments (0.296 -> 0.343): pooling pulls every track
toward the corpus centroid, yet retrieval improves, so the compaction is discarding
idiosyncratic noise rather than signal. That trend would eventually reverse, which is a
reason not to extrapolate past 5 segments without measuring.

**Tradeoffs accepted:**
- Song Describer ships **2-minute excerpts**, so this is established on excerpts. Five 10 s
  windows cover ~42% of a 2-minute clip but only ~17% of a 5-minute song. The direction
  should hold or strengthen on full tracks -- more of the track goes unsampled at
  n_segments=1 -- but that is an inference, not a measurement.
- The benchmark scores **exact-track identification from a caption**, which is narrower
  than aux's use case, where many tracks may legitimately satisfy a query. E1's conclusion
  is about representation quality under a proxy task.
- DESIGN.md's known tension stands: Music4All ships 30-second clips, so if behavioural work
  (Slice 5) uses single-segment representations while production pools five, the two
  diverge and the gap must be measured rather than assumed away.

**Removal/revisit condition:**
Re-run E1 against whichever encoder wins E0 before treating this as final -- the optimal
pooling depth is a property of the encoder, not of the task, and MuQ-MuLan may differ.

**Files updated:**
- EVALS.md: E1 result recorded
- STATUS.md: Slice 0 progress
- DESIGN.md: track representation resolved to 5-segment mean pooling

**Understanding check:**
Why is "5 beats 3" *not* a claim this evidence supports, even though 5 was chosen?

---

## DEC-012 — E0 result: MuQ-MuLan beats CLAP, and it is CC-BY-NC

**Status:** Accepted (2026-09-07) — **MuQ-MuLan**, option 1. Taken with the CC-BY-NC
constraint understood and accepted for a portfolio/research project, not because the
licence was judged unimportant.

**Current slice:** Slice 0

**Question:** CLAP or MuQ-MuLan as aux's joint music-text encoder?

**Experiment:** E0, via `scripts/eval_0b_retrieval.py` on Song Describer (1,106 captions,
706 candidate tracks), identical caption set and identical pooling for both encoders,
compared with paired McNemar (`scripts/compare_runs.py`).

**Result — MuQ-MuLan wins decisively, at every pooling depth:**

| Encoder | seg | R@1 | R@5 | R@10 | median | MRR |
|---|---:|---:|---:|---:|---:|---:|
| CLAP | 1 | 0.049 | 0.165 | 0.252 | 33 | 0.119 |
| CLAP | 5 | 0.063 | 0.193 | 0.307 | 27 | 0.141 |
| MuQ-MuLan | 1 | 0.077 | 0.246 | 0.362 | 20 | 0.169 |
| **MuQ-MuLan** | **5** | **0.090** | **0.270** | **0.407** | **15** | **0.189** |

Paired, at matched 5 segments: R@10 +0.099, 256 queries gained the top 10 against 146 lost,
p = 4.5e-08; sign test on rank change p = 1.6e-11. The same result holds at matched 1
segment (p = 2.6e-10), so it is a property of the encoder and not an artefact of one
pooling configuration. Median rank 27 -> 15 out of 706.

**E1 replicates on the new encoder**, which was DEC-011's revisit condition: multi-segment
beats single (R@10 0.362 -> 0.407, p = 1.05e-04), and **3 vs 5 remains unseparated**
(p = 0.89 / 0.85 / 0.31; sign test 0.23). DEC-011's framing -- five chosen on cost, not
because it beat three -- is unchanged and now replicated across two independent encoders.

**The cost of winning:**

| | CLAP | MuQ-MuLan | Ratio |
|---|---:|---:|---:|
| indexing (5 seg) | 0.26 s/track | 0.63 s/track | 2.4x |
| query | 3.0 ms | 8.7 ms | 2.9x |
| storage | 2048 B/track | 2048 B/track | same |

Both remain trivial at personal-library scale: 5,000 tracks index in ~53 minutes and query
in under 10 ms. Cost does not decide this.

**What does decide it: licensing.** MuQ-MuLan's released weights are **CC-BY-NC 4.0**.
DESIGN.md recorded this as a risk -- "MuQ-MuLan's CC-BY-NC weights constrain productization
if it wins the benchmark" -- and the risk has now materialised. The code is MIT; the
weights are not, and non-commercial terms restrict what a shipped product may do with them.

**A diversity cost, recorded for Slice 6.** MuQ-MuLan's space is more concentrated than
CLAP's: hubness 8.0% vs 4.9% of top-10 slots taken by the top 1% of tracks, and 55 of 706
tracks never retrieved vs 36. It retrieves better *and* covers less of the catalogue. That
tension is precisely Slice 6's subject, and it is better to know now than to discover it as
a surprise when diversity is measured.

**Options for Irene:**

1. **MuQ-MuLan everywhere.** Best retrieval. Acceptable for a portfolio and for research;
   constrains any commercial productization.
2. **MuQ-MuLan for development, CLAP as the shippable fallback.** Costs a documented
   -0.099 R@10, and the `EncoderAdapter` contract makes it a one-line swap. Every later
   slice would need to state which encoder its numbers came from.
3. **CLAP only.** Gives up a large, well-measured gain to avoid a constraint that may never
   bind on a personal-use local product.

**Chosen: option 1**, revisited if productization becomes a real goal. This
is a portfolio and research project; the non-commercial term does not bind current use, the
gap is too large to give up voluntarily, and the adapter contract means reversing costs one
config change and one re-index. Whichever is chosen, record the encoder and version
alongside every result -- that is already enforced by `EncoderAdapter.version`.

**Not legal advice.** The licence fact is verified from the primary repository; the
interpretation of what it permits is Irene's to make.

**Removal/revisit condition:**
Revisit if productization becomes a goal, if MuQ-MuLan weights are relicensed, or if Slice
6 finds MuQ-MuLan's hubness materially harms recommendation diversity.

**Files updated:**
- EVALS.md: E0 result and the E1 replication
- STATUS.md: Slice 0 outcome; encoder choice awaiting Irene
- DESIGN.md: pending Irene's call

**Understanding check:**
Why does the E0 result hold at both 1 and 5 segments, and why does that matter?

---

## DEC-013 — Context→acoustic translation as Slice 2's first mechanism

**Status:** Proposed, with its original precondition **falsified** (2026-09-08). The
mechanism still stands; the reason for building it has changed from "context is ignored" to
"context works partially". Gated on Slice 1 human ratings.

**Current slice:** Slice 1

**Question:**
Irene's own query set is dominated by *context*: 7 of her 12 queries name an activity or
setting ("for running", "for studying", "late night drive", "to get ready to"). Only 2 are
purely acoustic. If real queries are mostly context and the encoder was trained on
descriptive captions of *sound*, how does a context query reach the right audio?

**The observation that prompted this.** A probe over Irene's queries
(`evals/probe_query_response.json`) shows her compound genre+context queries separate poorly
from the corpus: z-top 1.48-1.55 for "hip hop for running", "pre-game and club dance hip
hop", "late night drive vibes", against 4.24 for "romantic classical piano". In a library
that is 59% hip-hop, naming the genre selects nearly everything, so all the discriminating
work falls on the context phrase — and there is little sign the encoder does much with it.

**Decision (design, not yet implementation):**
Slice 2's first mechanism is **context→acoustic translation**: rewrite a query that names a
situation into one that names sound, then retrieve. This refines DEC-007 rather than
replacing it — DEC-007 fixed that the planner is an LLM emitting schema-constrained output;
this fixes *what the first useful thing to emit is*.

Four rungs, cheapest first, each required to beat the one below it:

| # | Mechanism | Cost | Beats |
|---|---|---|---|
| 0 | whole-query embedding | none | — (Slice 1 baseline) |
| 1 | fixed lexicon: "for running" -> "fast tempo, driving percussion, high energy" | none, deterministic, offline | must beat 0 |
| 2 | LLM rewrite of context into acoustic description | one API call | must beat 1 |
| 3 | retrieve on both original and rewrite, fuse at rank level | 2x retrieval | must beat 2 |

**Rung 1 exists to stop rung 2 being assumed.** A hand-written lexicon of perhaps thirty
context terms is free, deterministic, needs no network, and captures the part of the mapping
that is genuinely universal. If an LLM cannot beat that, the LLM is not earning its place --
and per DEC-007's own framing, a measured "the simple version was enough" is a better
result than an unmeasured planner.

**The part no rewrite can fix.** Context→acoustic mapping splits in two:

- **Universal** -- running is fast, sleeping is slow and quiet. Rungs 1-3 handle this.
- **Personal** -- "for studying" means lo-fi hip-hop to one listener, solo piano to another,
  and silence to a third. No lexicon and no LLM knows which. **Only behaviour resolves it**,
  which is Slice 5's intent-specific preference layer.

This is why DEC-004's authority order matters here rather than abstractly: the explicit
query still wins, the intent profile only refines what "studying" means *for this user*.
Slice 1B's playback events are what make that learnable, which is the concrete payoff of
DEC-005 appearing in a specific mechanism rather than as a general argument.

**Also observed, and out of scope for this mechanism:**
- **"r&b songs about yearning"** has the highest mean score of any query (0.398) -- it
  matches everything moderately, because "yearning" is lyrical content an audio encoder
  cannot isolate. No acoustic rewrite fixes that; it is Slice 3.
- **"sung in Vietnamese" returns jazz** despite 13 v-pop tracks. Language identity is not
  represented. Also Slice 3, and a routing question for Slice 2.

**Precondition — RUN, and it falsified the prediction.**

The stated test was: if `hip hop for running`, `for studying`, `for falling asleep` and bare
`hip hop` return substantially the same tracks, context contributes nothing.

They do not. Top-10 Jaccard against bare `hip hop` is **0.05, 0.05 and 0.00**. Adding a
context phrase almost wholly replaces the top of the list.

Rank correlation against bare `hip hop` also falls in a sensible order --
running 0.783, studying 0.716, falling asleep 0.694 -- so sleeping moves the ranking
furthest from generic hip hop and running least.

**But low overlap alone could not settle it.** Low top-K Jaccard with high rank correlation
is equally consistent with a flat score distribution near the top, where any small shift
reshuffles a noisy head. Instability is not discrimination, and the earlier z-top of ~1.5 on
these queries made that explanation live. The Jaccard test as originally specified was the
wrong instrument -- it could not distinguish the two.

`scripts/acoustic_direction.py` settled it, using waveform features computed independently
of the encoder. Retrieved sets differ in the direction the context word implies, on
rhythmic density:

| Query | loudness z | onset-rate z | brightness z |
|---|---:|---:|---:|
| hip hop for a workout | +0.28 | +0.08 | +0.12 |
| hip hop for running | +0.09 | +0.05 | +0.04 |
| hip hop for studying | +0.06 | -0.37 | +0.21 |
| hip hop for falling asleep | +0.19 | -0.43 | -0.02 |
| hip hop to relax to | -0.13 | -0.69 | +0.20 |

running - sleeping = **+0.48** onset rate; workout - relax = **+0.77**. Two independently
constructed pairs, same direction, on exactly the dimension those words imply. Noise does
not do that.

Loudness is inconsistent (-0.11 on one pair, +0.41 on the other) and brightness is flat --
unsurprising in a library of modern loudness-normalised masters, where there is little
variance to exploit.

**Conclusion: context is not ignored. It is partially and unevenly used.** The encoder has
an energy/activity dimension that context words reach, and does not appear to reach much
else.

**Revised precondition, for Slice 1's human ratings:**
Build rung 1 only if context-bearing queries score **materially worse than acoustic
queries** in the per-category ratings. That is now the open question: context changes
retrieval, but changing it is not the same as improving it, and only a listener can say
whether the changed results are better.

If context queries rate comparably to acoustic ones, the baseline is adequate and this
whole ladder is complexity without a problem -- the same conclusion the original
precondition was reaching for, now via the right measurement.

### Evidence
- Direct evidence: probe z-scores above; Irene's query distribution (7 of 12 context-led).
- Engineering inference: **primary basis.** That translating situation into sound moves the
  query toward the language the encoder was trained on.
- Open hypothesis: that context words currently contribute little. **Testable now**, by the
  overlap control, and this decision should not be accepted before that result.

**Removal/revisit condition:**
If the overlap control shows context already works, this is withdrawn. If rung 1 beats
rung 0 and rungs 2-3 do not beat rung 1, ship the lexicon and record that the LLM did not
earn its place.

**Files updated:**
- STATUS.md: overlap control recorded as a Slice 1 deliverable
- DESIGN.md: pending the Slice 1 result
- EVALS.md: pending the Slice 1 result

**Understanding check:**
Why is the fixed lexicon in the ladder at all, given an LLM would almost certainly write a
better one?

---


### Result of the precondition test (2026-09-08)

The prediction recorded above -- that context phrases contribute little -- was **wrong**,
and recording it in advance is what made that visible rather than arguable.

The methodological lesson is the more useful one. The test as originally designed measured
*whether the results changed*, when the question was *whether they changed for a reason*.
Those need different instruments, and a set-overlap statistic cannot tell them apart. The
deciding evidence had to come from outside the system being tested -- waveform features the
encoder never sees.

This is the same failure shape as the 0.724 cosine in Eval 0C: an absolute statistic
crossing a threshold, where the real question needed a relative or independent measurement.
Twice now the naive instrument would have produced a confident and wrong conclusion.


### Slice 1 outcome (2026-09-09): still unsettled, and now we know why

The revised precondition was: build rung 1 only if context-bearing queries rate materially
worse than acoustic queries.

Context scored 3.14 and acoustic 3.42 — context is lower, but the intervals overlap almost
entirely ([2.23, 4.03] vs [2.45, 4.40]) and both categories are dragged down by unrelated
problems: acoustic by two negation failures and two library gaps, context by queries that
name no genre.

**The context control failed to discriminate, for a reason worth recording.** All four
variants rated 4.2-4.8, with bare "hip hop" (4.6) no worse than "hip hop for running" (4.8).
That is not evidence that context works — it is evidence that *a relevance rating cannot
measure this*. In a library that is 59% hip-hop, "hip hop for X" returns hip-hop regardless
of whether the context was honoured, and the rater scores it relevant on genre alone.

The measurement conflates two questions. Settling DEC-013 requires asking the second one
directly: **"is this good for falling asleep?"** rather than **"is this relevant?"** — a
small re-rating over genre-matched, context-varying queries.

What Eval 1 *did* establish about context: queries naming a context **without** a genre
anchor are the weakest in the set ("warming up before going out" 1.6, "background music
while reading" 1.6), while genre-anchored context queries average 4.18. The genre word
anchors retrieval; context alone does not carry it.

## DEC-014 — Fix negation, then ranking, then routed retrieval

**Status:** Accepted (2026-09-09) — option C, timeboxed. Negation first as a narrow fix,
then reranking, then reassess before committing to the Slice 2 planner.

**Current slice:** Slice 1 (verified)

**Question:**
The roadmap runs Slice 2 (routed retrieval) before Slice 4 (query-aware ranking). Eval 1
produced measured headroom for ranking and ambiguous evidence for routing. Should the order
change?

**Evidence:**

*For ranking, measured:* NDCG over the returned top 5 is **0.866 against 0.863** for the
same items shuffled. The system orders its own results no better than chance. Independently
corroborated by success@1 = 67% versus success@5 = 89% — in roughly a fifth of queries a
clearly-relevant track was retrieved and not placed first. **22 points of success@1 are
available from reranking alone**, with no improvement to retrieval.

*For routing, ambiguous:* the context control did not separate. "hip hop for running" 4.8,
"for studying" 4.4, "for falling asleep" 4.2, bare "hip hop" 4.6 — all high, and the bare
query no worse. That is not evidence context works; it is evidence **the measurement cannot
tell**, because a relevance rating conflates "right genre" with "right for this situation"
in a library that is 59% hip-hop.

*A third finding, orthogonal to both:* negation is broken and reproducible.

**Why this ordering question matters rather than being bookkeeping:**
DEC-003 requires each addition to beat a simpler baseline. Ranking currently *has* no
baseline — results are returned in raw cosine order, which is measurably arbitrary. Adding a
query planner on top of an arbitrary ranker means a planner improvement and a ranking
improvement become hard to attribute separately. Fixing ranking first gives Slice 2 a
non-trivial baseline to beat, which is the same argument DEC-003 already makes for why
Slice 1 precedes Slice 2.

**Argument against:** PROJECT.md frames Slice 2 as the test of whether structured query
decomposition beats whole-query embedding, and negation — the clearest defect found — is a
*query understanding* problem that ranking cannot fix. A reranker cannot rescue "solo piano,
no vocals" when retrieval returned hip-hop.

**Chosen: the middle option.** Negation comes out of Slice 2 and is handled first as a
narrow fix — split the query, retrieve on the positive part, penalise proximity to the
negated part. Then reranking, which has the measured headroom. Then reassess Slice 2 with a
real baseline to beat.

Taken with the timeline tension stated openly: the evidence favours this order, while the
portfolio argument favours building the LLM planner sooner, since Slice 2 is the only place
an LLM appears in this project. The compromise is that both earlier steps are small and
produce clean results, so the planner is reached with two documented findings behind it and
a non-trivial baseline — which is a better account of the planner than building it first
would give.

### Evidence classification
- Direct evidence: the Eval 1 numbers above, `evals/eval_1_relevance_20260909.json`.
- Engineering inference: that an arbitrary ranker makes later attribution harder.
- Not established: that fixing ranking would raise relevance in practice. NDCG-vs-random
  shows ordering is arbitrary; it does not prove a better ordering is learnable from what is
  available.

**Removal/revisit condition:**
If a reranking experiment cannot beat raw cosine order on held-out human ratings, ranking is
not the bottleneck and Slice 2 resumes its original position.

**Files updated:**
- STATUS.md: recorded as the open next action
- PROJECT.md, DESIGN.md, EVALS.md: pending Irene's call

**Understanding check:**
Why does an arbitrary ranker make a later planner result harder to attribute?

---

## DEC-015 — Reranking does not close the gap; proceed to Slice 2

**Status:** Proposed — Irene asked to be consulted before Slice 2 begins.

**Current slice:** Slice 1 (post-verification work under DEC-014)

**Question:**
Eval 1 found ordering within the returned top 5 no better than chance, with success@1 (67%)
trailing success@5 (89%). DEC-014 chose to attack that before building the Slice 2 planner.
Does reranking capture the gap?

**Experiment:** five reranking methods over **exactly the five already-rated candidates per
query**, so the existing ratings score them and no new rating effort was needed. Paired
across the same 36 queries (`scripts/eval_rerank.py`).

**Result — no method is distinguishable from the current behaviour:**

| Method | NDCG | 95% CI | succ@1 | vs base | p |
|---|---:|---|---:|---:|---:|
| cosine (current) | 0.866 | [0.814, 0.914] | 67% | — | — |
| max-segment | 0.860 | [0.803, 0.911] | 69% | +3% | 1.000 |
| mean+max | 0.863 | [0.810, 0.913] | 72% | +6% | 0.625 |
| CSLS (hubness) | **0.889** | [0.836, 0.935] | 72% | +6% | 0.625 |
| query-z | 0.877 | [0.819, 0.927] | **75%** | +8% | 0.250 |
| *ceiling* | 1.000 | | *89%* | | |

Every method is directionally positive and none is significant. Confidence intervals all
overlap the baseline's. With 36 queries, an 8-point change is three queries.

**Conclusion:** the headroom is real — 67% against an 89% ceiling — but it is **not
reachable by simple geometric reranking of the existing candidates**. The information that
would order these five correctly does not appear to live in segment maxima, local density,
or cross-query normalisation.

**Two caveats that matter more than the ranking:**

- **Underpowered, not disproven.** 36 queries cannot detect effects of this size. CSLS in
  particular targets hubness that was independently measured in this space (Eval 0C) and
  posts the best NDCG; it deserves re-testing on a larger rated set rather than dismissal.
- **query-z is not shippable anyway.** It standardises a track's score against how it scores
  for *other* queries, so it needs the query batch at serving time. It was included to
  probe whether the signal exists at all, not as a candidate.

The obvious next rung — a learned ranker — is explicitly blocked by EVALS.md's E9
precondition: 180 ratings over 36 queries is far too little to train and honestly evaluate
one.

**Recommendation: proceed to Slice 2.** DEC-014's premise was that ranking had cheap
measurable headroom and routing did not. Half of that has now been tested and failed. The
remaining evidence points at query understanding rather than ordering: negation was a
genuine defect and a cheap fix worked; pure-context queries are the weakest category; the
"sung in Vietnamese" failure is a routing problem. Those are Slice 2's subject.

**What this changes about Slice 2's framing.** It arrives with a *measured* baseline and two
documented findings behind it, which is the outcome DEC-014 was reaching for. It also
arrives with reranking excluded on evidence rather than left unexamined — so if Slice 2
improves results, the improvement is attributable to query understanding.

**Removal/revisit condition:**
Revisit reranking when a larger rated set exists — either from a wider Slice 2 evaluation or
from first-party behaviour after Slice 1B. Retest CSLS first.

**Files updated:**
- STATUS.md: reranking result; Slice 2 recorded as the recommended next step
- EVALS.md: Slice 4 gains the negative reranking result

**Understanding check:**
Why is "no method beat the baseline" not the same claim as "reranking cannot help here"?

---

## DEC-016 — Rung 1 (fixed context lexicon) does not beat the baseline

**Status:** Accepted (2026-09-09) — rung 1 is not adopted. It remains in the codebase as the
bar rung 2 must clear.

**Current slice:** Slice 2

**Experiment:** `scripts/eval_expansion.py`. A hand-written table of 30 context terms maps
situations to acoustic descriptions ("running" -> "fast tempo, driving percussion, high
energy"). The expansion is embedded separately and blended with the original query at
weight *b*. Two metrics, neither needing human ratings:

- **separation** -- how far apart two opposed contexts pull retrieval, on waveform features
  the encoder never sees (onset rate), over four minimal pairs;
- **genre fidelity** -- for genre-anchored queries, whether the named genre still dominates.

**Result:**

| weight | separation | genre fidelity |
|---:|---:|---:|
| **0.00 (baseline)** | 0.56 | **0.80** |
| 0.15 | 0.45 | 0.82 |
| 0.30 | 0.36 | 0.78 |
| 0.45 | 0.66 | 0.68 |
| 0.60 | 0.53 | 0.56 |
| 0.80 | 0.72 | 0.46 |

**Genre fidelity falls monotonically** as expansion weight rises — 0.80 to 0.46. Separation
does rise at the high end, but non-monotonically (0.56, 0.45, 0.36, 0.66, 0.53, 0.72), which
across only four pairs is not distinguishable from noise. The one clear effect is the cost.

**Decision:** do not adopt. `DEFAULT_EXPANSION_WEIGHT` stays 0.0, which reproduces the
Slice 1 baseline exactly.

**Why it probably fails, which matters for rung 2.** The encoder already responds to context
— that was established before this rung was built (running − sleeping = +0.48 onset rate at
weight 0). Adding a generic acoustic phrase therefore contributes something largely
redundant while pulling the query toward a generic "energetic music" direction and away from
what the user actually specified. The lexicon is not adding information; it is diluting.

**What this does and does not say about rung 2.** It does not predict the LLM will fail: a
rewrite conditioned on the *whole* query can be specific where a fixed table can only be
generic, and it could replace rather than dilute. It does mean the LLM's bar is **rung 0,
the plain baseline**, not rung 1 — and that DEC-013's premise that translation is obviously
the right mechanism is now weaker than when it was written.

**Limitation of the metric, recorded rather than hidden:** "margin" sums a z-score
difference and a proportion, which is not a principled combination. The conclusion does not
rest on it — genre fidelity's monotonic decline and separation's noisiness are each legible
on their own — but the summary number should not be quoted as if it were a score.

**Removal/revisit condition:**
Revisit if rung 2 shows that context translation helps when the rewrite is query-specific.
That would suggest the idea was right and the table too crude, rather than the idea being
wrong.

**Files updated:**
- STATUS.md: rung 1 result; rung 2 recorded as blocked on an API key
- EVALS.md: E2 gains the rung 1 negative result

**Understanding check:**
Why is rung 0, not rung 1, now the bar the LLM has to clear?

---


### Audit, 2026-09-09: half this evidence is void, the conclusion survives

`scripts/diagnose_depth.py` later established that most context queries have only one or two
genuinely good matches in the 160-track library, so any top-K separation metric over it was
comparing one real match against filler. The baseline's own separation swings between -0.04
and 1.01 depending on K, which is the signature of noise.

**The separation column in the table above is therefore not evidence.** It should not be
cited, and the "0.66 at w=0.45" figure quoted in later documents is unreliable.

**The genre-fidelity column stands**, and it alone is sufficient. Fidelity fell monotonically
across six weights — 0.82, 0.78, 0.68, 0.56, 0.46 — with an unambiguous direction and no
dependence on how many good matches a query had. A monotonic cost with no demonstrated
benefit is the disqualification case a cheap proxy handles reliably.

So the decision stands with **one supporting metric rather than two**, and the confidence
should be read down accordingly. Rung 1 was rejected for diluting the genre, not for failing
to separate contexts — that second claim was never actually measured.

## DEC-017 — E2: the planner helps genre-anchored context queries

**Status:** Proposed — objective evidence is positive; human relevance not yet measured.

**Current slice:** Slice 2

**Result** (`scripts/eval_2_depth_corpus.py`, 1,489 FMA tracks, 18 opposed pairs, K=25):

| group | n | baseline | planner | delta | se | wins |
|---|---:|---:|---:|---:|---:|---:|
| **genre-anchored context** | 6 | 0.16 | **0.51** | **+0.34** | 0.07 | **6/6** |
| context, no genre | 6 | 0.25 | 0.24 | -0.01 | 0.22 | 2/6 |
| mood only (control) | 6 | 0.21 | 0.51 | +0.30 | 0.21 | 5/6 |
| all | 18 | 0.21 | 0.42 | +0.21 | 0.11 | 13/18 |

The planner wins **6 of 6** genre-anchored context pairs at a delta near 5x its standard
error — the form Irene actually writes, 7 of her 12 queries. It does nothing for context
queries carrying no genre.

**This was a pre-registered test.** The six-pair run hinted that both its losses were the
genre-less "music for X" form; the grouping and the hypothesis were stated in the script
docstring and to Irene *before* this run returned. The result is a test of that hypothesis,
not a pattern found after the fact.

### Three runs, and why only the third counts

| run | corpus | pairs | result |
|---|---|---:|---|
| 1 | personal, 160 | 4 | planner worse |
| 2 | FMA, 1,489 | 6 | worse at low K, better at high K |
| 3 | FMA, 1,489 | 18 | **better, driven by genre-anchored context** |

The honest reading is not that three runs disagreed. It is that runs 1 and 2 were
**underpowered and uninformative**, and were reported as inconclusive at the time rather
than as findings. Run 1 was additionally invalid: the personal library holds one or two real
matches per context query, so its top-K compared one match against filler.

The metric never changed between runs — only the corpus depth and the number of pairs, each
for a stated reason before the run. That is what makes this a power correction rather than
a search for a favourable result.

### What this does and does not establish

**Does:** the planner makes genre-anchored context queries retrieve tracks that differ
acoustically in the direction the context implies, measured on waveform features the encoder
never sees.

**Does not:** that the results are more *relevant*. Onset-rate separation is a proxy for
"the context registered", not for "a listener prefers these". Those are different claims and
only ratings settle the second.

**Also does not:** say anything about latency, which Eval 2A puts at ~1.7 s p50 and which is
a real obstacle to interactive search regardless of quality.

**Recommendation:** a focused rating round — the genre-anchored context queries only, top-3
from each system, roughly 12 queries and 15 minutes. Narrow because the objective evidence
has already localised where the effect is, so rating the null groups would spend time
confirming nothing.

**Removal/revisit condition:**
If human ratings show no relevance gain on genre-anchored context queries, the planner has
moved retrieval without improving it and does not ship.

**Files updated:**
- STATUS.md: E2 result; rating round proposed
- EVALS.md: E2 gains the objective result and the power correction

**Understanding check:**
Why is "three runs gave three answers" not evidence that this result is unreliable?

---


### Corrected and reproducible result (2026-09-09)

Two defects were found while adding rung 3, and both invalidated the earlier numbers.

**1. The planner was re-sampling its own inputs.** No temperature was set, so plans differed
between runs — an E2 delta moved +0.34 to +0.40 while the baseline stayed at 0.21 to two
decimal places. That variance was the planner, not the result. This API version exposes no
temperature control, so determinism is provided instead by caching plans on disk
(`plan.cache`), keyed on planner version and query.

**2. The schema was requested, not enforced.** DEC-007 specified *schema-constrained*
output; what was built asked for JSON in a prompt and parsed whatever came back. The API's
`output_config.format` enforces a JSON schema server-side, which removes the failure rather
than recovering from it. `validate` still runs afterwards: the schema constrains shape,
`validate` constrains content.

**Result of record** — 1,489 FMA tracks, 18 pairs, K=25, schema-enforced, plans cached, and
verified byte-identical across two consecutive runs:

| group | n | baseline | planner | fused | delta | wins |
|---|---:|---:|---:|---:|---:|---:|
| genre-anchored context | 6 | 0.16 | **0.47** | 0.24 | +0.30 | **6/6** |
| mood only (control) | 6 | 0.21 | **0.64** | 0.51 | +0.43 | **6/6** |
| context, no genre | 6 | 0.25 | 0.29 | 0.49 | +0.04 | 3/6 |
| **all** | 18 | 0.21 | **0.47** | 0.41 | **+0.26** | 15/18 |

95% CI on the overall delta **[+0.08, +0.44]**, excluding zero.

The direction has now held across every run: the planner helps, most clearly where a genre
anchors the query, and does close to nothing for context with no genre. The magnitude moved
between runs until the two defects above were fixed; it no longer does.

**Rung 3 (fusion) is rejected.** Fusing the two rankings scores +0.20 against the planner's
+0.26 and wins fewer pairs. The K sweep had suggested the systems were complementary — the
baseline sharper at K=5, the planner better beyond — but combining them dilutes the
planner's advantage rather than adding to it. `reciprocal_rank_fusion` stays in the codebase
for the multi-modality fusion Slice 3 will need; it is simply not useful here.

# Decision entry template

## DEC-XXX — Short title

**Status:** Proposed / Accepted / Rejected / Revisit

**Current slice:**

**Question:**

**Current default:**

**Alternative(s):**

**Why this matters:**

### Evidence
- Direct evidence:
- Transfer evidence:
- Engineering inference:

**Experiment:**

**Result:**

**Decision:**

**Tradeoffs accepted:**

**Licensing/data implications:**

**Removal/revisit condition:**

**Files updated:**
- PROJECT.md:
- DESIGN.md:
- EVALS.md:
- STATUS.md:

**Understanding check:**  
Can Irene explain what changed and why?
