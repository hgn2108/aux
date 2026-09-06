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
