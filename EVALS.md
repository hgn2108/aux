# EVALS.md

> **Audience:** Shared — Claude uses this to implement evaluation; Irene should understand every metric/gate before advancing.
> **Purpose:** Evidence truth: how we decide whether a component or slice works.
> **Update frequency:** Medium/high as new slices become active.

# aux — Evaluation Plan

## Evaluation philosophy

"Vibe" is subjective.

No single metric is the ground truth.

Use layered evaluation:

1. system validity;
2. public retrieval benchmarks;
3. planner correctness;
4. modality-specific relevance;
5. human judgments;
6. personalization behavior;
7. diversity/discovery trade-offs.

Experiments should only exist when their outcome changes a design decision.

---

# Slice 0 — Media + representation

## Eval 0A — ingestion

Metrics:
- decode success rate;
- encoder success rate;
- preprocessing latency;
- deterministic repeatability;
- cross-format stability where equivalent audio is available.

Pass gate:
- >=95% supported-media success with failures categorized cleanly.

## Eval 0B — public text/music retrieval

Benchmark:
- Song Describer.

Metrics:
- Recall@1
- Recall@5
- Recall@10
- median rank
- optional mAP

Purpose:
Compare joint music-text encoders.

## Experiment E0 — CLAP vs MuQ-MuLan

Baseline:
- CLAP.

Treatment:
- MuQ-MuLan.

Decision factors:
- Song Describer retrieval;
- qualitative local-library results;
- compute;
- latency;
- memory;
- licensing.

## Experiment E1 — single vs multi-segment pooling

Baseline:
- one deterministic segment.

Treatment:
- 3–5 segment mean pooling.

Measure:
- retrieval quality;
- indexing latency/storage.

Keep multi-segment only if quality gain earns the extra cost.

---

# Slice 1 — Simple natural-language retrieval

## Query categories

At minimum:
- acoustic;
- mood;
- context;
- simple mixed descriptions.

Measure:
- human relevance;
- success@K;
- per-category performance.

Purpose:
Establish the simple whole-query baseline.

This baseline must exist before query routing is added.

---

# Slice 1B — Search -> play -> observe

The player is an enabling layer, so its evaluation is mostly correctness, not relevance.
The one thing that must be right is **attribution**: if an event cannot be traced to the
query and rank that produced it, every later personalization result is unreliable.

## Eval 1B-A — playback correctness

Metrics:
- playback success rate across supported formats (MP3/WAV/FLAC/M4A/MP4);
- correct decode-to-audible for each format the ingestion pipeline accepts;
- playback startup latency;
- seek accuracy;
- queue ordering correctness.

Pass gate:
- every format the indexer accepts is also playable, or the mismatch is explicitly
  documented as a known limitation.

## Eval 1B-B — event logging correctness

Metrics:
- event capture rate (no silently dropped events);
- impression -> play attribution accuracy;
- session/query id integrity across a multi-query session;
- position/duration accuracy for progress events;
- idempotency/replay safety of the append-only log.

Pass gate:
- attribution is correct on a scripted interaction trace with a known expected event
  sequence.

**Test with synthetic interaction traces.** They verify the plumbing. They must never be
reported as evidence about user behaviour.

## Product funnel — measurable only once real use exists

```text
query
 -> recommendation impression
 -> result selected
 -> play started
 -> continued / skipped
 -> completed / saved / repeated
```

Candidate metrics: result selection rate, search-to-play rate, early-skip rate, completion
rate, save/like rate, repeat rate, query reformulation rate, session depth.

These are **production-only**. Do not fabricate impact numbers, and do not report any of
them from synthetic traces.

---

# Slice 2 — Query planner / routed retrieval

## Planner eval dataset

Create ~100–150 manually labeled queries.

Labels:
- acoustic facets;
- lyrical facets;
- context;
- positive constraints;
- negative constraints;
- operator/relation;
- reference transform.

Metrics:
- facet extraction precision/recall/F1;
- modality-routing accuracy;
- negation accuracy;
- relation/operator accuracy.

## Experiment E2 — whole-query vs decomposition

Baseline:

```text
whole query -> one joint embedding -> retrieval
```

Treatment:

```text
query planner
 -> specialized subqueries
 -> retrieval
 -> union/rerank
```

Evaluate by:
- acoustic;
- emotional;
- context;
- negation;
- mixed/contrast;
- reference-transform queries.

Primary metrics:
- human relevance;
- NDCG;
- success@K;
- pairwise win rate.

Keep decomposition only where it adds measurable value.

---

# Slice 3 — Audio + lyric semantics

## Experiment E3 — modality ablation

Compare:
1. audio only;
2. lyrics only;
3. routed audio + lyrics.

Break out:
- acoustic queries;
- narrative queries;
- emotion;
- mixed;
- contrast.

Expected hypothesis:
- audio strongest for sonic queries;
- lyrics strongest for narrative queries;
- routed multimodal strongest for mixed/contrast queries.

## Experiment E4 — whole lyric vs section-aware

Baseline:
- whole-song lyric embedding.

Treatment:
- section/chunk retrieval.

Metric:
- narrative/theme relevance.

Keep section-level only if improvement earns storage/complexity.

## Experiment E5 — dense vs cross-encoder lyric reranking

Baseline:
- Qwen3 embedding similarity.

Treatment:
- Qwen3 reranker over top-K candidates.

Measure:
- NDCG;
- human relevance;
- latency.

Keep only if relevance gain earns cost.

---

# Slice 4 — Query-aware ranking

## Experiment E6 — fusion

Compare:
1. single retriever;
2. normalized weighted sum;
3. rank-level fusion such as RRF.

Metrics:
- candidate recall;
- NDCG;
- human win rate.

## Experiment E7 — fixed vs query-aware ranking

Baseline:
- same weights for every query.

Treatment:
- planner-conditioned weighting/features.

Measure:
- per-query-category NDCG;
- human preference.

## Experiment E8 — mood expert

Baseline:
- no supervised mood signal.

Treatment:
- frozen embedding + linear multi-label probe.

Optional:
- shallow MLP.

Metrics:
- mood mAP/F1;
- downstream retrieval/human relevance.

Removal condition:
No retrieval improvement and little explanatory value.

## Experiment E9 — learned ranker

Precondition:
Enough real query-track judgments.

Baseline:
- deterministic query-aware scorer.

Treatment:
- LambdaMART/LightGBM or pairwise model.

Metric:
- held-out NDCG.

Do not run if labeled data is too small for trustworthy evaluation.

---

# Slice 5 — Behavioral personalization

## Experiment E10 — personalization hierarchy

Compare:

A. no personalization  
B. global profile  
C. intent-specific profile  
D. intent + session profile

Metrics:
- relevance;
- NDCG;
- human/user preference;
- save/relevant rate when real use exists;
- diversity.

Core analysis:

> recommendation quality vs number of observed interactions

This is the cold-start curve.

## Data source

Use **real observed behaviour** logged by the Slice 1B player — playback events and
explicit feedback, attributed to the impression that produced them.

This is the reason Slice 1B exists: personalization can be evaluated on first-party
behaviour rather than assuming access to listening history from another platform.

Weight signals by the strength hypothesis in DESIGN.md (strong: explicit feedback, save,
repeat; medium: queued, high completion, deliberate play from a recommendation; weak:
early skip, pause, abandonment) — and treat that weighting as something to validate, not
as given.

Important:
Synthetic interactions may test code but must not be reported as effectiveness evidence.

---

# Slice 6 — Diversity/discovery

## Experiment E11 — diversity frontier

Baseline:
- pure relevance ranking.

Treatment:
- MMR-style reranking with lambda sweep.

Metrics:
- NDCG/relevance;
- intra-list diversity;
- novelty;
- catalog coverage.

Decision:
Choose an operating point on the relevance/diversity Pareto frontier.

---

# Human evaluation protocol

Because vibe is subjective, maintain a small human-judged query set.

Suggested scale:
- 30–100 expressive queries.

For each:
1. pool top results from competing systems;
2. hide system identity;
3. randomize;
4. collect 1–5 relevance or pairwise preference.

Track query category.

Report:
- mean relevance;
- graded NDCG;
- success@K;
- pairwise win rate;
- confidence intervals;
- inter-rater agreement if multiple raters.

---

# Stop conditions

Stop/revisit architecture if:

- neither CLAP nor MuQ-MuLan gives useful public retrieval;
- query decomposition does not improve mixed/constraint queries;
- lyric semantics cannot be sourced legally/reliably;
- mood model adds no measurable or interpretive value;
- learned ranking lacks labels;
- personalization evaluation depends mostly on synthetic behavior;
- playback events cannot be reliably attributed to the recommendation that produced them;
- advanced research begins consuming time before the core recommender is portfolio-ready.
