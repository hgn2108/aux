# INIT_RESEARCH.md

> **Audience:** Human-reference + Claude-on-demand.
> **Purpose:** One-off initialization research snapshot and evidence archive.
> **Update frequency:** Usually none. Append dated research notes only when new evidence materially changes architecture.
> **Claude should NOT load this every session unless a design/research question requires it.**

# aux — Initial Research Snapshot

Research snapshot date: 2026-08-29.

## How to read this file

A cited technique does **not** automatically belong in the product.

Each entry is classified as:
- **Direct evidence**
- **Transfer evidence**
- **Engineering inference**
- **Future research**

---

# 1. MTG-Jamendo

Primary source:
https://github.com/MTG/mtg-jamendo-dataset

## Verified
Official materials report:
- >55,000 full tracks;
- 195 tags across genre, instrument, mood/theme;
- 18,486-track mood/theme subset;
- official train/validation/test splits;
- downloadable audio;
- Creative Commons source catalog.

## Evidence type
Direct evidence for:
- public raw music audio;
- music tagging;
- auxiliary mood/theme supervision.

## Project implication
Use as primary public development ecosystem.

## Caveat
Uploader-provided tags are weak/proxy supervision, not a canonical definition of vibe.

---

# 2. Song Describer

Primary source:
https://github.com/mulab-mir/song-describer-dataset

Data card:
https://github.com/mulab-mir/song-describer-dataset/blob/main/docs/datacard.md

## Verified
- 706 tracks;
- 1,106 captions;
- validated subset: 546 tracks / 746 captions;
- ~2-minute audio;
- drawn from MTG-Jamendo split-0 test;
- intended for music-language evaluation.

## Evidence type
Direct evidence for free-form text->music evaluation.

## Project implication
Use as primary public retrieval benchmark.

## Caveat
Small benchmark and not a complete measure of real-user vibe satisfaction.

---

# 3. LAION-CLAP

Primary source:
https://github.com/LAION-AI/CLAP

## Verified
CLAP provides joint latent representations for audio and text and supports direct audio/text embedding.

## Evidence type
Direct evidence for joint audio-text retrieval.

## Project implication
Use as mature baseline.

## Open question
How competitive is CLAP specifically for this personal-library music use case?

Answer by benchmark, not assumption.

---

# 4. MuQ / MuQ-MuLan

Primary source:
https://github.com/tencent-ailab/MuQ

## Verified
Official repository states:
- MuQ is a self-supervised music representation model;
- MuQ-MuLan jointly represents music and text;
- MuQ-MuLan ~700M parameters;
- 24 kHz audio input;
- English and Chinese;
- code MIT;
- released weights CC-BY-NC 4.0.

## Evidence type
Direct evidence for music-specific music/text representation.

## Project implication
Benchmark against CLAP.

## Caveat
Non-commercial released weights affect productization.

---

# 5. Qwen3 Embedding / Reranker

Primary source:
https://qwenlm.github.io/blog/qwen3-embedding/

## Verified
Qwen reports:
- embedding and reranking families;
- 0.6B, 4B, 8B;
- 32K context;
- >100 languages;
- Apache 2.0;
- retrieval-oriented embedding model;
- cross-encoder reranking model.

## Evidence type
Direct evidence for general semantic retrieval/reranking.

Not direct evidence for lyrics specifically.

## Project implication
Qwen3-Embedding-0.6B is a strong practical lyric-semantic baseline candidate.

Qwen3-Reranker-0.6B is a later reranking candidate.

## Open question
Does it perform well on actual lyrical narrative/theme retrieval?

Must evaluate.

---

# 6. Query decomposition

Primary source:
ACL 2025:
https://aclanthology.org/2025.acl-srw.32/

## Verified
The work:
- decomposes complex questions;
- retrieves independently;
- merges candidates;
- reranks;
- reports gains on multi-hop QA.

## Evidence type
Transfer evidence.

Not music-specific.

## Project implication
Motivates:
- structured decomposition;
- independent modality retrieval;
- candidate union;
- reranking.

## Required validation
Whole-query retrieval vs decomposed routed retrieval.

If decomposition does not help aux, remove/simplify it.

---

# 7. Intent-based personalized music recommendation

Primary source:
Tsukuda et al. 2025:
https://doi.org/10.1007/s11042-025-21022-7

## Verified
A real music web service allowed users to maintain multiple recommendation models for distinct intents such as:
- "cool songs";
- "songs for concentrating on work";

using seed songs and repeated relevance feedback.

## Evidence type
Direct evidence for intent-specific music personalization.

## Project implication
Model:
- global preference;
- intent-specific profile;
- current session.

Avoid relying only on one global taste vector.

---

# 8. Open-vocabulary steerable music retrieval

Primary source:
https://arxiv.org/abs/2608.08757

## Verified
The work explores controls such as:
- more ambient;
- less distorted;
- without guitar;

using sparse representation techniques.

## Evidence type
Direct but advanced research evidence.

## Project implication
Future research extension only.

## Why not now
First prove:
- base retrieval;
- correct operator parsing;
- query-aware reranking.

---

# 9. YouTube / remote media

Primary source:
https://developers.google.com/youtube/terms/developer-policies

## Verified
YouTube API policies restrict downloading/caching audiovisual content without required permission/approval.

## Project implication
- local MP4 supported;
- arbitrary YouTube downloading is not MVP;
- remote adapters must be evaluated source-by-source.

---

# Current engineering inferences

These are not research findings.

## Exact vector search first
At personal-library scale, exact cosine/inner-product search is simpler and likely sufficient until scale proves otherwise.

## Rank fusion before learned score fusion
Different embedding-model scores are not naturally calibrated.

## SQLite + local vector index
Reasonable local-first system until scale/features require more.

## Sparse-profile personalization
Weighted centroids/Rocchio-style updates are appropriate before large real user histories exist.

## MMR as diversity baseline
Simple and interpretable before more complex list optimization.

---

# Research backlog

Investigate only when relevant:

1. newer music-text encoders with better quality/licensing;
2. music-specific lyric semantic models;
3. improved long-track aggregation;
4. learned query routers after enough labeled planner data;
5. calibrated multi-retriever fusion;
6. off-policy/causal evaluation after meaningful usage data exists;
7. sequential recommendation after enough real behavior exists;
8. concept steering after the core system is stable.
