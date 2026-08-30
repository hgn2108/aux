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

---

# Appendix A — Behavioral personalization research

Research snapshot date: 2026-08-30.

Appended, not replacing the 2026-08-29 snapshot above. Same evidence classification.

---

## A1. Music4All-Onion

Primary source:
https://zenodo.org/records/6609677

Paper (CIKM 2022):
https://dl.acm.org/doi/10.1145/3511808.3557656

### Verified
The Zenodo record states:
- 109,269 tracks;
- 119,140 Last.fm users;
- 252,984,396 listening records;
- 40 files, 15.0 GB;
- **Creative Commons Attribution 4.0**, openly downloadable, no access restriction;
- listening data in two forms — play counts per user-track pair, **and a timestamped file**
  (`userid_trackid_timestamp.tsv.bz2`);
- precomputed audio embeddings (ResNet, VGG19, i-vectors at 256/512/1024 dims);
- handcrafted audio features (MFCC statistics, spectral, chroma, voice characteristics);
- metadata features (genre/tag TF-IDF, lyrics Word2Vec, sentiment functionals).

Extends Music4All and LFM-2b.

### Evidence type
Direct evidence for large-scale music listening behaviour with content features and
timestamps, under a permissive licence.

### Project implication
Primary external behavioural dataset. Supports global taste (play counts), session and
temporal modelling (timestamps), and content-conditioned methods.

### Caveat
Its precomputed audio features are **not** aux's representation. Use them as benchmarks,
never as the transfer representation — see A2.

---

## A2. Music4All (base) — the transfer path

Primary sources:
https://sites.google.com/view/contact4music4all
http://www.din.uem.br/yandre/IWSSIP_2020_Music4All.pdf

### Verified
Published materials state:
- 30-second audio clips, 44.1 kHz stereo, for 109,269 tracks;
- clips cut deterministically — the midpoint of each song, ±15 seconds — explicitly to
  stay within fair use;
- lyrics and 16 metadata fields;
- 15,602 users with listening histories in the base release.

Same 109,269 tracks as Music4All-Onion.

### Evidence type
Direct evidence that raw audio exists for the tracks Music4All-Onion has behaviour for.

### Project implication — this resolves the item-representation problem

```text
Music4All 30s audio -> aux's own encoder -> item representation
                                      +
Music4All-Onion behaviour (same track ids)
                                      |
                                      v
          preference / session / intent mechanism
                                      |
                                      v
              transfers to arbitrary local audio
```

Item representations are computed by **our** encoder from raw audio, not taken from
dataset-specific IDs or dataset-specific features. This is the parity DEC-001 requires,
and it is the difference between a behavioural model that transfers to
`~/Music/random.mp3` and one that does not.

### Caveat — 30-second clips constrain the pooling decision
Music4All ships centre clips only. Behavioural research on it is therefore inherently
**single-segment**. If experiment E1 concludes multi-segment pooling is better for
retrieval, the production track representation and the behavioural training
representation diverge. Flagged as an open question; do not assume it is free.

### Open question
Access terms for the base audio need confirming — the official page is a contact form,
suggesting request-gated access rather than open download. Verify before this becomes
critical path.

---

## A3. Music4All A+A

Primary sources:
https://arxiv.org/pdf/2509.14891
https://ieeexplore.ieee.org/document/11339285/

### Verified
- multimodal dataset built on Music4All-Onion;
- includes user-item interaction data at track level;
- **CC BY-NC-SA 4.0**;
- distributed via GitHub.

### Evidence type
Direct evidence of a current (2025) multimodal successor.

### Project implication
Watch. The non-commercial share-alike licence is more restrictive than
Music4All-Onion's CC BY 4.0, so prefer Onion unless A+A offers something it lacks.

---

## A4. Datasets assessed and not recommended as critical path

### Spotify Million Playlist Dataset
Primary source:
https://www.aicrowd.com/challenges/spotify-million-playlist-dataset-challenge

**Verified:** no longer available for direct download; researchers must contact Spotify
Research. Non-commercial open research use.

**Implication:** the most attractive intent dataset — playlist titles are genuine weak
intent labels — but request-gated. Do not make it critical path. Request access in
parallel; treat as a bonus.

### Million Song Dataset Taste Profile
Primary source:
http://millionsongdataset.com/tasteprofile/

**Verified:** user/song/play-count triplets, ~1M users; Echo Nest API licence;
**no timestamps**.

**Implication:** usable as a classical collaborative-filtering benchmark to quantify how
much collaborative signal exists. No timestamps means no session modelling, and no audio
path means no transfer. Benchmark only.

### LFM-1b
**Verified:** as of January 2025 users report being unable to locate a download route;
status unclear.

**Implication:** do not depend on it. Its successor LFM-2b feeds Music4All-Onion, so the
lineage is available through Onion regardless.

### Last.fm 1K
**Verified:** ~19M timestamped listens, ~992 users.

**Implication:** small user count. Useful for quick sessionization prototyping; superseded
by Music4All-Onion for anything of scale.

---

## A5. Methods — session and sequential

Primary sources:
https://dl.acm.org/doi/10.1145/3523227.3548487 (RecSys 2022 BERT4Rec replicability study)
https://www.sciencedirect.com/science/article/pii/S0020025521005089 (Information Sciences)

### Verified
- a systematic review found BERT4Rec results **inconsistent across publications**, and
  the authors could not reproduce the original paper's results using its default
  configuration;
- nearest-neighbour methods such as V_SKNN outperform BERT4Rec, GRU4Rec and SASRec on
  some datasets across most accuracy metrics;
- the session-aware survey is titled, pointedly, "a surprising quest for the
  state-of-the-art".

### Evidence type
Direct evidence, though from general sequential recommendation rather than music
specifically.

### Project implication
**Start with simple baselines and make deep sequential models earn their place.** Last
track, mean of last N, recency-weighted centroid — all computable directly in the content
embedding space, all cheap. GRU4Rec/SASRec/BERT4Rec only if a simple baseline is beaten
on our own held-out evaluation.

This also matches the sparse-data reality: a single user's aux history will be small.

---

## A6. Methods — intent and context

Primary sources:
https://link.springer.com/article/10.1007/s12652-020-02777-3
https://dl.acm.org/doi/10.1145/3705328.3748053 (RecSys 2025)

### Verified
- playlist titles carry usable semantic intent, and capturing their meaning is
  **particularly beneficial in cold-start scenarios**;
- user-generated titles are a good starting point for inferring intended purpose;
- recent work applies language models to playlist generation, finding titles a valuable
  signal and that clustering plus fine-tuning improves generalisation across themes.

### Evidence type
Direct evidence for playlist-title-conditioned recommendation; transfer evidence for
aux, whose queries are free text rather than playlist titles.

### Project implication
An aux vibe query and a playlist title are structurally similar — both are short natural
language describing intended listening. This supports intent-conditioned retrieval, and
supports the existing intent-specific preference level. It does **not** establish that our
particular hierarchy is correct.

Combined with Tsukuda et al. 2025 (§7 above), intent-specific profiles have direct
real-world evidence behind them.

---

## A7. Item-representation transfer — the five options assessed

| Option | Verdict |
|---|---|
| 1. Behavioural dataset with legally available audio | **Recommended.** Music4All + Music4All-Onion. Exact parity. |
| 2. Metadata/text item representation | Fallback only. Creates the train/inference mismatch DEC-001 exists to prevent. |
| 3. Catalog matching to MusicBrainz/public audio | Unnecessary given option 1; keep as contingency if Music4All audio access fails. |
| 4. Pretrained public content embeddings alongside behaviour | Available via Onion (ResNet/VGG19/i-vector), but not our encoder's space. Benchmark only. |
| 5. Item-ID collaborative filtering | Benchmark only, as the brief states. Quantifies available collaborative signal; does not transfer to arbitrary local tracks. |

---

## A8. Unresolved research questions

1. Music4All base-audio access terms — open download or request-gated?
2. Does the 30-second-clip constraint conflict with a multi-segment pooling decision (E1)?
3. Do Last.fm listening counts transfer to in-product aux behaviour at all? Last.fm users
   scrobble passive listening; aux users act on explicit queries. Different behaviour,
   possibly different preference semantics.
4. Is the strong/medium/weak signal weighting empirically supported, or convention?
5. Does external behavioural pretraining actually help at small first-party history sizes,
   or does a simple content centroid match it? **This is the central experiment.**
