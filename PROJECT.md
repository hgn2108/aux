# Project

**aux** — a local-first, multimodal, intent-aware music search and recommendation
system for user-provided media, with a lightweight built-in player.

The intellectual centre is retrieval, ranking, recommendation and personalization.
Playback is an *enabling layer*: it makes the product usable end to end and closes the
recommendation→behaviour feedback loop that later personalization depends on. It is
deliberately minimal and must not displace the retrieval work.

## Problem

People own music but cannot search it by how it feels. Existing local players search by
filename, artist, or genre tag. Streaming services offer mood playlists but only over
their own catalogue, and their APIs expose proprietary features that cannot be reproduced
for a user's own files.

There is no way to point at a folder of audio and ask for *"quiet bittersweet nostalgia"*
or *"like this track, but dreamier"*.

## User

A single listener with a personal library of local audio/video files (MP3, WAV, FLAC,
M4A/AAC, MP4 containing music) who wants to rediscover their own collection by feel
rather than by metadata.

Designed for a single user or small-user environment — not a platform requiring
collaborative-filtering scale.

## Core Capability

Interpret a free-form music request, retrieve evidence from the appropriate acoustic and
semantic representations, rank candidates according to that specific intent, and later
adapt using intent-specific behavioural feedback.

Two retrieval modes, plus playback:

1. **Natural-language vibe search** — "energetic but not aggressive", "dark dreamy
   production with hopeful lyrics"
2. **Reference-track search** — "like this track, but less aggressive"
3. **Play what was found** — play a result in place, queue it, and mark it a good or bad
   match

The third exists because a recommender with no observable outcome cannot be improved.
Playing a result inside the product turns a recommendation into a behavioural event that
is attributable to the query and rank that produced it — first-party feedback, without
assuming access to listening history from another platform.

### Product principle

> "Vibe" is not one label or one embedding. It is a query-dependent composition of
> acoustic evidence, lyrical meaning, contextual intent, constraints, and user preference.

### Non-negotiable design decisions

1. User-provided media is the primary product input.
2. Decoded audio is the canonical acoustic ML input.
3. Training/inference parity is mandatory.
4. Do not fabricate a scalar "acoustic similarity" target.
5. Use pretrained music/audio foundation representations before training new encoders.
6. Build and measure a simple retrieval baseline before adding routing or personalization.
7. Lyrics are semantic evidence, not an MVP dependency.
8. Personalization must not override explicit user intent.
9. Behavioural personalization should eventually be intent-specific.
10. Every major complexity increase must be justified by an experiment.

## Success

### Product Success

A user can point the system at a local directory and retrieve tracks that genuinely match
an expressive natural-language request, plus plausible neighbours for a reference track,
with visible rationale for why each result matched.

### Technical Success

- ≥95% decode/encode success across supported media formats, with failures categorised.
- Text→music retrieval measurably above weak/random baseline on a public benchmark
  (Song Describer: Recall@1/5/10, median rank).
- Reference-track retrieval produces plausible neighbours on a personal library that was
  never used for development.
- At least one meaningful model/system comparison with a recorded decision.
- Practical indexing and query latency on local hardware.

## Hiring Signal

This project should prove that I can:

1. Build a multimodal cross-modal retrieval system on pretrained foundation models, with
   train/inference parity across public and user-owned data.
2. Design layered evaluation for a subjective objective, including human judgment, and
   make architecture decisions from measured evidence rather than assumption.
3. Take a recommender from retrieval through ranking, personalization, and
   relevance/diversity trade-offs as a deployable local product.

### What new evidence this adds

Beyond standard supervised ML, forecasting, or tabular modelling:

**Product DS / Recommender** — retrieval and ranking, product metric design,
multimodal relevance, experimentation, personalization and cold start,
relevance/diversity/novelty trade-offs, decision-making under imperfect proxy metrics.

**ML / AI Engineering** — pretrained/foundation models, cross-modal retrieval, vector
search, media ingestion, batch indexing, evaluation harnesses, model/version management,
latency and observability, deployable local packaging.

## MVP

### Must Have

- Ingest user-provided local media and extract/normalise audio
- Index music representations
- Natural-language text→music retrieval
- Reference-track→music retrieval
- Reported retrieval quality and latency
- At least one meaningful model/system comparison

### Explicitly Not Building

- Arbitrary YouTube downloading
- Spotify-dependent features
- Collaborative filtering requiring a large user base
- Training a music foundation model from scratch
- Social features
- Hosted vector infrastructure without a scale reason
- Advanced concept steering before the core recommender works
- Playlist generation, lyrics, personalization, or advanced UI as MVP requirements
- A full music-player product. Excluded unless later justified: sophisticated
  library-management UI, streaming, social features, crossfade, equalizer, elaborate
  playlists, synchronized lyrics, and general player polish

## Vertical Slice Roadmap

| Slice | Capability | Purpose |
|---|---|---|
| **0** | Local media → searchable music | Validate ingestion, raw-audio representations, track pooling, train/inference parity, basic retrieval |
| **1** | Natural language → music | Establish the simplest working text→music baseline |
| **1B** | Search → play → observe | Enabling slice. Play results in-product and log attributable behavioural events, so Slice 5 has real data rather than synthetic behaviour |
| **2** | Complex intent → routed retrieval | Test whether structured query decomposition beats whole-query embedding |
| **3** | Audio + lyrical semantics | Add a true semantic modality; test routed multimodal retrieval |
| **4** | Query-aware ranking | Move from generic fusion to intent-sensitive ranking |
| **5** | Intent-specific behavioural personalization | Turn semantic search into a personalized recommender |
| **6** | Relevance + diversity/discovery | Analyse relevance/diversity/novelty trade-offs |
| **7** | Harden + ship | Reproducibility, versioning, tests, observability, UI/API — after retrieval is validated |

## Constraints

- **Compute** — solo project on normal local development hardware, with occasional
  consumer/cloud GPU. No research-lab-scale training.
- **Data** — public music datasets for development; personal audio is an out-of-domain
  generalization test only, never training data and never redistributed.
- **Licensing** — model weights and datasets must permit the intended use; MuQ-MuLan
  weights are CC-BY-NC 4.0, which affects productization.
- **Parity** — every representation the production system needs must be reproducibly
  computable from arbitrary local audio files.

## Open Questions

1. CLAP vs MuQ-MuLan — which joint music-text encoder wins for this use case?
2. Single representative segment vs multi-segment pooled track representation?
3. Does structured query decomposition beat whole-query embedding?
4. Can lyrics be sourced legally and reliably for user-owned music without reintroducing
   train/inference mismatch?
5. Does a supervised mood/theme expert add ranking or interpretive value?
6. Which playback technology fits a local-first app without pulling in a heavy framework?
   Unresolved; research when Slice 1B becomes active.

> Canonical project scope and intent.
> User owns the substance; Claude may structure and maintain it.
> Update only when project scope, goals, success criteria, or MVP materially change.
