# Design

## System Goal

Turn a directory of user-owned media into a searchable music index, and answer two kinds
of request against it: a free-form natural-language description of a vibe, and a
reference track to find neighbours for. The same preprocessing and encoder must run over
public development audio and arbitrary user files, so that anything learned on public
data transfers to the user's library without a feature mismatch.

## Architecture

### Current approved foundation

```text
user/public media
 -> media probe
 -> audio decode/extraction
 -> AudioAsset
 -> model-specific resampling
 -> segment encoding
 -> track aggregation
 -> exact vector retrieval
```

### Media ingestion contract

```text
SourceAdapter -> MediaProbe -> AudioDecoder -> AudioAsset
```

`AudioAsset` preserves: source path/provenance, duration, native sample rate,
waveform/stream handle, content hash, media type.

For MP4/video: treat video as a container, extract and decode audio, then run the exact
same downstream pipeline as standalone audio.

**Do not permanently resample all stored media to one universal sample rate.** Preserve
source audio; each encoder adapter resamples to its own required rate. This keeps the
pipeline correct when a second encoder with a different input contract is added.

### Target long-term architecture

Approved direction, not yet implemented:

```text
user local media
        |
        v
library / audio pipeline
        |
        v
user query / reference track
        |
        v
structured query interpretation
        |
        +----------------+----------------+----------------+
        |                |                |                |
        v                v                v                v
acoustic facets     lyrical facets   context/operators   reference
        |                |                |                |
        v                v                |                v
music-text index    lyric index           |          audio similarity
        |                |                |                |
        +----------------+----------------+----------------+
                         |
                         v
                  candidate union
                         |
                         v
                 query-aware ranking
                         |
                         v
              intent-specific preference
                         |
                         v
                diversity reranking
                         |
                         v
                 recommendations
                         |
                         v
              local playback / queue
                         |
                         v
        event + explicit feedback logging
                         |
                         v
     session / intent / global preference state
                         |
                         +--> personalized reranking
                                      |
                                      +---- feedback loop
```

## Data Flow

**Indexing (batch).** Local directory → probe each file → decode to waveform → per-encoder
resample → select segment(s) → encode → aggregate to one track vector → L2-normalise →
persist embedding + metadata + content hash.

**Query — text.** Natural-language query → text encoder (same joint space) → exact cosine
search over track vectors → ranked results.

**Query — reference.** Reference track → look up its cached embedding → exact cosine
search excluding itself → ranked results.

Content hash keys the cache so re-indexing is incremental and a moved or renamed file is
not re-encoded.

## ML / AI Approach

### Core Intelligence

A pretrained joint music-text encoder. Similarity is an *emergent property* of that
representation, not a supervised target — there is no reliable canonical ground truth for
subjective musical similarity, so no fabricated pairwise similarity labels are created.

### Why a contrastive joint encoder

A joint music-text encoder is two towers — one audio, one text — trained to place matched
pairs at the same point in a single space. Training is contrastive: for a batch of N
(audio, caption) pairs, embed all of them, form the N x N cosine matrix, and optimise so
each clip scores highest against its own caption and each caption against its own clip.
The other N-1 pairs in the batch are the negatives.

**This is what makes DEC-002 coherent rather than a workaround.** The model is never
trained on a similarity score, and nothing in the objective ever asserts that two *tracks*
are similar. It learns only that a description belongs to an audio clip. Track-to-track
similarity is therefore a geometric consequence — two tracks are close when they would be
described in similar language — which is precisely the "emergent property" the project
requires instead of a fabricated scalar target.

Three properties follow directly:

- text and audio vectors are comparable, so text->music retrieval is one matrix multiply;
- reference search needs no second model, being audio<->audio cosine in the same space;
- parity holds by construction — the input is a raw waveform and nothing is fitted at
  inference, so the representation is reproducible from any local file.

### Known limits of that training signal

Four gaps between what the encoder was trained on and what aux asks of it. Each is a thing
Slice 0 and Slice 1 measure, not a thing to assume away.

1. **Caption language is not query language.** Training captions are descriptive ("a folk
   song with acoustic guitar and female vocals"); aux queries are expressive ("quiet
   bittersweet nostalgia"). Abstract, emotional and negated phrasing is plausibly
   underrepresented. This is why Slice 1 reports human relevance *per query category*
   rather than as one number, and part of the case for the Slice 2 planner: decomposition
   moves the query text closer to the shape the encoder was trained on.
2. **General-audio models are not music-first.** A general audio-text model must separate a
   dog bark from a siren as well as two indie tracks, so its musical resolution may be
   coarse. This is the substance of E0: general-audio maturity (CLAP) against
   music-specific resolution (MuQ-MuLan), decided by measurement.
3. **Clip length.** These encoders are trained on short windows, not whole songs. The model
   never learned to summarise a track; pooling is imposed afterwards. That is why E1 exists
   and why learned pooling is forbidden before it runs.
4. **The modality gap.** In contrastively trained dual encoders, text and audio embeddings
   occupy offset regions of the space rather than interleaving. A text->audio cosine of 0.4
   and an audio->audio cosine of 0.4 do not mean the same thing. This is a second and
   independent reason fusion is rank-level: scores are not comparable across modality pairs
   even within a single model.

Two further risks are properties of the space rather than of the training signal, and both
are cheap to instrument during indexing: **domain shift** (development audio is
Creative-Commons catalogue, a personal library is commercially mastered) and **hubness**
(contrastive spaces reliably produce a few vectors that are nearest neighbour to almost
everything). Symptoms and diagnostics are recorded under Risks / Assumptions.

### Inputs

Decoded audio waveform at the encoder's required sample rate; natural-language text for
the query side.

### Outputs

A fixed-length L2-normalised track embedding, in a space shared with text.

### Approach

Frozen pretrained encoders, benchmarked rather than assumed.

**LAION-CLAP** — mature joint audio/text baseline; supports text→audio and audio→audio.

**MuQ-MuLan** — newer music-specific comparator. ~700M parameters, 24 kHz input, code
MIT, released weights **CC-BY-NC 4.0** (affects productization).

Decision: benchmark both; do not assume a winner.

### Track representation — open decision

A 3–5 minute song must become one vector, but encoders accept much shorter windows.

- **Baseline** — one deterministic representative/centre segment.
- **Candidate** — 3–5 deterministic windows spread across the track, excluding obvious
  leading/trailing silence; normalise each segment embedding, mean pool, normalise the
  result.

Do not use learned pooling before this experiment runs.

### Auxiliary mood/theme expert — optional

Frozen music embedding → linear multi-label probe on MTG-Jamendo mood/theme; shallow MLP
only if justified. Purpose: interpretable attributes, auxiliary evidence for common mood
concepts, and a defensible trained component.

**Removal condition:** if it adds neither ranking benefit nor useful interpretability,
remove it.

## Query interpretation layer

Not implemented. Slice 2. Documented now because DEC-007 resolves what the planner *is*.

### What it does

```text
free-form query
  -> LLM, schema-constrained
  -> structured interpretation
  -> routed subqueries -> retrieval -> candidate union -> rerank
```

The structured interpretation carries the fields `EVALS.md` already labels: acoustic
facets, lyrical facets, context, positive constraints, negative constraints,
operator/relation, and reference transform.

### Why an LLM

Vibe language is open-vocabulary. A closed facet taxonomy fails on phrasing it has never
seen, and no labelled data exists yet to fine-tune a tagger on. Text→structure over
unbounded natural language is the one place in aux where a general language model is the
right tool rather than a fashionable one.

### Why not an agent

A query needs one or two interpretation calls. There is no long-horizon task, no tool
surface to plan over, and no failure-recovery requirement. A multi-step loop would be
complexity with no experiment behind it, which non-negotiable 10 forbids.

### Boundary

In scope: one schema-constrained generation per query, retry on schema violation, and a
deterministic fallback to whole-query embedding when generation fails or the structure is
empty. The baseline path must remain functional with the planner disabled.

Out of scope: multi-step planning, tool invocation, conversational state, self-critique
loops.

### Failure handling

Schema violation, timeout, and empty-extraction all fall back to the Slice 1 whole-query
path. The planner is an *enhancement over* a working baseline, never a dependency of it —
which is also what makes the E2 ablation possible.

### Local-first tension

Only the query string leaves the machine. Audio, library contents, file paths and
behavioural events never do. E2a's distilled local planner is the intended resolution: if
a small local model reproduces the hosted planner's structure closely enough, the hosted
dependency leaves the product path entirely.

### Evaluation coupling

The planner is evaluated on its own output (schema conformance, field-level P/R/F1) *and*
on downstream retrieval. Both are required — a planner that extracts fields correctly but
does not improve retrieval has still failed DEC-003.

## Playback layer

A deliberately minimal player. It exists so results can be heard in place and so the
recommendation→behaviour loop closes inside the product; it is not the intellectual
centre and must not accumulate music-player features.

### Boundary

In scope: play/pause, seek, next/previous, queue, play directly from a search or
recommendation result, basic metadata display, and explicit feedback (save, like/dislike,
or good match / bad match).

Out of scope unless later justified: library-management UI, streaming, social features,
crossfade, equalizer, elaborate playlists, synchronized lyrics, general polish.

### Playback and queue state

Transient application state — current track, position, queue contents and order, play/pause.
Not part of the index, not persisted as model input. Persisted only as the events below.

### Behavioural event model

Every event must be attributable to the query and rank that produced it, or personalization
later cannot distinguish "the user liked this" from "the user was shown this".

**SearchRequest** — `query_id`, raw query, parsed intent where applicable, timestamp.

**RecommendationImpression** — `impression_id`, `query_id`/`session_id`, `track_id`,
displayed rank, retrieval/ranking scores, model/version, timestamp.

**PlaybackEvent** — `track_id`, `session_id`, `impression_id` when attributable, event type,
playback position/duration where relevant, timestamp. Types: `play_started`, `paused`,
`resumed`, `seeked`, `early_skip`, `25_percent`, `50_percent`, `75_percent`, `completed`,
`replayed`, `queued`.

**ExplicitFeedback** — `liked`, `disliked`, `saved`, `relevant`, `not_relevant`.

Recording `model/version` on the impression is what makes a past result reproducible and
lets a ranking change be attributed rather than guessed at.

### Signal strength — hypotheses, not truths

Events are not equivalent preference evidence. Working hypothesis, to be evaluated rather
than assumed:

| Strength | Signals |
|---|---|
| Strong | explicit relevant/not-relevant, like/dislike, save, repeat |
| Medium | manually queued, high completion, deliberate play from a recommendation |
| Weak / ambiguous | early skip, no click, pause, abandonment |

A skip may mean dislike, a wrong moment, or an interruption. Treating weak signals as
strong is the most likely way to build a confidently wrong preference model.

### Behaviour store

A local append-only event log, separate from the track/embedding store. Interface:
append an event; read events by session, query, track, or time range. Preference state is
*derived* from this log rather than mutated in place, so a preference model can be rebuilt
or recomputed under a different weighting without losing history.

### Connection to personalization

Playback events feed the three preference levels in the existing hierarchy — global
(long-term listening/save/repeat), intent-specific (behaviour under query intents such as
study, workout, late night), and session (what the user prefers right now).

The authority order is unchanged and unchanged by playback: **explicit query > current
session > intent profile > global**. Behavioural history refines ambiguity; it never
overrides an explicit query constraint.

Do not implement personalization now.

## Behavioural personalization architecture

Not implemented. Slice 5. Documented now because the item-representation constraint
shapes decisions made earlier.

```text
external behaviour data (Music4All-Onion)
        |
        |  item ids joined to Music4All 30s audio
        v
   aux's own encoder  ------------------+
        |                               |
        v                               |
behavioural pretraining / research      |
        |                               |
        v                               |
global / session / intent mechanism     |
        |                               |
        +-------------------------+     |
                                  |     |
aux playback events --------------+     |
(same encoder, local audio) -------------+
                                  |
                                  v
                        user preference state
                                  |
                                  v
                      personalized reranking
```

The single encoder appearing on both paths is the point. External audio and the user's
own files enter the same encoder and produce representations of the same kind, so a
mechanism learned on one applies to the other.

### The transfer constraint

Any behavioural model intended for production must consume item representations
**computable from the user's audio at inference time**. A model learning
`user_id → external_song_id` cannot rank a file it has never seen.

This rules out collaborative filtering over dataset item IDs as the shipped architecture.
It remains admissible as a *benchmark* — it quantifies how much collaborative signal
exists, which bounds what any content-conditioned model could hope to recover.

### Three preference levels

**Global taste** — what this user generally prefers. Inputs: long-term play counts,
repeats, likes/saves, track content embeddings. Baseline: a weighted average of
positively-interacted track embeddings.

**Session preference** — what the user wants right now. Inputs: recent ordered listens,
recent selections and skips, current query context. Baselines: last-track representation,
mean of last N, recency-weighted average — all computable directly in the embedding space.

**Intent preference** — what "study" or "late night" means *for this user*. Inputs:
behaviour grouped by query intent. Playlist-title research shows short natural-language
intent labels carry usable semantics and help most in cold start; an aux query and a
playlist title are structurally similar.

Authority is unchanged and playback does not alter it:
**explicit query > current session > intent profile > global taste**.

### Method selection stance

Start simple and make complexity earn its place. A RecSys replicability study found
BERT4Rec's published results could not be reproduced under its default configuration, and
nearest-neighbour methods beat BERT4Rec, GRU4Rec and SASRec on some datasets. With a
single user's sparse history, simple content-space baselines are the honest starting
point; deep sequential models only after they beat one on our own held-out evaluation.

### Known tension — clip length

Music4All provides 30-second centre clips. Behavioural research on it is therefore
inherently single-segment. If E1 concludes multi-segment pooling is better for retrieval,
the production representation and the behavioural training representation diverge, and
that gap must be measured rather than assumed away. Open question.

## Evaluation

### Baseline

Whole-query text→music retrieval with a single frozen joint encoder and one segment per
track. Every added component must beat this.

### Metrics

- Decode/encode success rate, preprocessing latency, deterministic repeatability
- Song Describer retrieval: Recall@1, Recall@5, Recall@10, median rank
- Human relevance on a held query set: mean relevance, graded NDCG, success@K, pairwise
  win rate
- Indexing and query latency; embedding storage footprint

### Evaluation Design

Layered, because vibe is subjective and no single metric is ground truth: system validity
→ public retrieval benchmarks → planner correctness → modality-specific relevance → human
judgment → personalization behaviour → diversity trade-offs.

Full plan in `evals/EVALS.md`. Experiments exist only when their outcome changes a design
decision.

## Major Decisions

| Decision | Choice | Why | Alternative Considered |
|---|---|---|---|
| Canonical input | User-provided local media, decoded audio | Streaming APIs expose proprietary features that break train/inference parity | Spotify/streaming integration |
| Similarity target | Emergent from pretrained representation + ranking | Musical similarity is subjective and multi-dimensional; a fabricated scalar label would not be meaningful | Supervised pairwise similarity model |
| Encoder | Frozen pretrained, benchmarked | Training a music foundation model is out of compute scope and unnecessary | Train encoder from scratch |
| First retrieval | Whole-query text→music baseline | A query planner must prove incremental value against a strong simple baseline | Start with routed multimodal retrieval |
| Personalization authority | Query > session > intent profile > global | A user may explicitly request music outside their normal taste | Single global taste vector |
| Vector search | Exact cosine / brute-force or exact FAISS | Personal-library scale does not justify ANN or a hosted vector DB | Hosted vector database, ANN index |
| Fusion | Rank-level (e.g. RRF) initially | Raw cosine scores from independent embedding spaces are not calibrated | Learned score fusion |
| Sample rate | Preserve source; resample per encoder | A single universal rate breaks when a second encoder has a different contract | Resample everything once at ingest |

Full decision history and revisit conditions in `DECISIONS.md`.

## Risks / Assumptions

- **Assumption:** at least one pretrained joint music-text encoder gives useful retrieval
  over both public audio and arbitrary user media through the same raw-audio pipeline.
  This is the Slice 0 hypothesis and is unproven.
- **Risk:** MuQ-MuLan's CC-BY-NC weights constrain productization if it wins the benchmark.
- **Risk:** Song Describer is small (706 tracks / 1,106 captions; 546/746 validated) and is
  only a proxy for real-user vibe satisfaction.
- **Risk:** MTG-Jamendo tags are uploader-provided weak supervision, not a canonical
  definition of vibe.
- **Risk:** personal library is out-of-domain commercial music; public-data performance may
  not transfer.
- **Risk:** lyrics may not be legally or reliably obtainable for user-owned music, which
  would block Slice 3.

## Open Design Questions

1. CLAP vs MuQ-MuLan winner
2. Single vs multi-segment track pooling
3. Which LLM for the planner, and at what latency/cost budget? (Model choice open;
   DEC-007 fixes only that it is an LLM with schema-constrained output.)
4. Exact lyric chunking strategy
5. Final fusion method
6. Whether the mood expert adds value
7. Final learned ranker
8. UI/backend stack
9. Whether advanced concept steering is worth adding
10. Playback technology — which local audio playback approach fits without pulling in a
    heavy framework, and does it decode the same formats the ingestion pipeline accepts?
    Deliberately unresolved; research when Slice 1B becomes active.
11. Does external behavioural pretraining beat a simple first-party content centroid at
    small history sizes? Decides whether external pretraining ships or stays research.
12. Does the 30-second-clip constraint conflict with a multi-segment pooling decision?
13. Do Last.fm scrobbles and aux in-product actions carry the same preference semantics?
    Passive listening and deliberate action on an explicit query may differ.
14. Can the LLM planner be distilled into a small local model without material quality
    loss, removing the hosted dependency? (E2a)

## Design Revisions

None yet — initial design.

> Canonical technical design.
> User owns major decisions; Claude may document and maintain them.
> Update only when architecture, data flow, ML approach, evaluation, or material technical decisions change.
