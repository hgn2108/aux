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
               behavioral feedback
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
3. Exact query-planner model
4. Exact lyric chunking strategy
5. Final fusion method
6. Whether the mood expert adds value
7. Final learned ranker
8. UI/backend stack
9. Whether advanced concept steering is worth adding

## Design Revisions

None yet — initial design.

> Canonical technical design.
> User owns major decisions; Claude may document and maintain them.
> Update only when architecture, data flow, ML approach, evaluation, or material technical decisions change.
