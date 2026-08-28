# Track A — acoustic embeddings

Reasoning behind Model A: what it is for, what audio we can lawfully use, which
approaches are on the table, and which were rejected and why.

## What Model A is for

A function from a piece of music to a fixed-length vector, so "these two sound alike"
becomes distance arithmetic.

Two things depend on it:

- **Bridge Finder** must judge whether a candidate sits *between* two anchors on tempo,
  energy, key, and spectral character. With no acoustic vector there is no acoustic axis,
  and Bridge Finder collapses into co-listen statistics.
- **"Where does this track sit beyond its genre label"** is a PRD capability that requires
  acoustic position to be measurable.

Model A is one of three independent signals: acoustic (A), behavioral (B), lyrical (D).

## Where the audio comes from

Investigated and closed:

| Source | Verdict |
|---|---|
| YouTube (API or extraction) | Rejected. The Data API exposes no audio stream; extraction breaches ToS. Fails PRD principle 4. |
| iTunes / Apple previews | Rejected. Terms require previews be streamed only — never downloaded, saved, or cached — and only in a promotional context beside a purchase link. Feature extraction requires exactly what is prohibited. |
| Deezer previews | Rejected. Developer terms explicitly forbid harvesting and mining of data. |
| Spotify audio features | Rejected by the PRD already; endpoints deprecated and ML use restricted. |
| Unauthorized MP3 sources | Rejected. Same failure as YouTube, and it would undermine the provenance the rest of the project is establishing. |

What remains, and the architecture that follows:

- **AcousticBrainz** gives the user's library acoustic coverage without audio — CC0,
  ~7.5M recordings keyed by MusicBrainz ID, precomputed Essentia features, frozen June
  2022. Post-2022 releases are a known gap, reported under principle 2.
- **FMA** is the audio corpus for building and validating the pipeline.
- **Music the user owns** (purchased downloads, ripped CDs) adds coverage on top.

**The unifying move: extract with Essentia.** AcousticBrainz was built with Essentia's
`streaming_extractor_music`. Running the same extractor on any audio we lawfully hold
places those tracks in the *same feature space* as AcousticBrainz's 7.5M recordings — one
representation reached by several lawful routes. Essentia is AGPL: a dependency licence
to record, not a data licence.

**Coverage fallbacks** for tracks with neither audio nor an AcousticBrainz entry, each
surfaced with reduced confidence: MusicBrainz work-level substitution (a different
recording of the same work), artist centroid, and learned metadata→embedding translation.

## Approach ladder

Each rung must beat the previous one *on the same harness*.

| Tier | Approach | Status | Rationale |
|---|---|---|---|
| 0 | Hand-crafted features + PCA | Done | Interpretable — tempo/key/energy *are* the PRD's axes. Sets the floor. |
| 1 | Metric geometry + stronger eval targets | Done | Fixed a real defect and changed which configuration wins. |
| 2 | AcousticBrainz features | Next | The only representation that reaches the user's actual library. |
| 3 | Pretrained embeddings (CLAP first) | Planned | ~72% zero-shot perceptual agreement reported in recent work, no training required. |
| 4 | Metric learning on co-listen triplets | Blocked on Track B | The differentiator: acoustic space tuned to this listener. |

### Rejected, with reasons

**Training a CNN on mel-spectrograms from scratch.** At 8k tracks on a laptop it loses to
a pretrained encoder while costing days. The compute is better spent on Tier 4, which no
pretrained model can provide.

**MERT before CLAP.** LAION-CLAP-Music weights are CC0; MERT-v1-330M weights are reported
CC-BY-NC 4.0 while the MERT *repository* is Apache-2.0 — a code-vs-weights split like
FMA's. Licence hygiene applies to weights, not only to data. CLAP is also text-audio, so
it may serve Vibe Match natively rather than through hand-mapped prompt ranges.

**Self-supervised contrastive pretraining (CLMR, COLA).** Same objection as training from
scratch, plus it needs a far larger corpus than we hold.

## How similarity is measured

`StandardScaler` → `PCA(128, whiten=True)` → **cosine**.

Chosen empirically, and it contradicted the prior hypothesis. The expectation was that
whitened Euclidean would win, since Bridge Finder needs a metric where interpolation is
meaningful and an angle has no natural midpoint. The sweep said otherwise: cosine beat
Euclidean in all 10 paired configurations, and whitened Euclidean at 256 dims was the
worst configuration tested. See RESULTS.md for the numbers and the mechanism.

**Open question this creates for Track C:** if ranking is by cosine, "between two anchors"
needs definition on a sphere rather than a line — spherical interpolation, or
interpolating in whitened PCA space while ranking by cosine. Unresolved; flagged so it is
not silently assumed.

## How the embedding is evaluated

Retrieval, not classification: the product looks up similar tracks and never predicts a
genre. Measuring precision@k on nearest neighbours asks the question the product asks.

Label proxies, weakest to strongest:

1. **Genre** — coarse. Two folk tracks can sound nothing alike.
2. **Artist** — narrower; captures style.
3. **Album** — narrowest metadata proxy: shared production, instrumentation, session.
4. **Human similarity triplets** — the only target measuring perception rather than
   metadata. MagnaTagATune's "odd one out" game, 307 usable triplets after filtering.

Chance rates differ by three orders of magnitude across these (genre 0.125, album
0.00068), so lift is reported rather than the bare score.

Genre remains the *proxy*, never the target. The PRD explicitly wants a system that sees
past genre labels; a Model A that merely reproduced them would be a genre classifier, not
an acoustic embedding.
