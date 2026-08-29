# Track A — acoustic embeddings

What Model A is for, what audio we can lawfully use, and which approaches are worth
trying in what order. Dead ends and their evidence live in `lessons.md`; measured results
live in `results.md`.

## What Model A is for

A function from a piece of music to a fixed-length vector, so that "these two sound
alike" becomes distance arithmetic.

Two things depend on it:

- **Bridge Finder** must judge whether a candidate sits *between* two anchors on tempo,
  energy, key, and spectral character. With no acoustic vector there is no acoustic axis,
  and Bridge Finder collapses into co-listen statistics.
- **"Where does this track sit beyond its genre label"** is a PRD capability that
  requires acoustic position to be measurable.

Model A is one of three independent signals: acoustic (A), behavioral (B), lyrical (D).

## Where the audio comes from

Investigated and closed:

| Source | Verdict |
|---|---|
| YouTube (API or extraction) | Rejected. The Data API exposes no audio stream; extraction breaches ToS. Fails PRD principle 4. |
| iTunes / Apple previews | Rejected. Terms require previews be streamed only — never downloaded, saved, or cached — and only in a promotional context beside a purchase link. |
| Deezer previews | Rejected. Developer terms explicitly forbid harvesting and mining of data. |
| Spotify audio features | Rejected by the PRD; endpoints deprecated and ML use restricted. |
| Unauthorized MP3 sources | Rejected. Same failure as YouTube, and it would undermine the provenance the rest of the project establishes. |

What remains:

- **AcousticBrainz** — CC0, ~7.5M recordings keyed by MusicBrainz ID, precomputed
  Essentia features, frozen June 2022. Reaches the user's library without audio.
  Post-2022 releases are a known gap, reported under principle 2.
- **FMA** — the audio corpus for building and validating.
- **MagnaTagATune** — evaluation only.
- **Music the user owns** — purchased downloads and ripped CDs, added directly.

## One feature space across corpora

Sources arrive at different rates: FMA at 44.1 kHz, MagnaTagATune at 16 kHz. Loading each
natively puts descriptors on different frequency axes and makes corpora silently
incomparable — and combining corpora is the whole architecture.

Hand-crafted extraction therefore resamples everything to a fixed **22050 Hz**
(`features.SAMPLE_RATE`), passing `sr` explicitly to every librosa call taking `S=`.

**This constant does not apply to pretrained models.** Each carries its own input
contract — CLAP expects 48 kHz with a 10-second window — and must be given audio loaded
to its own specification, never the 22050 pipeline's output.

## What the literature establishes

Consulted before choosing what to try, per `AGENTS.md`.

**Published baselines on our exact inputs.** [FMA's paper](https://arxiv.org/abs/1612.01840)
reports 63% genre accuracy with an SVM on the same 518 features (16 genres, ~10× chance).
That is the reference point for hand-crafted features — and it is a *supervised classifier
on raw features*, not an unsupervised projection.

**Pretrained audio models dominate hand-crafted features.** Zero-shot LAION-CLAP and
MuQ-MuLan reach ~72% agreement with human listeners on perceptual similarity
([Interpretable and Perceptually-Aligned Music Similarity](https://arxiv.org/html/2601.19109)),
without task-specific training.

**Similarity is multi-axis, and decomposing it wins.** That same work lifts perceptual
agreement from ~72% to 90.4% by decomposing similarity per source-separated stem and
learning per-component weights. [Disentangled multidimensional metric
learning](https://arxiv.org/pdf/2008.03720) and [Conditional Similarity
Networks](https://arxiv.org/html/2404.06682) train one embedding with **separate
subspaces per similarity notion** — timbre, rhythm, tonal, mood — selected by masks and
trained with triplet loss.

This is the same shape as Model C's requirement to report per-axis relationships instead
of a scalar. The best-performing published design and the PRD's non-negotiable coincide,
which is a strong signal that multi-axis structure belongs in the model rather than in a
presentation layer.

**Model weights carry their own licences and defects.** LAION-CLAP is Apache 2.0;
MERT-v1-330M weights are reported CC-BY-NC 4.0 while the MERT repository is Apache-2.0.
`laion/larger_clap_music` has a degenerate text tower and cannot serve text→audio
retrieval — see `lessons.md`.

## Approach ladder

Each rung must beat the previous one on the same harness.

| Tier | Approach | Status | Why it earns a try |
|---|---|---|---|
| 0 | Hand-crafted features + PCA | Done | Interpretable floor. Tempo/key/energy *are* the PRD's axes. |
| 1 | Supervised ceiling + learned projection | Next | Establishes what the features contain, and replaces a variance objective with a similarity one. |
| 2 | Pretrained embeddings — CLAP, PANNs | Next | ~72% zero-shot perceptual agreement published, no training. Apache-2.0 weights. |
| 3 | Multi-axis subspaces | Next | Best published design, and what Model C requires. |
| 4 | Contrastive CNN (InfoNCE) on mel-spectrograms | Candidate | Tractable at FMA Small scale. Only if Tiers 1–3 leave a gap. |
| 5 | Metric learning on co-listen triplets | Blocked on Track B | The differentiator: acoustic space tuned to this listener. |

**Rejected: training a CNN from scratch on genre classification.** At 8k tracks on a
laptop it loses to a pretrained encoder while costing days, and it optimises
classification rather than similarity.

**Deferred: source separation.** The 90.4% result depends on separating stems, which is a
heavy pipeline. Tier 3's per-axis decomposition captures the same structural idea using
feature families we already compute, at a fraction of the cost. Revisit only if Tier 3
shows the decomposition is what pays.

## How similarity is measured

Currently `StandardScaler → PCA(128, whiten) → cosine`, chosen empirically from a sweep.

Cosine beat Euclidean in all ten paired configurations; whitening helps under cosine and
hurts under Euclidean. But PCA maximises variance, not similarity — Tier 1 replaces it
with a projection trained against the thing we actually want.

**Open question for Track C.** If ranking is by cosine, "between two anchors" needs a
definition on a sphere rather than a line — spherical interpolation, or interpolating in
a whitened space while ranking by cosine. Unresolved; flagged so it is not assumed.

## How the embedding is evaluated

Retrieval, not classification: the product looks up similar tracks and never predicts a
genre. Precision@k on nearest neighbours asks the question the product asks.

Targets, weakest to strongest:

1. **Genre** — coarse, but should show partial grouping. Two folk tracks can sound
   nothing alike, yet genre is not noise either.
2. **Artist** — narrower; captures style.
3. **Album** — narrowest metadata proxy: shared production, instrumentation, session.
4. **Human similarity triplets** — the only target measuring perception rather than
   metadata. MagnaTagATune's "odd one out" game, 307 usable triplets after filtering.

Chance rates span three orders of magnitude across these (genre 0.125, album 0.00068), so
lift is reported rather than the bare score.

**Evaluation must be per-axis too.** A single averaged number cannot distinguish a strong
timbre signal with a broken rhythm signal from a mediocre everything. Each axis is scored
against the target it should govern — tonal against key agreement, rhythm against tempo
agreement, timbre against genre and album — before any blend is reported.

Genre remains the *proxy*, never the target. A Model A that merely reproduced genre labels
would be a genre classifier, not an acoustic embedding.
