# Lessons

Approaches tried and abandoned, and mistakes worth not repeating. Kept because a
negative result is a result, and because the reasoning is harder to reconstruct than
the code.

Each entry records what was tried, what happened, and what it changed.

---

## Tuning depth inside the wrong method family

**Tried.** Twenty configurations of `standardize → PCA → distance`, varying dimensions
(16–256), whitening, and metric.

**What happened.** It produced a defensible configuration (128 dims, whitened, cosine)
and two genuine findings — cosine beats Euclidean in all ten paired configurations, and
whitening helps under cosine while hurting badly under Euclidean. But every run sat
inside one method family, and that family was never itself tested.

**Why it was the wrong shape of search.** PCA maximises variance. We need a space where
distance means perceived similarity. There is no reason the directions explaining the
most spread are the directions organising similarity — so twenty runs refined a
projection whose objective never matched the goal.

**Changed.** Breadth before depth: a few configurations across several method families
before tuning any one of them. Written into `AGENTS.md`.

---

## Optimising without knowing the ceiling

**Tried.** Reading weak retrieval and weak perceptual agreement as evidence that
hand-crafted spectral features are inherently weak.

**What happened.** [FMA's own paper](https://arxiv.org/abs/1612.01840) reports 63%
genre accuracy with an SVM on these same 518 features (16 genres, ~10× chance). Our
kNN-on-PCA reached ~50% on 8 genres (~4× chance). Relative to chance the published
baseline is substantially stronger — and we had never run a supervised classifier on the
raw features to find out.

**Why it mattered.** "The features are weak" and "our embedding discards what the
features contain" call for opposite responses, and nothing we had run distinguished them.

**Changed.** Establish the supervised ceiling on raw inputs before optimising any
projection. Written into `AGENTS.md`.

---

## Reading silhouette ≈ 0 as vindication

**Tried.** Interpreting near-zero silhouette across every configuration as "genres
overlap, which is the shape Bridge Finder needs."

**What happened.** That reading is consistent with the data — and so is "the embedding is
poorly organised and genuine genre structure is not being surfaced." The flattering
interpretation was recorded as though it were the only one.

**Changed.** Where two readings fit the same number, name both and design the experiment
that separates them. Here that is the supervised ceiling check.

---

## Native sample rates across corpora

**Tried.** Loading each audio file at its own native rate (`librosa.load(sr=None)`).

**What happened.** FMA arrives at 44.1 kHz and MagnaTagATune at 16 kHz, so their
descriptors landed on different frequency axes and were silently incomparable. Model A
depends on combining corpora — FMA for audio, MTAT for human judgements, AcousticBrainz
for the user's own library — so this would have invalidated every cross-corpus result.

**Changed.** All hand-crafted extraction resamples to a fixed 22050 Hz, and `sr` is
passed explicitly to every librosa call taking `S=` (librosa otherwise assumes 22050 and
builds its frequency axis from that regardless of the real rate).

**Note.** That constant governs hand-crafted features only. Pretrained models carry their
own input requirements — CLAP expects 48 kHz — and must not be fed the 22050 pipeline.

---

## Treating FMA's precomputed features as ground truth

**Tried.** Validating our extractor by correlating against FMA's published features,
expecting high agreement to mean correctness.

**What happened.** FMA's features were computed from 44.1 kHz audio while letting librosa
default to `sr=22050` in the feature calls, so their frequency axis is mislabelled by 2×.
Verified per track: their stored values track the native-rate/default-`sr` variant, and
the correctly-labelled variant is exactly double.

An earlier inference from the Nyquist ceiling — centroids never exceeding 7974 Hz,
rolloff capping at 10451 Hz — was consistent with this *and* with genuine resampling to
22050. Only extracting one track three ways and comparing against their stored values
separated the two.

**Changed.** Correlation with FMA measures how faithfully we reproduce their pipeline,
bug included — not correctness. The harness comparison is the validity check, and it
passes: our features score better on genre kNN (0.400 vs 0.325) while discarding
everything above 11 kHz. Aggregate statistics suggest; a concrete case decides.

---

## Checkpoint defects that model cards do not mention

**Found before use, not after.** `laion/larger_clap_music` has a degenerate text tower —
mean pairwise text-text cosine similarity 0.9993, scoring below random on audio-text
retrieval ([mteb#5069](https://github.com/embeddings-benchmark/mteb/issues/5069)). Audio
embeddings are unaffected.

**Consequence.** Usable for audio similarity, unusable for Vibe Match's text→audio
retrieval. `laion/clap-htsat-unfused` and `laion/larger_clap_general` have working text
towers.

**Changed.** Search for known failure modes of a candidate model before adopting it, and
confirm specifications from config files rather than model cards — the cards omitted
sample rate, embedding dimensionality, and this defect.

---

## Corrections to earlier records

- LAION-CLAP weights were recorded as CC0 from a secondary source. The model cards state
  **Apache 2.0**. Licence claims now come from the model card or repository directly.
- Self-supervised contrastive pretraining was rejected as needing a larger corpus than we
  hold. That was written about CLMR and COLA at research scale and over-generalised;
  InfoNCE with a small CNN is tractable at FMA Small scale. Reinstated as a candidate.
