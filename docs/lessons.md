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

## Fitting a supervised projection before splitting

**Tried.** Building every candidate embedding space over the full corpus, then handing
the transformed matrix to the harness, which does its own train/test split for kNN
accuracy.

**What happened.** For PCA that is merely transductive — no labels are involved. For LDA
and NCA it is label leakage: the projection has already seen the test set's labels, so
the split that follows protects nothing.

The cost was 8.7 points. LDA's genre kNN read **0.629** fit on everything and **0.542**
fit on the training split alone. The inflated figure happened to land on top of the RBF
SVM's 0.630, which produced a clean and completely false headline — *seven supervised
dimensions carry all the genre information an SVM finds in 518*. Honest LDA (0.542) is
barely above raw-feature kNN (0.540) and nowhere near the ceiling.

**How it surfaced.** A question about ensembling PCA and LDA prompted a re-read of the
code. Nothing in the numbers looked wrong — a supervised method beating an unsupervised
one is exactly what you expect, and matching the SVM felt like a satisfying result rather
than a suspicious one.

**Changed.** Every projection is fit on the training split and applied to the rest;
`aux/experiments/ceiling.py` takes the split before building any space. The general rule:
**a result that lands exactly where you hoped deserves the same scrutiny as one that does
not** — and any step fit with labels belongs inside the split, not before it.

**Kept as a diagnostic.** Running an unsupervised method beside supervised ones is what
made the leak visible: PCA moved 0.503 → 0.504 while LDA moved 0.629 → 0.542. A control
that *should not* change is a cheap way to detect a protocol error.

---

## Log-transforming heavy-tailed features: no effect

**Tried.** Applying `log1p` to the strictly-positive magnitude descriptors (rms,
spectral centroid, bandwidth, rolloff, ZCR) before standardising, on the reasoning that
these are roughly log-normal across tracks and a few loud or bright tracks would
otherwise dominate every distance.

**What happened.** Nothing measurable. Genre P@5 went 0.452 → 0.451 on raw features and
0.426 → 0.426 under PCA; kNN moved within a point. Artist and album retrieval were
unchanged to three decimals.

**Why the reasoning did not hold.** Only 35 of 518 columns are affected, and PCA's
whitening already equalises component scales, so the tail correction had little left to
do by the time distances were computed.

**Kept anyway.** `log_heavy_tails` stays available — it costs nothing, and it may matter
for a distance computed without whitening. Recorded here so nobody re-derives the
hypothesis and re-runs the experiment.

---

## Corrections to earlier records

- LAION-CLAP weights were recorded as CC0 from a secondary source. The model cards state
  **Apache 2.0**. Licence claims now come from the model card or repository directly.
- Self-supervised contrastive pretraining was rejected as needing a larger corpus than we
  hold. That was written about CLMR and COLA at research scale and over-generalised;
  InfoNCE with a small CNN is tractable at FMA Small scale. Reinstated as a candidate.
