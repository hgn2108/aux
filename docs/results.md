# Results

Index of experiments. Metrics live in MLflow (`sqlite:///mlflow.db`, experiment
`track-a-acoustic`); this file is the human-readable record of what was run and what it
meant. Each section names the command that reproduces it.

---

## A4a/A5 — Model A baseline, FMA small

**2026-08-29** · 8,000 tracks, 8 balanced genres · FMA precomputed features (518 dims)
· 20-configuration grid: dims × whitening × metric

Chance rates differ enormously by label, so raw numbers mislead: genre 0.125, artist
0.00199, album 0.00068. Lift matters more than the absolute value.

| dims | whiten | metric | genre P@5 | artist P@5 | album P@5 | artist lift | album lift | genre kNN |
|---|---|---|---|---|---|---|---|---|
| 64 | no | euclidean | 0.411 | 0.128 | 0.096 | 64× | 142× | 0.508 |
| 256 | no | cosine | **0.453** | 0.162 | 0.125 | 81× | 184× | **0.538** |
| 128 | yes | cosine | 0.426 | 0.188 | 0.151 | 94× | 223× | 0.503 |
| **256** | **yes** | **cosine** | 0.415 | **0.190** | **0.154** | **95×** | **228×** | 0.502 |
| 256 | yes | euclidean | 0.290 | 0.118 | 0.097 | 59× | 143× | 0.349 |

**Chosen: PCA 128, whitened, cosine.** Within noise of 256 on every metric at half the
dimensions.

### What it means

**Cosine beats Euclidean in all 10 paired configurations.** Not marginal, and consistent
enough not to be chance. Spectral feature vectors carry a strong overall-magnitude
component — roughly, how loud and dense a track is — and Euclidean lets that dominate.
Cosine compares *shape* instead.

**Whitening only helps under cosine, and hurts badly under Euclidean.** At 256 dims,
whitened Euclidean is the worst configuration in the grid (genre kNN 0.349, below the
unwhitened 16-dim result). Whitening rescales every PCA component to unit variance,
which lifts low-variance noise directions to the same weight as signal; Euclidean then
sums that noise across all dimensions, while cosine's normalization largely absorbs it.

**The evaluation target changed the answer.** The earlier "64 dims is optimal" finding
came from genre kNN accuracy under Euclidean. Against album and artist retrieval — the
tighter proxies for *sounds alike* — higher dimensions and whitening win instead. The
first conclusion was an artifact of the metric being measured, which is the argument for
having built the harness first.

**Silhouette stays ≈ 0 everywhere** (−0.055 to +0.003). Genres are not separable
clusters in acoustic space. But album retrieval runs 223× chance, so local neighbourhoods
are highly structured. Local structure is real; global genre boundaries are not — which
is the space the PRD actually wants, since Bridge Finder needs to traverse between
anchors without genre walls.

### Caveat

This is FMA's precomputed feature set, not ours. A4b recomputes features from audio and
scores them here for comparison.

---

## A4b — our own feature extraction vs FMA's reference

**2026-08-29** · 400 FMA tracks, seed 0 · run `ceba76ee`

### The headline: the reference has a mislabelled frequency axis

FMA's precomputed features were computed from 44.1 kHz audio while letting librosa
default to `sr=22050` in the feature calls. Their frequency values are half the true
figure.

Verified by extracting the same track three ways and comparing against their stored
values (`spectral_centroid` mean, Hz):

| track | FMA reference | native, no `sr` | native, correct `sr` | resampled 22050 |
|---|---|---|---|---|
| 2 | 1640 | 1842 | 3684 | 3057 |
| 5 | 1293 | 1464 | 2928 | 2430 |
| 10 | 1360 | 1393 | 2787 | 2358 |

FMA tracks the *native, no `sr`* column; the correctly-labelled variant is exactly
double it. An earlier inference from the Nyquist ceiling — their centroids never exceed
7974 Hz, rolloff caps at 10451 Hz — was consistent with this *and* with genuine
resampling to 22050. Only the per-track comparison separates them.

### The decision, and what it costs

Every corpus is now resampled to a fixed 22050 Hz. FMA arrives at 44100 and
MagnaTagATune at 16000; loading each natively put their features on different frequency
axes and made them silently incomparable — which the AcousticBrainz plan depends on.

This deliberately diverges from the reference, and correlation drops as a result:

| pipeline | overall median correlation |
|---|---|
| native rate, no `sr` (replicates FMA) | +0.702 |
| native rate, correct `sr` | +0.635 |
| **fixed 22050, correct `sr`** (chosen) | **+0.433** |

`spectral_contrast` falls to +0.004, because its octave-spaced bands cover entirely
different content once the top octave is gone.

**Correlation is therefore not a correctness test here.** It measures how faithfully we
reproduce FMA's pipeline, bug included. The harness is the real check:

| | genre P@5 | genre kNN | artist P@5 | album P@5 |
|---|---|---|---|---|
| ours | 0.228 | **0.400** | 0.009 | 0.007 |
| FMA reference | 0.236 | 0.325 | 0.019 | 0.013 |

Comparable on genre retrieval and **better on genre kNN** (0.400 vs 0.325), from a
feature set that discards everything above 11 kHz. The extractor is sound.

*(Retrieval numbers are far below the 8,000-track run because at n=400 there are 342
artists and 366 albums — almost every track is its own class, so artist and album P@5
have little room to score. Only the ours-vs-reference comparison is meaningful here.)*

### Per-statistic breakdown

Pooled across families, this explains why single-coefficient families looked weak:

| statistic | correlation |
|---|---|
| median | +0.892 |
| mean | +0.890 |
| std | +0.670 |
| skew | +0.423 |
| min | +0.337 |
| kurtosis | +0.315 |
| max | +0.256 |

Central tendency reproduces closely; higher moments and extrema are unstable across
decoders and framing. A family's median is taken over its seven statistics, so
`spectral_centroid` (1 coefficient) is dominated by the noisy four, while `mfcc`
(20 coefficients) and the chromas (12) average that noise away. The low family scores
were a measurement artifact, not broken descriptors.

---

## A10 — human similarity judgements, MagnaTagATune

**2026-08-29** · 307 triplets (of 533, after dropping 87 ties and low-vote rows) ·
embedding fit on 2,500 MTAT clips · run `08b4daca`

The "odd one out" task: given three clips, do we agree with listeners about which is
least like the other two? Chance is 1/3.

| metric | agreement | 95% CI | one-sided binomial p | before A4b fix |
|---|---|---|---|---|
| cosine | 0.397 (122/307) | [0.351, —] | **0.011** | 0.394 |
| euclidean | 0.371 (114/307) | [0.325, —] | 0.089 | 0.352 |

### The number is stable across two different feature pipelines

This was re-run after A4b changed extraction substantially — fixed 22050 Hz resampling,
explicit `sr` on every call, MTAT upsampled from 16 kHz. Cosine agreement moved 0.394 →
0.397. Euclidean improved more (0.352 → 0.371, p 0.265 → 0.089), narrowing the gap
between metrics from 0.042 to 0.026.

That stability matters: it rules out the possibility that marginal perceptual agreement
was an artifact of the sample-rate bug. **It is a real property of hand-crafted spectral
summaries.**

### Read the power analysis before the p-value

- Cohen's *h* ≈ 0.13, below the 0.2 conventional floor for a "small" effect.
- 388 triplets would be needed for 80% power; we have 307.
- Minimum rate reliably detectable at n=307 is 0.402. We observed 0.397 — still *below*
  our own detection threshold.

A significant p-value, on an underpowered test, at an effect the study cannot resolve.
The honest statement remains: weak evidence of above-chance agreement, not a result.

### The finding that matters

Metadata proxies said the embedding was excellent — album retrieval at 223× chance.
Human perceptual agreement says it is marginal. **Doing well on metadata does not mean
matching how people hear.** Album retrieval rewards shared production, mastering, and
session; listeners judging "odd one out" are not hearing those.

No metadata metric could have surfaced this, which is why the perceptual eval exists.
With the extractor now validated (A4b) and the number stable across pipelines, this is
the strongest argument for Tier 3: recent work reports ~72% zero-shot perceptual
agreement for pretrained models on a *different* benchmark — not directly comparable,
but not close to 0.397 either.

### Caveats

- MTAT audio is 16 kHz, 32 kbps mono, upsampled to 22050 for a shared axis. Its spectrum
  is empty above 8 kHz, so these features are genuinely thinner than FMA's. This should
  understate agreement rather than inflate it.
- The embedding is fit on MTAT clips, not the FMA-fitted space used elsewhere.
- MTAT is exhausted as an eval set at 533 raw comparisons. Power could reach 446 triplets
  by accepting margin-1 ties and single-vote rows, at the cost of noisier labels — which
  would likely lower the observed rate too. Not obviously a good trade.
