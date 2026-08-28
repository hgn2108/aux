# Results

Index of experiments. Metrics live in MLflow (`sqlite:///mlflow.db`, experiment
`track-a-acoustic`); this file is the human-readable record of what was run and what it
meant. Reproduce any row with `python -m aux.experiments.sweeps`.

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

## Pending

- **MagnaTagATune perceptual triplets** — loader and metric are implemented and tested
  (307 usable triplets of 533 after dropping 87 ties and low-vote rows). Running it needs
  embeddings for MTAT clips, which needs A4b's audio feature extraction. Blocked, not
  abandoned.
