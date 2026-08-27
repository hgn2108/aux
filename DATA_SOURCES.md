# Data sources

One entry per source, added when a track first needs it. Records what we verified,
when, and what it permits — so decisions are auditable and re-checkable.

| Source | Verified | Verdict | Used for |
|---|---|---|---|
| Last.fm API | 2026-08-27 | Usable | Live scrobble refresh (Track B) |
| FMA | 2026-08-27 | Usable, with conditions | Acoustic features (Track A) |
| Spotify data export | 2026-08-27 | Usable — own personal data | Behavioral history backfill (B0) |
| Spotify Web API audio features | — | **Not used** | — |
| Genius | — | **Not used** | — |
| Musixmatch | — | Unverified; gates Model D | Lyrical embeddings (D1) |

---

## FMA (Free Music Archive)

Verified 2026-08-27 against <https://github.com/mdeff/fma>.

Licensing is split three ways:

- **Code** — MIT.
- **Metadata** — CC BY 4.0. Attribution required wherever we publish results.
- **Audio** — no single license. FMA does not hold copyright and distributes each
  track under the license its artist chose. Per-track license must be recorded
  alongside any derived feature (roadmap A3).

Stated as "meant for research purposes," with no restriction on machine learning.

**Our use:** personal, non-commercial research and portfolio work. We extract
features and store derived vectors. We do not redistribute audio.

**Obligations:** cite the FMA paper and dataset for metadata (CC BY 4.0); carry
per-track license through the feature pipeline; re-check before any commercial or
redistributive use.

## Last.fm

Verified 2026-08-27. Read-only, unauthenticated endpoints for the account's own
listening history. No scrobble history exists yet (account created 2026-08-26), so
this is the live-refresh source rather than the backfill source.

## Spotify data export

Personal data export requested under GDPR/CCPA rights — the user's own listening
history, supplied to them by Spotify. Distinct from the Spotify Web API, whose
audio-features endpoints remain **unused and unverified** per the PRD.

---

## Results log

**A4a — Model A baseline, FMA small (8,000 tracks, 8 balanced genres), 2026-08-27**

Standardized FMA precomputed features (518 dims) reduced by PCA. Genre is a proxy
for musical structure here, not a prediction target.

| Embedding | dims | kNN acc | retrieval P@5 | silhouette |
|---|---|---|---|---|
| chance | — | 0.125 | 0.125 | — |
| raw features | 518 | 0.486 | 0.408 | -0.043 |
| PCA | 16 | 0.424 | 0.362 | -0.052 |
| PCA | 32 | 0.479 | 0.398 | -0.048 |
| **PCA** | **64** | **0.508** | **0.411** | -0.046 |
| PCA | 128 | 0.499 | 0.411 | -0.044 |
| PCA | 256 | 0.492 | 0.410 | -0.043 |

64 dims chosen: best kNN accuracy, ties best P@5, and 8x smaller than raw.
