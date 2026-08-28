# Data sources

Licensing register. One entry per source, added when a track first needs it, recording
what we verified, when, and what it permits — so decisions are auditable and re-checkable.
Results live in `RESULTS.md`; reasoning lives in `docs/`.

| Source | Verified | Verdict | Used for |
|---|---|---|---|
| Last.fm API | 2026-08-27 | Usable | Live scrobble refresh (Track B) |
| FMA | 2026-08-27 | Usable, with conditions | Acoustic features (Track A) |
| Spotify data export | 2026-08-27 | Usable — own personal data | Behavioral history backfill (B0) |
| AcousticBrainz | 2026-08-29 | Usable — CC0 | Acoustic coverage for the user's library (A, Tier 2) |
| MagnaTagATune | 2026-08-29 | Usable for research | Perceptual similarity evaluation only |
| Essentia (software) | 2026-08-29 | AGPL — dependency licence | Feature extraction |
| Spotify Web API audio features | — | **Not used** | — |
| Genius | — | **Not used** | — |
| Musixmatch | — | Unverified; gates Model D | Lyrical embeddings (D1) |

## Rejected audio sources

Investigated for Track A and closed. Recorded so the reasoning is not relitigated.

| Source | Reason |
|---|---|
| YouTube (API or extraction) | Data API exposes no audio stream; extraction breaches ToS. Fails principle 4. |
| iTunes / Apple previews | Terms require previews be streamed only — never downloaded, saved, or cached — and only in a promotional context beside a purchase link. Feature extraction requires what is prohibited. |
| Deezer previews | Developer terms explicitly forbid harvesting and mining of data. |
| Unauthorized MP3 sources | Same failure as YouTube; would undermine the project's provenance. |

---

## FMA (Free Music Archive)

Verified 2026-08-27 against <https://github.com/mdeff/fma>.

Licensing splits three ways:

- **Code** — MIT.
- **Metadata** — CC BY 4.0. Attribution required wherever we publish results.
- **Audio** — no single licence. FMA does not hold copyright and distributes each track
  under the licence its artist chose. Per-track licence must be recorded alongside any
  derived feature (roadmap A3).

Stated as "meant for research purposes," with no restriction on machine learning.

**Our use:** personal, non-commercial research and portfolio work. We extract features
and store derived vectors. We do not redistribute audio.

## AcousticBrainz

Verified 2026-08-29 against <https://musicbrainz.org/doc/AcousticBrainz>.

**CC0 (public domain)** — no restriction on commercial, research, or ML use. Indexed by
MusicBrainz recording MBID, which joins directly to the metadata resolution in roadmap
1.2. Precomputed Essentia features: low-level spectral descriptors plus high-level
rhythm and tonal features.

Submissions ended in 2022; the dump is frozen at June 2022. Anything released after that
has no entry — a coverage gap to report, not a blocker.

## MagnaTagATune

Verified 2026-08-29. Audio clips are Magnatune-published excerpts released for research
with the label's approval; annotations come from the TagATune game.

**Scope: evaluation only.** Audio is 16 kHz, 32 kbps mono — too compressed to serve as a
feature-extraction corpus. What it uniquely provides is 533 human "odd one out" similarity
judgements (7,650 votes), of which 307 survive filtering for ties and low vote counts.

## Last.fm

Verified 2026-08-27. Read-only, unauthenticated endpoints for the account's own listening
history. No scrobble history exists yet (account created 2026-08-26), so this is the
live-refresh source rather than the backfill source.

## Spotify data export

Personal data export requested under GDPR/CCPA rights — the user's own listening history,
supplied to them by Spotify. Distinct from the Spotify Web API, whose audio-features
endpoints remain **unused and unverified** per the PRD.
