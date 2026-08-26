# aux — Personal Music Intelligence Platform

Last synchronized from the PRD: 2026-08-22  
Canonical PRD: [Google Doc](https://docs.google.com/document/d/1qpTDbsf4_DMsZeh5HesfnpFR4fEOFbu0wbqYzn6zeTY/edit?usp=sharing)  
PRD title/current working name: **aux — Personal Music Intelligence Platform** (renamed from “Resonance” on 2026-08-27; Google Doc PRD not yet updated)  
PRD status: Pre-build / architecture defined  
Owner: Irene Nguyen

## Why this exists

This is a portfolio-quality, n-of-1 music-personalization product designed to demonstrate both ML engineering and AI engineering: genuinely trained and evaluated models, production-style serving/versioning/monitoring, an agent that calls models as tools, a mechanical regression/evaluation loop, and credible product framing around discovery, retention, and personalization.

## Product promise

The system learns the user's musical world from three independent signals:

- acoustic/content: what music sounds like;
- behavioral: how the user actually listens, sequences, and repeats;
- lyrical/thematic (optional/later): what a song is about.

Users interact conversationally and can:

- recall what they listened to in a period;
- find tracks similar to an anchor with a requested shift in mood or energy;
- use **Bridge Finder** to find tracks between two known tracks;
- use **Vibe Match** to find tracks for a natural-language mood, moment, or playlist idea;
- explore where a track sits acoustically beyond its genre label.

## Non-negotiable product principles

1. **Describe; never judge taste.** Do not score whether a user's taste, track, transition, or playlist is “good.” Explain measurable relationships such as tempo, energy, key, spectral proximity, behavioral proximity, and thematic similarity.
2. **Be honest about coverage and confidence.** If the library lacks a strong match, say “closest available” and explain caveats rather than forcing certainty.
3. **Keep evaluation mechanical.** Bridge Finder should reduce measurable gaps between anchors. Vibe Match results should satisfy acoustic/thematic ranges implied by a hand-labeled golden set. The eval harness must not be cut.
4. **Use data only when its terms permit the intended ML use.** Re-check new sources rather than relying on old music-API tutorials.
5. **Lyrics are embed-then-discard.** If Model D is built, raw lyric text must never be stored, displayed, logged, or redistributed. If this cannot be implemented correctly, omit Model D.

## Product capabilities and model boundaries

- **Model A — acoustic/content embeddings:** raw CC-licensed audio, feature extraction, fixed-length embeddings.
- **Model B — behavioral/taste embeddings:** Last.fm listening sequences/co-occurrence; evaluated on held-out history; later reweighted from user preference pairs.
- **Model C — Bridge Finder / Vibe Match compatibility:** outputs a multi-axis compatibility description, never a scalar quality score.
- **Model D — lyrical/thematic embeddings:** optional enhancement using transient, permitted lyric previews; vector retained, source text discarded.
- **Agent layer:** uses the models and metadata/retrieval functions as real tools and logs recommendation events.
- **Preference loop:** accept/reject/refine feedback becomes small-scale preference pairs used for contrastive reweighting of Model B, versioned in MLflow. It is explicitly not positioned as Spotify-scale DPO/RLHF.

## Data-source constraints from the PRD

- Last.fm: primary behavioral history.
- MusicBrainz: metadata enrichment.
- FMA/Jamendo: CC-licensed raw audio for content features.
- Spotify Million Playlist Dataset: optional historical co-occurrence pretraining prior; catalog ends in 2017 and is not current-catalog truth.
- Spotify Web API audio features/analysis: do not use without fresh verification; the PRD records endpoint removal/deprecation and ML-training restrictions.
- Genius lyrics: do not use.
- Musixmatch previews: only if current terms permit this personal/non-commercial derived-signal use; embed transiently and discard text.

## Architecture direction

- Data ingestion → trained models and serving → conversational/tool-calling agent.
- FastAPI + Docker for serving.
- MLflow for experiment/model/retraining tracking.
- Vector database for track embeddings; exact product remains open.
- Drift, distribution, and useful latency monitoring.
- Structured recommendation-event logging feeding the preference loop.

## Scope order

Ship a coherent Models A/B/C system before optional breadth. If time is constrained, reduce full-library audio coverage, Model A complexity, dashboard polish, number of agent tools, preference-loop sophistication, MPD pretraining, then Model D—in roughly that order.

Never cut:

- the eval harness;
- descriptive/non-evaluative Model C outputs;
- lyric embed-then-discard compliance if Model D exists.

## Open implementation decisions

- vector DB choice;
- Model A architecture;
- Model B sequence vs. simpler co-occurrence/factorization approach;
- batch vs. live scrobble refresh;
- preference reweighting method and cadence;
- preference-loop proxy metrics;
- MPD sample size/value;
- Vibe Match confidence/coverage representation;
- Model D embedding approach and minimum viable lyric-preview coverage;
- whether thematic distance belongs in Bridge Finder or remains Vibe Match-only.

## Decision log

Add entries in newest-first order.

| Date | Decision | Rationale | PRD impact |
|---|---|---|---|
| 2026-08-27 | Renamed the project from “Resonance” to **aux**. | Product naming decision by the owner. | Title/name change only; the canonical Google Doc PRD still says “Resonance” and needs a sync. |
| 2026-08-22 | Created a repository-local project context and update protocol. | Makes stable product rationale and constraints available to future coding sessions while retaining the Google Doc as canonical PRD. | None; documentation workflow only. |

## Sync protocol

- Read this file before project work.
- Treat the Google Doc as canonical product intent and this file as the repository's compact working memory.
- Record meaningful decisions here as they are made.
- Update the Google Doc as the product evolves when Irene asks to sync it; before editing, fetch the latest revision and preserve its structure and stable rationale.
- Repo code, README, changelog, and decision log are the source of truth for implemented details if they differ from old PRD Sections 6+.
