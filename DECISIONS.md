# DECISIONS.md

> **Audience:** Shared, but Claude is expected to maintain it.
> **Purpose:** Append-only history of material architecture/product decisions.
> **Update frequency:** Only when a meaningful decision is accepted/rejected.
> **Rule:** Do not duplicate transient status here.

# aux — Decision Log

## DEC-001 — Local/user-provided media is the canonical product input

**Status:** Accepted

**Decision:**  
Do not depend on Spotify or another streaming platform for the core system.

**Why:**  
The previous design risked train/inference mismatch because public datasets and streaming APIs exposed different features. Raw user-provided media lets the same preprocessing and encoding pipeline run in development and inference.

**Implication:**  
Support local audio/video files and treat decoded audio as the canonical acoustic input.

**Revisit if:**  
A future platform integration exposes legally permitted raw/compatible inputs without breaking train/inference parity.

---

## DEC-002 — Do not supervise a fabricated scalar acoustic-similarity target

**Status:** Accepted

**Decision:**  
Use pretrained representations and retrieval/ranking evaluation instead of inventing pairwise similarity labels.

**Why:**  
Musical similarity/vibe is subjective and multi-dimensional.

**Implication:**  
Similarity becomes a retrieval/ranking construct, not a fake universal target.

---

## DEC-003 — Establish a simple joint text/music baseline before routed multimodal retrieval

**Status:** Accepted

**Decision:**  
Build whole-query text->music retrieval first.

**Why:**  
The more complex query-planner architecture must prove incremental value against a strong simple baseline.

**Implication:**  
Slice 1 exists before routed retrieval.

---

## DEC-004 — Personalization must be subordinate to explicit query intent

**Status:** Accepted

**Decision:**  
Behavioral history may refine ambiguity but may not override explicit current-query constraints.

**Default preference hierarchy:**

1. current query
2. current session
3. intent-specific profile
4. global preference

**Why:**  
A user may explicitly request music outside their normal taste.

---

## DEC-005 — Include a lightweight local playback layer

**Status:** Accepted

**Decision:**
aux will include a minimal in-product music player: play/pause, seek, next/previous,
queue, play directly from a search or recommendation result, basic metadata, and explicit
feedback (save, like/dislike, good match / bad match).

Added to the roadmap as **Slice 1B — Search → play → observe**, positioned after simple
natural-language search works.

**Why:**
Local media is already the system's canonical input, so playback is cheap to add and
requires no new data source. More importantly it closes the feedback loop:

```text
query -> recommendation impression -> user selects track -> playback
      -> behavioural event -> later preference model
```

Without it, Slice 5 personalization would have to assume access to listening history from
another platform, or rely on synthetic behaviour — which can test software but cannot
support any claim about effectiveness.

**Tradeoffs accepted:**
Some engineering effort goes to a component that is not the intellectual centre. Bounded
by an explicit non-goal list: no library-management UI, streaming, social features,
crossfade, equalizer, elaborate playlists, synchronized lyrics, or general player polish.
The player must not displace retrieval, ranking and recommendation as the project's
technical story.

**Evidence:**
- Direct evidence: none. The research ledger does not establish this.
- Transfer evidence: intent-based personalization from repeated relevance feedback
  (INIT_RESEARCH §7) assumes a channel through which feedback arrives.
- Engineering inference: **primary basis.** This is a product/system-design judgement
  about closing the feedback loop, not a finding from the research ledger.

**Implication:**
DESIGN.md gains a playback boundary, playback/queue state, a behavioural event model
(SearchRequest, RecommendationImpression, PlaybackEvent, ExplicitFeedback), an append-only
behaviour store, and impression→play attribution. Event types are not equivalent
preference evidence; the strong/medium/weak weighting is a hypothesis to evaluate.

Personalization authority is unchanged: explicit query > current session > intent profile
> global preference. Behavioural history never overrides an explicit query constraint.

**Removal/revisit condition:**
Revisit if the player starts consuming time that belongs to retrieval/ranking work, or if
impression→play attribution proves unreliable enough that behavioural data cannot support
personalization evaluation.

**Files updated:**
- PROJECT.md: product framing, third capability, Slice 1B in roadmap, player non-goals
- DESIGN.md: playback layer section, updated target architecture, open question 10
- EVALS.md: Slice 1B evaluation, Slice 5 real-behaviour data source, new stop condition
- STATUS.md: Slice 1B recorded as upcoming; current slice unchanged

---

# Decision entry template

## DEC-XXX — Short title

**Status:** Proposed / Accepted / Rejected / Revisit

**Current slice:**

**Question:**

**Current default:**

**Alternative(s):**

**Why this matters:**

### Evidence
- Direct evidence:
- Transfer evidence:
- Engineering inference:

**Experiment:**

**Result:**

**Decision:**

**Tradeoffs accepted:**

**Licensing/data implications:**

**Removal/revisit condition:**

**Files updated:**
- PROJECT.md:
- DESIGN.md:
- EVALS.md:
- STATUS.md:

**Understanding check:**  
Can Irene explain what changed and why?
