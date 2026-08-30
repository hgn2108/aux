# DECISIONS.md

> **Audience:** Shared, but Claude is expected to maintain it.
> **Purpose:** Append-only history of material architecture/product decisions.
> **Update frequency:** Only when a meaningful decision is accepted/rejected.
> **Rule:** Do not duplicate transient status here.

# VibeSearch Decision Log

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
