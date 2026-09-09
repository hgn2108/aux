# Validating DEC-020, and testing how far it transfers

DEC-020 proposes gating the planner on **library-relative confidence**: rewrite the query
only when nothing already stands out. It rests on 10 rated queries from one library, and one
threshold (z-top 3.0) fitted to those same 10. Three separate things need testing, and they
are usually confused with each other:

1. **Does the rule work on queries it was not derived from?** (held-out queries, same library)
2. **Does the *measure* mean the same thing in a different library?** (transfer)
3. **Does the "library cannot answer" detector work at all?** (currently zero evidence)

## Why no new music is needed

Transfer is a claim about behaving consistently across **differently-composed collections**,
not about collection size. Four are already available without downloading anything:

| corpus | tracks | character |
|---|---:|---|
| personal — hip-hop only | 94 | one genre, dense, highly self-similar |
| personal — everything but hip-hop | 66 | five genres, no dominant one |
| personal — acoustic-leaning (jazz + classical + v-pop) | 40 | mostly acoustic |
| personal — electronic-leaning (edm + dnb) | 26 | mostly programmed |
| FMA small | 8,000 | diverse public catalogue, 50x the personal library |
| Song Describer | 706 | **has ground truth** — each caption names its track |

These differ on exactly the axis the rule is sensitive to: how much a given query stands out
against the collection. A hip-hop query is unremarkable in the hip-hop-only sub-library and
highly distinctive in the acoustic one — which is precisely the situation a library-relative
rule claims to handle.

**What new music would add:** a genuinely independent personal collection, which is the only
way to rule out that something about *these particular tracks* drives the result. That is
worth doing eventually and is not worth blocking on now — the sub-libraries share tracks but
differ in composition, and composition is the variable under test.

## Phase A — does the measure transfer at all? (no rating time)

Compute confidence for the same query set across all six collections above.

**The concern this addresses.** z-top is `(max - mean) / sd`. Its typical range depends on
collection size and shape — the maximum of 8,000 samples sits further into the tail than the
maximum of 26. So a threshold of 3.0 may mean different things in different libraries even
though the statistic is nominally normalised.

Alternatives to test alongside it, each scale-free by construction:

- **relative gap** — `(top1 - top10) / (top1 - mean)`: how much the winner beats the chasing
  pack, as a fraction of its own margin;
- **crowding** — fraction of the library scoring within 90% of the top;
- **top-K entropy** — how evenly the top scores are spread.

**Passes if** one measure's distribution is stable enough across collections that a single
threshold sits in the same place. **Fails if** none is, in which case the gate must be
calibrated per library — still workable, just a different design, and better known now.

## Phase B — the "cannot answer" detector, using synthetic gaps (no rating time)

The most valuable untested idea, and it can be tested with ground truth by **creating gaps
on purpose**: remove a genre from the index, then query for it.

Removing all 20 jazz tracks and querying "saxophone over an upright bass" produces a library
that provably cannot answer. Confidence should collapse. Repeat across genres and queries,
and it becomes a detection problem with known labels: precision and recall of "this library
cannot answer", with no human judgement anywhere.

**Passes if** the detector separates removed-genre queries from present-genre ones. **Fails
if** it cannot, in which case step 4 of the design is dropped rather than shipped on faith.

## Phase C — direction check on ground truth (no rating time)

Song Describer's captions are *already* written in sound-describing language — the encoder's
native distribution. So the rule predicts: **high confidence, do not rewrite**, and forcing
a rewrite should not help and may hurt.

This is measurable against known correct tracks (Recall@K), with no proxy and no rating.

**Why it matters:** every other test of the gate uses either a proxy metric or human
ratings. This one has a right answer. If forcing rewrites on already-sound-language queries
*improves* Recall@K, the whole premise — that rewriting closes a language gap — is wrong.

## Phase D — held-out queries (~8 minutes of rating)

Ten to twelve new queries, none seen before, with **the threshold fixed from Phase A** and
written down before the run. The gate applies automatically; a blind pooled A/B round scores
it.

**Passes if** the gate's calls track the ratings — helping where it fires, and where it
declines to fire the plain query holding up.

## Order, and why

A, B and C cost no rating time and can all fail in ways that change what D should even test.
Running D first would risk spending the only expensive resource on a threshold that Phase A
shows to be meaningless.

If Phase C contradicts the premise, stop: the design is wrong at the root and no amount of
threshold tuning fixes it.

## What none of this can establish

That the rule works for *other people's* libraries. Every collection here is either public
catalogue audio or a subset of one person's taste. That claim needs other users, which is
Slice 1B and beyond — and is worth stating plainly rather than being quietly implied by
"library-independent".
