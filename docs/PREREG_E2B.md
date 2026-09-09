# Pre-registration — E2b, planner on weak queries

Written and committed **before** the experiment runs. DEC-018 recorded that E2's first human
test was run on queries the baseline already answered well, leaving no headroom. This is the
corrected test, and the selection rule, metrics and predictions are fixed in advance so the
analysis cannot be shaped by the result.

## Query selection — mechanical, not chosen

All Slice 1 queries with mean relevance **< 3.5** (`evals/eval_1_relevance_20260909.json`).
That threshold sits below the Slice 1 overall mean of 3.78 and yields 11 queries. No query
was added or removed by hand.

| Slice 1 mean | category | query | why it is weak | planner expected to help? |
|---:|---|---|---|---|
| 1.20 | acoustic | solo piano, no vocals | negation — already fixed separately (DEC-014) | n/a, negation carries it |
| 1.20 | context | warming up before going out | context, no genre anchor | **yes** |
| 1.60 | acoustic | distorted electric guitar | library gap — no rock in the library | **no** |
| 1.60 | context | background music while reading | context, no genre anchor | **yes** |
| 2.40 | compound | jersey club and dancey vibes | probable library gap | no |
| 2.60 | acoustic | orchestral strings | library gap — only 7 classical tracks | **no** |
| 2.60 | compound | girly pop songs to get ready to | probable library gap + context | uncertain |
| 3.00 | acoustic | saxophone over an upright bass | thin jazz coverage | no |
| 3.00 | context | something for a rainy morning | context, no genre anchor | **yes** |
| 3.20 | mood | playful and light-hearted | abstract mood language | **yes** |
| 3.40 | context | something to fall asleep to | context, no genre anchor | **yes** |

## Predictions, recorded in advance

1. **The planner improves the five context/mood queries marked "yes"** — these are weak
   because the query language is far from the encoder's training language, which is the one
   thing the rewrite addresses.
2. **The planner does not improve the library-gap queries** — no rewrite conjures rock
   music into a library without any. If it appears to help here, suspect the metric.
3. **Overall improvement will be smaller than the per-query gains suggest**, because roughly
   half the set is library-limited.
4. **"solo piano, no vocals" improves regardless of the planner**, because negation was
   fixed after Slice 1 was rated. It is a check on the negation fix, not on the planner.

A result contradicting prediction 2 is more informative than one confirming prediction 1:
the first would mean the metric is measuring something other than what it claims.

## Metrics — objective first, ratings second

DEC-018's lesson was that ratings alone on 14 queries could not resolve anything. Three
objective metrics run first, none needing human time:

- **Intent match.** Does the retrieved audio actually have the acoustic properties the
  rewrite named? Terms like "quiet", "fast", "sparse" map to waveform features the encoder
  never sees. This is the closest objective proxy to "did the system deliver what was
  asked", and it needs no opposed pair.
- **Top-result confidence (z-top).** How far the best match stands above the corpus for that
  query. A vague query made specific should stand out more.
- **Separation** on opposed pairs, as in DEC-017.

Then a blind pooled A/B rating round on the same 11 queries.

**Both must be reported**, agreeing or not. Where they disagree, that disagreement is the
finding — it has already happened four times in this project, and each time the resolution
came from a third measurement rather than from trusting one.

## Stopping rule

The planner ships **selectively** — applied only where the baseline is weak — if it improves
both the objective intent-match and the human ratings on the "yes" queries. If only one
improves, that is recorded as unresolved rather than argued in either direction. If neither,
Slice 2 closes negative and the planner stays documented research.


---

## Amendment 1 (2026-09-09, after a first run returned no scores)

**What happened.** The intent-match metric scored the retrieved audio against terms found in
the *original query*. Every one of the 11 weak queries returned `n/a`, because none of them
names a measurable property — and that is exactly why they are weak. "Something to fall
asleep to" contains no word that maps to a waveform feature.

**What was rejected.** Scoring the planner against terms in *its own rewrite*. It would
grade the planner on its own homework: a rewrite could name "slow, quiet", retrieve slow
quiet music, and score perfectly while ignoring what the user meant.

**The fix.** A third source — the acoustic direction each query *implies*, written below from
the plain meaning of the query and applied identically to both systems. These are a
measurement standard, not a retrieval mechanism. DEC-016 rejected a hand-written
context→acoustic table as a way to *retrieve*; using one to define what a query means for
*scoring* is the same act as writing a relevance label by hand, which is how every human
evaluation works.

**Written before any intent score existed** — the first run produced `n/a` for every query,
so no result could have informed these.

| query | implied direction |
|---|---|
| solo piano, no vocals | quiet, sparse |
| warming up before going out | loud, fast, bright |
| distorted electric guitar | loud, bright |
| background music while reading | quiet, sparse |
| jersey club and dancey vibes | fast, loud |
| orchestral strings | *none — instrumentation, not a physical direction* |
| girly pop songs to get ready to | fast, bright, loud |
| saxophone over an upright bass | *none — instrumentation* |
| something for a rainy morning | quiet, slow |
| playful and light-hearted | bright, fast |
| something to fall asleep to | quiet, slow, sparse |

Two queries name instrumentation with no honest physical direction and are scored `n/a`
rather than assigned an invented mapping. Predictions from the original pre-registration are
unchanged.
