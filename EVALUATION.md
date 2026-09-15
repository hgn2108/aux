# Evaluation

The full method behind the numbers in [README.md](README.md). Everything here is reproducible
from the scripts named in each section; the raw output is committed under `results/`.

## Contents

1. [How relevance is defined](#1-how-relevance-is-defined)
2. [Audio recommendation](#2-audio-recommendation)
3. [The semantic benchmark](#3-the-semantic-benchmark)
4. [Validating the labels](#4-validating-the-labels)
5. [Why fusion appeared to fail](#5-why-fusion-appeared-to-fail)
6. [The crossover](#6-the-crossover)
7. [Routing](#7-routing)
8. [Failure analysis](#8-failure-analysis)
9. [What was measured and rejected](#9-what-was-measured-and-rejected)

---

## 1. How relevance is defined

Musical similarity is subjective and this project never fabricates a similarity label. Instead
relevance comes from metadata that is objective and complete for the corpus:

| label | meaning | weakness |
|---|---|---|
| **genre** | same top-level genre | broad, and partly a genre-classification test |
| **artist** | same artist | strict, not explainable by genre alone |
| **album** | same album | strictest; also rewards shared production |

All three are reported together because they are wrong in different directions. Agreement
across them is far more informative than a good score on any one.

**Same-artist exclusion.** Tracks from one album share production, mastering and
instrumentation, so a model can rank them together without having learned anything about
genre. The genre label is therefore reported twice: plainly, and with same-artist pairs
removed from both the ranking and the labels. This is standard practice in music IR and turns
a flattering number into an honest one.

**Baselines are computed, not derived.** The random baseline runs through the same
query-skipping and exclusion rules as the systems it is compared against, so the comparison
holds even where those rules bite.

**Queries with no relevant item are skipped.** They cannot distinguish a good system from a
bad one and would dilute every metric by a constant.

---

## 2. Audio recommendation

`python scripts/eval_recommendation.py --corpus fma --per-genre 250`

FMA small, 1,998 tracks that decoded cleanly out of 2,000, balanced at 250 per genre across 8
genres. Balance matters: in an unbalanced corpus, Precision@K is partly a measure of how
common a genre is.

| label | queries | P@5 | NDCG@5 | P@10 | NDCG@10 | P@20 | NDCG@20 | HR@10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| genre | 1,998 | 0.618 | 0.625 | 0.596 | 0.608 | 0.572 | 0.587 | 0.917 |
| genre (artist-filtered) | 1,998 | 0.573 | 0.576 | 0.560 | 0.566 | 0.545 | 0.553 | 0.902 |
| artist | 1,324 | 0.187 | 0.300 | 0.134 | 0.311 | 0.092 | 0.331 | 0.574 |
| album | 1,193 | 0.115 | 0.259 | 0.074 | 0.287 | 0.047 | 0.313 | 0.496 |
| *random (genre)* | *1,998* | *0.125* | *0.124* | *0.126* | *0.125* | *0.126* | *0.125* | *0.736* |
| *random (artist)* | *1,324* | *0.005* | *0.005* | *0.004* | *0.006* | *0.004* | *0.008* | *0.034* |
| *random (album)* | *1,193* | *0.002* | *0.002* | *0.001* | *0.003* | *0.001* | *0.005* | *0.013* |

Four metrics rather than one, because a recommender can look good on one and fail another.
Recall@K is reported but not used to compare systems on the genre label, where a query has
~249 relevant items and K is 10.

**Why the lyric arm is absent.** A coverage floor gates it: below 50% of tracks having a
reliable transcript, a "lyrics" row measures which tracks happened to be transcribed rather
than the modality. FMA sits at 2%.

---

## 3. The semantic benchmark

`python scripts/eval_semantic.py`

Genre, artist and album are acoustic constructs, so scoring lyrics against them asks whether
lyrics help identify something you can already hear. This asks the opposite question: given
*"songs about heartbreak"*, which signal finds them?

Twelve themes were chosen to be semantically distinct and **acoustically non-obvious**, so a
system that only hears production cannot guess them. Themes that map onto a genre were
deliberately excluded: they would let the audio channel win by recognising the genre rather
than understanding the query. Nine cleared the five-track floor.

| system | P@5 | NDCG@5 | P@10 | NDCG@10 | P@20 | NDCG@20 |
|---|---:|---:|---:|---:|---:|---:|
| sound | 0.444 | 0.472 | 0.311 | 0.367 | 0.267 | 0.326 |
| **lyrics** | **0.800** | **0.832** | **0.678** | **0.734** | **0.572** | **0.653** |
| fused α=0.25 | 0.778 | 0.813 | 0.667 | 0.724 | 0.550 | 0.637 |
| fused α=0.50 | 0.689 | 0.734 | 0.622 | 0.671 | 0.517 | 0.596 |
| fused α=0.75 | 0.600 | 0.600 | 0.500 | 0.529 | 0.422 | 0.477 |
| *random* | *0.156* | *0.161* | *0.178* | *0.175* | *0.194* | *0.191* |

**Per theme (NDCG@10):**

| theme | tracks | sound | lyrics | winner |
|---|---:|---:|---:|---|
| desire | 72 | 0.508 | 1.000 | lyrics |
| nightlife | 49 | 0.596 | 1.000 | lyrics |
| longing | 27 | 0.495 | 0.855 | lyrics |
| money | 48 | 0.176 | 0.805 | lyrics |
| substances | 32 | 0.220 | 0.797 | lyrics |
| doubt | 21 | 0.066 | 0.794 | lyrics |
| loyalty | 13 | 0.359 | 0.400 | lyrics |
| ambition | 18 | 0.161 | 0.284 | lyrics |
| **heartbreak** | 24 | **0.718** | 0.676 | **sound** |

`money` (0.176 → 0.805) and `doubt` (0.066 → 0.794) are where sound is nearly blind: neither
subject has an acoustic signature. `heartbreak` is the exception because sad songs sound sad.

An oracle allowed to pick the better modality per query scores 0.739 against the lyric
channel's 0.734. Within this family routing has almost nothing to add, because lyrics win
89% of the queries outright.

---

## 4. Validating the labels

`python scripts/rate_themes.py`

The theme labels are model-generated, so they are the measuring instrument and cannot be
assumed. A blind sample of 40 (track, theme) pairs was rated by hand: half of them the model had
called true and half false, with its answer hidden.

| | |
|---|---|
| raw agreement | 0.80 |
| **Cohen's κ** | **0.60** (substantial) |
| model said yes | 20 |
| human said yes | 22 |

κ rather than raw agreement, because most pairs are false and a labeller that always said
"no" would score 0.95 raw with κ ≈ 0. Errors are symmetric, 3 over-applied against 5 missed,
so the labeller is not simply marking everything true. That was the failure mode that would
have invalidated the benchmark.

Disagreement is concentrated rather than spread: six themes agree perfectly, `heartbreak`
reaches 0.86, and two fail: `substances` (0.20) and `doubt` (0.33). Both had scored strong
lyric wins, so the benchmark was rerun without them:

| | 9 themes | 7 themes (low-agreement dropped) |
|---|---:|---:|
| sound | 0.367 | 0.430 |
| **lyrics** | **0.734** | **0.717** |
| themes lyrics wins | 8/9 | 6/7 |

**The conclusion does not depend on the labels that failed the check.**

One concern the check retired: `desire` is applied to 57% of the library, which looked like an
over-broad theme inflating the lyric result. It agreed perfectly on every sampled pair. The
corpus really is that concentrated.

---

## 5. Why fusion appeared to fail

`python scripts/diagnose_fusion.py`

Fusing sound and lyrics scored below sound alone at every weight, even though the lyric
channel beat random comfortably. Three explanations, with different fixes:

1. **Combination**: the blend is miscalibrated.
2. **Routing**: lyrics win on some queries and a global weight averages winner with loser.
3. **Redundancy**: there is no complementary signal to recover.

The decisive test is an **oracle**: a cheating router allowed to see the labels and pick the
better modality per query. It bounds every router that could ever be written.

| | NDCG@10 |
|---|---:|
| sound only | 0.237 |
| lyrics only | 0.057 |
| oracle modality per query | 0.254 (+0.017) |
| oracle weight per query | 0.277 (+0.040) |

Lyrics beat sound on 11% of queries. Rank correlation between the two systems is **+0.273** -
partly independent, but not in a way that predicts *artist*.

**Calibration was a real defect.** Min-max normalisation takes its range from the two most
extreme candidates, so one outlier rescales everything else:

| α | min-max | z-score |
|---:|---:|---:|
| 0.25 | 0.088 | **0.120** |
| 0.50 | 0.106 | **0.143** |
| 0.75 | 0.152 | **0.183** |
| RRF | 0.162 | n/a |

z-score is now the default; min-max stays selectable because that comparison is the evidence
for the change. Restricting to the 127 tracks that all have transcripts, removing the
missing-modality fallback, narrows the gap to within noise.

None of it overturns the headline, because the labels being scored against are acoustic.

---

## 6. The crossover

`python scripts/eval_routing.py`

| α (weight on sound) | track → track (genre) | semantic ("songs about X") |
|---:|---:|---:|
| 0.00 | 0.555 | **0.734** |
| 0.25 | 0.759 | 0.724 |
| 0.50 | 0.802 | 0.671 |
| 0.75 | 0.829 | 0.529 |
| 1.00 | **0.832** | 0.367 |

| | |
|---|---|
| best single weight | α=0.25, mean 0.742 |
| choosing per family | 0.783 (**+0.042**) |
| cost of the wrong weight | 0.277 on similarity, 0.368 on semantic |

The two families use different labels and different query sets, so their mean compares
*policies*, not systems. The per-family columns are the evidence.

---

## 7. Routing

`python scripts/eval_router.py`

96 queries across three families, acoustic, semantic, and compound (*"Vietnamese pop songs
about missing someone"*, relevant only where both genre and theme hold). Scored **end to end**
against real retrieval rather than as classification accuracy, because a router can classify
well and still choose unhelpful weights.

| router | NDCG@10 | vs best fixed | % of oracle gap | ms/query |
|---|---:|---:|---:|---:|
| keyword rules | 0.500 | −0.053 | −45% | 0.0 |
| embedding prototypes | 0.570 | +0.017 | 15% | 2.3 |
| LLM (Haiku) | 0.571 | +0.019 | 16% | 1686 |
| *best fixed (α=0.25)* | *0.552* | n/a |, | n/a |
| *oracle* | *0.670* | n/a | *100%* | n/a |

**Significance** (paired permutation, Bonferroni threshold 0.0125):

| comparison | difference | p | verdict |
|---|---:|---:|---|
| rules vs best fixed | −0.053 | 0.028 | not significant |
| prototype vs best fixed | +0.017 | 0.173 | not significant |
| LLM vs best fixed | +0.019 | 0.379 | not significant |
| **family-level routing vs global** | **+0.051** | **0.0001** | **significant** |

Per family, the best weights are α=0.95 (acoustic), α=0.40 (compound), α=0.00 (semantic).

**The overfitting result.** On the original 24 hand-written queries the keyword router led at
+0.058. The set was regenerated with paraphrases instructed to avoid the constructions the
rules key on, generated from the label specification alone, never from any router's
behaviour, and deliberately biased *against* the router expected to win. It fell to −0.053:
best arm to worst, on held-out phrasing alone. It had been fitted to queries written by the
same person who wrote the rules.

A second bug surfaced here: the LLM arm was silently returning its fallback weight on all 24
queries, because the API rejects range keywords on a number type and a bare `except` turned
twenty-four failures into a confident-looking constant. Failures are now counted and the
evaluation refuses to report an arm that fell back.

---

## 8. Failure analysis

Per-genre NDCG@10 against the artist label, FMA:

| genre | queries | P@10 | NDCG@10 |
|---|---:|---:|---:|
| International | 180 | 0.193 | 0.535 |
| Rock | 120 | 0.086 | 0.403 |
| Folk | 172 | 0.133 | 0.362 |
| Instrumental | 225 | 0.256 | 0.347 |
| Electronic | 146 | 0.097 | 0.233 |
| Experimental | 123 | 0.083 | 0.215 |
| Hip-Hop | 194 | 0.101 | 0.203 |
| **Pop** | 164 | 0.051 | **0.165** |

The model does best where the acoustic signature is most distinctive and worst on Pop, the
broadest and most heterogeneous of the eight categories.

---

## 9. What was measured and rejected

Every added component had to beat the simpler baseline on a stated metric. These did not:

| component | verdict |
|---|---|
| **LLM query planner** | Measured against the plain baseline; off by default. The API exposes no temperature control, so it re-sampled its own inputs between runs, reproducibility needed plan caching. |
| **Context→acoustic lexicon** | 30-term mapping from context words to acoustic properties. Weight 0. |
| **Query expansion** | Rejected; weight 0. |
| **Reciprocal rank fusion** | Weights its inputs equally by construction, which makes a weight ablation impossible. Replaced by weighted-score fusion. |
| **Min-max normalisation** | Beaten by z-score at every interior weight. |
| **All three routers** | None significant against a fixed weight. |
| **LAION-CLAP** | Lost a paired test to MuQ-MuLan (p = 4.5e-08). A guard written during that comparison caught the published `larger_clap_music` checkpoint shipping untrained projection heads. |

Four times an objective proxy pointed the wrong way and was caught by a third, independent
measurement. Proxies proved reliable for cheap disqualification and unreliable for
confirmation, which is why no result here rests on a single metric.
