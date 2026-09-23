# aux

[![tests](https://github.com/hgn2108/aux/actions/workflows/tests.yml/badge.svg)](https://github.com/hgn2108/aux/actions/workflows/tests.yml)
[![python](https://img.shields.io/badge/python-3.11%2B-blue)](pyproject.toml)
[![licence](https://img.shields.io/badge/licence-MIT-green)](LICENSE)

**A local-first multimodal music search and content-based recommendation system for music
you already own.**

Streaming services recommend well because they watch millions of listeners. If your music
sits in a folder on your laptop, none of that applies: local players search filenames and
genre tags, and nothing lets you ask for *"something quiet and bittersweet"* or *"like this,
but dreamier"*.

`aux` answers those from the audio and the lyrics themselves. It models the music, not the
listener: there is no profile accumulating and no feedback loop narrowing, so a track nobody
has ever played competes on equal terms with a popular one, and the same query gives the same
answer a year from now. The taste stays with you. The system only has to be good at finding
things.

**Live demo:** https://aux-339224224982.us-central1.run.app
The first search takes about a minute while the model loads.

![Searching by description](docs/img/search.png)

---

## How it works

```mermaid
flowchart LR
    A[your audio files] --> B[decode<br/>5 windows per track]
    B --> C[MuQ-MuLan<br/>music + text encoder]
    C --> E[(sound vectors)]

    B --> F[Whisper<br/>transcribe]
    F --> G[Qwen3<br/>text encoder]
    G --> H[(lyric vectors)]

    I([your query]) --> C
    I --> G

    E --> J{blend<br/>weight α}
    H --> J
    J --> K[ranked tracks]

    style C fill:#e8f0fe
    style G fill:#e8f0fe
    style J fill:#fff3cd
```

Two independent signals, searched the same way:

- **Sound.** [MuQ-MuLan](https://huggingface.co/OpenMuQ/MuQ-MuLan-large) puts music and text
  in one embedding space, so a written description can be matched against a recording
  directly. Chosen over the main alternative, CLAP, after both were measured on this corpus
  and the difference was tested for significance.
- **Lyrics.** Whisper transcribes the track, Qwen3 embeds the words.

A slider blends the two. Which setting is right turns out to depend on the question, which is
the main finding below.

## Run it

```bash
git clone https://github.com/hgn2108/aux && cd aux
pip install -e ".[app]"
streamlit run app.py
```

Point it at your own folder by dropping files into `data/music/`, or use the **Your music**
tab to upload them straight into the browser.

---

## Results

Whether two tracks "sound similar" is a matter of opinion, so the system is not graded on
that. It is graded on facts: two tracks count as a match if they share a genre, an artist or
an album. None of those *is* musical similarity, so all three are reported, because each is
wrong in a different way.

Every score below is **NDCG@10**: of the ten tracks returned, how many were matches, and
whether the matches were near the top. 1.0 is perfect. Each is shown next to the score a random
shuffle gets on the same labels, so the number has a floor to be read against.

### Recommending from sound alone

Given a track, about 6 of the 10 returned share its genre, against about 1 of 10 by chance.
Artist is a far harder target, and it does 56× better than chance there.

| relevance label | queries | NDCG@10 | random | lift |
|---|---:|---:|---:|---:|
| genre | 1,998 | 0.608 | 0.125 | **4.9×** |
| genre, same-artist pairs excluded | 1,998 | 0.566 | 0.123 | **4.6×** |
| artist | 1,324 | 0.311 | 0.006 | **56×** |
| album | 1,193 | 0.287 | 0.003 | **88×** |

Row two is a control: tracks from one album share production and mastering, so a system can
score on *genre* by recognising an *album* instead. Removing same-artist pairs costs 0.04, so
the genre result is not that effect in disguise.

### The right blend depends on the question

The same two systems and the same music give opposite answers, decided only by what was
asked.

| α (weight on sound) | *"tracks like this one"* | *"songs about heartbreak"* |
|---:|---:|---:|
| 0.00, lyrics only | 0.555 | **0.734** |
| 0.50 | 0.802 | 0.671 |
| 1.00, sound only | **0.832** | 0.367 |

Genre and artist are audible, so lyrics add nothing when matching on those. On a private
160-track benchmark of "songs about X" queries, the lyrics score twice what sound does and
win 8 of the 9 subjects tested. The one exception is *heartbreak*, where sound wins: sad
songs sound sad. That benchmark is small and its labels are model-generated with a validated
sample, so it shows the effect on this collection rather than a general fact about lyrics.

Set the weight wrong for the question being asked and the system loses **a third to a half**
of its accuracy. No single setting is right for both.

### Setting that weight automatically did not work

If the right weight depends on the question, the obvious next step is to detect the question
and set the weight for the user. Three ways of doing that were tried: matching keywords,
comparing the query to example queries, and asking an LLM.

| approach | NDCG@10 | speed |
|---|---:|---:|
| fixed weight, no detection | 0.552 | instant |
| keyword matching | 0.500 | instant |
| compare to example queries | 0.580 | 2 ms |
| ask an LLM | 0.567 | 1.8 s |
| *perfect detection (upper bound)* | *0.688* | n/a |

The last row is the score if every question were classified correctly, so it bounds what any
such method could be worth. None of the three came close, and none beat a fixed weight by
enough to be significant over 96 queries.

**So the app exposes the weight as a control rather than guessing it.** Someone searching
already knows whether they are asking about sound or about meaning.

Knowing the question *would* help, though. Choosing one weight per query family, with the
weights fitted on training folds and scored on held-out queries, gains **+0.044 NDCG@10**
(95% CI +0.015 to +0.072, paired permutation p = 0.004, n = 96). Selecting and scoring on
the same queries, as an earlier version of this analysis did, reported +0.061; the gap
between those two numbers is what that shortcut was worth.

Keyword matching looked like the winner until the test set changed. Its rules and the test
queries had been written by the same person, so it was rewarded for recognising familiar
phrasing. Retested on differently worded queries meaning the same things, it went from the
best of the three to the worst.

![The findings tab](docs/img/findings.png)

Full tables, ablations and significance tests: **[EVALUATION.md](EVALUATION.md)**.

---

## What's next: a taste layer, opt in

Not built, and second on purpose. Content ranking answers *"find me this"*; it cannot answer
*"and I would like it"*. A listener model closes that gap, and doing it second means it can be
something a user switches on rather than something that happens to them.

The groundwork is the useful half. Every track already has a vector, so a listener is a point
in that same space that their plays pull around. No second encoder, no retraining, and a track
nobody has ever played still ranks, which is the failure collaborative filtering is known for.

**That layer needs listening events, not more audio.** One row per play: which track, when,
how long it ran, what ended it. The last two are the labels. A track played to the end and one
abandoned at nine seconds are a positive and a negative, and both fall out of ordinary
listening without anyone rating anything.

| where the rows come from | what they give | what it costs |
|---|---|---|
| plays inside this app | the query, the results, and the play or skip that followed: the only source that ties a preference to a *question* | the player has to exist first |
| a Spotify account export | years of history, with `ms_played` and `reason_end` on every play | a privacy data request, up to 30 days to arrive |
| Last.fm or ListenBrainz | scrobbles through a public API, continuously | start times and track identity only, no skips |

Syncing live from Spotify's API is not the route. Recommendations, related artists and audio
features were closed to apps registered after November 2024, and what is left
(`recently-played`, `top/tracks`) is a short window rather than a history. The export carries
more anyway, and it is a file, which suits a system that already runs offline.

Either way the events arrive as track names and have to be matched to files on disk before
they mean anything, and preference is only learnable for music actually present. That matching
is the first piece of work, not the model.

Then three stages, in the order their dependencies force:

1. **Learn from plays.** Results play in the app, so each play, skip or replay is recorded
   against the query and position that produced it.
2. **Keep preference specific to the request.** The finding above is that what counts as a good
   match changes with the question. Taste likely behaves the same way, so what is learned from
   *"music for running"* should not move results for *"music for studying"*.
3. **Balance familiarity against discovery.** Ranking on similarity alone returns more of what
   a listener already knows. Measuring that trade-off needs preference data to measure against,
   which is why it comes last.

---

## Scope and limitations

What this is: content-based retrieval and ranking. It models tracks, not listeners.

It does **not** do personalised recommendation from listening behaviour, collaborative
filtering, user-item interaction modelling, sequence models over listening history, or
learning from user feedback. None of those is implemented or claimed, and the
[What's next](#whats-next-a-taste-layer-opt-in) section is a plan rather than a result.

- **Relevance labels are proxies.** Genre, artist and album are not musical similarity. They
  are used because they are objective and complete, and reported together because they fail
  differently.
- **The lyric results use a separate 160-track collection.** No public corpus available here has
  a usable lyric channel: 56% of a sampled 75 Free Music Archive clips are instrumental, with
  transcripts running a median of 11 words.
- **The "songs about X" labels were generated by a model**, then checked against blind human
  judgement on a sample. Agreement was substantial but not perfect. Dropping the two themes
  where agreement was weakest narrows the lyric lead from 0.734 to 0.717 against 0.430, so
  the conclusion holds without them.
- **The router comparison uses 96 queries.** It rules out large effects, not small ones.
- **The blend weight is set by hand, and search compares against every track.** At this
  corpus size comparing everything is instant, so an approximate index would add complexity
  for no gain. A learned blend was not attempted because it would have to be trained on the
  same imperfect labels described above.

## Layout

The pipeline, in the order audio moves through it:

| path | what |
|---|---|
| `src/aux/ingest/` | find files, probe them, decode to audio, hash by content |
| `src/aux/encode/` | encoder adapters, windowing, pooling |
| `src/aux/lyrics/` | transcription, lyric embedding, transcript index |
| `src/aux/index/` | the embedding cache, keyed by content hash and encoder version |
| `src/aux/recommend.py` | scoring, the blend, per-result explanations |
| `src/aux/data/` | the two corpora and their relevance labels |
| `src/aux/app/` | what the demo loads, and its precomputed example queries |
| `app.py` | the demo |

Measurement:

| path | what |
|---|---|
| `src/aux/eval/` | ranking metrics, paired significance tests |
| `scripts/` | one file per evaluation, plus corpus building and deployment |
| `results/` | the JSON behind every number above, read by the app and the write-up |
| `queries/` | the query sets the router comparison runs on |

Three packages hold work that was measured and **not** shipped. They stay because
[EVALUATION.md](EVALUATION.md) reports those results, and a rejection is worth no more than
the code that backs it:

| path | what, and why it is not in the product |
|---|---|
| `src/aux/route/` | three ways to pick the blend weight automatically. None beat a fixed weight. |
| `src/aux/plan/` | an LLM query planner. Measured against the plain baseline; off by default. |
| `src/aux/query/`, `src/aux/rank/` | negation handling, a context-to-acoustic lexicon, rank fusion. The lexicon and expansion scored zero weight; rank fusion was replaced by the weighted blend. |

## Licence

MIT, for the code. The model weights downloaded at runtime carry their own licences
(MuQ-MuLan is CC-BY-NC). No music is distributed here. See [LICENSE](LICENSE).
