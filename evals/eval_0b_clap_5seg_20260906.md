# Eval 0B — text→music retrieval (clap_5seg)

- Encoder: `clap` / `laion/larger_clap_music_and_speech` on mps
- Segments per track: 5
- Captions: 1106 · Candidate tracks: 706
- Failed to embed: 0

## Retrieval

| Metric | Measured | Random | Lift |
|---|---:|---:|---:|
| Recall@1 | 0.063 | 0.0014 | 45× |
| Recall@5 | 0.193 | 0.0071 | 27× |
| Recall@10 | 0.307 | 0.0142 | 22× |
| Median rank | 27.0 | 353.5 | — |
| MRR | 0.1415 | — | — |

## Diagnostics

- track–track cosine: mean 0.343, sd 0.196, p95 0.683
- hubness: top 1% of tracks take 4.9% of all top-10 slots; single largest 0.8%
- tracks never in any top-10: 36
- within-track segment cosine: 0.8213765996036043

## Cost

- indexing: 0.26 s/track (182 s total)
- query: 3.0 ms/query
- storage: 2048 bytes/track

Raw: `eval_0b_clap_5seg_20260906.json`
