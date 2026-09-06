# Eval 0B — text→music retrieval (clap_1seg_valid)

- Encoder: `clap` / `laion/larger_clap_music_and_speech` on mps
- Segments per track: 1
- Captions: 746 · Candidate tracks: 547 · human-validated subset only
- Failed to embed: 0

## Retrieval

| Metric | Measured | Random | Lift |
|---|---:|---:|---:|
| Recall@1 | 0.070 | 0.0018 | 38× |
| Recall@5 | 0.208 | 0.0091 | 23× |
| Recall@10 | 0.310 | 0.0183 | 17× |
| Median rank | 23.0 | 274.0 | — |
| MRR | 0.1497 | — | — |

## Diagnostics

- track–track cosine: mean 0.306, sd 0.188, p95 0.635
- hubness: top 1% of tracks take 4.2% of all top-10 slots; single largest 0.9%
- tracks never in any top-10: 36
- within-track segment cosine: n/a (single segment)

## Cost

- indexing: 0.18 s/track (98 s total)
- query: 2.6 ms/query
- storage: 2048 bytes/track

Raw: `eval_0b_clap_1seg_valid_20260906.json`
