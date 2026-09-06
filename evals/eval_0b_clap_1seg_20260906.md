# Eval 0B — text→music retrieval (clap_1seg)

- Encoder: `clap` / `laion/larger_clap_music_and_speech` on mps
- Segments per track: 1
- Captions: 1106 · Candidate tracks: 706
- Failed to embed: 0

## Retrieval

| Metric | Measured | Random | Lift |
|---|---:|---:|---:|
| Recall@1 | 0.049 | 0.0014 | 34× |
| Recall@5 | 0.165 | 0.0071 | 23× |
| Recall@10 | 0.252 | 0.0142 | 18× |
| Median rank | 33.0 | 353.5 | — |
| MRR | 0.1193 | — | — |

## Diagnostics

- track–track cosine: mean 0.296, sd 0.190, p95 0.630
- hubness: top 1% of tracks take 5.1% of all top-10 slots; single largest 0.8%
- tracks never in any top-10: 40
- within-track segment cosine: n/a (single segment)

## Cost

- indexing: 0.18 s/track (126 s total)
- query: 3.0 ms/query
- storage: 2048 bytes/track

Raw: `eval_0b_clap_1seg_20260906.json`
