# Eval 0B — text→music retrieval (clap_3seg)

- Encoder: `clap` / `laion/larger_clap_music_and_speech` on mps
- Segments per track: 3
- Captions: 1106 · Candidate tracks: 706
- Failed to embed: 0

## Retrieval

| Metric | Measured | Random | Lift |
|---|---:|---:|---:|
| Recall@1 | 0.062 | 0.0014 | 44× |
| Recall@5 | 0.190 | 0.0071 | 27× |
| Recall@10 | 0.286 | 0.0142 | 20× |
| Median rank | 27.5 | 353.5 | — |
| MRR | 0.1386 | — | — |

## Diagnostics

- track–track cosine: mean 0.334, sd 0.194, p95 0.673
- hubness: top 1% of tracks take 4.7% of all top-10 slots; single largest 0.9%
- tracks never in any top-10: 41
- within-track segment cosine: 0.824606849012564

## Cost

- indexing: 0.22 s/track (152 s total)
- query: 3.0 ms/query
- storage: 2048 bytes/track

Raw: `eval_0b_clap_3seg_20260906.json`
