# Eval 0B — text→music retrieval (muq-mulan_1seg)

- Encoder: `muq-mulan` / `OpenMuQ/MuQ-MuLan-large` on mps
- Segments per track: 1
- Captions: 1106 · Candidate tracks: 706
- Failed to embed: 0

## Retrieval

| Metric | Measured | Random | Lift |
|---|---:|---:|---:|
| Recall@1 | 0.077 | 0.0014 | 54× |
| Recall@5 | 0.246 | 0.0071 | 35× |
| Recall@10 | 0.362 | 0.0142 | 26× |
| Median rank | 20.0 | 353.5 | — |
| MRR | 0.1686 | — | — |

## Diagnostics

- track–track cosine: mean 0.371, sd 0.183, p95 0.712
- hubness: top 1% of tracks take 8.1% of all top-10 slots; single largest 1.5%
- tracks never in any top-10: 83
- within-track segment cosine: n/a (single segment)

## Cost

- indexing: 0.25 s/track (176 s total)
- query: 7.6 ms/query
- storage: 2048 bytes/track

Raw: `eval_0b_muq-mulan_1seg_20260907.json`
