# Eval 0B — text→music retrieval (muq-mulan_5seg)

- Encoder: `muq-mulan` / `OpenMuQ/MuQ-MuLan-large` on mps
- Segments per track: 5
- Captions: 1106 · Candidate tracks: 706
- Failed to embed: 0

## Retrieval

| Metric | Measured | Random | Lift |
|---|---:|---:|---:|
| Recall@1 | 0.090 | 0.0014 | 63× |
| Recall@5 | 0.270 | 0.0071 | 38× |
| Recall@10 | 0.407 | 0.0142 | 29× |
| Median rank | 15.0 | 353.5 | — |
| MRR | 0.1889 | — | — |

## Diagnostics

- track–track cosine: mean 0.405, sd 0.176, p95 0.732
- hubness: top 1% of tracks take 8.0% of all top-10 slots; single largest 1.5%
- tracks never in any top-10: 55
- within-track segment cosine: 0.8766767523190793

## Cost

- indexing: 0.63 s/track (446 s total)
- query: 8.7 ms/query
- storage: 2048 bytes/track

Raw: `eval_0b_muq-mulan_5seg_20260907.json`
