# Eval 0B — text→music retrieval (muq-mulan_3seg)

- Encoder: `muq-mulan` / `OpenMuQ/MuQ-MuLan-large` on mps
- Segments per track: 3
- Captions: 1106 · Candidate tracks: 706
- Failed to embed: 0

## Retrieval

| Metric | Measured | Random | Lift |
|---|---:|---:|---:|
| Recall@1 | 0.088 | 0.0014 | 62× |
| Recall@5 | 0.268 | 0.0071 | 38× |
| Recall@10 | 0.397 | 0.0142 | 28× |
| Median rank | 17.0 | 353.5 | — |
| MRR | 0.1858 | — | — |

## Diagnostics

- track–track cosine: mean 0.398, sd 0.178, p95 0.729
- hubness: top 1% of tracks take 8.0% of all top-10 slots; single largest 1.5%
- tracks never in any top-10: 56
- within-track segment cosine: 0.8817411590230364

## Cost

- indexing: 0.45 s/track (315 s total)
- query: 8.2 ms/query
- storage: 2048 bytes/track

Raw: `eval_0b_muq-mulan_3seg_20260907.json`
