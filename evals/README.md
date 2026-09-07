# Evaluation evidence

Every number quoted in `STATUS.md`, `DESIGN.md`, `EVALS.md` or `DECISIONS.md` is produced by
a script in `scripts/` and lands here. Nothing is measured by an ad-hoc terminal command,
so any figure can be regenerated.

## How to regenerate

| Evidence | Command |
|---|---|
| Eval 0A — ingestion | `python scripts/eval_0a_ingestion.py data/raw/fma/fma_small --label fma_small` |
| Song Describer fetch | `python scripts/fetch_song_describer.py --audio --extract` |
| Eval 0B / E0 / E1 | `python scripts/eval_0b_retrieval.py --encoder muq --n-segments 5` |
| Paired comparison | `python scripts/compare_runs.py <baseline.json> <treatment.json>` |
| Library audit | `python scripts/audit_library.py data/music` |
| Eval 0C — out-of-domain | `python scripts/eval_0c_library.py data/music --encoder muq --n-segments 5` |

## Current evidence

### Eval 0A — ingestion (Slice 0)

`eval_0a_fma_small_arm64_20260906.*` — 8,000 FMA MP3s, **99.92%** decode success,
deterministic, ~1098× realtime. Six failures split cleanly into three truncated files
(`container_unparseable`) and three with corrupt payloads (`decode_failed`).

`eval_0a_fma_small_full_20260906.md` — the same corpus under x86_64/Rosetta. Kept because
DEC-010 cites the cross-architecture comparison: identical success rate and identical
failures at ~40% worse latency. Its raw JSON is gitignored as superseded.

### Eval 0B / E0 / E1 — text→music retrieval (Slice 0)

Song Describer, 1,106 captions over 706 candidate tracks. `eval_0b_<encoder>_<n>seg_*.json`
carries per-query ranks so two runs can be compared with a **paired** test.

| Run | R@1 | R@5 | R@10 | Median |
|---|---:|---:|---:|---:|
| `clap_1seg` | 0.049 | 0.165 | 0.252 | 33 |
| `clap_3seg` | 0.062 | 0.190 | 0.286 | 28 |
| `clap_5seg` | 0.063 | 0.193 | 0.307 | 27 |
| `clap_1seg_valid` | 0.070 | 0.208 | 0.310 | 23 |
| `muq-mulan_1seg` | 0.077 | 0.246 | 0.362 | 20 |
| `muq-mulan_3seg` | 0.088 | 0.268 | 0.397 | 17 |
| **`muq-mulan_5seg`** | **0.090** | **0.270** | **0.407** | **15** |

- **E0** (DEC-012): MuQ-MuLan beats CLAP, paired p = 4.5e-08, holding at both 1 and 5
  segments.
- **E1** (DEC-011): multi-segment beats single on both encoders; 3 vs 5 is *not* separated.
- `clap_1seg_valid` is the human-validated caption subset. It scores higher than the full
  set on identical audio, which is why E0 holds the caption set fixed.

### Eval 0C — out-of-domain personal library (Slice 0)

`eval_0c_library_muq-mulan_20260907.*` and `library_audit_20260907.json` — 160 tracks over
six genres. Within-genre cosine 0.710 vs between-genre 0.401; top-5 genre purity 82.2%
against 16.1% chance; same-artist retrieval 6.0× chance. No collapse.

## A note on the personal library

The personal library is an out-of-domain test set and is **never published or
redistributed** (PROJECT.md). Track titles identify a listener as surely as the audio does,
so any output naming a file is written to a `*.private.json` that `.gitignore` excludes.
Committed reports pseudonymise tracks as `<genre>/tNNN`, which preserves the finding —
neighbour structure and genre membership — without the titles.
