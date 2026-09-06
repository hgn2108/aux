# Eval 0A — ingestion (fma_small_full)

- Run: 2026-09-06T08:41:09.554761+00:00
- Root: `data/raw/fma/fma_small`
- Platform: Darwin x86_64 py3.13.5

## Gate

**PASS** — 7994/8000 supported files (99.92%); gate is >=95%.

## Per-format

| Format | Total | OK | Success | Failures |
|---|---:|---:|---:|---|
| `.mp3` | 8000 | 7994 | 99.92% | decode_failed=3, container_unparseable=3 |

## Latency

- probe p50/p95: 0.9 / 1.1 ms
- decode p50/p95: 39.6 / 42.6 ms
- decode realtime factor: 767.4x

## Repeatability

- 20 files decoded twice; deterministic: **True**

## Library shape

- sample rates: {44100: 7569, 48000: 411, 22050: 14}
- channels: {2: 7909, 1: 85}
- codecs: {'mp3float': 7994}
- total audio: 66.59 h

Raw per-file outcomes: `eval_0a_fma_small_full_20260906.json`
