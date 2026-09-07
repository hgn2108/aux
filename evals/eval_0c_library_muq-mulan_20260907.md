# Eval 0C — personal-library out-of-domain test (muq-mulan)

- Encoder: `OpenMuQ/MuQ-MuLan-large` on mps, 5 segments
- Tracks: 160

Track identities are pseudonymised as `<genre>/tNNN`; the personal library is never published.

## Space health

- track–track cosine: mean 0.518, sd 0.263, p95 0.875
- benchmark reference (`muq-mulan_5seg`): mean 0.405, sd 0.176
- hubness: top 1% take 2.2% of neighbour slots; 12 tracks never a neighbour

## Genre structure

Within-genre cosine 0.710 vs between-genre 0.401 (separation +0.309). Top-5 genre purity 82.2% vs 16.1% chance (5.1×).

| Genre | n | Within | Between | Purity | Chance |
|---|---:|---:|---:|---:|---:|
| hiphop_rnb | 94 | 0.724 | 0.424 | 92.3% | 58.5% |
| jazz | 20 | 0.405 | 0.311 | 69.0% | 11.9% |
| edm | 14 | 0.702 | 0.465 | 78.6% | 8.2% |
| vpop | 13 | 0.723 | 0.500 | 80.0% | 7.5% |
| dnb | 12 | 0.732 | 0.433 | 85.0% | 6.9% |
| classical | 7 | 0.606 | 0.157 | 88.6% | 3.8% |

## Same-artist structure

Same artist in top 5: 36.4% observed vs 6.0% chance (6.0×), over 44 tracks with a same-artist counterpart.

Free ground truth from filenames. It is what separates a self-similar library from a collapsed embedding space: a collapsed space ranks same-artist tracks at chance.

## Text probe queries

Probes, not a benchmark — eight queries prove nothing statistically; they exist to be read.

| Query | Top result | Score |
|---|---|---:|
| aggressive hard-hitting trap with heavy 808s | `hiphop_rnb/t060` | 0.430 |
| melodic and melancholy, sung rather than rapped | `vpop/t013` | 0.523 |
| slow smooth late-night R&B | `hiphop_rnb/t047` | 0.491 |
| upbeat energetic party track | `edm/t003` | 0.424 |
| dreamy atmospheric production with reverb | `hiphop_rnb/t001` | 0.189 |
| stripped back and minimal, few instruments | `hiphop_rnb/t077` | 0.323 |
| dark and menacing | `dnb/t009` | 0.280 |
| warm and nostalgic | `classical/t005` | 0.350 |

Raw: `eval_0c_library_muq-mulan_20260907.json` (pseudonymised) · `eval_0c_library_muq-mulan_20260907.private.json` (local only)
