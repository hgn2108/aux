"""Does a context phrase move retrieval in the direction it implies?

The overlap control found that adding "for running" or "for falling asleep" almost wholly
replaces the top 10 while leaving the broad ranking correlated. That has two very different
explanations, and Jaccard alone cannot separate them:

- the context phrase genuinely redirects retrieval; or
- the score distribution is flat near the top, so any small shift in query direction
  reshuffles a noisy head. Instability is not discrimination.

They separate on a question needing no human ratings: do the retrieved tracks *differ
acoustically in the direction the word implies*? "For running" should return denser,
more rhythmically active material than "for falling asleep".

Three crude proxies, deliberately simple and computed from the waveform rather than from
the encoder, so they are independent of the thing being tested:

- **loudness** -- RMS;
- **onset rate** -- fraction of short-time energy frames showing a sharp positive jump, a
  rough stand-in for rhythmic density;
- **brightness** -- zero-crossing rate.

Reported as z-scores against the library mean, so the numbers are comparable across
queries and independent of the library's absolute character.

    python scripts/acoustic_direction.py
"""

import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, "src")
from aux.encode.base import l2_normalise
from aux.ingest import decode, discover, IngestError
from aux.encode.muq import MuQMuLanAdapter

root = Path("data/music")
paths = [f.path for f in discover(root)]
enc = MuQMuLanAdapter()

vecs, feats, kept = [], [], []
for p in paths:
    try:
        a = decode(p); v,_ = enc.embed_track(a, n_segments=5)
    except Exception:
        continue
    x = a.samples
    sr = a.sample_rate
    rms = float(np.sqrt(np.mean(x**2)))
    # crude onset-rate proxy: rate of positive jumps in a short-time energy envelope
    hop = sr//100
    n = (x.size//hop)*hop
    env = np.abs(x[:n].reshape(-1,hop)).max(axis=1)
    d = np.diff(env); onset = float((d > (d.std()*1.5)).mean()) if d.size else 0.0
    # crude brightness proxy: zero-crossing rate
    zcr = float(np.mean(np.abs(np.diff(np.sign(x))) > 0))
    vecs.append(v); feats.append((rms, onset, zcr)); kept.append(p)

V = np.stack(vecs); F = np.array(feats)
Fz = (F - F.mean(0)) / F.std(0)

queries = ["hip hop", "hip hop for running", "hip hop for studying",
           "hip hop for falling asleep", "hip hop for a workout",
           "hip hop to relax to"]
T = l2_normalise(enc.embed_text(queries))
S = T @ V.T

print(f"{'query':32}{'loudness':>10}{'onset rate':>12}{'brightness':>12}")
print("-"*66)
rows=[]
for i,q in enumerate(queries):
    top = np.argsort(-S[i])[:10]
    m = Fz[top].mean(0)
    rows.append((q, *m))
    print(f"{q:32}{m[0]:>10.2f}{m[1]:>12.2f}{m[2]:>12.2f}")
print("\n(z-scores vs library mean; top-10 per query)")
run = rows[1]; sleep = rows[3]; work = rows[4]; relax = rows[5]
print(f"\nrunning - sleeping : loudness {run[1]-sleep[1]:+.2f}, onset {run[2]-sleep[2]:+.2f}, brightness {run[3]-sleep[3]:+.2f}")
print(f"workout - relax    : loudness {work[1]-relax[1]:+.2f}, onset {work[2]-relax[2]:+.2f}, brightness {work[3]-relax[3]:+.2f}")
json.dump({"queries":[{"query":r[0],"loudness_z":r[1],"onset_z":r[2],"brightness_z":r[3]} for r in rows]},
          open("evals/acoustic_direction.json","w"), indent=2)
