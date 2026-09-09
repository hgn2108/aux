"""Does the encoder handle negation, and which weak queries are library gaps?

Eval 1's weakest query was "solo piano, no vocals" (mean 1.2), yet bare "solo piano" probes
cleanly at z=4.97 returning classical. That gap is worth isolating, because "acoustic
queries are weak" and "negation is broken" call for completely different responses.

Compares minimal pairs that differ only by a negation or a modifier, and reports what genre
each retrieves. A joint encoder trained contrastively has no negation operator: "no vocals"
contains the token "vocals", so the phrase can pull *toward* what it was meant to exclude.

    python scripts/negation_check.py
"""

import sys, json
from pathlib import Path
import numpy as np
sys.path.insert(0, "src")
from aux.encode.base import l2_normalise
from aux.encode.muq import MuQMuLanAdapter
from aux.ingest import decode, discover

root = Path("data/music")
enc = MuQMuLanAdapter()
paths, vecs = [], []
for p in [f.path for f in discover(root)]:
    try:
        v,_ = enc.embed_track(decode(p), n_segments=5)
    except Exception: continue
    paths.append(p); vecs.append(v)
V = np.stack(vecs)
genre = [p.relative_to(root).parts[0] if len(p.relative_to(root).parts)>1 else "hiphop_rnb" for p in paths]

pairs = [
    ("solo piano", "solo piano, no vocals"),
    ("acoustic guitar and soft vocals", "acoustic guitar, no vocals"),
    ("hip hop", "hip hop, no rapping"),
    ("distorted electric guitar", "electric guitar"),
    ("orchestral strings", "strings"),
]
qs = [q for pr in pairs for q in pr]
T = l2_normalise(enc.embed_text(qs))
S = T @ V.T
print(f"{'query':38}{'top genres (top-5)':38}{'z-top':>7}")
print("-"*84)
for q, s in zip(qs, S):
    top = np.argsort(-s)[:5]
    z = (s.max()-s.mean())/s.std()
    print(f"{q:38}{', '.join(genre[i] for i in top):38}{z:>7.2f}")
print()
for a,b in pairs:
    ia, ib = qs.index(a), qs.index(b)
    ta, tb = set(np.argsort(-S[ia])[:5]), set(np.argsort(-S[ib])[:5])
    print(f"{a:38} vs {b:34} q-cos {float(T[ia]@T[ib]):.3f}  J@5 {len(ta&tb)/len(ta|tb):.2f}")
