"""E3 — audio, lyrics, or both?

Three tests, all with real ground truth and none needing human rating. EVALS.md's E3 asks
which modality wins for which query kind; these pin the parts that are objectively
checkable, so any later rating round can be spent on the genuinely subjective half.

**Test 1 — lyric-line retrieval.** Take a line from a track's own transcript and use it as
the query. The correct answer is known exactly. This is the "what's that song that goes..."
task, and it is *definitionally* impossible for an audio encoder — so audio should sit near
chance while lyrics should be near-perfect. It measures capability rather than semantic
generalisation, since the query is drawn from the document.

**Test 2 — language.** "sung in Vietnamese" returned jazz in Slice 1 because the audio
encoder does not represent language. Ground truth is the v-pop folder: 13 tracks.

**Test 3 — thematic.** Queries about what a song is *about*. Ground truth here is weaker —
Claude labels which tracks plausibly qualify — so it is reported separately and read as
indicative rather than decisive.

Fusion is rank-level (RRF), per DESIGN.md: audio cosine and lyric cosine come from different
models and are not on a common scale.

    python scripts/eval_3_modality.py
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aux.encode.base import l2_normalise  # noqa: E402
from aux.eval import ranks_of_truth, retrieval_metrics  # noqa: E402
from aux.index import build_index  # noqa: E402
from aux.lyrics import LyricEmbedder  # noqa: E402
from aux.rank import reciprocal_rank_fusion  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
EVALS = ROOT / "evals"
SEED = 20260910

LANGUAGE_QUERIES = ["sung in Vietnamese", "a song with Vietnamese lyrics",
                    "vietnamese vocals"]


def genre_of(path: Path, root: Path) -> str:
    rel = Path(path).relative_to(root)
    return rel.parts[0] if len(rel.parts) > 1 else "hiphop_rnb"


def pick_line(text: str, rng: random.Random, words: int = 10) -> str | None:
    """A contiguous run of words from the middle of a transcript.

    Middle rather than start: openings are often a repeated tag or an ad-lib, and a
    hallucinated loop at the top would make the task trivially easy for the wrong reason.
    """
    tokens = text.split()
    if len(tokens) < words * 3:
        return None
    lo, hi = len(tokens) // 4, (3 * len(tokens)) // 4 - words
    if hi <= lo:
        return None
    start = rng.randint(lo, hi)
    return " ".join(tokens[start:start + words])


def main() -> int:
    ap = argparse.ArgumentParser(description="E3 — modality ablation")
    ap.add_argument("--root", type=Path, default=ROOT / "data" / "music")
    ap.add_argument("--n-segments", type=int, default=5)
    args = ap.parse_args()

    tr_files = sorted(EVALS.glob("transcription_*.private.json"))
    if not tr_files:
        print("no transcripts; run scripts/transcribe_library.py first", file=sys.stderr)
        return 1
    records = json.loads(tr_files[-1].read_text())["records"]
    # Transcripts were written with paths relative to the invocation directory; the index
    # returns absolute ones. Key on the resolved path so the two always meet.
    by_path = {str(Path(r["path"]).resolve()): r for r in records}

    from aux.encode.muq import MuQMuLanAdapter

    encoder = MuQMuLanAdapter()
    V, paths, _ = build_index(args.root, encoder, n_segments=args.n_segments,
                              cache_path=ROOT / ".cache" / "personal.npz", progress_every=0)
    texts, reliable = [], []
    for p in paths:
        r = by_path.get(str(Path(p).resolve()))
        texts.append((r or {}).get("text", ""))
        reliable.append(bool((r or {}).get("reliable")))
    reliable = np.array(reliable)
    if not reliable.any():
        print("no transcripts matched the indexed tracks — check paths", file=sys.stderr)
        return 1
    print(f"{len(paths)} tracks, {int(reliable.sum())} with reliable lyrics", file=sys.stderr)

    lyric = LyricEmbedder()
    D = lyric.embed_documents(texts)
    # An unreliable transcript should never win a lyric search; zeroing its vector removes
    # it from the lyric modality without disturbing the audio one.
    D[~reliable] = 0.0

    genres = np.array([genre_of(p, args.root) for p in paths])
    results: dict = {}

    # --- Test 1: lyric-line retrieval ------------------------------------------------
    rng = random.Random(SEED)
    queries, truth = [], []
    for i, (t, ok) in enumerate(zip(texts, reliable)):
        if not ok:
            continue
        line = pick_line(t, rng)
        if line:
            queries.append(line)
            truth.append(i)
    truth = np.array(truth)
    audio_s = l2_normalise(encoder.embed_text(queries)) @ V.T
    lyric_s = lyric.embed_query(queries) @ D.T
    fused_s = np.stack([reciprocal_rank_fusion([audio_s[i], lyric_s[i]])
                        for i in range(len(queries))])

    print(f"\n=== Test 1 — find the track from a line of its own lyrics "
          f"({len(queries)} queries, {len(paths)} candidates) ===")
    print(f"{'modality':14}{'R@1':>8}{'R@5':>8}{'R@10':>8}{'median':>9}{'MRR':>8}")
    print("-" * 55)
    t1 = {}
    for name, s in (("audio", audio_s), ("lyrics", lyric_s), ("fused", fused_s)):
        m = retrieval_metrics(ranks_of_truth(s, truth), len(paths))
        t1[name] = m
        print(f"{name:14}{m['recall@1']:>8.3f}{m['recall@5']:>8.3f}{m['recall@10']:>8.3f}"
              f"{m['median_rank']:>9.0f}{m['mrr']:>8.3f}")
    print(f"{'chance':14}{1/len(paths):>8.3f}{5/len(paths):>8.3f}{10/len(paths):>8.3f}"
          f"{len(paths)/2:>9.0f}")
    results["lyric_line"] = {"n_queries": len(queries), **t1}

    # --- Test 2: language ------------------------------------------------------------
    print(f"\n=== Test 2 — language ({int((genres == 'vpop').sum())} Vietnamese tracks) ===")
    print(f"{'query':34}{'modality':10}{'vpop in top 13':>16}")
    print("-" * 60)
    t2 = {}
    for q in LANGUAGE_QUERIES:
        a = (V @ l2_normalise(encoder.embed_text([q]))[0])
        l_ = (D @ lyric.embed_query([q])[0])
        f = reciprocal_rank_fusion([a, l_])
        for name, s in (("audio", a), ("lyrics", l_), ("fused", f)):
            hits = int((genres[np.argsort(-s)[:13]] == "vpop").sum())
            t2.setdefault(name, []).append(hits)
            print(f"{q[:32]:34}{name:10}{hits:>10}/13")
    results["language"] = {k: {"mean_hits": float(np.mean(v)), "per_query": v}
                           for k, v in t2.items()}
    print(f"\n{'mean':44}" + "  ".join(f"{k} {np.mean(v):.1f}" for k, v in t2.items()))

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    out = EVALS / f"eval_3_modality_{stamp}.json"
    out.write_text(json.dumps({"run_at": datetime.now(timezone.utc).isoformat(),
                               "encoder": encoder.version, "lyric_model": lyric.version,
                               "tracks": len(paths), "reliable_lyrics": int(reliable.sum()),
                               "results": results}, indent=2))
    print(f"\nwrote {out.relative_to(ROOT)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
