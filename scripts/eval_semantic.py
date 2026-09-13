"""Text-to-track retrieval on *semantic* queries: where does each modality actually win?

The recommendation evaluation scores track-to-track similarity against genre, artist and
album. Those labels are acoustic constructs, so audio wins and fusion cannot help — there is
nothing for the lyric channel to contribute that audio has not already captured.

That is a statement about the task, not about the modalities. This script asks the opposite
question with the same two systems: given "songs about heartbreak", which channel finds
them? The audio encoder is a joint music-text model, so it can answer directly and is not a
strawman; the lyric encoder answers by matching the query against transcripts.

Three systems plus two bounds:

- **audio** — the query embedded in MuQ-MuLan's text tower, matched against audio vectors.
- **lyrics** — the query embedded by Qwen3, matched against transcript vectors.
- **fused** — both, z-score normalised per query, blended at `alpha` (the audio share).
- **oracle** — per query, whichever single modality scored better. An upper bound on every
  possible router, so it says how much a router could ever be worth.
- **random** — computed under the same labels and exclusions, not derived.

Labels come from `scripts/label_themes.py` and are weak: a model read each transcript and
marked the themes it is about. Agreement with blind human labels is measured separately by
`scripts/rate_themes.py` and belongs next to every number printed here.

    python scripts/eval_semantic.py
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aux.data import load_personal_tracks  # noqa: E402
from aux.eval import ndcg_at_k, precision_at_k, hit_rate_at_k, recall_at_k  # noqa: E402
from aux.index import build_index  # noqa: E402
from aux.ingest.asset import content_hash  # noqa: E402
from aux.recommend import NORMALISERS  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / ".cache"
RESULTS = ROOT / "results"

ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)
KS = (5, 10, 20)
MIN_TRACKS_PER_THEME = 5
"""A theme matching fewer tracks than this cannot separate systems: with two relevant items
in a 160-track corpus, one lucky hit swings NDCG by more than the effect being measured."""


def score_queries(query_vectors: np.ndarray, item_vectors: np.ndarray,
                  usable: np.ndarray | None = None) -> np.ndarray:
    """Cosine scores for every query against every track; unusable items get -inf."""
    scores = np.asarray(query_vectors, np.float32) @ np.asarray(item_vectors, np.float32).T
    if usable is not None:
        scores[:, ~usable] = -np.inf
    return scores.astype(float)


def fuse(audio: np.ndarray, lyric: np.ndarray, alpha: float, has_lyrics: np.ndarray,
         norm) -> np.ndarray:
    """Blend two score matrices per query, falling back to audio where lyrics are missing.

    Identical in shape to `Recommender.score`: a track with no transcript keeps its audio
    score rather than being zeroed, which would turn fusion into a vocal-music filter.
    """
    out = np.empty_like(audio)
    for i in range(audio.shape[0]):
        a = norm(audio[i])
        row = np.array(a)
        if has_lyrics.any():
            lz = np.zeros_like(a)
            lz[has_lyrics] = norm(lyric[i][has_lyrics])
            row = alpha * a + (1 - alpha) * lz
            row[~has_lyrics] = a[~has_lyrics]
        out[i] = row
    return out


def per_query_metrics(scores: np.ndarray, relevant: np.ndarray, k: int) -> np.ndarray:
    """NDCG@k for each query, so an oracle bound can be taken before averaging."""
    order = np.argsort(-scores, axis=1)
    totals = relevant.sum(axis=1)
    return np.array([ndcg_at_k(relevant[i, order[i]], k, int(totals[i]))
                     for i in range(scores.shape[0])])


def summarise(scores: np.ndarray, relevant: np.ndarray) -> dict:
    order = np.argsort(-scores, axis=1)
    totals = relevant.sum(axis=1)
    out: dict = {"n_queries": int(scores.shape[0]),
                 "mean_relevant_per_query": float(totals.mean())}
    for k in KS:
        ranked = [relevant[i, order[i]] for i in range(scores.shape[0])]
        out[f"precision@{k}"] = float(np.mean([precision_at_k(r, k) for r in ranked]))
        out[f"recall@{k}"] = float(np.mean(
            [recall_at_k(r, k, int(t)) for r, t in zip(ranked, totals)]))
        out[f"hit_rate@{k}"] = float(np.mean([hit_rate_at_k(r, k) for r in ranked]))
        out[f"ndcg@{k}"] = float(np.mean(
            [ndcg_at_k(r, k, int(t)) for r, t in zip(ranked, totals)]))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Semantic text-to-track retrieval evaluation")
    ap.add_argument("--n-segments", type=int, default=5)
    ap.add_argument("--whisper", default="small")
    ap.add_argument("--normaliser", default="zscore", choices=sorted(NORMALISERS))
    ap.add_argument("--k", type=int, default=10, help="K used for the oracle bound")
    args = ap.parse_args()

    from label_themes import THEMES

    labels = json.loads((CACHE / "themes.json").read_text())

    tracks = load_personal_tracks()
    from aux.encode.muq import MuQMuLanAdapter

    encoder = MuQMuLanAdapter()
    V, kept, _ = build_index(None, encoder, paths=[t.path for t in tracks],
                             n_segments=args.n_segments, cache_path=CACHE / "personal.npz",
                             progress_every=0)
    kept_set = {str(p) for p in kept}
    tracks = [t for t in tracks if str(t.path) in kept_set]

    sys.path.insert(0, str(ROOT / "scripts"))
    from eval_recommendation import load_lyric_vectors

    L, has_lyrics = load_lyric_vectors(tracks, args.whisper, "personal")

    # Build the query set: one query per theme with enough labelled tracks to be measurable.
    hashes = [content_hash(t.path) for t in tracks]
    themes, relevance_rows, queries = [], [], []
    for theme, description in sorted(THEMES.items()):
        row = np.array([theme in labels.get(h, {}).get("themes", []) for h in hashes])
        if row.sum() >= MIN_TRACKS_PER_THEME:
            themes.append(theme)
            relevance_rows.append(row)
            queries.append(f"songs about {description}")
    if not themes:
        print("no theme has enough labelled tracks — run scripts/label_themes.py",
              file=sys.stderr)
        return 1
    R = np.stack(relevance_rows)
    print(f"{len(tracks)} tracks, {int(has_lyrics.sum())} with lyrics, "
          f"{len(themes)} themes with >= {MIN_TRACKS_PER_THEME} tracks", file=sys.stderr)

    from aux.lyrics import LyricEmbedder

    audio_scores = score_queries(encoder.embed_text(queries), V)
    lyric_scores = score_queries(LyricEmbedder().embed_query(queries), L, has_lyrics)

    norm = NORMALISERS[args.normaliser]
    systems = {"audio": audio_scores, "lyrics": np.where(np.isfinite(lyric_scores),
                                                         lyric_scores, -1e9)}
    for a in ALPHAS:
        if a not in (0.0, 1.0):
            systems[f"fused a={a}"] = fuse(audio_scores, lyric_scores, a, has_lyrics, norm)

    rng = np.random.default_rng(0)
    systems["random"] = rng.random(audio_scores.shape)

    rows = {name: summarise(S, R) for name, S in systems.items()}

    # Oracle: per query, the better of the two single modalities. Not a system that could be
    # built — it reads the labels — but an upper bound on every router that could be.
    nd_audio = per_query_metrics(systems["audio"], R, args.k)
    nd_lyric = per_query_metrics(systems["lyrics"], R, args.k)
    oracle = float(np.mean(np.maximum(nd_audio, nd_lyric)))

    print(f"\n=== semantic queries ({len(themes)} themes, "
          f"{R.sum(1).mean():.1f} relevant each) ===")
    print(f"{'system':18}" + "".join(f"{'P@' + str(k):>9}{'NDCG@' + str(k):>11}" for k in KS)
          + f"{'HR@10':>8}")
    print("-" * (18 + 20 * len(KS) + 8))
    for name, m in rows.items():
        line = "".join(f"{m[f'precision@{k}']:>9.3f}{m[f'ndcg@{k}']:>11.3f}" for k in KS)
        print(f"{name:18}{line}{m['hit_rate@10']:>8.3f}")
    print(f"\nORACLE modality per query, NDCG@{args.k}: {oracle:.3f}  "
          f"(audio {float(nd_audio.mean()):.3f}, lyrics {float(nd_lyric.mean()):.3f})")
    print(f"lyrics beat audio on {float((nd_lyric > nd_audio).mean()):.0%} of queries")

    print(f"\n=== per theme, NDCG@{args.k} ===")
    print(f"{'theme':14}{'n':>5}{'audio':>9}{'lyrics':>9}{'winner':>10}")
    print("-" * 47)
    per_theme = {}
    for i, theme in enumerate(themes):
        winner = "lyrics" if nd_lyric[i] > nd_audio[i] else "audio"
        per_theme[theme] = {"n": int(R[i].sum()), "audio": float(nd_audio[i]),
                            "lyrics": float(nd_lyric[i]), "winner": winner}
        print(f"{theme:14}{int(R[i].sum()):>5}{nd_audio[i]:>9.3f}"
              f"{nd_lyric[i]:>9.3f}{winner:>10}")

    RESULTS.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    out = RESULTS / f"semantic_{stamp}.json"
    out.write_text(json.dumps({
        "run_at": datetime.now(timezone.utc).isoformat(),
        "corpus": {"name": "personal library", "tracks": len(tracks),
                   "with_lyrics": int(has_lyrics.sum())},
        "encoder": encoder.version, "normaliser": args.normaliser,
        "themes": themes, "queries": queries, "ks": list(KS),
        "results": rows, "oracle_ndcg": oracle, "oracle_k": args.k,
        "per_theme": per_theme,
    }, indent=2))
    print(f"\nwrote {out.relative_to(ROOT)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
