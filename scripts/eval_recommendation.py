"""Evaluate track-to-track recommendation: audio, lyrics, and the two fused.

Runs on either of two corpora, because no single available corpus supports both halves of
the evaluation:

    python scripts/eval_recommendation.py --corpus fma --per-genre 250
    python scripts/eval_recommendation.py --corpus personal

**fma** — 2,000 Creative Commons tracks, balanced at 250 per genre across 8 genres. Public,
fully reproducible, and the corpus for the headline audio result. It has no usable lyric
channel: a sample of 75 clips was 56% instrumental with a median transcript of 11 words, and
the metadata alternatives are no better (tags are populated for 17% of tracks and are mostly
place names; `track_genres` is the label itself). So this corpus runs audio-only.

**personal** — 160 commercially released tracks, 79% with a reliable transcript at a median
of 376 words. Not redistributable and far smaller, but it is the only corpus here where both
modalities exist, so it carries the audio-vs-lyrics-vs-fused comparison and the fusion-weight
sweep. Results identify tracks by pseudonym; no audio, title or artist is published.

**Proxy labels, not ground truth.** Relevance means "shares a genre / artist / album" with
the query. None of these is musical similarity. Three definitions are used rather than one
because each is wrong in a different direction, and agreement between them says far more than
a good score on any one.

The genre label is the weakest and is reported twice: plainly, and excluding same-artist
pairs. Tracks from one album share production, mastering and instrumentation, so without that
exclusion a model scores well on genre by recognising an album rather than a genre.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aux.data import balanced_subset, load_personal_tracks, load_tracks  # noqa: E402
from aux.data import relevance_matrix, same_artist_matrix  # noqa: E402
from aux.eval import evaluate_ranking, random_ranking_baseline  # noqa: E402
from aux.index import build_index  # noqa: E402
from aux.ingest.asset import content_hash  # noqa: E402
from aux.recommend import Recommender  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / ".cache"
RESULTS = ROOT / "results"

ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)
KS = (5, 10, 20)

#: Minimum share of tracks needing a reliable transcript before the lyric arm is reported.
#: Below this, a "lyrics" row measures coverage rather than the modality: most candidates
#: have no text at all, so the ranking is driven by which tracks happen to have been
#: transcribed. FMA sits far under the floor and is therefore evaluated audio-only.
MIN_LYRIC_COVERAGE = 0.5


def load_lyric_vectors(tracks, whisper_model: str, cache_tag: str):
    """Embed transcripts for the corpus, caching the embedding matrix.

    Returns `(vectors, has_lyrics)`, or `(None, None)` when no track has a reliable
    transcript. A track without one gets a zero vector and a False flag; the recommender
    reads the flag and never the zero.
    """
    tpath = CACHE / f"transcripts_{whisper_model}.json"
    if not tpath.exists():
        return None, None
    transcripts = json.loads(tpath.read_text())
    by_hash = {k.split("|")[0]: v for k, v in transcripts.items()}

    texts, has = [], []
    for t in tracks:
        rec = by_hash.get(content_hash(t.path))
        if rec and rec.get("reliable"):
            texts.append(rec["text"])
            has.append(True)
        else:
            texts.append("")
            has.append(False)
    has = np.array(has, dtype=bool)
    if not has.any():
        return None, None

    vec_path = CACHE / f"lyrics_{cache_tag}.npy"
    if vec_path.exists():
        vectors = np.load(vec_path)
        if vectors.shape[0] == len(tracks):
            return vectors, has

    from aux.lyrics import LyricEmbedder

    vectors = LyricEmbedder().embed_documents(texts)
    vectors[~has] = 0.0
    np.save(vec_path, vectors)
    return vectors, has


def build_systems(rec: Recommender, has_lyrics) -> dict[str, np.ndarray]:
    """Score matrices for every system under comparison.

    Audio-only when the corpus has no lyric channel; otherwise audio, lyrics, and the fused
    system at each interior fusion weight. `alpha` is the audio share, so alpha=0 and
    alpha=1 would duplicate the two single-modality systems and are skipped here.
    """
    systems = {"audio": rec.score_matrix(modality="audio")}
    if has_lyrics is None:
        return systems
    systems["lyrics"] = rec.score_matrix(modality="lyrics")
    for a in ALPHAS:
        if a not in (0.0, 1.0):
            systems[f"fused a={a}"] = rec.score_matrix(modality="fused", alpha=a)
    return systems


def print_table(name: str, rows: dict) -> None:
    head = rows["random"]
    print(f"\n=== {name} ({head['n_queries']} queries, "
          f"{head['mean_relevant_per_query']:.1f} relevant each) ===")
    print(f"{'system':18}" + "".join(f"{'P@' + str(k):>9}{'NDCG@' + str(k):>11}" for k in KS)
          + f"{'HR@10':>8}")
    print("-" * (18 + 20 * len(KS) + 8))
    for sys_name, m in rows.items():
        line = "".join(f"{m[f'precision@{k}']:>9.3f}{m[f'ndcg@{k}']:>11.3f}" for k in KS)
        print(f"{sys_name:18}{line}{m['hit_rate@10']:>8.3f}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Track-to-track recommendation evaluation")
    ap.add_argument("--corpus", choices=("fma", "personal"), default="fma")
    ap.add_argument("--per-genre", type=int, default=250, help="fma only")
    ap.add_argument("--n-segments", type=int, default=5)
    ap.add_argument("--whisper", default="small")
    args = ap.parse_args()

    if args.corpus == "fma":
        tracks = balanced_subset(load_tracks(), per_genre=args.per_genre)
        labels = ("genre", "artist", "album")
        tag = f"fma_{args.per_genre}"
        cache_path = CACHE / f"fma_audio_{args.per_genre}.npz"
        corpus_name = "FMA small"
    else:
        tracks = load_personal_tracks()
        labels = ("genre", "artist")  # filenames carry no album
        tag = "personal"
        cache_path = CACHE / "personal.npz"
        corpus_name = "personal library"

    from aux.encode.muq import MuQMuLanAdapter

    encoder = MuQMuLanAdapter()
    V, kept, _ = build_index(None, encoder, paths=[t.path for t in tracks],
                             n_segments=args.n_segments, cache_path=cache_path,
                             progress_every=0)
    kept_set = {str(p) for p in kept}
    tracks = [t for t in tracks if str(t.path) in kept_set]
    print(f"{corpus_name}: {len(tracks)} tracks indexed", file=sys.stderr)

    L, has_lyrics = load_lyric_vectors(tracks, args.whisper, tag)
    coverage = float(has_lyrics.mean()) if L is not None else 0.0
    if L is not None:
        print(f"{int(has_lyrics.sum())}/{len(tracks)} tracks have usable lyrics "
              f"({coverage:.0%})", file=sys.stderr)
    if coverage < MIN_LYRIC_COVERAGE:
        print(f"lyric coverage below {MIN_LYRIC_COVERAGE:.0%} — evaluating audio only",
              file=sys.stderr)
        L, has_lyrics = None, None

    rec = Recommender(V, lyric_vectors=L, has_lyrics=has_lyrics,
                      paths=[t.path for t in tracks])
    systems = build_systems(rec, has_lyrics)
    same_artist = same_artist_matrix(tracks)

    results: dict = {}
    for label in labels:
        R = relevance_matrix(tracks, label)
        variants = {label: None}
        if label == "genre":
            variants["genre (artist-filtered)"] = same_artist
        for name, exclude in variants.items():
            rows = {s: evaluate_ranking(S, R, KS, exclude) for s, S in systems.items()}
            rows["random"] = random_ranking_baseline(R, KS, exclude)
            results[name] = rows
            print_table(name, rows)

    # Per-genre breakdown of the strongest system, for failure analysis. Scored against the
    # artist label: genre is constant within each slice, so it cannot discriminate there.
    genres = np.array([t.genre for t in tracks])
    R_art = relevance_matrix(tracks, "artist")
    best = max(systems, key=lambda s: results["artist"][s]["ndcg@10"])
    per_genre = {}
    for g in sorted(set(genres)):
        idx = np.flatnonzero(genres == g)
        m = evaluate_ranking(systems[best], R_art, (10,), queries=idx)
        if m["n_queries"] == 0:
            continue
        per_genre[g] = {"n": m["n_queries"], "ndcg@10": m["ndcg@10"],
                        "precision@10": m["precision@10"]}

    print(f"\n=== per-genre, artist label, {best} ===")
    print(f"{'genre':16}{'n':>5}{'P@10':>9}{'NDCG@10':>10}")
    print("-" * 40)
    for g, m in sorted(per_genre.items(), key=lambda x: -x[1]["ndcg@10"]):
        print(f"{g:16}{m['n']:>5}{m['precision@10']:>9.3f}{m['ndcg@10']:>10.3f}")

    RESULTS.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    out = RESULTS / f"recommendation_{args.corpus}_{stamp}.json"
    out.write_text(json.dumps({
        "run_at": datetime.now(timezone.utc).isoformat(),
        "corpus": {"name": corpus_name, "tracks": len(tracks),
                   "per_genre": args.per_genre if args.corpus == "fma" else None,
                   "with_lyrics": int(has_lyrics.sum()) if L is not None else 0,
                   "lyric_coverage": coverage,
                   "modalities": sorted(systems)},
        "encoder": encoder.version, "n_segments": args.n_segments,
        "whisper": args.whisper, "alphas": list(ALPHAS), "ks": list(KS),
        "results": results, "per_genre": per_genre, "best_system": best,
    }, indent=2))
    print(f"\nwrote {out.relative_to(ROOT)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
