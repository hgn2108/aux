"""Eval 0B -- text->music retrieval on Song Describer.

Each caption is a query; the track it describes is the single relevant item; every track in
the benchmark is a candidate. Reports Recall@1/5/10, median rank and MRR against the random
baseline that the same task implies.

The same harness serves E0 and E1: `--encoder` swaps the model and `--n-segments` swaps the
pooling strategy, holding everything else fixed, so a difference between two runs is
attributable to the one thing that changed.

    python scripts/eval_0b_retrieval.py --n-segments 1
    python scripts/eval_0b_retrieval.py --n-segments 3 --label clap_3seg

Two diagnostics run alongside the metrics, both cheap and both aimed at Slice 0's named
risks:

- **embedding spread** -- mean pairwise cosine between track vectors. A space that has
  collapsed on out-of-domain audio shows high mean similarity with little spread, which is
  a representation failure and not something ranking work can fix.
- **hubness** -- how concentrated top-10 appearances are across tracks. Contrastive spaces
  reliably produce a few vectors that are nearest neighbour to almost everything, and a
  hub inflates nothing while quietly capping achievable recall.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aux.encode.base import l2_normalise  # noqa: E402
from aux.eval import (  # noqa: E402
    random_baseline,
    ranks_of_truth,
    retrieval_metrics,
    space_diagnostics,
)
from aux.ingest import IngestError, decode  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SDD = ROOT / "data" / "raw" / "song_describer"
EVALS_DIR = ROOT / "evals"


def load_captions(valid_only: bool) -> list[dict]:
    rows = list(csv.DictReader((SDD / "song_describer.csv").open()))
    if valid_only:
        rows = [r for r in rows if r["is_valid_subset"] == "True"]
    return rows


def _resolve_audio(audio_root: Path, rel: str) -> Path | None:
    """Map a caption row's `path` onto the released audio file.

    The CSV names full tracks (`34/1004034.mp3`) but the distributed audio is 2-minute
    excerpts (`34/1004034.2min.mp3`). Worth stating plainly rather than papering over: the
    benchmark measures retrieval over 2-minute excerpts, so any pooling conclusion drawn
    from it (E1) is a conclusion about excerpts, not about whole 3-5 minute tracks. The
    production library has no such truncation.
    """
    direct = audio_root / rel
    if direct.exists():
        return direct
    excerpt = direct.with_suffix("").with_suffix(".2min.mp3")
    if excerpt.exists():
        return excerpt
    stem = direct.stem
    matches = sorted(direct.parent.glob(f"{stem}.*.mp3"))
    return matches[0] if matches else None


def build_encoder(name: str, checkpoint: str | None):
    if name == "clap":
        from aux.encode.clap import DEFAULT_CHECKPOINT, ClapAdapter

        return ClapAdapter(checkpoint or DEFAULT_CHECKPOINT)
    raise ValueError(f"unknown encoder {name!r}; add an adapter for it")


def embed_tracks(encoder, track_paths: dict[str, Path], n_segments: int):
    """Embed every benchmark track. Returns vectors, ids, per-track segment spread, failures."""
    ids, vectors, spreads, failures = [], [], [], []
    t0 = time.perf_counter()
    for i, (track_id, path) in enumerate(track_paths.items(), 1):
        try:
            asset = decode(path)
            vector, segments = encoder.embed_track(asset, n_segments=n_segments)
        except (IngestError, Exception) as exc:  # noqa: BLE001
            failures.append({"track_id": track_id, "error": f"{type(exc).__name__}: {exc}"[:200]})
            continue
        ids.append(track_id)
        vectors.append(vector)
        # Mean pairwise cosine between a track's own segments: low spread means the track
        # is internally uniform, high spread means pooling is averaging away real contrast.
        if segments.shape[0] > 1:
            sims = segments @ segments.T
            off = ~np.eye(segments.shape[0], dtype=bool)
            spreads.append(float(sims[off].mean()))
        if i % 50 == 0:
            print(f"  {i}/{len(track_paths)}", file=sys.stderr)
    elapsed = time.perf_counter() - t0
    return np.stack(vectors), ids, spreads, failures, elapsed


def main() -> int:
    ap = argparse.ArgumentParser(description="Eval 0B — text->music retrieval")
    ap.add_argument("--encoder", default="clap")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--n-segments", type=int, default=1)
    ap.add_argument("--label", default=None)
    ap.add_argument("--valid-only", action="store_true",
                    help="restrict to the human-validated caption subset")
    ap.add_argument("--limit-tracks", type=int, default=None, help="quick pass")
    args = ap.parse_args()

    audio_root = SDD / "audio"
    if not audio_root.is_dir():
        print(f"missing benchmark audio at {audio_root}\n"
              "run: python scripts/fetch_song_describer.py --audio --extract", file=sys.stderr)
        return 1

    rows = load_captions(args.valid_only)
    track_paths: dict[str, Path] = {}
    for r in rows:
        p = _resolve_audio(audio_root, r["path"])
        if p is not None:
            track_paths.setdefault(r["track_id"], p)
    if args.limit_tracks:
        track_paths = dict(list(track_paths.items())[: args.limit_tracks])
    rows = [r for r in rows if r["track_id"] in track_paths]
    if not rows:
        print("no captions with matching audio on disk", file=sys.stderr)
        return 1

    encoder = build_encoder(args.encoder, args.checkpoint)
    label = args.label or f"{encoder.name}_{args.n_segments}seg" + ("_valid" if args.valid_only else "")
    print(f"{encoder.name} {encoder.version} on {encoder.device}; "
          f"{len(track_paths)} tracks, {len(rows)} captions, n_segments={args.n_segments}",
          file=sys.stderr)

    V, ids, spreads, failures, index_seconds = embed_tracks(encoder, track_paths, args.n_segments)
    index_of = {t: i for i, t in enumerate(ids)}
    rows = [r for r in rows if r["track_id"] in index_of]

    t0 = time.perf_counter()
    T = l2_normalise(encoder.embed_text([r["caption"] for r in rows]))
    query_seconds = time.perf_counter() - t0

    scores = T @ V.T
    order = np.argsort(-scores, axis=1)
    truth = np.array([index_of[r["track_id"]] for r in rows])
    ranks = ranks_of_truth(scores, truth)

    summary = retrieval_metrics(ranks, len(ids))
    payload = {
        "eval": "0B",
        "label": label,
        "run_at": datetime.now(timezone.utc).isoformat(),
        "platform": f"{platform.system()} {platform.machine()} py{platform.python_version()}",
        "encoder": {"name": encoder.name, "version": encoder.version,
                    "device": encoder.device, "dim": encoder.embedding_dim,
                    "sample_rate": encoder.sample_rate,
                    "segment_seconds": encoder.segment_seconds},
        "config": {"n_segments": args.n_segments, "valid_only": args.valid_only},
        "metrics": summary,
        "random_baseline": random_baseline(len(ids)),
        "diagnostics": space_diagnostics(V, order[:, :10], spreads),
        "cost": {
            "index_seconds": index_seconds,
            "seconds_per_track": index_seconds / max(1, len(ids)),
            "query_seconds_total": query_seconds,
            "ms_per_query": query_seconds * 1000 / max(1, len(rows)),
            "embedding_bytes_per_track": int(V.dtype.itemsize * V.shape[1]),
        },
        # Per-query ranks are persisted so two runs can be compared with a *paired*
        # test. Aggregate Recall@K alone cannot distinguish a real improvement from
        # sampling noise on 1,106 queries, and E1 is a decision, not a readout.
        "per_query": [
            {"caption_id": r["caption_id"], "track_id": r["track_id"], "rank": int(rank)}
            for r, rank in zip(rows, ranks)
        ],
        "failures": failures,
    }

    EVALS_DIR.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    json_path = EVALS_DIR / f"eval_0b_{label}_{stamp}.json"
    json_path.write_text(json.dumps(payload, indent=2))

    m, b, d, c = summary, payload["random_baseline"], payload["diagnostics"], payload["cost"]
    lines = [
        f"# Eval 0B — text→music retrieval ({label})", "",
        f"- Encoder: `{encoder.name}` / `{encoder.version}` on {encoder.device}",
        f"- Segments per track: {args.n_segments}",
        f"- Captions: {m['n_queries']} · Candidate tracks: {m['n_candidates']}"
        + (" · human-validated subset only" if args.valid_only else ""),
        f"- Failed to embed: {len(failures)}", "",
        "## Retrieval", "",
        "| Metric | Measured | Random | Lift |",
        "|---|---:|---:|---:|",
        f"| Recall@1 | {m['recall@1']:.3f} | {b['recall@1']:.4f} | {m['recall@1'] / b['recall@1']:.0f}× |",
        f"| Recall@5 | {m['recall@5']:.3f} | {b['recall@5']:.4f} | {m['recall@5'] / b['recall@5']:.0f}× |",
        f"| Recall@10 | {m['recall@10']:.3f} | {b['recall@10']:.4f} | {m['recall@10'] / b['recall@10']:.0f}× |",
        f"| Median rank | {m['median_rank']:.1f} | {b['median_rank']:.1f} | — |",
        f"| MRR | {m['mrr']:.4f} | — | — |", "",
        "## Diagnostics", "",
        f"- track–track cosine: mean {d['track_cosine_mean']:.3f}, "
        f"sd {d['track_cosine_std']:.3f}, p95 {d['track_cosine_p95']:.3f}",
        f"- hubness: top 1% of tracks take {d['hubness_top1pct_share']:.1%} "
        f"of all top-10 slots; single largest {d['hubness_max_track_share']:.1%}",
        f"- tracks never in any top-10: {d['tracks_never_retrieved']}",
        f"- within-track segment cosine: {d['within_track_segment_cosine_mean']}"
        if d["within_track_segment_cosine_mean"] is not None
        else "- within-track segment cosine: n/a (single segment)",
        "", "## Cost", "",
        f"- indexing: {c['seconds_per_track']:.2f} s/track ({c['index_seconds']:.0f} s total)",
        f"- query: {c['ms_per_query']:.1f} ms/query",
        f"- storage: {c['embedding_bytes_per_track']} bytes/track", "",
        f"Raw: `{json_path.name}`", "",
    ]
    md_path = EVALS_DIR / f"eval_0b_{label}_{stamp}.md"
    md_path.write_text("\n".join(lines))
    print("\n" + md_path.read_text())
    print(f"wrote {md_path} and {json_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
