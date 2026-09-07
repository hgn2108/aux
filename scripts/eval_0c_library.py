"""Eval 0C -- out-of-domain test on a personal library.

Slice 0's remaining pass condition: "reference-track retrieval produces plausible
neighbours on a personal library that was never used for development."

This is the condition the whole design leans on. Song Describer is Creative-Commons
catalogue audio; a personal library is commercially mastered, loudness-normalised and
skewed to one person's taste. An encoder can score well on the benchmark and be useless
here, and that failure was named in advance as the most likely way Slice 0 breaks.

The output has two halves, kept apart on purpose:

- **Quantitative and objective** -- does the embedding space *collapse* on this audio?
  Compared directly against the same encoder's spread on the benchmark corpus, because the
  absolute number means little without that reference. Plus hubness, the other named risk.
- **Qualitative and for Irene** -- neighbour listings and text-query results. There is no
  ground truth for "correct neighbour", so this half is evidence to judge, never a score.

    python scripts/eval_0c_library.py data/music --encoder muq --n-segments 5
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aux.encode.base import l2_normalise  # noqa: E402
from aux.eval import space_diagnostics  # noqa: E402
from aux.ingest import IngestError, decode, discover  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
EVALS_DIR = ROOT / "evals"

PROBE_QUERIES = [
    "aggressive hard-hitting trap with heavy 808s",
    "melodic and melancholy, sung rather than rapped",
    "slow smooth late-night R&B",
    "upbeat energetic party track",
    "dreamy atmospheric production with reverb",
    "stripped back and minimal, few instruments",
    "dark and menacing",
    "warm and nostalgic",
]
"""Probe queries, not a benchmark.

Deliberately spanning the axes DESIGN.md cares about -- acoustic, mood, production,
energy -- so the listing shows whether the space responds to *different kinds* of language,
not only whether it responds at all. Eight queries prove nothing statistically; they exist
to be read.
"""


def artist_of(path: Path) -> str | None:
    """Best-effort artist from a scraped filename ("Artist - Title - Uploader")."""
    stem = re.sub(r"\(youtube\)$", "", path.stem).strip()
    if " - " not in stem:
        return None
    artist = stem.split(" - ")[0].strip().lower()
    artist = re.sub(r"[^a-z0-9 &.]+", "", artist).strip()
    return artist or None


def same_artist_structure(paths: list[Path], order: np.ndarray, k: int) -> dict:
    """Do same-artist tracks retrieve each other above chance?

    This is the test that separates the two explanations for a high mean pairwise cosine:

    - **collapse** -- the encoder cannot separate this audio, so ranking is meaningless and
      same-artist tracks land no better than chance;
    - **homogeneity** -- the library genuinely is one narrow slice of music, so everything
      is similar to everything *and the ranking is still informative*.

    Artist identity is free ground truth already sitting in the filenames. It is a weak
    proxy for musical similarity -- artists vary across their own catalogue -- so a modest
    lift is expected rather than a large one. Its value is the direction, not the size.
    """
    artists = [artist_of(p) for p in paths]
    counts: dict[str, int] = {}
    for a in artists:
        if a:
            counts[a] = counts.get(a, 0) + 1

    eligible = [i for i, a in enumerate(artists) if a and counts[a] > 1]
    if not eligible:
        return {"eligible_tracks": 0}

    hits, chance_sum = 0, 0.0
    n = len(paths)
    for i in eligible:
        top = order[i, :k]
        if any(artists[j] == artists[i] for j in top):
            hits += 1
        # P(at least one of k draws from the other n-1 tracks shares the artist)
        others = counts[artists[i]] - 1
        miss = 1.0
        for step in range(k):
            miss *= max(0.0, (n - 1 - others - step)) / max(1, (n - 1 - step))
        chance_sum += 1 - miss

    observed = hits / len(eligible)
    expected = chance_sum / len(eligible)
    return {
        "eligible_tracks": len(eligible),
        "k": k,
        "observed_rate": observed,
        "chance_rate": expected,
        "lift": (observed / expected) if expected else None,
        "artists_with_multiple_tracks": sum(1 for c in counts.values() if c > 1),
    }


def genre_of(path: Path, root: Path) -> str:
    """Genre label from the folder a track sits in.

    Files at the root belong to the original library, which predates the genre folders and
    was left in place rather than refiled -- the label is derived here instead.
    """
    rel = path.relative_to(root)
    return rel.parts[0] if len(rel.parts) > 1 else "hiphop_rnb"


def genre_structure(labels: list[str], sims: np.ndarray, order: np.ndarray, k: int) -> dict:
    """Within-genre vs between-genre similarity, and genre purity of the top-k.

    A far sharper collapse test than one global mean cosine. A high overall similarity is
    expected in a narrow library; what a *working* space must also show is that tracks sit
    closer to their own genre than to others, and that a track's nearest neighbours mostly
    share its genre. A collapsed space cannot do that regardless of how the global mean
    happens to land.

    Genre is a coarse proxy -- an acoustic jazz ballad and a horn-led uptempo number share
    a label while sounding little alike -- so purity well below 1.0 is expected and is not
    itself a failure.
    """
    arr = np.array(labels)
    n = len(labels)
    same = arr[:, None] == arr[None, :]
    off = ~np.eye(n, dtype=bool)

    within = float(sims[same & off].mean()) if (same & off).any() else float("nan")
    between = float(sims[~same].mean()) if (~same).any() else float("nan")

    per_genre = {}
    for g in sorted(set(labels)):
        idx = np.flatnonzero(arr == g)
        if idx.size < 2:
            continue
        block = sims[np.ix_(idx, idx)]
        blk_off = ~np.eye(idx.size, dtype=bool)
        others = np.flatnonzero(arr != g)
        purity = float(np.mean([np.mean(arr[order[i, :k]] == g) for i in idx]))
        per_genre[g] = {
            "n": int(idx.size),
            "within_cosine": float(block[blk_off].mean()),
            "between_cosine": float(sims[np.ix_(idx, others)].mean()) if others.size else None,
            "top_k_purity": purity,
            "chance_purity": float((idx.size - 1) / (n - 1)),
        }

    overall_purity = float(np.mean([per_genre[g]["top_k_purity"] for g in per_genre]))
    overall_chance = float(np.mean([per_genre[g]["chance_purity"] for g in per_genre]))
    return {
        "within_cosine": within,
        "between_cosine": between,
        "separation": within - between,
        "top_k_purity": overall_purity,
        "chance_purity": overall_chance,
        "purity_lift": overall_purity / overall_chance if overall_chance else None,
        "per_genre": per_genre,
    }


def build_encoder(name: str, checkpoint: str | None):
    if name == "clap":
        from aux.encode.clap import DEFAULT_CHECKPOINT, ClapAdapter

        return ClapAdapter(checkpoint or DEFAULT_CHECKPOINT)
    if name in {"muq", "muq-mulan"}:
        from aux.encode.muq import DEFAULT_CHECKPOINT, MuQMuLanAdapter

        return MuQMuLanAdapter(checkpoint or DEFAULT_CHECKPOINT)
    raise ValueError(f"unknown encoder {name!r}")


def short(path: Path) -> str:
    """Trim scraped filename noise so a neighbour listing is readable."""
    stem = path.stem
    for marker in (" (Official", " [Official", " (Audio", " (Lyric", " (youtube"):
        idx = stem.find(marker)
        if idx > 0:
            stem = stem[:idx]
    return stem.strip()[:64]


def main() -> int:
    ap = argparse.ArgumentParser(description="Eval 0C — personal-library out-of-domain test")
    ap.add_argument("root", type=Path)
    ap.add_argument("--encoder", default="muq")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--n-segments", type=int, default=5)
    ap.add_argument("--neighbours", type=int, default=5)
    ap.add_argument("--examples", type=int, default=12, help="reference tracks to list")
    ap.add_argument("--reference-json", type=Path, default=None,
                    help="an Eval 0B run to compare the cosine distribution against")
    args = ap.parse_args()

    encoder = build_encoder(args.encoder, args.checkpoint)
    paths = [f.path for f in discover(args.root)]
    print(f"{encoder.name} {encoder.version} on {encoder.device}; {len(paths)} tracks",
          file=sys.stderr)

    vectors, kept, spreads = [], [], []
    for i, path in enumerate(paths, 1):
        try:
            vec, segs = encoder.embed_track(decode(path), n_segments=args.n_segments)
        except (IngestError, Exception) as exc:  # noqa: BLE001
            print(f"  skip {path.name}: {exc}"[:120], file=sys.stderr)
            continue
        vectors.append(vec)
        kept.append(path)
        if segs.shape[0] > 1:
            off = ~np.eye(segs.shape[0], dtype=bool)
            spreads.append(float((segs @ segs.T)[off].mean()))
        if i % 20 == 0:
            print(f"  {i}/{len(paths)}", file=sys.stderr)

    V = np.stack(vectors)
    sims = V @ V.T
    np.fill_diagonal(sims, -np.inf)
    order = np.argsort(-sims, axis=1)
    diag = space_diagnostics(V, order[:, : args.neighbours], spreads)

    reference = None
    if args.reference_json and args.reference_json.exists():
        ref = json.loads(args.reference_json.read_text())
        reference = {
            "label": ref["label"],
            "track_cosine_mean": ref["diagnostics"]["track_cosine_mean"],
            "track_cosine_std": ref["diagnostics"]["track_cosine_std"],
        }

    neighbours = []
    step = max(1, len(kept) // max(1, args.examples))
    for i in range(0, len(kept), step):
        neighbours.append({
            "track": short(kept[i]),
            "neighbours": [
                {"track": short(kept[j]), "cosine": round(float(sims[i, j]), 3)}
                for j in order[i, : args.neighbours]
            ],
        })

    T = l2_normalise(encoder.embed_text(PROBE_QUERIES))
    query_scores = T @ V.T
    query_results = [
        {"query": q,
         "top": [{"track": short(kept[j]), "score": round(float(query_scores[qi, j]), 3)}
                 for j in np.argsort(-query_scores[qi])[:5]]}
        for qi, q in enumerate(PROBE_QUERIES)
    ]

    artist_test = same_artist_structure(kept, order, args.neighbours)
    labels = [genre_of(p, args.root) for p in kept]
    genre_test = genre_structure(labels, sims, order, args.neighbours)

    # Pseudonymous, stable, and genre-bearing: "classical/t003". This keeps the committed
    # evidence scientifically complete -- neighbour structure and genre membership are the
    # findings, not the titles -- while never publishing what Irene listens to.
    alias = {}
    per_genre_count: dict[str, int] = {}
    for path, label in zip(kept, labels):
        per_genre_count[label] = per_genre_count.get(label, 0) + 1
        alias[short(path)] = f"{label}/t{per_genre_count[label]:03d}"

    def anonymise(obj):
        if isinstance(obj, dict):
            return {k: (alias.get(v, v) if k == "track" else anonymise(v)) for k, v in obj.items()}
        if isinstance(obj, list):
            return [anonymise(v) for v in obj]
        return obj

    payload = {
        "eval": "0C",
        "same_artist_structure": artist_test,
        "genre_structure": genre_test, "run_at": datetime.now(timezone.utc).isoformat(),
        "encoder": {"name": encoder.name, "version": encoder.version,
                    "device": encoder.device, "dim": encoder.embedding_dim},
        "config": {"n_segments": args.n_segments, "tracks": len(kept)},
        "diagnostics": diag, "benchmark_reference": reference,
        "neighbours": neighbours, "queries": query_results,
    }
    EVALS_DIR.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    private = EVALS_DIR / f"eval_0c_library_{encoder.name}_{stamp}.private.json"
    private.write_text(json.dumps(payload, indent=2))

    out = EVALS_DIR / f"eval_0c_library_{encoder.name}_{stamp}.json"
    public = dict(payload)
    public["neighbours"] = anonymise(payload["neighbours"])
    public["queries"] = anonymise(payload["queries"])
    public["note"] = ("Track identities are pseudonymised as <genre>/tNNN. Real titles are "
                      "in the .private.json alongside this file, which is gitignored: the "
                      "personal library is an out-of-domain test set and is never published.")
    out.write_text(json.dumps(public, indent=2))

    md = EVALS_DIR / f"eval_0c_library_{encoder.name}_{stamp}.md"
    g = genre_test
    lines = [
        f"# Eval 0C — personal-library out-of-domain test ({encoder.name})", "",
        f"- Encoder: `{encoder.version}` on {encoder.device}, {args.n_segments} segments",
        f"- Tracks: {len(kept)}", "",
        "Track identities are pseudonymised as `<genre>/tNNN`; the personal library is "
        "never published.", "",
        "## Space health", "",
        f"- track–track cosine: mean {diag['track_cosine_mean']:.3f}, "
        f"sd {diag['track_cosine_std']:.3f}, p95 {diag['track_cosine_p95']:.3f}",
    ]
    if reference:
        lines.append(f"- benchmark reference (`{reference['label']}`): "
                     f"mean {reference['track_cosine_mean']:.3f}, "
                     f"sd {reference['track_cosine_std']:.3f}")
    lines += [
        f"- hubness: top 1% take {diag['hubness_top1pct_share']:.1%} of neighbour slots; "
        f"{diag['tracks_never_retrieved']} tracks never a neighbour",
        "", "## Genre structure", "",
        f"Within-genre cosine {g['within_cosine']:.3f} vs between-genre "
        f"{g['between_cosine']:.3f} (separation {g['separation']:+.3f}). "
        f"Top-{args.neighbours} genre purity {g['top_k_purity']:.1%} vs "
        f"{g['chance_purity']:.1%} chance ({g['purity_lift']:.1f}×).", "",
        "| Genre | n | Within | Between | Purity | Chance |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, v in sorted(g["per_genre"].items(), key=lambda kv: -kv[1]["n"]):
        lines.append(f"| {name} | {v['n']} | {v['within_cosine']:.3f} | "
                     f"{v['between_cosine']:.3f} | {v['top_k_purity']:.1%} | "
                     f"{v['chance_purity']:.1%} |")
    if artist_test.get("eligible_tracks"):
        lines += ["", "## Same-artist structure", "",
                  f"Same artist in top {artist_test['k']}: "
                  f"{artist_test['observed_rate']:.1%} observed vs "
                  f"{artist_test['chance_rate']:.1%} chance "
                  f"({artist_test['lift']:.1f}×), over {artist_test['eligible_tracks']} "
                  f"tracks with a same-artist counterpart.", "",
                  "Free ground truth from filenames. It is what separates a self-similar "
                  "library from a collapsed embedding space: a collapsed space ranks "
                  "same-artist tracks at chance."]
    lines += ["", "## Text probe queries", "",
              "Probes, not a benchmark — eight queries prove nothing statistically; they "
              "exist to be read.", "",
              "| Query | Top result | Score |", "|---|---|---:|"]
    for r in public["queries"]:
        top = r["top"][0]
        lines.append(f"| {r['query']} | `{top['track']}` | {top['score']:.3f} |")
    lines += ["", f"Raw: `{out.name}` (pseudonymised) · `{private.name}` (local only)", ""]
    md.write_text("\n".join(lines))

    print(f"\n=== SPACE HEALTH ({len(kept)} tracks) ===")
    print(f"track–track cosine: mean {diag['track_cosine_mean']:.3f} "
          f"sd {diag['track_cosine_std']:.3f} p95 {diag['track_cosine_p95']:.3f}")
    if reference:
        print(f"  benchmark ({reference['label']}): mean {reference['track_cosine_mean']:.3f} "
              f"sd {reference['track_cosine_std']:.3f}")
        delta = diag["track_cosine_mean"] - reference["track_cosine_mean"]
        print(f"  delta vs benchmark: {delta:+.3f}")
    print(f"hubness: top 1% take {diag['hubness_top1pct_share']:.1%} of neighbour slots; "
          f"{diag['tracks_never_retrieved']} tracks never a neighbour")
    if diag["within_track_segment_cosine_mean"] is not None:
        print(f"within-track segment cosine: {diag['within_track_segment_cosine_mean']:.3f}")

    if artist_test.get("eligible_tracks"):
        print(f"\nsame-artist in top {artist_test['k']}: "
              f"{artist_test['observed_rate']:.1%} observed vs "
              f"{artist_test['chance_rate']:.1%} chance "
              f"({artist_test['lift']:.1f}x) over {artist_test['eligible_tracks']} tracks")

    g = genre_test
    print(f"\nwithin-genre cosine {g['within_cosine']:.3f} vs between-genre "
          f"{g['between_cosine']:.3f}  (separation {g['separation']:+.3f})")
    print(f"top-{args.neighbours} genre purity {g['top_k_purity']:.1%} vs "
          f"{g['chance_purity']:.1%} chance ({g['purity_lift']:.1f}x)")
    print(f"{'genre':14}{'n':>4}{'within':>8}{'between':>9}{'purity':>8}{'chance':>8}")
    for name, v in sorted(g["per_genre"].items(), key=lambda kv: -kv[1]["n"]):
        print(f"{name:14}{v['n']:>4}{v['within_cosine']:>8.3f}"
              f"{v['between_cosine']:>9.3f}{v['top_k_purity']:>8.1%}{v['chance_purity']:>8.1%}")

    # A raised mean cosine alone does NOT indicate collapse -- a genre-narrow library is
    # genuinely self-similar. The two are separated by whether *ranking* still carries
    # information, which is what the same-artist lift measures. Reporting the cosine delta
    # on its own would raise a false alarm on any focused library.
    lift = artist_test.get("lift")
    if reference is not None and lift is not None:
        raised = diag["track_cosine_mean"] - reference["track_cosine_mean"] > 0.25
        if lift < 1.5:
            verdict = "COLLAPSE — ranking carries little information"
        elif raised:
            verdict = "no collapse — self-similar library, but ranking is informative"
        else:
            verdict = "no collapse"
        print(f"verdict: {verdict}")

    print("\n=== NEIGHBOURS (for Irene to judge) ===")
    for entry in neighbours:
        print(f"\n{entry['track']}")
        for n in entry["neighbours"]:
            print(f"    {n['cosine']:.3f}  {n['track']}")

    print("\n=== TEXT QUERIES ===")
    for r in query_results:
        print(f"\n\"{r['query']}\"")
        for t in r["top"]:
            print(f"    {t['score']:.3f}  {t['track']}")
    print(f"\nreport: {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
