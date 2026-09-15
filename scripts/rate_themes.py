"""Blind spot-check of the model-generated theme labels, and the agreement they earn.

`scripts/label_themes.py` labels every transcript with a model. Those labels are the ruler
the semantic evaluation is measured with, and an unchecked ruler is worth nothing, so this
samples pairs, asks a human, and reports agreement. Every number the semantic benchmark
produces should be quoted with the kappa from here next to it.

Blind. The model's own label is never shown, and the sample is drawn half from pairs the
model marked true and half from pairs it marked false. Rating only the positives would
measure precision and call it agreement, which flatters a labeller that marks everything.

Cohen's kappa is reported rather than raw agreement because these classes are unbalanced:
most (track, theme) pairs are false, so a labeller that always said "no" would score high
raw agreement and a kappa near zero.

    python scripts/rate_themes.py            # rate a fresh sample
    python scripts/rate_themes.py --report   # agreement so far, no prompting
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from aux.data import load_personal_tracks  # noqa: E402
from aux.ingest.asset import content_hash  # noqa: E402
from label_themes import THEMES, load_transcripts  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / ".cache"
EVALS = ROOT / "evals"
#: Private: pairs a rating to a transcript from the user's own library.
ANSWERS = EVALS / "theme_ratings.private.json"

EXCERPT_CHARS = 1200
DEFAULT_SAMPLE = 40
SEED = 20260913


def sample_pairs(labels: dict, transcripts: dict[str, str], n: int, seed: int = SEED):
    """Half pairs the model called true, half it called false.

    Balanced on the model's own answer, never on the human's, so the sample is drawn without
    reference to the thing being measured.
    """
    positives, negatives = [], []
    for h, rec in sorted(labels.items()):
        if h not in transcripts or not rec.get("confident"):
            continue
        marked = set(rec.get("themes", []))
        for theme in sorted(THEMES):
            (positives if theme in marked else negatives).append((h, theme))

    rng = random.Random(seed)
    rng.shuffle(positives)
    rng.shuffle(negatives)

    # Aim for half and half, but top up from the other pool when one is short rather than
    # returning a smaller sample: a thinly labelled theme set would otherwise quietly
    # shrink the check at exactly the point where it matters most.
    take_pos = min(n // 2, len(positives))
    take_neg = min(n - take_pos, len(negatives))
    take_pos = min(len(positives), take_pos + (n - take_pos - take_neg))

    pairs = positives[:take_pos] + negatives[:take_neg]
    rng.shuffle(pairs)
    return pairs


def cohen_kappa(pairs: list[tuple[bool, bool]]) -> tuple[float, float]:
    """Returns (kappa, raw agreement) for a list of (model, human) judgements."""
    n = len(pairs)
    if n == 0:
        return 0.0, 0.0
    agree = sum(a == b for a, b in pairs) / n
    p_model = sum(a for a, _ in pairs) / n
    p_human = sum(b for _, b in pairs) / n
    chance = p_model * p_human + (1 - p_model) * (1 - p_human)
    kappa = (agree - chance) / (1 - chance) if chance < 1 else 0.0
    return float(kappa), float(agree)


def report(labels: dict) -> int:
    if not ANSWERS.exists():
        print("no ratings yet, run scripts/rate_themes.py", file=sys.stderr)
        return 1
    answers = json.loads(ANSWERS.read_text())
    judged = [(theme in labels.get(h, {}).get("themes", []), bool(v))
              for key, v in answers.items()
              for h, theme in [key.split("|")]]
    kappa, agree = cohen_kappa(judged)
    model_yes = sum(a for a, _ in judged)
    human_yes = sum(b for _, b in judged)
    print(f"{len(judged)} pairs rated")
    print(f"  model said yes: {model_yes}")
    print(f"  human said yes: {human_yes}")
    print(f"  raw agreement:  {agree:.2f}")
    print(f"  Cohen's kappa:  {kappa:.2f}")
    verdict = ("substantial" if kappa >= 0.6 else
               "moderate" if kappa >= 0.4 else "weak, labels are not trustworthy")
    print(f"  interpretation: {verdict}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Blind spot-check of theme labels")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--n", type=int, default=DEFAULT_SAMPLE)
    args = ap.parse_args()

    labels = json.loads((CACHE / "themes.json").read_text())
    if args.report:
        return report(labels)

    transcripts = load_transcripts()
    # Restrict to tracks in the library, so nothing from another corpus is rated.
    library = {content_hash(t.path) for t in load_personal_tracks()}
    labels = {h: rec for h, rec in labels.items() if h in library}

    EVALS.mkdir(exist_ok=True)
    answers = json.loads(ANSWERS.read_text()) if ANSWERS.exists() else {}
    pairs = [p for p in sample_pairs(labels, transcripts, args.n) if f"{p[0]}|{p[1]}" not in answers]
    if not pairs:
        print("sample already fully rated\n")
        return report(labels)

    print(f"\n{len(pairs)} pairs to rate. y = yes, n = no, s = skip, q = save and quit.")
    print("The model's own answer is hidden on purpose.\n")
    for i, (h, theme) in enumerate(pairs, 1):
        excerpt = " ".join(transcripts[h].split())[:EXCERPT_CHARS]
        print("=" * 72)
        print(f"[{i}/{len(pairs)}]  Is this song about: {THEMES[theme].upper()}?\n")
        print(excerpt + ("..." if len(transcripts[h]) > EXCERPT_CHARS else ""))
        print()
        while True:
            reply = input("  y / n / s / q > ").strip().lower()
            if reply in {"y", "n", "s", "q"}:
                break
        if reply == "q":
            break
        if reply == "s":
            continue
        answers[f"{h}|{theme}"] = reply == "y"
        ANSWERS.write_text(json.dumps(answers, indent=2, sort_keys=True))

    ANSWERS.write_text(json.dumps(answers, indent=2, sort_keys=True))
    print(f"\nsaved {len(answers)} ratings to {ANSWERS.relative_to(ROOT)}\n")
    return report(labels)


if __name__ == "__main__":
    raise SystemExit(main())
