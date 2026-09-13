"""Weak-label each transcript with the lyrical themes it is about.

The recommendation evaluation scores against genre, artist and album. All three are
*acoustic* constructs, so they ask "do lyrics help identify the artist?" — and mostly they
do not, which is why fusion loses there. That result says nothing about the queries a lyric
channel exists to serve: "songs about heartbreak", "songs about proving people wrong". This
script builds the labels needed to measure those.

**Why a model and not a keyword list.** Matching keywords against the transcript would label
a track using the exact text the lyric encoder reads, biasing the benchmark toward the system
under test. A separate model reading for meaning is not independent either, but its errors
are far less aligned with cosine similarity in Qwen3 space. The honest check is human
agreement, which `scripts/rate_themes.py` measures on a blind sample and which is reported
alongside every number these labels produce.

**Privacy.** Only transcript text leaves the machine — never audio, never a file path, never
a filename, never the library listing. Each request carries one transcript and nothing that
links it to a person or to the rest of the collection. This is a wider concession than the
planner's (DEC-007, query strings only) and is recorded as such.

    python scripts/label_themes.py            # labels, caching by content hash
    python scripts/label_themes.py --summary  # coverage per theme, no API calls
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aux.data import load_personal_tracks  # noqa: E402
from aux.ingest.asset import content_hash  # noqa: E402
from aux.plan.claude import load_api_key  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / ".cache"
THEMES_PATH = CACHE / "themes.json"

MODEL = "claude-sonnet-5"
"""Labelling is the measurement instrument, so accuracy matters more than latency here —
the opposite tradeoff from the planner, which runs inside a search and uses Haiku."""

#: Themes chosen to be semantically distinct and *acoustically* non-obvious: a system that
#: only hears production should not be able to guess them. Deliberately excludes themes that
#: map onto a genre ("songs about DJing"), which would let the audio channel win by
#: recognising the genre instead of understanding the query.
THEMES: dict[str, str] = {
    "heartbreak": "heartbreak, being left, or the end of a relationship",
    "money": "money, wealth, luxury or material success",
    "loyalty": "loyalty, betrayal, or trust between friends",
    "longing": "missing someone, distance, or waiting for someone",
    "nightlife": "partying, clubs, drinking or going out at night",
    "grief": "grief, death, or losing someone permanently",
    "doubt": "self-doubt, depression, anxiety or inner struggle",
    "ambition": "ambition, hustle, or proving doubters wrong",
    "desire": "physical attraction, lust or seduction",
    "roots": "hometown, family, childhood or where someone is from",
    "substances": "drugs, alcohol or getting high",
    "faith": "God, prayer, faith or the spiritual",
}

MAX_TRANSCRIPT_CHARS = 4000

SYSTEM = """You label song lyrics with the themes they are about.

You will be given a transcript of one song's lyrics. It comes from automatic speech
recognition, so expect mistakes, repeated lines and missing words.

Return every theme the song is genuinely about. Judge the song as a whole: a single passing
mention is not a theme. Most songs have one to three. Return an empty list if the transcript
is too garbled or too sparse to judge, rather than guessing."""

SCHEMA = {
    "type": "object",
    "properties": {
        "themes": {"type": "array", "items": {"type": "string", "enum": sorted(THEMES)}},
        "confident": {"type": "boolean",
                      "description": "false when the transcript is too poor to judge"},
    },
    "required": ["themes", "confident"],
    "additionalProperties": False,
}


def theme_prompt(transcript: str) -> str:
    catalogue = "\n".join(f"- {key}: {desc}" for key, desc in sorted(THEMES.items()))
    return (f"Themes:\n{catalogue}\n\n"
            f"Lyrics:\n{transcript[:MAX_TRANSCRIPT_CHARS]}")


def answer_text(response) -> str:
    """The first text block of a response.

    Not `content[0]`: the model may emit a thinking block first, and indexing blindly into
    position zero reads that instead of the answer.
    """
    for block in response.content:
        if getattr(block, "type", None) == "text":
            return block.text
    raise ValueError("response contained no text block")


def load_transcripts(whisper_model: str = "small") -> dict[str, str]:
    """Reliable transcripts only, keyed by content hash."""
    path = CACHE / f"transcripts_{whisper_model}.json"
    records = json.loads(path.read_text())
    return {k.split("|")[0]: v["text"] for k, v in records.items() if v.get("reliable")}


def summarise(labels: dict) -> None:
    counts: dict[str, int] = {t: 0 for t in THEMES}
    confident = 0
    for rec in labels.values():
        confident += bool(rec.get("confident"))
        for theme in rec.get("themes", []):
            counts[theme] = counts.get(theme, 0) + 1
    print(f"{len(labels)} tracks labelled, {confident} confidently\n")
    print(f"{'theme':14}{'tracks':>8}")
    print("-" * 22)
    for theme, n in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"{theme:14}{n:>8}")
    thin = [t for t, n in counts.items() if n < 5]
    if thin:
        print(f"\nunder 5 tracks, excluded from evaluation: {', '.join(sorted(thin))}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Weak-label transcripts with lyrical themes")
    ap.add_argument("--summary", action="store_true", help="report coverage, no API calls")
    ap.add_argument("--whisper", default="small")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    labels = json.loads(THEMES_PATH.read_text()) if THEMES_PATH.exists() else {}
    if args.summary:
        summarise(labels)
        return 0

    transcripts = load_transcripts(args.whisper)
    tracks = load_personal_tracks()
    wanted = [(content_hash(t.path)) for t in tracks]
    todo = [h for h in wanted if h in transcripts and h not in labels]
    if args.limit:
        todo = todo[: args.limit]
    print(f"{len(wanted)} tracks, {len(transcripts)} with transcripts, "
          f"{len(labels)} already labelled, {len(todo)} to do", file=sys.stderr)
    if not todo:
        summarise(labels)
        return 0

    import anthropic

    key = load_api_key()
    if not key:
        print("no ANTHROPIC_API_KEY in the environment or .env", file=sys.stderr)
        return 1
    client = anthropic.Anthropic(api_key=key)

    failures = 0
    for i, h in enumerate(todo, 1):
        try:
            response = client.messages.create(
                model=MODEL, max_tokens=1000, system=SYSTEM,
                output_config={"format": {"type": "json_schema", "schema": SCHEMA}},
                messages=[{"role": "user", "content": theme_prompt(transcripts[h])}],
            )
            labels[h] = json.loads(answer_text(response))
        except Exception as exc:  # noqa: BLE001 - one bad track must not lose the run
            failures += 1
            print(f"  {h[:8]}: {type(exc).__name__}: {exc}", file=sys.stderr)
        if i % 10 == 0 or i == len(todo):
            THEMES_PATH.write_text(json.dumps(labels, indent=2, sort_keys=True))
            print(f"  {i}/{len(todo)}", file=sys.stderr)
    if failures:
        print(f"{failures} tracks failed and are simply unlabelled; rerun to retry them",
              file=sys.stderr)

    THEMES_PATH.write_text(json.dumps(labels, indent=2, sort_keys=True))
    summarise(labels)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
