"""Expand the router query set with paraphrases, so the comparison has enough power.

The first router comparison used 24 hand-written queries and found nothing significant: the
leading router was +0.058 NDCG@10 with p=0.066 against a Bonferroni threshold of 0.0125. That
is an underpowered test, not a null result, and the fix is more queries rather than a softer
claim.

**Paraphrases are generated deliberately against the rule router.** The rule baseline keys on
phrasings — "songs about", genre nouns, tempo words — so the model is told to avoid exactly
those constructions and to write the way people actually type. Every paraphrase therefore
makes the test harder for the router most likely to win, which is the direction a bias should
point. Paraphrases are generated from the label specification alone, never from any router's
behaviour, and are committed so the set is fixed.

    python scripts/build_routing_queries.py --per-spec 3
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aux.plan.claude import load_api_key  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
QUERIES = ROOT / "queries"
SOURCE = QUERIES / "routing_queries.json"
EXPANDED = QUERIES / "routing_queries_expanded.json"

MODEL = "claude-sonnet-5"

SYSTEM = """You rewrite music search queries the way different people would type them.

Given one query, write alternative phrasings that ask for exactly the same thing.

Rules:
- Keep the meaning identical. If the original asks for a genre, every rewrite must still ask
  for that genre; if it asks what songs are about, every rewrite must still ask that.
- Vary the construction. Do NOT reuse the opening of the original. In particular, avoid
  starting with "songs about" unless the original did not, and vary how the request is framed
  -- a question, a fragment, an instruction, a description of a moment.
- Write how people actually type into a search box: lowercase is fine, short is fine.
- No track names, no artist names."""

SCHEMA = {
    "type": "object",
    "properties": {"rewrites": {"type": "array", "items": {"type": "string"}}},
    "required": ["rewrites"],
    "additionalProperties": False,
}


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate paraphrases for the router query set")
    ap.add_argument("--per-spec", type=int, default=3)
    args = ap.parse_args()

    source = json.loads(SOURCE.read_text())
    existing = json.loads(EXPANDED.read_text())["queries"] if EXPANDED.exists() else []
    done = {q["source"] for q in existing if "source" in q}

    import anthropic

    key = load_api_key()
    if not key:
        print("no ANTHROPIC_API_KEY in the environment or .env", file=sys.stderr)
        return 1
    client = anthropic.Anthropic(api_key=key)

    out = list(existing)
    for spec in source["queries"]:
        # The original is always kept, tagged as its own source.
        if spec["query"] not in done:
            out.append({**spec, "source": spec["query"], "paraphrase": False})
            done.add(spec["query"])
        made = sum(1 for q in out if q.get("source") == spec["query"] and q.get("paraphrase"))
        if made >= args.per_spec:
            continue

        response = client.messages.create(
            model=MODEL, max_tokens=1000, system=SYSTEM,
            output_config={"format": {"type": "json_schema", "schema": SCHEMA}},
            messages=[{"role": "user", "content":
                       f"Write {args.per_spec} rewrites of: {spec['query']}"}],
        )
        text = next(b.text for b in response.content if getattr(b, "type", None) == "text")
        for rewrite in json.loads(text)["rewrites"][: args.per_spec]:
            out.append({"query": rewrite, "family": spec["family"], "genre": spec["genre"],
                        "theme": spec["theme"], "source": spec["query"], "paraphrase": True})
        print(f"  {spec['query'][:46]:48} +{args.per_spec}", file=sys.stderr)

    EXPANDED.write_text(json.dumps({
        "_comment": ("Router query set: the hand-written originals plus model paraphrases "
                     "generated from the label specification alone. Paraphrases are "
                     "instructed to avoid the constructions the rule router keys on, so "
                     "they make the test harder for it rather than easier."),
        "queries": out,
    }, indent=2))
    families: dict[str, int] = {}
    for q in out:
        families[q["family"]] = families.get(q["family"], 0) + 1
    print(f"\n{len(out)} queries: " + ", ".join(f"{k} {v}" for k, v in sorted(families.items())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
