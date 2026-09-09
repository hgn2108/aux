"""The structured interpretation a planner must produce — DEC-007's schema.

Two things come out of a plan, serving different purposes:

- **`rewritten`** drives retrieval today. It restates the query in the descriptive,
  sound-naming language the encoder was trained on, which is the mechanism DEC-013 proposed
  and the fixed lexicon failed to deliver generically.
- **the facets** are the structured decomposition DEC-007 committed to. They are what makes
  the planner *evaluable* field by field (Eval 2A) rather than only end to end, and they are
  what a later routing step would consume.

Validation is strict and total: a plan that does not parse is a measured failure with a
recorded rate, never a silent fallback to something half-formed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

MAX_ITEMS = 12
MAX_STR = 300


class PlanValidationError(ValueError):
    """The model returned something the schema does not accept."""


@dataclass(frozen=True, slots=True)
class QueryPlan:
    """A parsed, validated interpretation of one query."""

    original: str
    rewritten: str
    """The query restated as a description of sound. Drives retrieval."""
    acoustic: tuple[str, ...] = ()
    """Instrumentation, production, tempo, texture."""
    mood: tuple[str, ...] = ()
    genre: tuple[str, ...] = ()
    context: tuple[str, ...] = ()
    """Situations named in the query — "for running", "late night"."""
    exclude: tuple[str, ...] = ()
    """Concepts to push away from, fed to the existing score-level negation."""
    lyrical: tuple[str, ...] = ()
    """Thematic or narrative content. Recorded but unused: Slice 1 showed the audio encoder
    cannot isolate it ("r&b songs about yearning" matched everything moderately), and it is
    Slice 3's subject. Captured now so the planner's output does not need reshaping later."""
    raw: dict = field(default_factory=dict, compare=False, repr=False)

    def as_dict(self) -> dict:
        d = asdict(self)
        d.pop("raw", None)
        return d


def _clean_list(value, name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        raise PlanValidationError(f"{name} must be a list of strings, got {type(value).__name__}")
    out = []
    for item in value[:MAX_ITEMS]:
        if not isinstance(item, str):
            raise PlanValidationError(f"{name} contains a non-string: {item!r}")
        item = item.strip()
        if item:
            out.append(item[:MAX_STR])
    return tuple(out)


def validate(payload: dict, original: str) -> QueryPlan:
    """Turn a model's JSON into a QueryPlan, or raise.

    `rewritten` is the one required field, because it is the only one retrieval depends on.
    An empty rewrite falls back to the original query rather than searching for nothing —
    a planner that declines to rewrite should degrade to the baseline, not to noise.
    """
    if not isinstance(payload, dict):
        raise PlanValidationError(f"expected a JSON object, got {type(payload).__name__}")

    rewritten = payload.get("rewritten")
    if rewritten is not None and not isinstance(rewritten, str):
        raise PlanValidationError("rewritten must be a string")
    rewritten = (rewritten or "").strip()[:MAX_STR] or original

    return QueryPlan(
        original=original,
        rewritten=rewritten,
        acoustic=_clean_list(payload.get("acoustic"), "acoustic"),
        mood=_clean_list(payload.get("mood"), "mood"),
        genre=_clean_list(payload.get("genre"), "genre"),
        context=_clean_list(payload.get("context"), "context"),
        exclude=_clean_list(payload.get("exclude"), "exclude"),
        lyrical=_clean_list(payload.get("lyrical"), "lyrical"),
        raw=payload,
    )


def passthrough(query: str) -> QueryPlan:
    """The plan meaning "no interpretation" — used when a planner fails.

    Retrieval on this is byte-identical to the Slice 1 baseline, so a planner outage
    degrades to measured behaviour rather than to something unknown.
    """
    return QueryPlan(original=query, rewritten=query)
