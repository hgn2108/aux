"""The planner contract.

One interface over a hosted model and a local one, so E2a, does a local model match the
hosted one on quality, latency and cost, is a config change rather than a rewrite. Same
pattern as `EncoderAdapter`, and for the same reason: a comparison is only clean when
everything except the thing being compared is held fixed.
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod

from .schema import PlanValidationError, QueryPlan, passthrough, validate

_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


def extract_json(text: str) -> dict:
    """Pull a JSON object out of a model response.

    Models sometimes wrap JSON in a code fence or add a sentence despite instructions. That
    is a formatting slip, not a refusal, so it is recovered rather than counted as failure  -
    but the recovery is narrow: the first outermost braces, parsed strictly.
    """
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.MULTILINE).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = _JSON_BLOCK.search(text)
    if not match:
        raise PlanValidationError(f"no JSON object in response: {text[:120]!r}")
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        raise PlanValidationError(f"malformed JSON: {exc}") from exc


class Planner(ABC):
    """Turns a free-form query into a validated QueryPlan."""

    name: str
    version: str

    @abstractmethod
    def _complete(self, query: str) -> tuple[str, dict]:
        """Return the raw model text and a usage/latency record."""

    def plan(self, query: str, *, retries: int = 1) -> tuple[QueryPlan, dict]:
        """Plan one query, falling back to the baseline rather than to noise.

        A schema violation is retried once, models mostly recover on a second attempt  -
        and then degrades to `passthrough`, which retrieves identically to Slice 1. The
        fallback is counted: Eval 2A reports the rate, because a planner that quietly
        fails half the time while scoring well on the half that works is not a planner.
        """
        meta: dict = {"planner": self.name, "version": self.version, "attempts": 0,
                      "fallback": False, "error": None}
        last: Exception | None = None
        for _ in range(retries + 1):
            meta["attempts"] += 1
            try:
                text, usage = self._complete(query)
                meta.update(usage)
                return validate(extract_json(text), query), meta
            except Exception as exc:  # noqa: BLE001, any failure degrades, none propagates
                last = exc
        meta["fallback"] = True
        meta["error"] = f"{type(last).__name__}: {last}"[:200]
        return passthrough(query), meta
