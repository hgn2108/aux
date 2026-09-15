"""The paid arm: ask a model for the weight.

Included because the free baselines have a known weakness, they recognise phrasings, and
a listener can ask about meaning without using any of the constructions a rule lists. A model
that understands the request should route those correctly, and should be able to grade a
compound query rather than picking a side.

Whether that is worth a network call per search is exactly what the evaluation decides. The
same question was asked of the query planner and answered no, so nothing here is assumed.

Only the query string leaves the machine, the same boundary the planner holds, and narrower
than the theme labeller's.
"""

from __future__ import annotations

import json
import sys
import time

from ..plan.claude import load_api_key
from .base import Route, Router

DEFAULT_MODEL = "claude-haiku-4-5-20251001"
"""A short string to one number: well within Haiku, and this sits inside a search where
latency is visible to the user."""

SYSTEM = """You choose how a music search should be answered.

Two signals are available:
- SOUND: how the recording actually sounds, genre, instruments, tempo, production, mood
  carried by the music itself.
- LYRICS: what the words are about, topics, stories, feelings stated in the text.

Return `alpha`, the weight on SOUND, between 0 and 1.

- 1.0 when the query describes only sound.
- 0.0 when the query asks only what songs are about.
- In between when it asks for both, weighted toward whichever it leans on more.

Judge the request, not its wording: a query can ask about meaning without saying "about".
Also return a short `reason`, phrased for the person searching, never naming these rules."""

SCHEMA = {
    "type": "object",
    "properties": {
        # No `minimum`/`maximum`: this API rejects range keywords on a number. The range is
        # stated in the system prompt and enforced by clamping the reply.
        "alpha": {"type": "number"},
        "reason": {"type": "string"},
    },
    "required": ["alpha", "reason"],
    "additionalProperties": False,
}

FALLBACK = 0.9
"""Where an unroutable query goes. Matches the rule router's default: audio is the stronger
channel overall on this corpus, so a failed call degrades to the better single system rather
than to an even split."""


class ClaudeRouter(Router):
    name = "claude"

    def __init__(self, model: str = DEFAULT_MODEL, *, api_key: str | None = None,
                 cache: dict | None = None) -> None:
        import anthropic

        key = api_key or load_api_key()
        if not key:
            raise RuntimeError("no ANTHROPIC_API_KEY in the environment or .env")
        self.model = model
        self._client = anthropic.Anthropic(api_key=key)
        self.api_calls = 0
        self.api_seconds = 0.0
        """Latency is measured over calls that actually went out. Timing the wrapper
        instead reports ~0 ms on a warm cache, which would claim a network round trip is
        free, true for a repeated query, false for the novel ones a router exists to
        handle."""
        self.failures = 0
        """Counted, not swallowed. A router that falls back on every query returns a clean
        constant, which is indistinguishable from a deliberate routing decision unless the
        failures are surfaced, an earlier run reported the fallback weight as a measured
        result because nothing recorded that all 24 calls had failed."""
        self._cache: dict[str, tuple[float, str]] = cache if cache is not None else {}
        """Routing the same query twice must give the same answer. This API exposes no
        temperature control, so determinism comes from caching rather than sampling  -
        the same resolution the planner reached."""

    def route(self, query: str) -> Route:
        if query in self._cache:
            alpha, reason = self._cache[query]
            return Route(alpha, reason, self.name)
        started = time.perf_counter()
        try:
            response = self._client.messages.create(
                model=self.model, max_tokens=1000, system=SYSTEM,
                output_config={"format": {"type": "json_schema", "schema": SCHEMA}},
                messages=[{"role": "user", "content": query}],
            )
            text = next(b.text for b in response.content
                        if getattr(b, "type", None) == "text")
            answer = json.loads(text)
            alpha = min(1.0, max(0.0, float(answer["alpha"])))
            reason = str(answer["reason"])
        except Exception as exc:  # noqa: BLE001 - a routing failure must not fail a search
            self.failures += 1
            if self.failures == 1:
                print(f"routing call failed, falling back to alpha={FALLBACK}: "
                      f"{type(exc).__name__}: {exc}", file=sys.stderr)
            alpha, reason = FALLBACK, "could not read the query; matching on sound"
        self.api_calls += 1
        self.api_seconds += time.perf_counter() - started
        self._cache[query] = (alpha, reason)
        return Route(alpha, reason, self.name)
