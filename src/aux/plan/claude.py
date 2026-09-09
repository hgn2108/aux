"""Anthropic-backed planner.

**Privacy boundary, enforced here rather than by convention.** Only the query string leaves
the machine. Never audio, never file paths, never library contents, never ratings. This is
the concession DEC-007 accepted in a local-first project, and keeping it to a single
`content=query` line is what makes it checkable.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

from .base import Planner
from .prompt import SYSTEM, build_messages
from .schema import JSON_SCHEMA

DEFAULT_MODEL = "claude-haiku-4-5-20251001"
"""Short text to a small JSON object is well within Haiku, and latency matters for search.
Sonnet is run over the identical set for the quality/latency/cost comparison (E2a) rather
than assumed better."""


def load_api_key() -> str | None:
    """Read the key from the environment, falling back to a gitignored .env."""
    if key := os.environ.get("ANTHROPIC_API_KEY"):
        return key
    env = Path(__file__).resolve().parents[3] / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if line.startswith("ANTHROPIC_API_KEY="):
                return line.split("=", 1)[1].strip()
    return None


class ClaudePlanner(Planner):
    def __init__(self, model: str = DEFAULT_MODEL, *, max_tokens: int = 400,
                 structured: bool = True, api_key: str | None = None) -> None:
        import anthropic

        key = api_key or load_api_key()
        if not key:
            raise RuntimeError(
                "no ANTHROPIC_API_KEY in the environment or .env; "
                "see STATUS.md for how Slice 2 rung 2 is configured"
            )
        self.name = "claude"
        self.version = model
        self.max_tokens = max_tokens
        self.structured = structured
        """Enforce the JSON schema server-side rather than asking for it in the prompt.

        This API version exposes no temperature control, so the planner cannot be made
        deterministic by sampling settings — and it is not deterministic: an E2 measurement
        moved from +0.34 to +0.40 between two runs whose baseline was identical to two
        decimal places, which was the planner re-sampling its own inputs. Reproducibility is
        therefore handled by caching plans (`plan.cache`), and schema enforcement removes the
        other source of variation, which is response *shape*."""
        self._client = anthropic.Anthropic(api_key=key)

    def _complete(self, query: str) -> tuple[str, dict]:
        started = time.perf_counter()
        kwargs = {}
        if self.structured:
            kwargs["output_config"] = {
                "format": {"type": "json_schema", "schema": JSON_SCHEMA}
            }
        response = self._client.messages.create(
            model=self.version,
            max_tokens=self.max_tokens,
            system=SYSTEM,
            # The query string is the only user data in this request.
            messages=build_messages(query),
            **kwargs,
        )
        text = "".join(block.text for block in response.content if block.type == "text")
        return text, {
            "latency_ms": (time.perf_counter() - started) * 1000,
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
        }
