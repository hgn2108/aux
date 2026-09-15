"""Ollama-backed planner, the local half of E2a.

Exists so the question "does this need a hosted model?" is answered by measurement rather
than assumption. If a local model matches the hosted one, DEC-007's hosted-API concession is
withdrawn and the planner runs entirely on the user's machine, which is what PROJECT.md's
local-first framing wants.

Talks to Ollama over its local HTTP API rather than through a client library, to avoid a
dependency for what is one POST.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request

from .base import Planner
from .prompt import SYSTEM, build_messages

DEFAULT_MODEL = "qwen3:8b"
DEFAULT_HOST = "http://127.0.0.1:11434"


class OllamaPlanner(Planner):
    def __init__(self, model: str = DEFAULT_MODEL, *, host: str = DEFAULT_HOST,
                 timeout: float = 60.0) -> None:
        self.name = "ollama"
        self.version = model
        self.host = host.rstrip("/")
        self.timeout = timeout

    def available(self) -> bool:
        try:
            with urllib.request.urlopen(f"{self.host}/api/tags", timeout=3) as r:
                tags = json.load(r)
        except (urllib.error.URLError, OSError, json.JSONDecodeError):
            return False
        return any(m.get("name", "").startswith(self.version.split(":")[0])
                   for m in tags.get("models", []))

    def _complete(self, query: str) -> tuple[str, dict]:
        payload = {
            "model": self.version,
            "system": SYSTEM,
            "messages": build_messages(query),
            "stream": False,
            # Deterministic, so a re-run of the evaluation gives the same plans and a
            # difference between two runs is a real difference.
            "options": {"temperature": 0.0},
            "format": "json",
        }
        request = urllib.request.Request(
            f"{self.host}/api/chat",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        started = time.perf_counter()
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            body = json.load(response)
        return body.get("message", {}).get("content", ""), {
            "latency_ms": (time.perf_counter() - started) * 1000,
            "input_tokens": body.get("prompt_eval_count"),
            "output_tokens": body.get("eval_count"),
        }
