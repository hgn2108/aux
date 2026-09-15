"""On-disk cache of planner outputs.

Two reasons, and the second is the important one:

- a re-run should not pay for the same queries again;
- a re-run should get the same plans. Even at temperature 0 an API is not contractually
  deterministic, and an evaluation that silently re-samples its own inputs cannot confirm a
  previous result.

Keyed on `(planner version, structured-output flag, query)`.
"""

from __future__ import annotations

import json
from pathlib import Path

from .schema import QueryPlan, validate


class PlanCache:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._plans: dict[str, dict] = {}
        self.hits = 0
        self.misses = 0
        if self.path.exists():
            self._plans = json.loads(self.path.read_text())

    @staticmethod
    def key(version: str, temperature: float, query: str) -> str:
        return f"{version}|{temperature}|{query}"

    def get(self, key: str, query: str) -> QueryPlan | None:
        if key not in self._plans:
            self.misses += 1
            return None
        self.hits += 1
        return validate(self._plans[key], query)

    def put(self, key: str, plan: QueryPlan) -> None:
        self._plans[key] = {
            "rewritten": plan.rewritten,
            "acoustic": list(plan.acoustic), "mood": list(plan.mood),
            "genre": list(plan.genre), "context": list(plan.context),
            "exclude": list(plan.exclude), "lyrical": list(plan.lyrical),
        }

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.name + ".tmp")
        tmp.write_text(json.dumps(self._plans, indent=2, sort_keys=True))
        tmp.replace(self.path)

    def __len__(self) -> int:
        return len(self._plans)


def plan_all(planner, queries: list[str], cache_path: Path) -> dict[str, QueryPlan]:
    """Plan every query, reusing cached plans so a re-run is free and reproducible."""
    cache = PlanCache(cache_path)
    temperature = getattr(planner, "structured", True)
    out: dict[str, QueryPlan] = {}
    for query in queries:
        key = cache.key(planner.version, temperature, query)
        if (found := cache.get(key, query)) is not None:
            out[query] = found
            continue
        plan, _ = planner.plan(query)
        cache.put(key, plan)
        out[query] = plan
    cache.save()
    return out
