"""Query planning — DEC-007's LLM rung, behind one interface."""

from .base import Planner, extract_json
from .schema import PlanValidationError, QueryPlan, passthrough, validate


def build_planner(name: str = "claude", model: str | None = None) -> Planner:
    """Construct a planner by name, so evaluation can swap backends by config."""
    if name == "claude":
        from .claude import DEFAULT_MODEL, ClaudePlanner

        return ClaudePlanner(model or DEFAULT_MODEL)
    if name in {"ollama", "local"}:
        from .local import DEFAULT_MODEL, OllamaPlanner

        return OllamaPlanner(model or DEFAULT_MODEL)
    raise ValueError(f"unknown planner {name!r}")


__all__ = [
    "PlanValidationError",
    "Planner",
    "QueryPlan",
    "build_planner",
    "extract_json",
    "passthrough",
    "validate",
]
