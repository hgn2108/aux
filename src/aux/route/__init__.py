"""Query-dependent fusion weighting."""

from .base import Route, Router
from .prototype import PrototypeRouter
from .rules import RuleRouter

__all__ = ["PrototypeRouter", "Route", "RuleRouter", "Router"]
