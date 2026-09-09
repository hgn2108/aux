"""Planner schema, parsing and failure-handling tests.

None of these need a live model. That is the point: the parts that must not break —
validation, JSON recovery, and degrading to the baseline instead of to noise — are exactly
the parts that should be testable without a network or a bill.
"""

from __future__ import annotations

import pytest

from aux.plan import PlanValidationError, extract_json, passthrough, validate
from aux.plan.base import Planner


# --- schema validation ---------------------------------------------------------------

def test_minimal_payload_validates():
    plan = validate({"rewritten": "solo acoustic piano, sparse"}, "solo piano")
    assert plan.rewritten == "solo acoustic piano, sparse"
    assert plan.original == "solo piano"
    assert plan.acoustic == ()


def test_empty_rewrite_falls_back_to_the_original_query():
    """A planner declining to rewrite must degrade to the baseline, not search for nothing."""
    assert validate({"rewritten": ""}, "jazz").rewritten == "jazz"
    assert validate({}, "jazz").rewritten == "jazz"


def test_all_facets_are_parsed():
    plan = validate({
        "rewritten": "mellow r&b, steady groove",
        "acoustic": ["steady groove", "soft vocals"],
        "mood": ["mellow"], "genre": ["r&b"], "context": ["studying"],
        "exclude": ["aggressive"], "lyrical": ["longing"],
    }, "chill r&b for studying")
    assert plan.acoustic == ("steady groove", "soft vocals")
    assert plan.genre == ("r&b",)
    assert plan.exclude == ("aggressive",)
    assert plan.lyrical == ("longing",)


def test_a_bare_string_is_accepted_where_a_list_is_expected():
    """Models emit a string for a single-item list often enough to be worth absorbing."""
    assert validate({"rewritten": "x", "genre": "jazz"}, "q").genre == ("jazz",)


def test_blank_entries_are_dropped():
    plan = validate({"rewritten": "x", "acoustic": ["piano", "", "  "]}, "q")
    assert plan.acoustic == ("piano",)


@pytest.mark.parametrize("payload", [
    {"rewritten": "x", "acoustic": {"not": "a list"}},
    {"rewritten": "x", "genre": [1, 2]},
    {"rewritten": 42},
])
def test_wrong_types_raise(payload):
    with pytest.raises(PlanValidationError):
        validate(payload, "q")


def test_non_object_payload_raises():
    with pytest.raises(PlanValidationError):
        validate(["not", "an", "object"], "q")


def test_lists_are_capped():
    plan = validate({"rewritten": "x", "acoustic": [f"t{i}" for i in range(50)]}, "q")
    assert len(plan.acoustic) <= 12


# --- JSON recovery -------------------------------------------------------------------

def test_plain_json_parses():
    assert extract_json('{"rewritten": "a"}') == {"rewritten": "a"}


def test_code_fenced_json_parses():
    assert extract_json('```json\n{"rewritten": "a"}\n```') == {"rewritten": "a"}


def test_json_with_surrounding_prose_parses():
    """A formatting slip is not a refusal, so it is recovered rather than counted a failure."""
    assert extract_json('Sure! {"rewritten": "a"} Hope that helps.') == {"rewritten": "a"}


@pytest.mark.parametrize("text", ["", "no json here", "{ broken: ", "```json\n{oops\n```"])
def test_unrecoverable_responses_raise(text):
    with pytest.raises(PlanValidationError):
        extract_json(text)


# --- failure handling ----------------------------------------------------------------

class FlakyPlanner(Planner):
    def __init__(self, responses):
        self.name, self.version = "flaky", "test"
        self.responses = list(responses)
        self.calls = 0

    def _complete(self, query):
        self.calls += 1
        r = self.responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r, {"latency_ms": 1.0}


def test_a_good_response_needs_one_call():
    p = FlakyPlanner(['{"rewritten": "solo piano"}'])
    plan, meta = p.plan("piano")
    assert plan.rewritten == "solo piano"
    assert meta["attempts"] == 1
    assert not meta["fallback"]


def test_a_bad_response_is_retried_once_then_succeeds():
    p = FlakyPlanner(["garbage", '{"rewritten": "solo piano"}'])
    plan, meta = p.plan("piano")
    assert plan.rewritten == "solo piano"
    assert meta["attempts"] == 2
    assert not meta["fallback"]


def test_repeated_failure_degrades_to_the_baseline():
    """A planner outage must retrieve exactly as Slice 1 did, not as something unknown."""
    p = FlakyPlanner(["garbage", "still garbage"])
    plan, meta = p.plan("hip hop for running")
    assert plan.rewritten == "hip hop for running"
    assert plan == passthrough("hip hop for running")
    assert meta["fallback"] is True
    assert meta["error"]


def test_a_transport_error_also_degrades_rather_than_propagating():
    p = FlakyPlanner([RuntimeError("connection reset"), RuntimeError("connection reset")])
    plan, meta = p.plan("jazz")
    assert plan == passthrough("jazz")
    assert meta["fallback"] is True
    assert "RuntimeError" in meta["error"]


def test_fallback_is_counted_not_hidden():
    """A planner that quietly fails half the time is not a planner; the rate must be visible."""
    p = FlakyPlanner(["bad", "bad"])
    _, meta = p.plan("q")
    assert set(meta) >= {"planner", "version", "attempts", "fallback", "error"}


def test_passthrough_preserves_the_query_exactly():
    plan = passthrough("west coast hip hop drive")
    assert plan.rewritten == plan.original == "west coast hip hop drive"
    assert plan.acoustic == plan.exclude == ()
