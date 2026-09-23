"""Turn exit for non-converging tool loops + the repeated-call detector
(core/runner.py, core/turn_recovery.py).

UI benchmark 2026-09-22: D19 ran 8 tool rounds for 421 s and ended in a
spinner; L06 issued five near-identical datalab_color_magnitude_diagram calls
(~70 s ReadTimeout each) that differed only in title/labels. The runner now
(a) caps a turn at QUASAR_MAX_TOOL_ROUNDS logical tool rounds / a
QUASAR_TURN_TOOL_BUDGET_SECONDS wall budget and forces ONE final answer round
with an explicit summarise-what-happened instruction, and (b) refuses to
re-execute a canonically identical call after a failure (or after a success,
serving the cached result).
"""
from __future__ import annotations

import json

import pytest

from core import runner
from core.turn_recovery import (
    COSMETIC_ARG_KEYS,
    canonical_tool_call_key,
    tool_call_key,
    tool_result_failure_reason,
)


# ── canonical key ────────────────────────────────────────────────────────


def test_cosmetic_only_changes_produce_the_same_canonical_key():
    base = {"catalog": "gaia_dr3", "ra": 229.02, "dec": -0.11, "radius": 2.0, "title": "WD candidates"}
    variants = [
        {**base, "title": "White dwarf CMD (attempt 2)"},
        {**base, "x_label": "BP-RP", "y_label": "G", "title": "third try"},
        {**base, "label": "x", "caption": "y", "cmap": "viridis"},
    ]
    k0 = canonical_tool_call_key("datalab_color_magnitude_diagram", base)
    for v in variants:
        assert canonical_tool_call_key("datalab_color_magnitude_diagram", v) == k0
        assert tool_call_key("datalab_color_magnitude_diagram", v) != tool_call_key("datalab_color_magnitude_diagram", base)


def test_whitespace_case_and_trailing_semicolon_are_normalised_in_strings():
    a = {"sql": "SELECT ra, dec FROM gaia_dr3.gaia_source WHERE parallax > 5;"}
    b = {"sql": "select  ra,   dec\nFROM gaia_dr3.gaia_source\n  WHERE parallax > 5 /* retry */"}
    assert canonical_tool_call_key("datalab_sql_query", a) == canonical_tool_call_key("datalab_sql_query", b)


def test_real_argument_changes_produce_a_different_key():
    a = {"ra": 10.0, "dec": 20.0, "radius": 0.5}
    assert canonical_tool_call_key("t", a) != canonical_tool_call_key("t", {**a, "radius": 1.0})
    assert canonical_tool_call_key("t", a) != canonical_tool_call_key("u", a)
    assert canonical_tool_call_key("t", {"sql": "select 1"}) != canonical_tool_call_key("t", {"sql": "select 2"})


def test_cosmetic_key_set_covers_the_l06_variants():
    assert {"title", "x_label", "y_label", "label"} <= COSMETIC_ARG_KEYS


# ── failure classification ───────────────────────────────────────────────


@pytest.mark.parametrize(
    "result",
    [
        {"success": False, "error": "Read timed out"},
        {"error": "Unknown column zmag"},
        {"success": False},
        {"timeout": True, "error": "TIMEOUT: did not finish"},
        {"success": False, "circuit_breaker": True, "infrastructure_failure": True, "error": "host down"},
        {"success": False, "cancelled": True, "error": "turn cancelled"},
        json.dumps({"success": False, "error": "as text"}),
        {"repeated_call": True, "previous_result": {"success": False, "error": "inner"}},
    ],
)
def test_failed_results_yield_a_reason(result):
    assert tool_result_failure_reason(result)


@pytest.mark.parametrize(
    "result",
    [
        {"success": True, "rows": 10},
        {"rows": [1, 2, 3]},
        {"success": True, "error": None},
        {"success": True, "warnings": ["row cap hit"]},
        "not json at all",
        [1, 2, 3],
        None,
    ],
)
def test_successful_or_unstructured_results_yield_none(result):
    assert tool_result_failure_reason(result) is None


# ── turn-exit configuration ─────────────────────────────────────────────


def test_defaults_max_rounds_8_and_tool_budget_300s(monkeypatch):
    for var in ("QUASAR_MAX_TOOL_ROUNDS", "QUASAR_TURN_TOOL_BUDGET_SECONDS"):
        monkeypatch.delenv(var, raising=False)
    assert runner._env_int("QUASAR_MAX_TOOL_ROUNDS", 8) == 8
    assert runner._env_seconds("QUASAR_TURN_TOOL_BUDGET_SECONDS", 300.0) == 300.0


def test_env_overrides_and_garbage_fall_back(monkeypatch):
    monkeypatch.setenv("QUASAR_MAX_TOOL_ROUNDS", "3")
    monkeypatch.setenv("QUASAR_TURN_TOOL_BUDGET_SECONDS", "120")
    assert runner._env_int("QUASAR_MAX_TOOL_ROUNDS", 8) == 3
    assert runner._env_seconds("QUASAR_TURN_TOOL_BUDGET_SECONDS", 300.0) == 120.0
    monkeypatch.setenv("QUASAR_MAX_TOOL_ROUNDS", "0")
    assert runner._env_int("QUASAR_MAX_TOOL_ROUNDS", 8) == 1, "at least one tool round"
    monkeypatch.setenv("QUASAR_TURN_TOOL_BUDGET_SECONDS", "nan")
    assert runner._env_seconds("QUASAR_TURN_TOOL_BUDGET_SECONDS", 300.0) == 300.0
    monkeypatch.setenv("QUASAR_TURN_TOOL_BUDGET_SECONDS", "-5")
    assert runner._env_seconds("QUASAR_TURN_TOOL_BUDGET_SECONDS", 300.0) == 300.0
    monkeypatch.setenv("QUASAR_TURN_TOOL_BUDGET_SECONDS", "0")
    assert runner._env_seconds("QUASAR_TURN_TOOL_BUDGET_SECONDS", 300.0) == 0.0, "0 disables"


def test_final_note_demands_a_structured_honest_summary():
    note = runner.TOOL_BUDGET_FINAL_NOTE.format(reason="max tool rounds reached (8)")
    for phrase in ("what was tried", "what succeeded", "what failed", "next step", "Do not call more tools",
                   "Do not claim results", "max tool rounds reached (8)"):
        assert phrase in note, phrase


def test_material_image_arguments_are_not_cosmetic():
    """Guard CX-13: a 300 px and a 1200 px cutout are different requests."""
    a = {"target": "M31", "width": 300, "height": 300, "dpi": 100}
    for k, v in (("width", 1200), ("height", 1200), ("dpi", 300), ("figsize", [12, 8])):
        assert canonical_tool_call_key("hips_cutout", a) != canonical_tool_call_key("hips_cutout", {**a, k: v})
    assert canonical_tool_call_key("create_ads_library", {"name": "x", "description": "A"}) != \
        canonical_tool_call_key("create_ads_library", {"name": "x", "description": "B"})


def test_short_breaker_skip_is_reissued_exactly_once_per_turn(monkeypatch):
    """Guard CX-14 (+ the exact-call cache): an identical call refused by a
    short-cooldown breaker may be re-issued ONCE after the cooldown -- and only
    once per turn, even though the ledger entry is overwritten by the retry."""
    import copy
    import json
    from types import SimpleNamespace as NS

    from core.llm_client import LLMClient
    from tests.unit.test_discovery_recovery import _tool_agent

    clock = [0.0]
    monkeypatch.setattr("core.runner.time.monotonic", lambda: clock[0])
    calls = []

    def create(**kwargs):
        calls.append(copy.deepcopy(kwargs))
        clock[0] += 3.0  # each model round takes longer than the 2 s cooldown
        n = len(calls)
        if n <= 3:
            delta = NS(content=None, tool_calls=[NS(index=0, id=f"call-{n}", function=NS(name="query_archive", arguments=json.dumps({"x": 1})))])
        else:
            delta = NS(content="The archive was unavailable; here is what I have.", tool_calls=None)
        return iter([NS(choices=[NS(delta=delta, finish_reason="stop")])])

    client = LLMClient(model="gpt-oss-120b")
    client._get_tacc_client = lambda: NS(chat=NS(completions=NS(create=create)))
    agent, executed = _tool_agent(client.responses)

    def execute(*a, **kw):
        executed.append(kw)
        return {"success": False, "circuit_breaker": True, "infrastructure_failure": True, "retry_after_s": 2,
                "error": "INFRASTRUCTURE FAILURE: archive.example is unreachable right now"}

    agent._execute_tool_with_progress = execute
    agent.stream_response_api("hello there", conversation_id="cx14-reissue")
    assert len(calls) >= 4
    assert len(executed) == 2, "first call + exactly one re-issue after the cooldown"
