"""R7 — premature-give-up classifier in the RecoveryEngine (ReplicationBench
failure mode #1: false infeasibility claims citing compute limits)."""

import asyncio

import pytest

from core.recovery import (
    _BUDGET_STATEMENT,
    RecoveryEngine,
    _GIVE_UP_RE,
)


class _Node:
    def __init__(self, sla=10.0):
        self.description = "measure the flux of the source"
        self.sla_seconds = sla
        self.id = "n1"
        self.agent_type = "analysis"
        self.depends_on = []


GIVE_UP_TEXTS = [
    "This task cannot be completed due to computational constraints on my end.",
    "The analysis can't be performed because of resource limitations.",
    "Unfortunately this is too computationally expensive to run here.",
    "Computational limits prevent running the full pipeline.",
    "This exceeds my computational capabilities.",
    "As an AI language model, I cannot run simulations.",
    "The fit would require excessive time and memory to converge.",
]

HONEST_TEXTS = [
    "The ALMA archive query returned 15 observations of M87 in Band 6.",
    "The FITS download failed with HTTP 503 from the CADC server.",
    "No sources brighter than 1 mJy were detected in the field.",
    "The cone search completed in 2.3 s and found 1523 rows.",
    # Mentions compute neutrally without declaring infeasibility:
    "The computation used the q3c index, keeping the query cheap.",
]


@pytest.mark.parametrize("text", GIVE_UP_TEXTS)
def test_give_up_regex_matches_give_up_shapes(text):
    assert _GIVE_UP_RE.search(text), text


@pytest.mark.parametrize("text", HONEST_TEXTS)
def test_give_up_regex_ignores_honest_answers(text):
    assert not _GIVE_UP_RE.search(text), text


def test_give_up_triggers_exactly_one_replan_with_budget_statement():
    engine = RecoveryEngine(client=None, max_retries=3)
    node = _Node()
    calls = []

    async def _executor(n):
        calls.append(n.description)
        return "This task cannot be completed due to computational constraints."

    result = asyncio.run(engine.execute_with_recovery(node, _executor))

    # Exactly one retry: initial attempt + one challenged attempt.
    assert len(calls) == 2
    # The second attempt carried the explicit budget statement.
    assert _BUDGET_STATEMENT.strip() in calls[1]
    assert _BUDGET_STATEMENT.strip() not in calls[0]
    # The persistent give-up on attempt 2 is accepted as the honest answer.
    assert "cannot be completed" in result


def test_give_up_recovered_answer_is_returned():
    engine = RecoveryEngine(client=None, max_retries=3)
    node = _Node()
    state = {"n": 0}

    async def _executor(n):
        state["n"] += 1
        if state["n"] == 1:
            return "I am sorry but this exceeds my computational capabilities."
        return "Measured flux: 2.4 mJy at 230 GHz."

    result = asyncio.run(engine.execute_with_recovery(node, _executor))
    assert result == "Measured flux: 2.4 mJy at 230 GHz."
    assert state["n"] == 2


def test_honest_result_is_untouched():
    engine = RecoveryEngine(client=None, max_retries=3)
    node = _Node()
    calls = []

    async def _executor(n):
        calls.append(1)
        return "The ALMA archive query returned 15 observations."

    result = asyncio.run(engine.execute_with_recovery(node, _executor))
    assert result == "The ALMA archive query returned 15 observations."
    assert len(calls) == 1


def test_dict_answer_field_is_checked():
    engine = RecoveryEngine(client=None, max_retries=3)
    node = _Node()
    calls = []

    async def _executor(n):
        calls.append(n.description)
        return {"success": True, "total_results": 1,
                "answer": "Cannot be executed due to compute restrictions."}

    asyncio.run(engine.execute_with_recovery(node, _executor))
    assert len(calls) == 2  # challenged once


def test_give_up_check_ignores_non_text_results():
    engine = RecoveryEngine(client=None, max_retries=3)
    assert engine._check_premature_give_up(42) is None
    assert engine._check_premature_give_up({"success": True, "count": 3}) is None
    assert engine._check_premature_give_up(None) is None


# ─────────────────────────────────────────────────────────────────────────────
# Guard round CX-11/12/13 — classifier coverage + suppressor + max_retries=1
# ─────────────────────────────────────────────────────────────────────────────
CX11_GIVE_UP_TEXTS = [
    "I can't do this because of my compute limits.",
    "This is infeasible because of computational limits.",
    "Limited compute prevents running the pipeline.",
]


@pytest.mark.parametrize("text", CX11_GIVE_UP_TEXTS)
def test_cx11_additional_refusal_wordings_match(text):
    assert _GIVE_UP_RE.search(text), text


def test_cx12_completion_context_suppresses_false_positive():
    engine = RecoveryEngine(client=None, max_retries=3)
    text = ("The task would require excessive memory in the naive approach, "
            "but the query completed successfully using the q3c index.")
    assert engine._check_premature_give_up(text) is None


def test_cx12_plain_refusal_still_detected():
    engine = RecoveryEngine(client=None, max_retries=3)
    assert engine._check_premature_give_up(
        "The fit would require excessive time and memory to converge.") is not None


def test_cx13_max_retries_one_returns_answer_not_exhaustion():
    engine = RecoveryEngine(client=None, max_retries=1)
    node = _Node()
    calls = []

    async def _executor(n):
        calls.append(1)
        return "This task cannot be completed due to computational constraints."

    result = asyncio.run(engine.execute_with_recovery(node, _executor))
    # With no retry budget the answer is returned as-is — never
    # "[All recovery strategies failed...]".
    assert result == "This task cannot be completed due to computational constraints."
    assert len(calls) == 1
