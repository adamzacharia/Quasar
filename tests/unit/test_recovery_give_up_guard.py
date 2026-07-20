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


def test_cx12_verify_negated_completion_is_still_a_refusal():
    # "cannot be completed successfully due to computational constraints" is
    # a refusal — the negated completion phrase must not feed the suppressor.
    engine = RecoveryEngine(client=None, max_retries=3)
    text = "This cannot be completed successfully due to computational constraints."
    assert engine._check_premature_give_up(text) is not None


def test_cx12_verify_mixed_negated_and_affirmative_completion():
    # An affirmative completion elsewhere still suppresses.
    engine = RecoveryEngine(client=None, max_retries=3)
    text = ("A naive scan would require excessive memory. "
            "Instead the indexed query completed successfully with 1523 rows.")
    assert engine._check_premature_give_up(text) is None


def test_cx12_second_reopen_unrelated_negation_does_not_negate_completion():
    engine = RecoveryEngine(client=None, max_retries=3)
    # An unrelated "not"/"Without" earlier in the sentence must not turn an
    # affirmative completion report into a refusal.
    t1 = ("A full scan would require excessive memory. The query did not use "
          "the naive scan and completed successfully.")
    t2 = ("A naive join would require excessive memory. Without materializing "
          "the table, the query completed successfully.")
    assert engine._check_premature_give_up(t1) is None
    assert engine._check_premature_give_up(t2) is None


def test_cx12_second_reopen_directly_negated_completion_still_refusal():
    engine = RecoveryEngine(client=None, max_retries=3)
    for text in [
        "This cannot be completed successfully due to computational constraints.",
        "The pipeline could not be completed successfully — it would require excessive memory.",
        "The job hasn't completed successfully; it would require excessive compute.",
    ]:
        assert engine._check_premature_give_up(text) is not None, text


def test_cx12_third_round_perfect_passive_negations_are_refusals():
    engine = RecoveryEngine(client=None, max_retries=3)
    for text in [
        "The analysis could not have been completed successfully; it would require excessive memory.",
        "The job has not been completed successfully — computational limits prevent running it.",
    ]:
        assert engine._check_premature_give_up(text) is not None, text


def test_cx01_r9round_adverb_separated_negation_is_still_refusal():
    engine = RecoveryEngine(client=None, max_retries=3)
    for text in [
        "The analysis was not fully completed successfully; it would require excessive memory.",
        "The job could not have been fully completed successfully — computational limits prevent it.",
    ]:
        assert engine._check_premature_give_up(text) is not None, text
    # And an unrelated mid-sentence negation still doesn't suppress-proof it.
    ok = ("A scan would require excessive memory. The query did not use the "
          "naive scan and completed successfully.")
    assert engine._check_premature_give_up(ok) is None


def test_cx01_r9round2_typographic_apostrophes_detected():
    engine = RecoveryEngine(client=None, max_retries=3)
    for text in [
        "This couldn’t be completed successfully due to computational constraints.",
        "The job wasn’t fully completed successfully; it would require excessive memory.",
        "It can’t be done because of my compute limits.",
    ]:
        assert engine._check_premature_give_up(text) is not None, text


def test_cx01_r9round2_not_only_is_affirmative():
    engine = RecoveryEngine(client=None, max_retries=3)
    text = ("The pipeline not only completed successfully but also stayed "
            "cheap, though a naive approach would require excessive memory.")
    assert engine._check_premature_give_up(text) is None


@pytest.mark.parametrize("adverb", ["only", "merely", "simply", "solely", "purely", "exclusively"])
def test_cx01_r9round3_focusing_adverbs_are_affirmative(adverb):
    engine = RecoveryEngine(client=None, max_retries=3)
    text = (f"The pipeline was not {adverb} completed successfully, but also "
            "stayed cheap, though a naive approach would require excessive memory.")
    assert engine._check_premature_give_up(text) is None, adverb


def test_cx01_r9round3_manner_adverb_negation_still_refusal():
    engine = RecoveryEngine(client=None, max_retries=3)
    text = "The job was not fully completed successfully; it would require excessive memory."
    assert engine._check_premature_give_up(text) is not None


def test_cx01_r9round4_barely_is_a_manner_negation():
    # "not barely completed" negates completion — it must NOT be exempted.
    engine = RecoveryEngine(client=None, max_retries=3)
    text = ("The job was not barely completed successfully due to "
            "computational constraints.")
    assert engine._check_premature_give_up(text) is not None


def test_cx01_r9round5_uniquely_and_novel_adverbs_via_continuation():
    # The structural rule needs NO word list: any "-ly" adverb negation with
    # a correlative/additional-success continuation is affirmative — incl.
    # adverbs never seen before.
    engine = RecoveryEngine(client=None, max_retries=3)
    affirmatives = [
        ("The workflow was not uniquely completed successfully; two "
         "independent methods succeeded, although a naive method would "
         "require excessive memory."),
        ("The run was not predominantly completed successfully, but also "
         "cross-checked, though a full rerun would require excessive memory."),
    ]
    for text in affirmatives:
        assert engine._check_premature_give_up(text) is None, text
    # Manner-adverb negation WITHOUT a continuation stays a refusal.
    refusal = ("The job was not fully completed successfully; it would "
               "require excessive memory.")
    assert engine._check_premature_give_up(refusal) is not None


def test_cx01_r9round6_bare_but_refusal_is_still_challenged():
    # "…, but retrying would require excessive memory" is a refusal — a bare
    # "but" must not count as an additive-success continuation.
    engine = RecoveryEngine(client=None, max_retries=3)
    text = ("The job was not fully completed successfully, but retrying "
            "would require excessive memory.")
    assert engine._check_premature_give_up(text) is not None


def test_cx01_r9round7_failure_continuations_stay_refusals():
    # Additive markers WITHOUT success content must not suppress: "also
    # failed", "No fallback succeeded", "Additionally, …", "as well" after a
    # failure are all still refusals.
    engine = RecoveryEngine(client=None, max_retries=3)
    for text in [
        "The job was not fully completed successfully; it also failed validation, and retrying would require excessive memory.",
        "The job was not fully completed successfully; no fallback succeeded, and retrying would require excessive memory.",
        "The job was not fully completed successfully; additionally, retrying would require excessive memory.",
        "The job was not fully completed successfully; validation failed as well, and retrying would require excessive memory.",
    ]:
        assert engine._check_premature_give_up(text) is not None, text


def test_cx01_r9round8_two_word_negations_after_also_stay_refusals():
    engine = RecoveryEngine(client=None, max_retries=3)
    for text in [
        "The job was not fully completed successfully; it also did not pass validation, and retrying would require excessive memory.",
        "The job was not fully completed successfully; it also could not pass validation, and retrying would require excessive memory.",
    ]:
        assert engine._check_premature_give_up(text) is not None, text
