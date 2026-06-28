import json
import re
from dataclasses import asdict

import pytest

from Benchmark.run_benchmark import QUESTIONS, QuestionResult, score_golden


pytestmark = pytest.mark.unit

ALLOWED_GOLDEN_KEYS = {"must_include", "must_include_any", "regex", "numbers", "forbid"}
ALLOWED_NUMBER_KEYS = {"value", "unit", "tol", "why"}
ALLOWED_UNITS = {"GHz", "MHz"}

FROZEN_GOLDEN_RESPONSES = {
    "GK-M-02": (
        "ALMA has 10 bands, Band 1 through Band 10. For context, "
        "Band 6 spans 211-275 GHz."
    ),
    "DS-H-01": (
        "For HH 212, use Band 7 continuum observations. ALMA Band 7 spans "
        "275-373 GHz, and the products should be filtered for sub-arcsecond imaging."
    ),
    "AM-M-03": (
        "A search for M83 in Band 6 should query ALMA observations covering "
        "the 211-275 GHz receiver range."
    ),
    "SQ-M-01": (
        "Require 12CO(2-1), 13CO, and C18O in Band 6 in the same project. "
        "The rest frequencies are 230.538 GHz, 220.398684 GHz, and "
        "219.560354 GHz."
    ),
    "SQ-H-01": (
        "For z=1 to z=2, use nu_obs = nu_rest / (1+z). CO(2-1) has rest "
        "frequency 230.538 GHz and appears at 76.8-115.3 GHz. CO(3-2) "
        "has rest frequency 345.796 GHz and reaches 172.9 GHz at z=1."
    ),
}


def _golden_questions():
    return [question for question in QUESTIONS if question.get("golden")]


def _assert_non_empty_strings(values, label):
    assert isinstance(values, list), label
    assert values, label
    for value in values:
        assert isinstance(value, str), label
        assert value.strip(), label


def _synthetic_response_for(golden):
    parts = []
    parts.extend(golden.get("must_include", []))
    for group in golden.get("must_include_any", []):
        parts.append(group[0])
    for pattern in golden.get("regex", []):
        if "proposal_id" in pattern:
            parts.append("proposal_id")
        else:
            parts.append("regex example")
    for number in golden.get("numbers", []):
        parts.append(f"{number['value']} {number['unit']}")
    return " ".join(str(part) for part in parts)


def test_golden_schema_validity():
    golden_questions = _golden_questions()
    assert len(golden_questions) >= 5

    for question in golden_questions:
        golden = question["golden"]
        assert isinstance(golden, dict), question["id"]
        assert set(golden) <= ALLOWED_GOLDEN_KEYS, question["id"]
        assert any(golden.get(key) for key in ALLOWED_GOLDEN_KEYS), question["id"]

        if "must_include" in golden:
            _assert_non_empty_strings(golden["must_include"], f"{question['id']} must_include")

        if "forbid" in golden:
            _assert_non_empty_strings(golden["forbid"], f"{question['id']} forbid")

        if "must_include_any" in golden:
            groups = golden["must_include_any"]
            assert isinstance(groups, list), question["id"]
            assert groups, question["id"]
            for group in groups:
                _assert_non_empty_strings(group, f"{question['id']} must_include_any")

        if "regex" in golden:
            _assert_non_empty_strings(golden["regex"], f"{question['id']} regex")
            for pattern in golden["regex"]:
                re.compile(pattern)

        if "numbers" in golden:
            numbers = golden["numbers"]
            assert isinstance(numbers, list), question["id"]
            assert numbers, question["id"]
            for entry in numbers:
                assert isinstance(entry, dict), question["id"]
                assert set(entry) <= ALLOWED_NUMBER_KEYS, question["id"]
                assert "value" in entry and "unit" in entry, question["id"]
                assert isinstance(entry["value"], (int, float)) and not isinstance(entry["value"], bool)
                assert entry["unit"] in ALLOWED_UNITS
                if "tol" in entry:
                    assert isinstance(entry["tol"], (int, float)) and entry["tol"] >= 0
                assert isinstance(entry.get("why"), str) and entry["why"].strip()


def test_scorer_accepts_each_question_golden():
    for question in _golden_questions():
        response = _synthetic_response_for(question["golden"])
        score = score_golden(response, question["golden"])
        assert score["passed"], (question["id"], score["failures"], response)
        assert score["hits"] == score["total"]


def test_scorer_reports_failure_kinds():
    cases = [
        ({"must_include": ["Band 6"]}, "Band 7 only", "must_include"),
        ({"must_include_any": [["M83", "M 83"]]}, "M82 only", "must_include_any"),
        ({"regex": [r"\bproposal_id\b"]}, "project code column", "regex"),
        (
            {"numbers": [{"value": 230.538, "unit": "GHz", "tol": 0.001, "why": "CO(2-1)"}]},
            "The line is at 230.540 GHz.",
            "number",
        ),
        ({"forbid": ["Band 11"]}, "This answer incorrectly claims Band 11.", "forbid"),
    ]

    for golden, response, expected_kind in cases:
        score = score_golden(response, golden)
        assert not score["passed"]
        assert any(failure["kind"] == expected_kind for failure in score["failures"])


def test_scorer_numeric_conversion_and_ranges():
    mhz_score = score_golden(
        "CO(2-1) is at 230538 MHz.",
        {"numbers": [{"value": 230.538, "unit": "GHz", "tol": 0.001, "why": "CO(2-1)"}]},
    )
    assert mhz_score["passed"]

    range_score = score_golden(
        "Band 6 covers 211-275 GHz.",
        {"numbers": [
            {"value": 211.0, "unit": "GHz", "tol": 0.0, "why": "Band 6 lower edge"},
            {"value": 275.0, "unit": "GHz", "tol": 0.0, "why": "Band 6 upper edge"},
        ]},
    )
    assert range_score["passed"]

    boundary_score = score_golden(
        "The line is 230.539 GHz.",
        {"numbers": [{"value": 230.538, "unit": "GHz", "tol": 0.001, "why": "boundary"}]},
    )
    assert boundary_score["passed"]

    beyond_score = score_golden(
        "The line is 230.540 GHz.",
        {"numbers": [{"value": 230.538, "unit": "GHz", "tol": 0.001, "why": "beyond"}]},
    )
    assert not beyond_score["passed"]


def test_frozen_golden_responses_pass():
    questions_by_id = {question["id"]: question for question in QUESTIONS}
    assert set(FROZEN_GOLDEN_RESPONSES) <= set(questions_by_id)

    for question_id, response in FROZEN_GOLDEN_RESPONSES.items():
        question = questions_by_id[question_id]
        score = score_golden(response, question["golden"])
        assert score["passed"], (question_id, score["failures"])


def test_question_bank_text_is_present():
    for question in QUESTIONS:
        if question.get("criteria"):
            assert question.get("title", "").strip(), question["id"]
            assert question.get("question", "").strip(), question["id"]


def test_question_result_golden_fields_are_json_serializable():
    result = QuestionResult(
        id="T-1",
        category="Test",
        difficulty="Easy",
        title="Title",
        question="Question?",
        golden_passed=False,
        golden_hits=1,
        golden_total=2,
        golden_failures=[{"kind": "number", "expected": "230.538 GHz", "detail": "missing"}],
    )
    json.dumps(asdict(result))
