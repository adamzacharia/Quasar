"""R6 — DataLabBench v1.2 additive axes (faithfulness/correctness/fabrication)."""

import sys
from pathlib import Path

import pytest

BENCH_DIR = str(Path(__file__).resolve().parents[2] / "Benchmark" / "datalabbench")
if BENCH_DIR not in sys.path:
    sys.path.insert(0, BENCH_DIR)

from dlb_dataset_v1 import (  # noqa: E402
    AXIS_OVERRIDES,
    BENCH_VERSION,
    CHECKPOINT_AXES,
    QUESTIONS,
    axis_for_checkpoint,
    fabrication_guard_ids,
    get_question,
    validate_dataset,
)
from run_datalabbench import (  # noqa: E402
    CheckpointScore,
    PenaltyHit,
    QuestionResult,
    compute_axis_rollup,
)


def test_bench_version_is_1_2():
    assert BENCH_VERSION == "1.2"


def test_dataset_still_validates():
    assert validate_dataset() == []


def test_default_axis_rule():
    q = get_question("DLB-01")
    cps = {cp["id"]: cp for cp in q["checkpoints"]}
    assert axis_for_checkpoint("DLB-01", cps["C1"]) == "faithfulness"  # auto
    assert axis_for_checkpoint("DLB-01", cps["C3"]) == "correctness"   # judge


def test_axis_overrides_apply():
    q = get_question("DLB-02")
    cps = {cp["id"]: cp for cp in q["checkpoints"]}
    assert axis_for_checkpoint("DLB-02", cps["C5"]) == "faithfulness"  # override
    assert axis_for_checkpoint("DLB-02", cps["C6"]) == "correctness"


def test_every_checkpoint_resolves_to_a_known_axis():
    for q in QUESTIONS:
        for cp in q["checkpoints"]:
            assert axis_for_checkpoint(q["id"], cp) in CHECKPOINT_AXES


def test_overrides_reference_real_checkpoints():
    by_id = {q["id"]: {cp["id"] for cp in q["checkpoints"]} for q in QUESTIONS}
    for qid, overrides in AXIS_OVERRIDES.items():
        assert qid in by_id
        for cid in overrides:
            assert cid in by_id[qid]


def test_fabrication_guards_derived_from_guidance():
    assert "C6" in fabrication_guard_ids(get_question("DLB-02"))
    assert "C4" in fabrication_guard_ids(get_question("DLB-01"))


def _result_for_dlb02(c6_credit, with_gp00):
    q = get_question("DLB-02")
    r = QuestionResult(id="DLB-02", tier=1, title=q["title"], prompt=q["prompt"])
    r.checkpoints = [
        CheckpointScore("C1", "auto", "", 20, credit=1.0, earned=20.0),
        CheckpointScore("C2", "auto", "", 20, credit=1.0, earned=20.0),
        CheckpointScore("C3", "auto", "", 15, credit=0.5, earned=7.5),
        CheckpointScore("C4", "auto", "", 15, credit=1.0, earned=15.0),
        CheckpointScore("C5", "judge", "", 15, credit=1.0, earned=15.0),
        CheckpointScore("C6", "judge", "", 15, credit=c6_credit,
                        earned=(15.0 * c6_credit) if c6_credit is not None else None),
    ]
    if with_gp00:
        r.penalties = [PenaltyHit("GP-00", "fabricated", 20, "no dl tool ok")]
    return q, r


def test_axis_rollup_sums_and_fabrication():
    q, r = _result_for_dlb02(c6_credit=0.0, with_gp00=True)
    axes = compute_axis_rollup(q, r)
    assert axes["faithfulness"]["possible"] == 85.0
    assert axes["faithfulness"]["earned"] == 77.5
    assert axes["faithfulness"]["pct"] == pytest.approx(91.2, abs=0.05)
    assert axes["correctness"]["possible"] == 15.0
    assert axes["correctness"]["pct"] == 0.0
    fab = axes["fabrication"]
    assert fab["gp00_applied"] and fab["points_docked"] == 20
    assert fab["guard_checkpoints_zeroed"] == ["C6"] and fab["signal"]


def test_axis_rollup_unjudged_axis_has_none_pct():
    q, r = _result_for_dlb02(c6_credit=None, with_gp00=False)
    axes = compute_axis_rollup(q, r)
    # C6 unjudged → correctness axis pct undefined; faithfulness unaffected.
    assert axes["correctness"]["pct"] is None
    assert axes["faithfulness"]["pct"] is not None
    assert axes["fabrication"]["signal"] is False


def test_axis_rollup_never_alters_v11_score():
    # The axes are derived FROM the checkpoint scores; total_earned stays the
    # v1.1 computation (sum earned − penalties, floored at 0).
    q, r = _result_for_dlb02(c6_credit=0.0, with_gp00=True)
    compute_axis_rollup(q, r)
    assert r.total_earned == max(0.0, (20 + 20 + 7.5 + 15 + 15 + 0) - 20)


# ─────────────────────────────────────────────────────────────────────────────
# Guard round CX-09/CX-10 — report rendering + all-question additivity sweep
# ─────────────────────────────────────────────────────────────────────────────
def _full_credit_result(q):
    r = QuestionResult(id=q["id"], tier=q["tier"], title=q["title"], prompt=q["prompt"])
    r.checkpoints = [
        CheckpointScore(cp["id"], cp["type"], cp["desc"], cp["points"],
                        credit=1.0, earned=float(cp["points"]))
        for cp in q["checkpoints"]
    ]
    return r


def test_cx10_axes_partition_every_question_and_preserve_totals():
    # For EVERY question: the two axes partition exactly the 100 rubric
    # points, a full-credit result scores exactly 100 on both the v1.1 total
    # and the axis sums, and the fabrication axis stays silent.
    for q in QUESTIONS:
        r = _full_credit_result(q)
        axes = compute_axis_rollup(q, r)
        possible = axes["faithfulness"]["possible"] + axes["correctness"]["possible"]
        earned = axes["faithfulness"]["earned"] + axes["correctness"]["earned"]
        assert possible == 100.0, q["id"]
        assert earned == 100.0, q["id"]
        assert r.total_earned == 100.0, q["id"]          # v1.1 math untouched
        assert axes["fabrication"]["signal"] is False, q["id"]


def test_cx10_partial_credit_sums_match_v11_total():
    # Axis earned-sums must always equal the v1.1 pre-penalty raw sum.
    for q in QUESTIONS:
        r = QuestionResult(id=q["id"], tier=q["tier"], title=q["title"], prompt=q["prompt"])
        r.checkpoints = [
            CheckpointScore(cp["id"], cp["type"], cp["desc"], cp["points"],
                            credit=0.5, earned=round(cp["points"] * 0.5, 2))
            for cp in q["checkpoints"]
        ]
        axes = compute_axis_rollup(q, r)
        raw = sum(c.earned for c in r.checkpoints)
        assert axes["faithfulness"]["earned"] + axes["correctness"]["earned"] == \
            pytest.approx(raw, abs=0.01), q["id"]


def test_cx09_report_renders_axes_and_citation_sections(tmp_path):
    from types import SimpleNamespace

    from run_datalabbench import generate_report, overall_rollup

    q = get_question("DLB-02")
    r = _full_credit_result(q)
    r.usage = {"total": 1000, "cost_usd": 0.01, "unpriced_tokens": 0}
    r.axes = compute_axis_rollup(q, r)
    r.citation_metrics = {"citation_recall": 0.75, "citation_precision": 1.0,
                          "claim_sentences": 4, "citations_total": 2,
                          "method": "mechanical_v1"}
    results = [r]
    rollup = overall_rollup(results)
    args = SimpleNamespace(api_url="http://test", model="test-model",
                           judge_model="judge", skip_judge=False)
    generate_report(results, tmp_path, args, rollup)
    report = (tmp_path / "DataLabBench_report.md").read_text(encoding="utf-8")
    assert "## Axes (v1.2, additive)" in report
    assert "Faithfulness (task/approach compliance)" in report
    assert "Fabrication signals" in report
    assert "Citation metrics (R3, informational — not scored)" in report
    assert "| DLB-02 | 0.75 | 1.0 |" in report
