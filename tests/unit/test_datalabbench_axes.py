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


# ─────────────────────────────────────────────────────────────────────────────
# Verify-round CX-10 — execute the REAL v1.1 scoring path on deterministic
# evidence and pin its exact outputs (a scoring-behavior snapshot), then prove
# the v1.2 axis rollup neither mutates nor re-derives any of it.
# ─────────────────────────────────────────────────────────────────────────────
import copy

from run_datalabbench import Evidence, ToolCall, apply_penalties, score_auto_checkpoints


def _dlb02_good_evidence():
    ev = Evidence()
    ev.response_text = "There are 1523 Gaia DR3 sources within 10 arcminutes of Palomar 5."
    sql = ("SELECT COUNT(*) FROM gaia_dr3.gaia_source "
           "WHERE q3c_radial_query(ra, dec, 229.022, -0.112, 0.166667)")
    ev.calls = [ToolCall(
        name="datalab_cone_count",
        arguments={"catalog": "gaia_dr3", "ra": 229.022, "dec": -0.112,
                   "radius_deg": 0.166667},
        output='{"success": true, "reported_count": 1523}',
        ok=True,
        sql=sql,
    )]
    ev.sql_texts = [sql]
    ev.trace_source = "tool_trace"
    return ev


def test_cx10_v11_scoring_snapshot_on_executed_evidence():
    """The v1.1 scoring pipeline (score_auto_checkpoints + apply_penalties +
    total_earned) produces EXACTLY these values on fixed evidence. Any change
    to v1.1 scoring behavior — which v1.2 promised not to touch — breaks this
    snapshot."""
    q = get_question("DLB-02")
    ev = _dlb02_good_evidence()

    r = QuestionResult(id=q["id"], tier=q["tier"], title=q["title"], prompt=q["prompt"])
    r.checkpoints = score_auto_checkpoints(q, ev)
    r.penalties = apply_penalties(q, ev)

    # v1.1 snapshot: all four auto checkpoints pass in full, no penalties.
    earned_by_id = {c.id: c.earned for c in r.checkpoints}
    assert earned_by_id["C1"] == 20.0
    assert earned_by_id["C2"] == 20.0
    assert earned_by_id["C3"] == 15.0
    assert earned_by_id["C4"] == 15.0
    assert earned_by_id["C5"] is None and earned_by_id["C6"] is None  # judge
    assert r.penalties == []
    assert r.auto_earned == 70.0 and r.auto_max == 70.0

    # Simulate judge verdicts, then pin the v1.1 total.
    for c in r.checkpoints:
        if c.id == "C5":
            c.credit, c.earned = 1.0, 15.0
        if c.id == "C6":
            c.credit, c.earned = 0.5, 7.5
    assert r.total_earned == 92.5

    # v1.2 rollup: derives from the SAME numbers and mutates nothing.
    checkpoints_before = copy.deepcopy([vars(c) for c in r.checkpoints])
    penalties_before = copy.deepcopy(r.penalties)
    axes = compute_axis_rollup(q, r)
    assert [vars(c) for c in r.checkpoints] == checkpoints_before
    assert r.penalties == penalties_before
    assert r.total_earned == 92.5
    assert axes["faithfulness"]["earned"] + axes["correctness"]["earned"] == 92.5
    assert axes["faithfulness"]["earned"] == 85.0    # C1-C5 (C5 override)
    assert axes["correctness"]["earned"] == 7.5      # C6


def test_cx10_v11_fabrication_snapshot_still_fires_gp00():
    """Fabrication scenario through the REAL penalty path: claim-shaped text
    with no successful Data Lab call docks GP-00 exactly as in v1.1, and the
    v1.2 fabrication axis merely REPORTS it."""
    q = get_question("DLB-02")
    ev = Evidence()
    ev.response_text = "There are 1523 sources within the cone."
    ev.calls = []

    r = QuestionResult(id=q["id"], tier=q["tier"], title=q["title"], prompt=q["prompt"])
    r.checkpoints = score_auto_checkpoints(q, ev)
    r.penalties = apply_penalties(q, ev)

    assert r.auto_earned == 0.0
    gp = {p.id: p.points for p in r.penalties}
    assert gp.get("GP-00") == 20                     # v1.1 penalty unchanged
    axes = compute_axis_rollup(q, r)
    assert axes["fabrication"]["gp00_applied"] is True
    assert axes["fabrication"]["points_docked"] == 20


# ─────────────────────────────────────────────────────────────────────────────
# Verify-round CX-10 (2nd reopen) — an EXECUTED per-question v1.1 snapshot
# sweep. Every question's full check tree runs through the real evaluator on
# fixed (empty) evidence; the exact auto_max / auto_earned / penalty-count
# triple is pinned per question. Any change to v1.1 checkpoint structure,
# check evaluation, or penalty gating on ANY question breaks this literal.
# ─────────────────────────────────────────────────────────────────────────────
_V11_EMPTY_EVIDENCE_SNAPSHOT = {
    # qid: (auto_max, auto_earned, n_penalties) on empty evidence
    "DLB-01": (40, 0, 0),
    "DLB-02": (70, 0, 0),
    "DLB-03": (65, 0, 0),
    "DLB-04": (50, 0, 0),
    "DLB-05": (70, 0, 0),
    "DLB-06": (60, 0, 0),
    "DLB-07": (55, 0, 0),
    "DLB-08": (60, 0, 0),
    "DLB-09": (65, 0, 0),
    "DLB-10": (80, 0, 0),
    "DLB-11": (70, 0, 0),
    "DLB-12": (65, 0, 0),
    "DLB-13": (55, 0, 0),
    "DLB-14": (75, 0, 0),
    "DLB-15": (55, 0, 0),
}


def test_cx10_v11_executed_snapshot_sweep_all_questions():
    assert set(_V11_EMPTY_EVIDENCE_SNAPSHOT) == {q["id"] for q in QUESTIONS}
    for q in QUESTIONS:
        ev = Evidence()
        r = QuestionResult(id=q["id"], tier=q["tier"], title=q["title"], prompt=q["prompt"])
        r.checkpoints = score_auto_checkpoints(q, ev)
        r.penalties = apply_penalties(q, ev)
        expected = _V11_EMPTY_EVIDENCE_SNAPSHOT[q["id"]]
        assert (r.auto_max, r.auto_earned, len(r.penalties)) == expected, q["id"]
        # Every auto checkpoint individually earns exactly zero on empty
        # evidence, and every judge checkpoint stays unjudged.
        for c in r.checkpoints:
            if c.type == "auto":
                assert c.earned == 0.0, f"{q['id']}.{c.id}"
            else:
                assert c.earned is None, f"{q['id']}.{c.id}"
        # The v1.2 rollup on top of the executed scores mutates nothing.
        import copy as _copy
        before = _copy.deepcopy([vars(c) for c in r.checkpoints])
        compute_axis_rollup(q, r)
        assert [vars(c) for c in r.checkpoints] == before, q["id"]


def test_cx10_v11_positive_snapshot_dlb01():
    """Second positive-evidence executed snapshot (DLB-01): valid evidence
    earns exactly the v1.1 auto credit."""
    q = get_question("DLB-01")
    ev = Evidence()
    ev.response_text = (
        "The near-infrared catalog is VHS DR5 (vhs_dr5.vhs_cat_v3, J/H/Ks). "
        "Optical LMC coverage comes from smash_dr2.object and nsc_dr2.object; "
        "gaia_dr3.gaia_source is all-sky."
    )
    ev.calls = [ToolCall(name="datalab_list_catalogs", arguments={}, ok=True,
                         output='{"success": true}')]
    ev.trace_source = "tool_trace"

    r = QuestionResult(id=q["id"], tier=q["tier"], title=q["title"], prompt=q["prompt"])
    r.checkpoints = score_auto_checkpoints(q, ev)
    r.penalties = apply_penalties(q, ev)

    earned_by_id = {c.id: c.earned for c in r.checkpoints}
    assert earned_by_id["C1"] == 20.0   # schema tooling used + succeeded
    assert earned_by_id["C2"] == 20.0   # candidate catalogs named
    assert r.penalties == []            # honest tool-backed answer: no GP-00
    assert r.auto_earned == 40.0 and r.auto_max == 40.0


# ─────────────────────────────────────────────────────────────────────────────
# R9-round CX-03 — a NON-DEGENERATE executed per-question snapshot: each
# question is scored against its own reference_sql as executed evidence,
# which exercises the sql/trace regex checks, radius/position parsing, and
# penalty gating with real content. The per-checkpoint earned values are
# pinned as literals; any v1.1 scoring-behavior change on ANY question
# breaks the corresponding entry.
# ─────────────────────────────────────────────────────────────────────────────
_V11_REFERENCE_SQL_SNAPSHOT = {
    # qid: (auto_earned, {auto checkpoint id: earned}, [penalty ids])
    "DLB-01": (0, {'C1': 0.0, 'C2': 0.0}, []),
    "DLB-02": (70, {'C1': 20.0, 'C2': 20.0, 'C3': 15.0, 'C4': 15.0}, []),
    "DLB-03": (45, {'C1': 15.0, 'C2': 15.0, 'C3': 15.0, 'C4': 0.0}, []),
    "DLB-04": (0, {'C1': 0.0, 'C2': 0.0, 'C3': 0.0}, []),
    "DLB-05": (60, {'C1': 15.0, 'C2': 10.0, 'C3': 20.0, 'C4': 15.0, 'C5': 0.0}, []),
    "DLB-06": (50, {'C1': 15.0, 'C2': 20.0, 'C3': 15.0, 'C4': 0.0}, []),
    "DLB-07": (30, {'C1': 15.0, 'C2': 15.0, 'C3': 0.0, 'C4': 0.0}, []),
    "DLB-08": (50, {'C1': 15.0, 'C2': 20.0, 'C3': 15.0, 'C5': 0.0}, []),
    "DLB-09": (65, {'C1': 20.0, 'C2': 15.0, 'C3': 10.0, 'C4': 10.0, 'C5': 10.0}, []),
    "DLB-10": (50, {'C1': 15.0, 'C2': 10.0, 'C3': 15.0, 'C4': 10.0, 'C5': 0.0, 'C6': 0.0}, []),
    "DLB-11": (60, {'C1': 15.0, 'C2': 20.0, 'C3': 10.0, 'C4': 15.0, 'C5': 0.0}, []),
    "DLB-12": (30, {'C1': 0.0, 'C2': 10.0, 'C3': 10.0, 'C4': 10.0, 'C5': 0.0}, []),
    "DLB-13": (40, {'C1': 15.0, 'C2': 15.0, 'C4': 0.0, 'C7': 10.0}, []),
    "DLB-14": (35, {'C1': 15.0, 'C2': 10.0, 'C3': 10.0, 'C4': 0.0, 'C5': 0.0, 'C7': 0.0}, []),
    "DLB-15": (13.33, {'C2': 0.0, 'C3': 13.33, 'C4': 0.0, 'C6': 0.0}, []),
}


def test_cx03_r9round_reference_sql_executed_snapshot_all_questions():
    assert set(_V11_REFERENCE_SQL_SNAPSHOT) == {q["id"] for q in QUESTIONS}
    for q in QUESTIONS:
        ref_sql = q.get("reference_sql") or ""
        ev = Evidence()
        if ref_sql:
            ev.calls = [ToolCall(name="datalab_sql_query",
                                 arguments={"sql": ref_sql},
                                 output='{"success": true}', ok=True, sql=ref_sql)]
            ev.sql_texts = [ref_sql]
        ev.trace_source = "tool_trace"

        r = QuestionResult(id=q["id"], tier=q["tier"], title=q["title"], prompt=q["prompt"])
        r.checkpoints = score_auto_checkpoints(q, ev)
        r.penalties = apply_penalties(q, ev)

        exp_total, exp_by_id, exp_pens = _V11_REFERENCE_SQL_SNAPSHOT[q["id"]]
        assert r.auto_earned == pytest.approx(exp_total, abs=0.01), q["id"]
        got_by_id = {c.id: c.earned for c in r.checkpoints if c.type == "auto"}
        assert got_by_id == pytest.approx(exp_by_id, abs=0.01), q["id"]
        assert sorted(p.id for p in r.penalties) == exp_pens, q["id"]
        # And the v1.2 rollup on the executed scores mutates nothing.
        import copy as _copy
        before = _copy.deepcopy([vars(c) for c in r.checkpoints])
        compute_axis_rollup(q, r)
        assert [vars(c) for c in r.checkpoints] == before, q["id"]


# ─────────────────────────────────────────────────────────────────────────────
# R9-round CX-03 (2nd reopen) — executed PENALTY-trigger snapshots (GP-01/02/
# 03 + question-specific QP-10a) and a positive DLB-04 (no-reference_sql)
# snapshot, closing the ungated paths.
# ─────────────────────────────────────────────────────────────────────────────
def _pens_for(qid, sql, text="There are 1523 sources in the result."):
    q = get_question(qid)
    ev = Evidence()
    ev.response_text = text
    ev.calls = [ToolCall(name="datalab_sql_query", arguments={"sql": sql},
                         output='{"success": true}', ok=True, sql=sql)]
    ev.sql_texts = [sql]
    ev.trace_source = "tool_trace"
    return sorted(p.id for p in apply_penalties(q, ev))


def test_cx03_penalty_trigger_snapshots():
    # GP-01: row-level scan with no q3c bound and no key equality.
    assert _pens_for(
        "DLB-02", "SELECT ra, dec FROM gaia_dr3.gaia_source LIMIT 100"
    ) == ["GP-01"]
    # GP-02: flat q3c_join without a MATERIALIZED small-side reduction.
    assert _pens_for(
        "DLB-09",
        "SELECT * FROM small s JOIN nsc_dr2.object o "
        "ON q3c_join(s.ra, s.dec, o.ra, o.dec, 0.0003)",
    ) == ["GP-02"]
    # GP-03 stacks with GP-01: a BETWEEN box as the only spatial bound is
    # still an unbounded row scan for the q3c index.
    assert _pens_for(
        "DLB-02",
        "SELECT ra, dec FROM nsc_dr2.object "
        "WHERE ra BETWEEN 10 AND 20 AND dec BETWEEN -5 AND 5",
    ) == ["GP-01", "GP-03"]
    # QP-10a (DLB-10 only): joining an external WISE catalog; the flat join
    # also trips GP-02.
    assert _pens_for(
        "DLB-10",
        "SELECT * FROM ls_dr9.tractor t JOIN allwise.source w "
        "ON q3c_join(t.ra, t.dec, w.ra, w.dec, 0.0003)",
    ) == ["GP-02", "QP-10a"]
    # Clean reference-style SQL triggers nothing.
    assert _pens_for(
        "DLB-02",
        "SELECT COUNT(*) FROM gaia_dr3.gaia_source "
        "WHERE q3c_radial_query(ra, dec, 229.022, -0.112, 0.166667)",
    ) == []


def test_cx03_dlb04_positive_snapshot_without_reference_sql():
    # DLB-04 has no reference_sql (image workflow) — pin its scoring against
    # tool-call evidence instead of the degenerate empty case.
    q = get_question("DLB-04")
    ev = Evidence()
    ev.response_text = "Rendered a gri color image of the center of M31."
    ev.calls = [ToolCall(name="datalab_color_image",
                         arguments={"ra": 10.6847, "dec": 41.2687, "fov_deg": 0.2},
                         output='{"success": true}', ok=True)]
    ev.trace_source = "tool_trace"
    r = QuestionResult(id=q["id"], tier=q["tier"], title=q["title"], prompt=q["prompt"])
    r.checkpoints = score_auto_checkpoints(q, ev)
    r.penalties = apply_penalties(q, ev)
    assert r.auto_earned == 50.0
    assert {c.id: c.earned for c in r.checkpoints if c.type == "auto"} == \
        {"C1": 20.0, "C2": 15.0, "C3": 15.0}
    assert r.penalties == []
