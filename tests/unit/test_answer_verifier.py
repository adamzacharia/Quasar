"""core/answer_verifier.py — prose must match the trace.

Fixtures are the saved UI answers and executed queries of the 2026-09-22
benchmark (tmp/ui-bench-2026-09-22/**/answer.md, queries.md). The four
evidence turns must be flagged for the reason the graders gave; the clean
turns (L02, L14, L10 re-run) must produce no unsupported claims.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from core.answer_verifier import (
    TraceSummary,
    build_trace_summary,
    format_verification_block,
    verify_answer,
)

REPO = Path(__file__).resolve().parents[2]
BENCH = REPO / "tmp" / "ui-bench-2026-09-22"


def _read(rel: str) -> str:
    path = BENCH / rel
    if not path.exists():
        pytest.skip(f"benchmark fixture missing: {rel}")
    return path.read_text(encoding="utf-8", errors="replace")


def _queries(rel: str):
    """Parse queries.md: 'ADQL\\n<tool>\\n<sql...>' and 'ARGS\\n<tool>\\n{json}' blocks."""
    text = _read(rel)
    sql, args = [], []
    lines = text.split("\n")
    i = 0
    while i < len(lines):
        tag = lines[i].strip()
        if tag in ("ADQL", "ARGS", "PARAMS"):
            i += 1
            if i < len(lines) and re.match(r"^[a-z_]+$", lines[i].strip()):
                i += 1  # tool name line
            body = []
            while i < len(lines) and lines[i].strip() not in ("ADQL", "ARGS", "PARAMS", "Show query", "Hide query", "==="):
                body.append(lines[i])
                i += 1
            blob = "\n".join(body).strip()
            if tag == "ADQL":
                sql.append(blob)
            else:
                args.append(blob)
        else:
            i += 1
    return sql, args


def _summary(sql=(), args=(), counts=(), cards=(), results=()):
    ts = build_trace_summary(list(results), [], [dict(c) for c in cards], extra_sql=list(sql))
    for a in args:
        ts.arg_texts.append(a)
        ts.query_arg_texts.append(a)  # fixture args come from data tools
    for c in counts:
        ts.counts.add(int(c))
    return ts


# ── evidence turns: must be flagged ──────────────────────────────────────


def test_l09_rerun_flags_cuts_the_sql_never_applied():
    answer = _read("datalab-rerun/L09-DLB-09/answer.md")
    sql, args = _queries("datalab-rerun/L09-DLB-09/queries.md")
    ts = _summary(sql=sql, args=args, counts=[5000, 10000],
                  cards=[{"type": "image", "title": "Palomar 5 tidal tail candidate stars (Gaia+NSC)", "requestTool": "plot_sky_map"}])
    report = verify_answer(answer, ts)
    cuts = " | ".join(c.text for c in report.by_kind("cut"))
    assert "class_star > 0.5" in cuts, cuts
    assert re.search(r"16 < phot_g_mean_mag < 20", cuts), cuts
    assert re.search(r"0\.3 < bp_rp < 0\.9", cuts), cuts
    # The proper-motion window WAS applied in the SQL -> not flagged.
    assert not re.search(r"pmra\s*<\s*0(?![.\d])", cuts) or "pmra > -4" not in cuts
    assert not any("pmra" in c.text and "-4" in c.text.replace("–", "-") for c in report.by_kind("cut")), cuts
    assert not report.by_kind("artifact"), "the sky map card exists"
    block = format_verification_block(report)
    assert "Verification" in block and "class_star > 0.5" in block


def test_l11_flags_the_map_that_was_never_rendered_but_accepts_the_histogram():
    answer = _read("datalab/L11-DLB-11/answer.md")
    sql, args = _queries("datalab/L11-DLB-11/queries.md")
    ts = _summary(sql=sql, args=args, counts=[335438, 368905, 418387, 484787, 4, 5000],
                  cards=[{"type": "plotly", "title": "Redshift histogram of DESI DR1 LRGs", "requestTool": "datalab_catalog_scatter"}])
    report = verify_answer(answer, ts)
    artifacts = report.by_kind("artifact")
    assert any("map" in c.text.lower() for c in artifacts), [c.text for c in artifacts]
    assert not any("histogram" in c.text.lower() and "map" not in c.text.lower() for c in artifacts), [c.text for c in artifacts]
    # The z window and the LRG bit ARE in the executed SQL.
    assert not any("0.4" in c.text and "0.8" in c.text for c in report.by_kind("cut")), [c.text for c in report.by_kind("cut")]


def test_d18_flags_a_null_result_stated_over_a_timed_out_alma_phase():
    answer = _read("domain/D18-AM-H-01/answer.md")
    result = {
        "success": True, "partial": True,
        "archive_errors": {"ALMA": "ALMA phase timed out (budget exhausted) — 0 of 12 sources checked"},
        "archive_status": {"ALMA": "timeout", "JWST": "ok"},
        "matches": [], "n_sources": 12, "sources_with_jwst": 12,
    }
    ts = build_trace_summary([{"output": json.dumps(result)}], [], [])
    assert ts.timed_out_phases
    report = verify_answer(answer, ts)
    nulls = report.by_kind("null_result")
    assert any("no Perseus protostars" in c.text for c in nulls), [c.text for c in nulls]


def test_d10_flags_a_calibrator_presented_as_the_science_target():
    answer = _read("domain/D10-DS-E-01/answer.md")
    rows = [
        {"target_name": "J0238+1636", "science_observation": "F", "scan_intent": "CALIBRATE_PHASE", "proposal_id": "2023.1.01040.S"},
        {"target_name": "Sun", "science_observation": "T", "scan_intent": "TARGET", "proposal_id": "2023.1.01040.S"},
    ]
    result = {"success": True, "rows": rows, "n_projects": 1, "unique_projects": 1, "rowcount": 80, "n_mous": 7, "n_eb": 7}
    ts = build_trace_summary([{"output": json.dumps(result)}], [], [])
    assert "J0238+1636" in ts.calibrator_targets and "Sun" in ts.science_targets
    report = verify_answer(answer, ts)
    assert any(c.kind == "calibrator" and c.text == "J0238+1636" for c in report.unsupported), [(c.kind, c.text) for c in report.unsupported]


def test_d11_flags_the_window_count_reported_as_the_matching_count():
    answer = _read("domain/D11-DS-M-01/answer.md")
    result = {"success": True, "n_projects": 30, "unique_projects": 1, "rowcount": 5000, "n_mous": 225, "n_eb": 264,
              "note": "TOP 5000 rows", "partial": True}
    ts = build_trace_summary([{"output": json.dumps(result)}], [], [])
    assert (30, 1) in ts.count_pairs
    report = verify_answer(answer, ts)
    assert any(c.kind == "count" and c.text == "30 projects" for c in report.unsupported), [(c.kind, c.text) for c in report.unsupported]


def test_l15_placeholders_and_generic_plot_claims_are_not_artifact_failures():
    answer = _read("datalab/L15-DLB-15/answer.md")
    cards = [{"type": "image", "title": f"CMD {i}"} for i in (1, 2, 3)] + [{"type": "image", "title": f"g-band cutout {i}"} for i in (1, 2, 3)]
    ts = _summary(counts=[221, 199, 188, 5, 3, 2, 1000, 5000], cards=cards,
                  args=[json.dumps({"ra": 180, "dec": 30, "radius": 2.0, "class_star_min": 0.5, "radius_deg": 0.2, "size_deg": 0.05})])
    report = verify_answer(answer, ts)
    assert not report.by_kind("artifact"), [c.text for c in report.by_kind("artifact")]


# ── clean turns: nothing flagged ─────────────────────────────────────────


def test_l02_cone_count_is_clean():
    answer = _read("datalab/L02-DLB-02/answer.md")
    sql, args = _queries("datalab/L02-DLB-02/queries.md")
    ts = _summary(sql=sql, args=args, counts=[1313], cards=[{"type": "data", "title": "Cone count"}])
    report = verify_answer(answer, ts)
    assert report.ok, [(c.kind, c.text, c.detail) for c in report.unsupported]


def test_l14_light_curve_turn_is_clean():
    answer = _read("datalab/L14-DLB-14/answer.md")
    sql, args = _queries("datalab/L14-DLB-14/queries.md")
    args = list(args) + [json.dumps({"ra": 185.4311, "dec": -31.9953, "radius_arcsec": 1, "size_deg": 0.1, "band": "g", "min_period": 0.1, "max_period": 1.0})]
    ts = _summary(sql=sql, args=args, counts=[381, 127, 3],
                  cards=[{"type": "image", "title": "Phase-folded g-band light curve (P = 0.6487 d)", "requestTool": "datalab_period_fold"},
                         {"type": "image", "title": "Legacy Surveys DR9 g-band cutout", "requestTool": "datalab_image_cutout"}])
    report = verify_answer(answer, ts)
    assert report.ok, [(c.kind, c.text, c.detail) for c in report.unsupported]


def test_l10_rerun_sed_turn_is_clean():
    answer = _read("datalab-rerun/L10-DLB-10/answer.md")
    sql, args = _queries("datalab-rerun/L10-DLB-10/queries.md")
    ts = _summary(sql=sql, args=args, counts=[300], cards=[{"type": "image", "title": "SED: magnitude vs wavelength (300 red galaxies)"}])
    report = verify_answer(answer, ts)
    assert report.ok, [(c.kind, c.text, c.detail) for c in report.unsupported]


# ── unit behaviour ───────────────────────────────────────────────────────


def test_knowledge_answer_without_tools_is_never_flagged():
    ts = TraceSummary()
    answer = "ALMA has 10 receiver bands from 35 GHz to 950 GHz. Band 6 covers 211–275 GHz; sensitivity scales as 1/sqrt(N(N-1) Δν t)."
    assert verify_answer(answer, ts).ok


def test_cuts_inside_code_blocks_are_not_claims():
    ts = _summary(sql=["SELECT ra FROM gaia_dr3.gaia_source WHERE parallax > 5 LIMIT 100"], counts=[100])
    answer = "Here is the query you can run:\n```sql\nSELECT * FROM t WHERE pmra > 10 AND class_star > 0.9 LIMIT 5000\n```\nIt keeps sources with parallax > 5."
    report = verify_answer(answer, ts)
    assert report.ok, [(c.kind, c.text) for c in report.unsupported]


def test_shown_below_with_no_card_is_flagged_and_block_is_short():
    ts = _summary(sql=["SELECT COUNT(*) FROM x"], counts=[10])
    answer = "The colour–magnitude diagram is shown below.\n\nWe found 10 sources."
    report = verify_answer(answer, ts)
    assert [c.kind for c in report.unsupported] == ["artifact"]
    block = format_verification_block(report)
    assert block.count("\n> - ") == 1 and "figure/card claimed" in block


def test_hedged_null_statement_over_a_timeout_is_accepted():
    ts = build_trace_summary([{"output": json.dumps({"success": False, "timeout": True, "error": "ALMA TAP timed out"})}], [], [])
    answer = "The ALMA archive could not be queried (timed out), so whether these sources have ALMA data is unknown."
    assert verify_answer(answer, ts).ok


def test_count_not_in_any_result_is_flagged_but_band_and_cycle_numbers_are_ignored():
    ts = _summary(sql=["SELECT proposal_id FROM ivoa.obscore WHERE band_list LIKE '%6%'"], counts=[12])
    answer = "Band 6 data from Cycle 9: the archive lists 95 projects; 12 datasets were public."
    report = verify_answer(answer, ts)
    assert [c.text for c in report.by_kind("count")] == ["95 projects"]



def test_numbers_quoted_from_tool_result_text_are_supported():
    """Live 2026-09-23 L10: '5 000 rows' came from a platform row-cap note."""
    ts = build_trace_summary([{"output": json.dumps({"success": True, "rowcount": 500,
                                                     "warnings": ["Platform row cap: at most 5000 rows per query"]})}], [], [])
    assert verify_answer("The query can return at most 5 000 rows; 500 rows came back.", ts).ok


def test_verification_block_closes_its_list():
    ts = _summary(sql=["SELECT COUNT(*) FROM x"], counts=[10])
    block = format_verification_block(verify_answer("The map is shown below.", ts))
    assert "\n>\n> Treat these as unverified" in block
