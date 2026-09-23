"""core/answer_verifier.py -- WP1 guard findings CX-15..CX-23.

Each test pins one way the first verifier either supported a claim it should
not have (value borrowed from another predicate, any numeric leaf as a count,
a failed call's SQL) or flagged a correct one (a documentation fact, a MAST
null result beside an ALMA timeout, a calibrator described as a calibrator,
code in a tilde fence).
"""
from __future__ import annotations

import json

from core.answer_verifier import build_trace_summary, verify_answer


def _trace(name, args=None, output=None, ok=True, sql=None):
    rec = {"name": name, "arguments": args or {}, "output": json.dumps(output or {"success": True}), "ok": ok}
    if sql:
        rec["sql"] = sql
    return rec


# CX-15 -- a cut is supported only by ITS OWN predicate
def test_cut_value_cannot_be_borrowed_from_another_predicate():
    sql = "SELECT ra, dec FROM gaia_dr3.gaia_source WHERE pmra > -4 AND class_star > 0.2 LIMIT 100"
    ts = build_trace_summary([], [_trace("datalab_sql_query", {"sql": sql}, {"success": True, "rowcount": 100}, sql=sql)], [])
    bad = verify_answer("I kept point sources with class_star > -4.", ts)
    assert [c.text for c in bad.by_kind("cut")] == ["class_star > -4"]
    good = verify_answer("I kept point sources with class_star > 0.2 and pmra > -4.", ts)
    assert not good.by_kind("cut"), [c.text for c in good.by_kind("cut")]


def test_cut_direction_must_match_and_same_column_ranges_combine():
    sql = "SELECT * FROM desi_dr1.zpix WHERE z >= 0.4 AND z <= 0.8 AND parallax < 5"
    ts = build_trace_summary([], [_trace("datalab_sql_query", {"sql": sql}, {"success": True, "rowcount": 10}, sql=sql)], [])
    assert verify_answer("Selected 0.4 < z < 0.8.", ts).ok
    flagged = verify_answer("Only stars with parallax > 5 were kept.", ts)
    assert [c.text for c in flagged.by_kind("cut")] == ["parallax > 5"]


def test_json_argument_bounds_support_cuts():
    args = {"catalog": "gaia_dr3", "pm_total_min_mas_yr": 50, "z_range": [0.4, 0.8], "radius_deg": 5}
    ts = build_trace_summary([], [_trace("datalab_selection_diagram", args, {"success": True, "rowcount": 18})], [])
    assert verify_answer("Total proper motion pm > 50 mas/yr and z between 0.4 and 0.8 within a 5 deg radius.", ts).ok


# CX-16 / CX-17 -- counts need a COUNT field or a tool's own sentence
def test_arbitrary_numeric_leaves_do_not_support_counts():
    out = {"success": True, "rows": [{"ra": 185.4, "dec": 30, "band": 6}], "rowcount": 1}
    ts = build_trace_summary([{"output": json.dumps(out)}], [], [])
    report = verify_answer("The archive lists 30 projects.", ts)
    assert [c.text for c in report.by_kind("count")] == ["30 projects"]


def test_zero_and_one_are_checked():
    ts = build_trace_summary([{"output": json.dumps({"success": True, "rowcount": 4, "n_projects": 4})}], [], [])
    report = verify_answer("There is 1 match and 0 projects in Band 3.", ts)
    assert sorted(c.text for c in report.by_kind("count")) == ["0 projects", "1 match"]
    ts1 = build_trace_summary([{"output": json.dumps({"success": True, "rowcount": 1, "n_projects": 0})}], [], [])
    assert not verify_answer("There is 1 match and 0 projects in Band 3.", ts1).by_kind("count")


# CX-18 -- failed / rejected calls substantiate nothing
def test_failed_call_sql_and_args_do_not_support_claims():
    sql = "SELECT * FROM t WHERE class_star > 0.9 LIMIT 5000"
    failed = _trace("datalab_sql_query", {"sql": sql}, {"success": False, "error": "Unknown column class_star"}, ok=False, sql=sql)
    ok_sql = "SELECT COUNT(*) FROM t WHERE gmag < 22"
    ran = _trace("datalab_sql_query", {"sql": ok_sql}, {"success": True, "rowcount": 12, "count": 12}, sql=ok_sql)
    ts = build_trace_summary([], [failed, ran], [])
    report = verify_answer("Sources with class_star > 0.9 and gmag < 22: 12 sources.", ts)
    assert [c.text for c in report.by_kind("cut")] == ["class_star > 0.9"]
    assert "Unknown column class_star" in " ".join(ts.errors)


# CX-19 -- documentation facts are not selection cuts
def test_documentation_turn_facts_are_not_flagged_as_cuts():
    doc = {"success": True, "reference": "Band 6 receivers cover a frequency range of 211 - 275 GHz."}
    ts = build_trace_summary([], [_trace("alma_reference", {"topic": "band 6 frequency"}, doc)], [])
    assert verify_answer("Band 6 observes at frequency between 211 and 275 GHz.", ts).ok
    # a mixed turn: the doc fact is stated verbatim in a tool result -> supported
    sql = "SELECT proposal_id FROM ivoa.obscore WHERE band_list LIKE '%6%'"
    ts2 = build_trace_summary([], [_trace("alma_reference", {"topic": "band 6"}, doc),
                                   _trace("query_alma_science_archive", {"adql": sql}, {"success": True, "rowcount": 3}, sql=sql)], [])
    assert not verify_answer("Band 6 observes at frequency between 211 and 275 GHz.", ts2).by_kind("cut")


# CX-20 -- only the archive that timed out makes a null result unknown
def test_null_result_about_a_healthy_archive_is_not_flagged():
    out = {"success": True, "partial": True, "archive_errors": {"ALMA": "ALMA phase timed out"},
           "archive_status": {"ALMA": "timeout", "MAST": "ok"}}
    ts = build_trace_summary([{"output": json.dumps(out)}], [], [])
    assert "alma" in ts.timed_out_archives
    assert verify_answer("No MAST sources were found near the target.", ts).ok
    flagged = verify_answer("There are no ALMA observations of these protostars.", ts)
    assert flagged.by_kind("null_result")


# CX-21 -- a calibrator described as a calibrator is correct
def test_calibrator_named_as_calibrator_is_not_flagged():
    rows = [{"target_name": "J0238+1636", "science_observation": "F", "scan_intent": "CALIBRATE_PHASE"},
            {"target_name": "Sun", "science_observation": "T", "scan_intent": "TARGET"}]
    ts = build_trace_summary([{"output": json.dumps({"success": True, "rows": rows, "rowcount": 2})}], [], [])
    assert verify_answer("J0238+1636 is the phase calibrator, not the science target; the target is the Sun.", ts).ok
    wrong = verify_answer("The observed target was J0238+1636.", ts)
    assert [c.text for c in wrong.unsupported if c.kind == "calibrator"] == ["J0238+1636"]


# CX-22 -- every figure noun of the claim needs a card; 'distribution' is not a map
def test_compound_figure_claims_need_every_card():
    only_cmd = [{"type": "image", "title": "Colour-magnitude diagram", "requestTool": "datalab_color_magnitude_diagram"}]
    ts = build_trace_summary([], [], only_cmd)
    report = verify_answer("The CMD and sky-density map are shown above.", ts)
    assert report.by_kind("artifact") and "map" in report.by_kind("artifact")[0].detail
    hist = [{"type": "plotly", "title": "Redshift distribution", "requestTool": "datalab_catalog_scatter"}]
    assert verify_answer("The sky map is shown above.", build_trace_summary([], [], hist)).by_kind("artifact")
    both = only_cmd + [{"type": "image", "title": "Sky density map", "requestTool": "datalab_healpix_density_map"}]
    assert verify_answer("The CMD and sky-density map are shown above.", build_trace_summary([], [], both)).ok
    # nouns AFTER the claim phrase are not claimed
    cut = [{"type": "image", "title": "g-band cutouts", "requestTool": "datalab_cutout_grid"}]
    assert verify_answer("The cutouts below show the peaks found in the density map.", build_trace_summary([], [], cut)).ok


# CX-23 -- tilde and longer fences are code, not claims
def test_tilde_and_long_fences_are_stripped():
    sql = "SELECT ra FROM gaia_dr3.gaia_source WHERE parallax > 5 LIMIT 100"
    ts = build_trace_summary([], [_trace("datalab_sql_query", {"sql": sql}, {"success": True, "rowcount": 100}, sql=sql)], [])
    answer = ("Run this yourself:\n~~~sql\nSELECT * FROM t WHERE pmra > 10 AND class_star > 0.9\n~~~\n"
              "or\n````python\nq = \"WHERE ruwe < 1.2 LIMIT 5000\"\n```\nstill code\n````\n"
              "The executed query kept parallax > 5 (100 rows).")
    report = verify_answer(answer, ts)
    assert report.ok, [(c.kind, c.text) for c in report.unsupported]


def test_count_in_a_field_named_for_the_noun_is_supported():
    """Live 2026-09-23 D18: '252 (19 projects)' came from alma_projects=19 in the
    matcher's per-source rows; a same-valued field named for ANOTHER noun
    (dec=19) still does not support it."""
    rows = [{"source_name": "SVS 13", "alma_observations": 252, "alma_projects": 19, "jwst_observations": 4}]
    ts = build_trace_summary([{"output": json.dumps({"success": True, "rows": rows, "n_sources": 12})}], [], [])
    assert verify_answer("SVS 13 has 252 observations in 19 projects.", ts).ok
    other = build_trace_summary([{"output": json.dumps({"success": True, "rows": [{"dec": 19, "alma_observations": 3}]})}], [], [])
    assert [c.text for c in verify_answer("It has 19 projects.", other).by_kind("count")] == ["19 projects"]


def test_row_count_equal_to_an_executed_limit_is_supported():
    """Live 2026-09-23 L06: "limited to 5 000 rows" describes the executed LIMIT;
    SQL numbers still never support other counts."""
    sql = "SELECT ra, dec FROM gaia_dr3.gaia_source WHERE parallax > 5 LIMIT 5000"
    ts = build_trace_summary([], [_trace("datalab_sql_query", {"sql": sql}, {"success": True, "rowcount": 256}, sql=sql)], [])
    assert verify_answer("The query was limited to 5 000 rows and returned 256 rows.", ts).ok
    assert [c.text for c in verify_answer("We found 5000 sources.", ts).by_kind("count")] == ["5000 sources"]



def test_generic_image_noun_is_satisfied_by_any_figure_card():
    """Live 2026-09-23 L08: "The image attached above displays the log-scaled
    density field" was flagged although the HEALPix map card was attached."""
    cards = [{"type": "image", "title": "nsc_dr2 stellar density, HEALPix nside 256", "requestTool": "datalab_healpix_density_map"}]
    assert verify_answer("The image attached above displays the log-scaled density field.", build_trace_summary([], [], cards)).ok
    assert verify_answer("The image attached above shows it.", build_trace_summary([], [], [])).by_kind("artifact")



def test_equality_on_an_echoed_tool_parameter_is_supported_but_not_an_inequality():
    """Live 2026-09-23 L08: "nside = 256" came from the tool's own nside=256."""
    out = {"success": True, "nside": 256, "cells": 2475, "rowcount": 2475}
    ts = build_trace_summary([], [_trace("datalab_healpix_density_map", {"catalog": "nsc_dr2", "preset": "ngp"}, out)], [])
    assert not verify_answer("The map uses HEALPix nside = 256.", ts).by_kind("cut")
    assert verify_answer("Only cells with nside > 256 were kept.", ts).by_kind("cut")
