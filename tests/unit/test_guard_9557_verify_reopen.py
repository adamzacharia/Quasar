"""Regression tests for the Codex verify of task-25bee13-9557 (2026-09-24,
gpt-6-sol high): the ten reopened findings CX-02 CX-03 CX-04 CX-09 CX-10
CX-11 CX-12 CX-14 CX-15 CX-16."""
from __future__ import annotations

import json

import pytest

from core.answer_verifier import build_trace_summary, verify_answer


def _counts(report):
    return [c.text for c in report.by_kind("count")]


# CX-02: an unkeyed row count of observation rows is not a project count;
# plural stems match their own named count fields.
def test_cx02_generic_rowcount_does_not_support_projects_without_project_grain():
    obs = build_trace_summary([{"output": json.dumps({"success": True, "rowcount": 19})}], [], [],
                              extra_sql=["SELECT obs_id, target_name FROM ivoa.obscore WHERE band_list LIKE '%6%'"])
    assert "19 projects" in _counts(verify_answer("The archive lists 19 projects.", obs))
    # the grain SQL must belong to the SAME call as the count (guard 12894 CX-10)
    grain = build_trace_summary([{"output": json.dumps({"success": True, "rowcount": 19,
                                 "validated_sql": "SELECT DISTINCT proposal_id FROM ivoa.obscore WHERE band_list LIKE '%6%'"})}], [], [])
    assert _counts(verify_answer("The archive lists 19 projects.", grain)) == []
    # a generic row count still supports rows / sources
    assert _counts(verify_answer("The query returned 19 rows.", obs)) == []


def test_cx02_plural_stem_matches_named_count_field():
    ts = build_trace_summary([{"output": json.dumps({"success": True, "alma_galaxies": 19})}], [], [],
                             extra_sql=["SELECT COUNT(*) FROM x"])
    assert _counts(verify_answer("There are 19 galaxies with ALMA data.", ts)) == []


# CX-03: a cap word does not turn the following RESULT claim into a cap claim.
def test_cx03_cap_then_returned_count_is_checked_separately():
    ts = build_trace_summary([{"output": json.dumps({"success": True, "rowcount": 256})}], [], [],
                             extra_sql=["SELECT ra, dec FROM gaia_dr3.gaia_source WHERE q3c_radial_query(ra, dec, 1, 2, 0.1) LIMIT 5000"])
    flagged = _counts(verify_answer("The query was capped at 5000 rows and returned 5000 rows.", ts))
    assert flagged == ["5000 rows"]
    assert _counts(verify_answer("The query was capped at 5000 rows and returned 256 rows.", ts)) == []


# CX-04: a failed call supports nothing even when its output has no success field.
def test_cx04_failed_trace_output_does_not_support_via_tool_results():
    failed = json.dumps({"n_projects": 30, "rowcount": 30})
    trace = [{"name": "alma_project_census", "ok": False, "output": failed, "arguments": {"cycle": 9}}]
    ts = build_trace_summary([{"output": failed}], trace, [], extra_sql=["SELECT COUNT(*) FROM ivoa.obscore"])
    assert 30 not in ts.counts
    assert "30 projects" in _counts(verify_answer("Cycle 9 has 30 projects.", ts))


# CX-11: survey names in front of a plain count are not object names.
def test_cx11_survey_name_before_a_count_is_still_checked():
    ts = build_trace_summary([{"output": json.dumps({"success": True, "rowcount": 7})}], [], [],
                             extra_sql=["SELECT COUNT(*) FROM sdss_dr17.specobj"])
    assert "19 galaxies" in _counts(verify_answer("SDSS 19 galaxies were returned.", ts))
    # a survey-named cluster (one digit) is still an object name
    assert _counts(verify_answer("We selected DES 1 members from the field.", ts)) == []


# CX-12: a zero budget never submits a server job.
def test_cx12_zero_budget_never_submits():
    from services import datalab_orchestration as orch

    class _Client:
        token = "real-token"

        def submit(self, **k):
            raise AssertionError("must not submit")

    out = orch.run_async_sql("SELECT 1", client=_Client(), max_seconds=0)
    assert out["state"] == "SKIPPED"


# CX-14: the card-direction rewrite leaves fenced code alone.
def test_cx14_direction_rewrite_skips_code_blocks():
    import threading
    from types import SimpleNamespace as NS

    import core.runner as runner
    from core.agent import QuasarAgent

    agent = QuasarAgent.__new__(QuasarAgent)
    agent._tls = threading.local()
    agent.tool_registry = NS(list_tools=lambda: [])
    agent._accumulated_tool_trace = []
    agent._accumulated_run_results = [{"type": "image", "image_url": "/plots/cmd.png", "caption": "CMD"}]
    agent.last_run_result = None
    text = "The CMD is shown above.\n\n```python\n# See the CMD above\nprint('shown above')\n```\n"
    out = runner._finalize_answer_text(agent, text, on_token=None, user_query="plot a CMD", url_sources=[],
                                       all_tool_results=[], had_tool_calls=False)
    assert "The CMD is shown in the card." in out
    assert "# See the CMD above" in out and "print('shown above')" in out


# CX-10: other radius phrasings route; a tiny radius is refused, not enlarged.
def test_cx10_radius_phrasings_and_small_radius():
    from capabilities import datalab_tools
    from capabilities.base import CallContext
    from core.oneshot_routing import detect_oneshot_intent

    q = "Find the densest stellar clumps in a 0.2 degree radius around Hydra II and pull cutouts so I can eyeball them."
    intent = detect_oneshot_intent(q)
    assert intent["tool"] == "datalab_satellite_search" and intent["args"]["radius_deg"] == pytest.approx(0.2)
    cap = datalab_tools.SatelliteSearch()
    out = cap.run(cap.InputModel(preset="hydra2", radius_deg=0.01), CallContext()).to_native()
    assert out["success"] is False and out["status"] == "invalid_region"


# CX-15 / CX-16: the async job survives a fallback; the map publishes the SQL that ran.
def test_cx15_cx16_healpix_async_job_survives_fallback(monkeypatch):
    import pandas as pd

    from capabilities import datalab_tools
    from capabilities.base import CallContext
    from services import datalab_orchestration as orch
    from services.datalab_result_store import DatalabResultStore

    store = DatalabResultStore(enable_disk_cache=False)
    monkeypatch.setenv("DATALAB_HEALPIX_ASYNC", "1")
    monkeypatch.setattr(orch, "async_density_aggregate", lambda *a, **k: {
        "success": False, "jobid": "job-42", "job_state": "UNKNOWN", "error": "status polling failed",
        "note": "fetch it with datalab_job_results jobid=job-42"})
    rid = store.put(pd.DataFrame({"healpix": [1, 2], "source_count": [10, 20]}), {"provenance": {}})
    monkeypatch.setattr(orch, "tiled_density_aggregate", lambda *a, **k: {
        "success": True, "result_id": rid, "tiles_completed": 4, "tiles_total": 4, "warnings": []})
    monkeypatch.setattr(datalab_tools, "_run_analysis_plot",
                        lambda name, args, ctx: type("R", (), {"to_native": lambda self: {"success": True}})())
    cap = datalab_tools.HealpixDensityMap()
    class _AbortableClient:
        def abort(self, jobid):  # CX-09 (12894): an UNKNOWN job is aborted before the fallback
            return None

    ctx = CallContext(services={"datalab_client": _AbortableClient()}, result_store=store)
    out = cap.run(cap.InputModel(preset="south_gradient"), ctx).to_native()
    assert out["async_job"]["jobid"] == "job-42" and "job_results" in out["async_job"]["note"]
    assert out["validated_sql"].startswith("-- run as concurrent sub-cones")

    monkeypatch.setattr(orch, "async_density_aggregate", lambda *a, **k: {
        "success": True, "result_id": rid, "jobid": "job-43", "sql_example": "SELECT ring256 AS healpix ... LIMIT 20000"})
    out2 = cap.run(cap.InputModel(preset="south_gradient"), ctx).to_native()
    assert out2["validated_sql"] == "SELECT ring256 AS healpix ... LIMIT 20000"
    assert "async_job" not in out2
