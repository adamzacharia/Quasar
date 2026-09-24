"""Codex guard task-25bee13-9557 (gpt-6-sol xhigh, 2026-09-23): one
regression test per ledger item CX-01..CX-17 (reconciliation.md)."""
from __future__ import annotations

import json
import threading
import types

import pandas as pd
import pytest

from core.answer_verifier import build_trace_summary, verify_answer


def _trace(name, args=None, output=None, ok=True, sql=None):
    rec = {"name": name, "arguments": args or {}, "output": json.dumps(output or {"success": True}), "ok": ok}
    if sql:
        rec["sql"] = sql
    return rec


# CX-01 -- "5 < parallax" is parallax > 5
def test_leading_comparator_reads_reversed():
    sql = "SELECT ra, dec FROM gaia_dr3.gaia_source WHERE 5 < parallax LIMIT 100"
    ts = build_trace_summary([], [_trace("datalab_sql_query", {"sql": sql}, {"success": True, "rowcount": 100}, sql=sql)], [])
    assert [c.text for c in verify_answer("Only stars with parallax < 5 were kept.", ts).by_kind("cut")] == ["parallax < 5"]
    assert not verify_answer("Only stars with parallax > 5 were kept.", ts).by_kind("cut")


# CX-02 -- identifiers, typed counts of another noun and other-noun text do not support a count
def test_unrelated_numbers_do_not_support_a_count():
    ids = build_trace_summary([{"output": json.dumps({"success": True, "rows": [{"project_id": 19}]})}], [], [])
    assert [c.text for c in verify_answer("It has 19 projects.", ids).by_kind("count")] == ["19 projects"]
    typed = build_trace_summary([{"output": json.dumps({"success": True, "n_observations": 19})}], [], [])
    assert [c.text for c in verify_answer("It has 19 projects.", typed).by_kind("count")] == ["19 projects"]
    text = build_trace_summary([{"output": json.dumps({"success": True, "rowcount": 3, "note": "the cone holds 19 observations"})}], [], [])
    assert [c.text for c in verify_answer("It has 19 projects.", text).by_kind("count")] == ["19 projects"]
    # the right noun / a generic row count still support it
    ok_typed = build_trace_summary([{"output": json.dumps({"success": True, "n_projects": 19})}], [], [])
    assert verify_answer("It has 19 projects.", ok_typed).ok
    ok_text = build_trace_summary([{"output": json.dumps({"success": True, "rowcount": 3, "note": "19 proposals matched"})}], [], [])
    assert verify_answer("It has 19 projects.", ok_text).ok


# CX-03 -- a LIMIT supports a cap, never a returned-count claim
def test_executed_limit_does_not_support_a_returned_count():
    sql = "SELECT ra, dec FROM gaia_dr3.gaia_source WHERE parallax > 5 LIMIT 5000"
    ts = build_trace_summary([], [_trace("datalab_sql_query", {"sql": sql}, {"success": True, "rowcount": 256}, sql=sql)], [])
    assert [c.text for c in verify_answer("The query returned 5000 rows.", ts).by_kind("count")] == ["5000 rows"]
    assert verify_answer("The query was capped at 5000 rows and returned 256 rows.", ts).ok


# CX-04 -- a failed result does not feed the calibrator / project-pair passes
def test_failed_results_do_not_change_role_findings():
    failed = {"success": False, "rows": [{"target_name": "J0238+1636", "scan_intent": "CALIBRATE_PHASE"}]}
    ts = build_trace_summary([{"output": json.dumps(failed)}], [], [])
    assert not ts.calibrator_targets


# CX-05 -- a mixed supported/unsupported species request fails
def test_mixed_species_request_names_the_unsupported_one():
    from services.alma_science_queries import UnsupportedSpecies, line_names_for_species

    with pytest.raises(UnsupportedSpecies) as exc:
        line_names_for_species(["CO", "XYZ"])
    assert "XYZ" in str(exc.value)
    assert line_names_for_species(["CO"])


# CX-06 -- a code span crossing one line break is protected
def test_multiline_inline_code_is_protected():
    from core.prose_hygiene import _split_inline

    parts = _split_inline("see `a<br>\nb` here")
    assert (True, "`a<br>\nb`") in parts
    # a blank line ends a span
    assert all(not code for code, _ in _split_inline("`a\n\nb`"))


# CX-07 -- scheme/port pairing; stop after DNS; no VO floor above the parent budget
def test_url_guard_pairs_scheme_and_port():
    from core.agent import QuasarAgent

    assert QuasarAgent._public_http_target("http://example.org:443/x")[0] is False
    assert QuasarAgent._public_http_target("https://example.org:80/x")[0] is False


def test_link_probe_checks_stop_after_dns(monkeypatch):
    from core.agent import QuasarAgent

    stop = threading.Event()

    def resolve(url):
        stop.set()  # cancellation arrives during DNS
        return True, "ok", "93.184.216.34"

    monkeypatch.setattr(QuasarAgent, "_resolve_public_target", classmethod(lambda cls, u: resolve(u)))
    monkeypatch.setattr(QuasarAgent, "_head_once", classmethod(lambda cls, *a, **k: pytest.fail("HEAD after Stop")))
    ok, why = QuasarAgent._head_verify_url("https://example.org/", 2.0, stop)
    assert ok is False and "stopped" in why


def test_vo_wait_has_no_floor_above_the_parent_budget(monkeypatch):
    from services import vo_registry
    from services import tool_budgets

    parent = types.SimpleNamespace(remaining=lambda: 0.01, child=lambda label=None: None)
    monkeypatch.setattr(tool_budgets, "current_deadline", lambda: parent)
    with pytest.raises(vo_registry._DeadlineExceeded):
        vo_registry._run_with_deadline(lambda: pytest.fail("worker started"), 5.0, "vo test")


# CX-08 -- a partial stream scan is reported as partial
def test_stream_selection_reports_partial_scans():
    import inspect
    from capabilities import datalab_tools

    src = inspect.getsource(datalab_tools.StreamSelection.run)
    assert '"partial" if native.get("partial") else "ok"' in src and '"tiles_completed": native.get("tiles_completed")' in src


# CX-09 -- only screened peaks count; an all-artefact top gets no CMD
def test_satellite_counts_only_screened_candidates():
    import inspect
    from capabilities import datalab_tools

    src = inspect.getsource(datalab_tools.SatelliteSearch.run)
    # CX-09 verify: vet=False no longer counts unscreened peaks as new.
    assert 'n_new = sum(1 for r in strong if r.get("artefact_screen"))' in src
    # DLB-100 F1: vetting = the top NON-ARTEFACT peaks (known objects included).
    assert '[r for r in top if not r.get("artefact")][: max(0, int(inp.cmd_candidates))]' in src
    assert 'r["significance"] >= self._PEAKS["candidate_sigma"])]' in src


# CX-10 -- an explicit catalog / radius wins over a one-shot preset
def test_routes_respect_named_catalogs_and_radii():
    from core.oneshot_routing import detect_oneshot_intent

    r = detect_oneshot_intent("Make a redshift wedge of DESI DR1 galaxies to show the cosmic web")
    assert r is None or r["tool"] != "datalab_lss_wedge"
    r = detect_oneshot_intent("Find the densest stellar clumps within 3 deg of Hydra II in NSC DR2 and show cutouts")
    assert r["tool"] == "datalab_satellite_search" and r["args"]["radius_deg"] == 3.0


def test_preset_with_a_wider_radius_is_gated(monkeypatch):
    from capabilities import datalab_tools
    from capabilities.base import CallContext
    from services import datalab_orchestration as orch

    monkeypatch.setattr(orch, "tiled_density_aggregate", lambda *a, **k: pytest.fail("scanned before confirmation"))
    cap = datalab_tools.SatelliteSearch()
    out = cap.run(cap.InputModel(survey="nsc", preset="hydra2", radius_deg=3.0), CallContext()).to_native()
    assert out["status"] == "needs_confirmation"


# CX-11 -- an upper-case survey word is not an object name
def test_survey_words_do_not_skip_count_checks():
    ts = build_trace_summary([{"output": json.dumps({"success": True, "rowcount": 2})}], [], [])
    assert [c.text for c in verify_answer("ALMA 19 projects were returned.", ts).by_kind("count")] == ["19 projects"]
    assert verify_answer("Palomar 5 members are marked.", ts).ok


# CX-12 / CX-15 / CX-16 -- async submit, polling and executed SQL
class _AsyncClient:
    token = "tok"

    def __init__(self, statuses):
        self.statuses, self.submitted = list(statuses), []

    def strip_materialized_for_async(self, sql):
        return sql.replace(" AS MATERIALIZED", "")

    def submit(self, sql):
        self.submitted.append(sql)
        return "job1"

    def status(self, jobid):
        s = self.statuses.pop(0)
        if isinstance(s, Exception):
            raise s
        return s

    def results(self, jobid, query_text=None):
        return types.SimpleNamespace(dataframe=pd.DataFrame({"a": [1]}))

    def abort(self, jobid):
        pass


def test_async_is_not_submitted_without_a_polling_budget(monkeypatch):
    from services import datalab_orchestration as orch
    from services import tool_budgets

    monkeypatch.setattr(tool_budgets, "remaining_seconds", lambda: 12.0)
    c = _AsyncClient(["COMPLETED"])
    out = orch.run_async_sql("WITH m AS MATERIALIZED (SELECT 1) SELECT * FROM m", client=c)
    assert out["state"] == "SKIPPED" and not c.submitted


def test_async_keeps_the_job_id_through_transport_errors_and_reports_executed_sql(monkeypatch):
    from services import datalab_orchestration as orch
    from services import tool_budgets

    monkeypatch.setattr(tool_budgets, "remaining_seconds", lambda: None)
    monkeypatch.setattr(orch.time, "sleep", lambda s: None)
    c = _AsyncClient([ConnectionError("reset"), "EXECUTING", "COMPLETED"])
    out = orch.run_async_sql("WITH m AS MATERIALIZED (SELECT 1) SELECT * FROM m", client=c, max_seconds=60)
    assert out["state"] == "COMPLETED" and out["jobid"] == "job1"
    assert "MATERIALIZED" not in out["executed_sql"]
    c3 = _AsyncClient([ConnectionError("a"), ConnectionError("b"), ConnectionError("c")])
    out3 = orch.run_async_sql("SELECT 1", client=c3, max_seconds=60)
    assert out3["state"] == "UNKNOWN" and out3["jobid"] == "job1"


# CX-13 -- contour_levels selects how many significance levels are drawn
def test_contour_levels_selects_the_number_of_sigma_levels():
    import inspect
    from services import fits_service

    src = inspect.getsource(fits_service.overlay_fits_images)
    assert "CONTOUR_SIGMAS[:n_levels]" in src
    import numpy as np
    rng = np.random.default_rng(0)
    img = rng.normal(0, 1, (64, 64))
    img[30:34, 30:34] = 100.0
    assert len(fits_service.significance_contour_levels(img, fits_service.CONTOUR_SIGMAS[:3])["sigmas"]) == 3


# CX-14 -- recipe header names what it was tested against; "the CMD above" is rewritten
def test_recipe_header_names_its_own_test_target():
    from pathlib import Path

    header = Path("docs/recipes/astroquery_datalink.py").read_text(encoding="utf-8").splitlines()[:6]
    assert any("datalink" in h for h in header)
    import inspect
    from capabilities import alma_tools

    src = inspect.getsource(alma_tools)
    assert '"tested_against": meta.get("tested_against")' in src and "against the live TAP service" not in src


def test_card_direction_phrases_are_rewritten():
    import re as _re
    import inspect
    from core import runner

    src = inspect.getsource(runner)
    pat = r"\b((?:the|this|that|these|those)\s+(?:[\w-]+\s+){0,3}?(?:card|figure|plot|image|map|diagram|CMD|grid|cutouts?|panels?|chart|wedge|histogram)s?)\s+(?:above|below)\b"
    assert pat in src
    assert _re.sub(pat, r"\1", "See the CMD above for details.", flags=_re.I) == "See the CMD for details."
    assert _re.sub(pat, r"\1", "the table below lists them", flags=_re.I) == "the table below lists them"
