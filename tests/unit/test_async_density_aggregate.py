"""Whole-region density aggregate as ONE Data Lab async job (UI 2026-09-23,
L08): dense wide fields such as the LMC never finish in the 60 s sync window
whatever the tiling (a 0.5 deg HEALPix tile at the LMC centre timed out),
while the whole 20x20 deg aggregate completed as one async job in 199 s.
Network is faked."""
from __future__ import annotations

import pandas as pd

from integrations.datalab_client import ANON_TOKEN, DatalabResult
from services import datalab_orchestration as orch
from services import tool_budgets as tb


class _Store:
    def __init__(self):
        self.saved = []

    def put(self, df, meta):
        self.saved.append((df, meta))
        return f"dlr_async{len(self.saved)}"


class _AsyncClient:
    timeout = 60.0

    def __init__(self, states, token="real.user.token", cells=3):
        self.token = token
        self.states = list(states)
        self.submitted = []
        self.aborted = []
        self.cells = cells

    def submit(self, *, sql=None, adql=None, timeout=None):
        self.submitted.append(sql)
        return "job123"

    def status(self, jobid):
        return self.states.pop(0) if self.states else "EXECUTING"

    def error(self, jobid):
        return "syntax error near GROUP"

    def abort(self, jobid):
        self.aborted.append(jobid)
        return "ABORTED"

    def results(self, jobid, *, fmt="pandas", query_text=None):
        df = pd.DataFrame({"healpix": list(range(self.cells)), "source_count": [50000 - i for i in range(self.cells)]})
        return DatalabResult.from_dataframe(df, {"query": query_text})

    def query(self, **kw):  # pragma: no cover - the async path never uses sync
        raise AssertionError("sync query must not be used")


def _run(client, **kw):
    store = _Store()
    out = orch.async_density_aggregate(
        "nsc_dr2", "object", mode="healpix", healpix_column="ring256", ra=80.894, dec=-69.756, radius_deg=10.0,
        predicates=["class_star > 0.5"], client=client, result_store=store, poll_seconds=0.01, **kw,
    )
    return out, store


def test_completed_job_is_stored_with_async_provenance_and_the_whole_cone_sql():
    client = _AsyncClient(["QUEUED", "EXECUTING", "COMPLETED"])
    out, store = _run(client, max_seconds=5.0)
    assert out["success"] and out["result_id"] == "dlr_async1" and out["rowcount"] == 3 and not out["partial"]
    sql = client.submitted[0]
    assert "q3c_radial_query(ra, dec, 80.894, -69.756, 10)" in sql and "GROUP BY ring256" in sql
    assert "LIMIT 20000" in sql, "whole-region maps are not clipped to the 5000-cell per-tile cap"
    df, meta = store.saved[0]
    assert meta["provenance"]["mode"] == "healpix_async" and meta["provenance"]["jobid"] == "job123"
    assert meta["provenance"]["healpix"]["nside"] == 256


def test_a_job_still_running_at_the_budget_is_reported_with_its_jobid():
    client = _AsyncClient(["EXECUTING"] * 1000)
    out, store = _run(client, max_seconds=0.2)
    assert out["success"] is False and out["jobid"] == "job123" and out["budget_exhausted"] is True
    assert "datalab_job_results" in out["note"] and not store.saved


def test_job_error_and_anonymous_token():
    out, _ = _run(_AsyncClient(["ERROR"]), max_seconds=5.0)
    assert out["success"] is False and "syntax error" in out["error"] and not out.get("budget_exhausted")
    out, _ = _run(_AsyncClient([], token=ANON_TOKEN), max_seconds=5.0)
    assert out.get("async_unavailable") is True


def test_turn_cancellation_aborts_the_job():
    turn = tb.TurnCancellation("async")
    d = tb.make_tool_deadline("datalab_healpix_density_map", 60.0, turn=turn)
    tb.adopt_deadline(d)
    try:
        client = _AsyncClient(["EXECUTING"] * 100)
        real_status = client.status

        def _status(jobid):  # Stop is pressed while the job is polling
            turn.cancel("stop pressed")
            return real_status(jobid)

        client.status = _status
        out, _ = _run(client, max_seconds=5.0)
    finally:
        tb.end_tool_deadline()
    assert out["success"] is False and "cancelled" in out["error"] and client.aborted == ["job123"]


def test_a_cancelled_turn_never_submits_a_job():
    """Guard task-25bee13-9557 CX-12: no server job is started for a turn that
    is already cancelled (or has no polling budget)."""
    turn = tb.TurnCancellation("async")
    d = tb.make_tool_deadline("datalab_healpix_density_map", 60.0, turn=turn)
    tb.adopt_deadline(d)
    try:
        turn.cancel("stop pressed")
        client = _AsyncClient(["EXECUTING"] * 100)
        out, _ = _run(client, max_seconds=5.0)
    finally:
        tb.end_tool_deadline()
    assert out["success"] is False and "cancelled" in out["error"] and client.aborted == [] and not getattr(client, "submitted", [])


def test_healpix_tool_uses_sync_tiles_by_default_and_async_only_when_enabled(monkeypatch):
    """Live 2026-09-23: the Data Lab async queue left even a sparse NGP
    aggregate EXECUTING after 240 s, while 69 concurrent sync tiles finished in
    82 s -- so the async whole-region job is opt-in (DATALAB_HEALPIX_ASYNC=1)."""
    from capabilities import datalab_tools
    from capabilities.base import CallContext

    calls = []
    monkeypatch.delenv("DATALAB_HEALPIX_ASYNC", raising=False)
    monkeypatch.setattr(orch, "async_density_aggregate", lambda *a, **k: calls.append("async") or {"success": False})
    monkeypatch.setattr(orch, "tiled_density_aggregate", lambda *a, **k: calls.append("tiled") or
                        {"success": False, "error": "no tiles", "budget_exhausted": False})
    cap = datalab_tools.HealpixDensityMap()
    cap.run(cap.InputModel(catalog="nsc_dr2"), CallContext(services={"datalab_client": object()}, result_store=_Store()))
    assert calls == ["tiled"]


def test_healpix_tool_prefers_the_async_job_for_wide_regions_when_enabled(monkeypatch):
    from capabilities import datalab_tools
    from capabilities.base import CallContext

    monkeypatch.setenv("DATALAB_HEALPIX_ASYNC", "1")
    calls = []
    monkeypatch.setattr(orch, "async_density_aggregate", lambda *a, **k: calls.append("async") or
                        {"success": False, "async_unavailable": True})
    monkeypatch.setattr(orch, "tiled_density_aggregate", lambda *a, **k: calls.append("tiled") or
                        {"success": False, "error": "no tiles", "budget_exhausted": False})
    cap = datalab_tools.HealpixDensityMap()
    out = cap.run(cap.InputModel(catalog="nsc_dr2", preset="lmc"), CallContext(services={"datalab_client": object()}, result_store=_Store()))
    out = out.to_native() if hasattr(out, "to_native") else out
    assert calls == ["async", "tiled"], "async first; tiled fallback when async is unavailable"
    assert out["success"] is False


def test_async_submission_drops_postgres_casts_the_async_parser_rejects():
    """Live 2026-09-23: the async query manager (JSQLParser) failed with
    "Lexical error ... Encountered ':'" on the NaN guards `col < 'Infinity'::float8`.
    A bare special-float literal keeps the semantics; other text is untouched."""
    from integrations.datalab_client import DatalabClient

    q = ("SELECT x FROM t WHERE (gmag > 16 AND gmag < 'Infinity'::float8) AND note = 'a::float8' "
         "AND y < 'NaN'::double precision")
    out = DatalabClient.strip_materialized_for_async(q)
    assert out == "SELECT x FROM t WHERE (gmag > 16 AND gmag < 'Infinity') AND note = 'a::float8' AND y < 'NaN'"
    assert "::" not in out.replace("'a::float8'", "")
