# tests/unit/test_datalab_async_my_tables.py
"""First-class async Data Lab jobs (async_submit → job_id up front; server jobs
with a real token, local threaded runner with the anonymous token; prefix-routed
job tools) and My-tables (durable named tables on the result store)."""

from __future__ import annotations

import time

import pandas as pd
import pytest

import capabilities.datalab as dl
from capabilities.base import CallContext
from integrations.datalab_client import ANON_TOKEN, DatalabResult
from services.datalab_job_service import DatalabJobService
from services.datalab_result_store import DatalabResultStore


# ── fakes ─────────────────────────────────────────────────────────────────────
class _FakeClient:
    """Anonymous-token Data Lab client fake; records SQL, serves one frame."""

    def __init__(self, df=None, token=ANON_TOKEN):
        self.token = token
        self.df = pd.DataFrame({"ra": [1.0], "dec": [2.0]}) if df is None else df
        self.last_sql = None
        self.submitted = []
        self.aborted = []
        self.status_value = "COMPLETED"

    def query(self, *, sql=None, adql=None, fmt="pandas", async_fallback=True, **kw):
        self.last_sql = sql or adql
        return DatalabResult.from_dataframe(
            self.df.copy(), {"catalog": "gaia_dr3", "table": "gaia_source", "query": sql, "rowcount": len(self.df)}
        )

    def submit(self, *, sql=None, adql=None, timeout=None):
        self.submitted.append(sql or adql)
        return "9876543"

    def status(self, jobid):
        return self.status_value

    def results(self, jobid, *, fmt="pandas"):
        return DatalabResult.from_dataframe(self.df.copy(), {"jobid": str(jobid), "rowcount": len(self.df)})

    def abort(self, jobid):
        self.aborted.append(str(jobid))
        return "OK"


def _ctx(client, *, job_service=None, store=None, counts=None):
    store = store if store is not None else DatalabResultStore(enable_disk_cache=False)
    services = {
        "datalab_client": client,
        "datalab_job_service": job_service or DatalabJobService(max_workers=2),
        "datalab_job_poll_counts": {} if counts is None else counts,
        "datalab_agg_timeout_tables": set(),
    }
    return CallContext(services=services, result_store=store), services["datalab_job_service"], store


def _wait_terminal(job_service, job_id, timeout=8.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        rec = job_service.status(job_id)
        if rec["status"] in {"succeeded", "failed", "canceled"}:
            return rec
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} never finished: {job_service.status(job_id)}")


_ARGS = dict(catalog="gaia_dr3", table="gaia_source", ra=10.0, dec=10.0, radius_deg=0.05)


# ── async_submit: local threaded path (anonymous token) ───────────────────────
def test_select_rows_async_submit_local_runs_and_stores():
    client = _FakeClient()
    ctx, jobs, store = _ctx(client)
    out = dl.SelectCatalogRows().run(
        dl.SelectCatalogRowsInput(**_ARGS, async_submit=True), ctx
    ).to_native()
    assert out["success"] is True and out["job_kind"] == "local"
    assert out["job_id"].startswith("dlj_")
    assert "job_id" in out and "datalab_job_status" in out["note"]

    rec = _wait_terminal(jobs, out["job_id"])
    assert rec["status"] == "succeeded"
    result_id = rec["result"]["result_id"]
    assert result_id.startswith("dlr_")
    assert client.last_sql and "q3c_radial_query" in client.last_sql
    loaded = store.get(result_id)
    assert int(len(loaded.dataframe)) == 1


def test_sql_query_async_submit_still_requires_expert_ack():
    client = _FakeClient()
    ctx, _, _ = _ctx(client)
    out = dl.SqlQuery().run(
        dl.SqlQueryInput(sql="SELECT 1", async_submit=True), ctx
    ).to_native()
    assert out["success"] is False and "expert_ack" in out["error"]


def test_sql_query_async_submit_local():
    client = _FakeClient()
    ctx, jobs, _ = _ctx(client)
    out = dl.SqlQuery().run(
        dl.SqlQueryInput(
            sql="SELECT ra, dec FROM gaia_dr3.gaia_source WHERE q3c_radial_query(ra, dec, 10.0, 10.0, 0.05)",
            expert_ack=True, reason="async test", async_submit=True,
        ),
        ctx,
    ).to_native()
    assert out["success"] is True and out["job_kind"] == "local"
    rec = _wait_terminal(jobs, out["job_id"])
    assert rec["status"] == "succeeded"


def test_density_aggregate_async_submit_skips_sync_and_tiling():
    class _NeverSync(_FakeClient):
        def query(self, **kw):
            raise AssertionError("async_submit must not run a sync query")

    client = _NeverSync()
    ctx, jobs, _ = _ctx(client)
    # radius 3.0 > the 2.5-degree unconfirmed cap (F-7 HITL gate, which fires
    # before the async_submit branch), so model a user-confirmed wide scan.
    out = dl.DensityAggregate().run(
        dl.DensityAggregateInput(catalog="gaia_dr3", table="gaia_source", ra=10.0, dec=10.0, radius_deg=3.0,
                                 async_submit=True, confirm=True),
        ctx,
    ).to_native()
    # The job was queued; the sync attempt (which would raise) never ran on this thread.
    assert out["success"] is True and out["job_id"].startswith("dlj_")
    rec = _wait_terminal(jobs, out["job_id"])
    assert rec["status"] == "failed"  # the worker hit the _NeverSync guard — expected here


# ── async_submit: server-side path (real token) ───────────────────────────────
def test_async_submit_server_job_with_real_token():
    client = _FakeClient(token="real.login.token")
    ctx, jobs, _ = _ctx(client)
    out = dl.SelectCatalogRows().run(
        dl.SelectCatalogRowsInput(**_ARGS, async_submit=True), ctx
    ).to_native()
    assert out["success"] is True and out["job_kind"] == "server"
    assert out["job_id"] == "9876543"
    assert client.submitted and "q3c_radial_query" in client.submitted[0]
    # Registered so list_jobs()/the Jobs panel can see it.
    listed = jobs.list_jobs()
    assert listed and listed[0]["job_id"] == "9876543" and listed[0]["external"] is True


# ── job tools: prefix routing ─────────────────────────────────────────────────
def test_job_status_routes_server_ids_to_the_client():
    client = _FakeClient(token="real.login.token")
    client.status_value = "EXECUTING"
    ctx, _, _ = _ctx(client)
    out = dl.JobStatus().run(dl.JobIdInput(job_id="9876543"), ctx).to_native()
    assert out["success"] is True
    assert out["status"] == "running" and out["server_status"] == "EXECUTING"


def test_job_status_server_id_completed_maps_to_succeeded():
    client = _FakeClient(token="real.login.token")
    client.status_value = "COMPLETED"
    ctx, _, _ = _ctx(client)
    out = dl.JobStatus().run(dl.JobIdInput(job_id="9876543"), ctx).to_native()
    assert out["status"] == "succeeded"
    assert "datalab_job_results" in out["note"]


def test_job_status_server_id_with_anon_token_is_typed_error():
    ctx, _, _ = _ctx(_FakeClient())  # anon token
    out = dl.JobStatus().run(dl.JobIdInput(job_id="9876543"), ctx).to_native()
    assert out["success"] is False and "DATALAB_TOKEN" in out["error"]


def test_job_results_server_id_stores_rows_as_result_id():
    client = _FakeClient(token="real.login.token")
    ctx, _, store = _ctx(client)
    out = dl.JobResults().run(dl.JobIdInput(job_id="9876543"), ctx).to_native()
    assert out["success"] is True and out["result_id"].startswith("dlr_")
    assert out["rowcount"] == 1 and out["preview"]
    assert int(len(store.get(out["result_id"]).dataframe)) == 1


def test_job_cancel_server_id_aborts_via_client():
    client = _FakeClient(token="real.login.token")
    ctx, _, _ = _ctx(client)
    out = dl.JobCancel().run(dl.JobIdInput(job_id="9876543"), ctx).to_native()
    assert out["success"] is True and out["status"] == "canceled"
    assert client.aborted == ["9876543"]


def test_local_dlj_ids_still_use_the_job_service():
    class _JobSvc:
        def status(self, jid):
            return {"job_id": jid, "status": "succeeded"}

    ctx = CallContext(services={"datalab_job_service": _JobSvc()})
    out = dl.JobStatus().run(dl.JobIdInput(job_id="dlj_abc"), ctx).to_native()
    assert out["success"] is True and out["status"] == "succeeded"


# ── job service: list + external records ─────────────────────────────────────
def test_job_service_list_and_external_lifecycle():
    svc = DatalabJobService(max_workers=1)
    svc.register_external("111", params={"sql": "SELECT 1"})
    local_id = svc.start("async_query", lambda cancel: {"ok": True})
    _wait_terminal(svc, local_id)
    listed = svc.list_jobs()
    assert {rec["job_id"] for rec in listed} == {"111", local_id}
    assert all("_cancel" not in rec and "_thread" not in rec for rec in listed)
    # cancel on an external (thread-less) record must not blow up
    rec = svc.cancel("111")
    assert rec["status"] == "canceled"
    svc.update_external("111", status="failed", error="server said no")
    assert svc.status("111")["error"] == "server said no"


# ── My-tables ─────────────────────────────────────────────────────────────────
def _stored_result(store):
    df = pd.DataFrame({"ra": [1.0, 2.0], "dec": [3.0, 4.0], "gmag": [20.1, 21.2]})
    return store.put(df, {"provenance": {"catalog": "nsc_dr2", "table": "object"}})


def test_my_tables_save_list_load_delete_roundtrip():
    store = DatalabResultStore(enable_disk_cache=False)
    result_id = _stored_result(store)
    entry = store.save_result(result_id, "LMC Candidates")  # spaces + case normalize
    assert entry["name"] == "lmc_candidates" and entry["rowcount"] == 2
    assert entry["catalog"] == "nsc_dr2"

    tables = store.list_my_tables()
    assert [t["name"] for t in tables] == ["lmc_candidates"]

    loaded = store.load_my_table("lmc_candidates")
    assert int(len(loaded.dataframe)) == 2
    assert loaded.provenance["my_table"] == "lmc_candidates"
    assert loaded.provenance["saved_from"] == result_id

    deleted = store.delete_my_table("lmc_candidates")
    assert deleted["name"] == "lmc_candidates"
    assert store.list_my_tables() == []
    with pytest.raises(KeyError):
        store.load_my_table("lmc_candidates")


def test_my_table_survives_result_ttl_expiry():
    store = DatalabResultStore(enable_disk_cache=False, ttl_seconds=1)
    result_id = _stored_result(store)
    store.save_result(result_id, "keeper")
    # Age the ephemeral result past its TTL.
    store._memory[result_id]["created_at"] -= 10
    with pytest.raises(KeyError):
        store.get(result_id)
    assert int(len(store.load_my_table("keeper").dataframe)) == 2


def test_my_table_durable_across_store_restart(tmp_path):
    first = DatalabResultStore(cache_dir=tmp_path / "results", ttl_seconds=3600)
    result_id = _stored_result(first)
    first.save_result(result_id, "survivor", description="across restarts")
    second = DatalabResultStore(cache_dir=tmp_path / "results", ttl_seconds=3600)
    names = [t["name"] for t in second.list_my_tables()]
    assert names == ["survivor"]
    assert int(len(second.load_my_table("survivor").dataframe)) == 2


def test_my_table_name_validation_and_unknown_ids():
    store = DatalabResultStore(enable_disk_cache=False)
    with pytest.raises(ValueError):
        store.save_result("dlr_x", "1starts-with-digit")
    with pytest.raises(ValueError):
        store.save_result("dlr_x", "bad;name")
    with pytest.raises(KeyError):
        store.save_result("dlr_never_existed", "fine_name")


def test_my_tables_limit(monkeypatch):
    import services.datalab_result_store as mod
    monkeypatch.setattr(mod, "_MY_TABLES_MAX", 2)
    store = DatalabResultStore(enable_disk_cache=False)
    store.save_result(_stored_result(store), "one")
    store.save_result(_stored_result(store), "two")
    with pytest.raises(ValueError):
        store.save_result(_stored_result(store), "three")
    # Overwriting an existing name is allowed at the cap.
    store.save_result(_stored_result(store), "two")


# ── My-tables capabilities ────────────────────────────────────────────────────
def test_my_table_capabilities_roundtrip():
    store = DatalabResultStore(enable_disk_cache=False)
    ctx = CallContext(services={}, result_store=store, user_id="alice")
    result_id = _stored_result(store)

    saved = dl.SaveResult().run(
        dl.SaveResultInput(result_id=result_id, name="picks", description="test set"), ctx
    ).to_native()
    assert saved["success"] is True and saved["my_table"]["name"] == "picks"

    listed = dl.ListMyTables().run(dl.ListMyTablesInput(), ctx).to_native()
    assert listed["count"] == 1 and listed["my_tables"][0]["description"] == "test set"
    assert store.list_my_tables() == []
    assert [entry["name"] for entry in store.list_my_tables(user_id="alice")] == ["picks"]

    loaded = dl.LoadMyTable().run(dl.MyTableNameInput(name="picks"), ctx).to_native()
    assert loaded["success"] is True and loaded["result_id"].startswith("dlr_")
    assert loaded["rowcount"] == 2 and loaded["preview"]
    # The fresh result_id is a first-class result for follow-up tools.
    assert int(len(store.get(loaded["result_id"]).dataframe)) == 2


def test_load_unknown_my_table_is_typed_error():
    store = DatalabResultStore(enable_disk_cache=False)
    ctx = CallContext(services={}, result_store=store)
    out = dl.LoadMyTable().run(dl.MyTableNameInput(name="ghost"), ctx).to_native()
    assert out["success"] is False and "Unknown my-table" in out["error"]


# ── guard-review regressions (task-097819d-1643) ─────────────────────────────
def test_job_results_failed_local_job_is_not_a_success():
    """CX-08: a failed job's results must carry success=False, not wrap the
    failure in a success envelope."""
    svc = DatalabJobService(max_workers=1)

    def _boom(cancel_check):
        raise RuntimeError("worker exploded")

    job_id = svc.start("async_query", _boom)
    _wait_terminal(svc, job_id)
    ctx = CallContext(services={"datalab_job_service": svc})
    out = dl.JobResults().run(dl.JobIdInput(job_id=job_id), ctx).to_native()
    assert out["success"] is False
    assert out["status"] == "failed" and "worker exploded" in out["error"]
    # A succeeded job still reports success=True.
    ok_id = svc.start("async_query", lambda cancel: {"ok": True})
    _wait_terminal(svc, ok_id)
    ok = dl.JobResults().run(dl.JobIdInput(job_id=ok_id), ctx).to_native()
    assert ok["success"] is True and ok["status"] == "succeeded"
