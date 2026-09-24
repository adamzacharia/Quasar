"""Offline tests for async TAP on any TAP service: vo_adql_query modes
(sync / async / auto) and the vo_tap_job tool (status / results / abort).
pyvo is never touched: TAP services and UWS jobs are injected fakes."""

from __future__ import annotations

import pytest
import requests

import services.vo_registry as vr
from capabilities.base import CallContext
from capabilities.vo import VoAdqlQuery, VoTapJob
from services.vo_registry import VoRegistryService

TAP = "https://tap.example.org/tap"
JOB = "https://tap.example.org/tap/async/abc123"


@pytest.fixture(autouse=True)
def _clean_jobs():
    vr._SUBMITTED_JOBS.clear()
    yield
    vr._SUBMITTED_JOBS.clear()


class _Table:
    """Minimal astropy-like table for _normalize_dal_result."""

    def __init__(self, colnames, rows):
        self.colnames = colnames
        self._rows = rows

    def __iter__(self):
        return iter(self._rows)


class _Summary:
    def __init__(self, text):
        self.message = type("M", (), {"content": text})()


class FakeJob:
    def __init__(self, url=JOB, phase="EXECUTING", table=None, error=None):
        self.url = url
        self.phase = phase
        self._table = table
        self._delete_on_exit = True
        self.ran = False
        self.deleted = False
        self.aborted = False
        self._job = type("T", (), {"errorsummary": _Summary(error) if error else None})()

    def run(self):
        self.ran = True
        self.phase = "QUEUED"

    def fetch_result(self):
        return self._table

    def delete(self):
        self.deleted = True

    def abort(self):
        self.aborted = True


class FakeTap:
    def __init__(self, job=None, sync_exc=None, sync_table=None):
        self.job = job or FakeJob()
        self.sync_exc = sync_exc
        self.sync_table = sync_table or _Table(["a"], [{"a": 1}])
        self.submitted = []
        self.sync_calls = []

    def run_sync(self, query, maxrec=None):
        self.sync_calls.append((query, maxrec))
        if self.sync_exc is not None:
            raise self.sync_exc
        return self.sync_table

    def submit_job(self, query, maxrec=None):
        self.submitted.append((query, maxrec))
        return self.job


def _svc(tap, jobs=None):
    jobs = jobs or {}
    return VoRegistryService(tap_factory=lambda url: tap,
                             job_factory=lambda url: jobs[url])


# ── service: modes ──────────────────────────────────────────────────────────
def test_sync_default_unchanged():
    tap = FakeTap()
    out = _svc(tap).run_adql(TAP, "SELECT a FROM t")
    assert out["success"] and out["rows"] == [{"a": 1}]
    assert tap.submitted == []


def test_bad_mode_rejected():
    out = _svc(FakeTap()).run_adql(TAP, "SELECT 1", mode="turbo")
    assert out["success"] is False and "mode must be one of" in out["error"]


def test_async_submits_runs_and_remembers_without_pyvo_cleanup():
    tap = FakeTap()
    out = _svc(tap).run_adql(TAP, "SELECT a FROM big", max_rows=300, mode="async")
    assert out["success"] and out["job_url"] == JOB and out["phase"] == "QUEUED"
    assert out["status"] == "job_submitted"
    assert tap.job.ran and tap.job._delete_on_exit is False
    assert tap.submitted == [("SELECT a FROM big", 300)]
    assert tap.sync_calls == []
    assert out["provenance"]["query"] == "SELECT a FROM big"
    assert JOB in vr._SUBMITTED_JOBS


def test_async_default_maxrec_is_async_cap_and_clamped():
    tap = FakeTap()
    svc = _svc(tap)
    svc.run_adql(TAP, "SELECT 1", max_rows=None, mode="async")
    assert tap.submitted[-1][1] == svc.async_maxrec_cap
    out = svc.run_adql(TAP, "SELECT 1", max_rows=10**9, mode="async")
    assert tap.submitted[-1][1] == svc.async_maxrec_cap
    assert any("clamped" in w for w in out["warnings"])


def test_async_still_rejects_non_select_and_private_urls():
    tap = FakeTap()
    assert "Only SELECT" in _svc(tap).run_adql(TAP, "DELETE FROM t", mode="async")["error"]
    out = _svc(tap).run_adql("http://10.0.0.1/tap", "SELECT 1", mode="async")
    assert out["success"] is False and "public network address" in out["error"]
    assert tap.submitted == []


# ── service: job lifecycle ──────────────────────────────────────────────────
def test_status_running_completed_and_error():
    svc = _svc(FakeTap(), {JOB: FakeJob(phase="EXECUTING")})
    out = svc.job_status(JOB)
    assert out["success"] and out["phase"] == "EXECUTING" and "still running" in out["message"]

    svc = _svc(FakeTap(), {JOB: FakeJob(phase="ERROR", error="Table big does not exist")})
    out = svc.job_status(JOB)
    assert out["phase"] == "ERROR" and out["job_error"] == "Table big does not exist"


def _submitted(svc, owner=None):
    svc.run_adql(TAP, "SELECT x FROM t", mode="async", owner=owner)


def test_results_fetch_rows_and_cap():
    table = _Table(["x"], [{"x": i} for i in range(10)])
    svc = _svc(FakeTap(), {JOB: FakeJob(phase="COMPLETED", table=table)})
    _submitted(svc)
    out = svc.job_results(JOB, max_rows=4)
    assert out["success"] and out["count"] == 4 and out["total_rows"] == 10
    assert out["truncated"] is True and "first 4" in out["warnings"][0]


def test_results_flags_job_that_hit_its_row_cap():
    tap = FakeTap(job=FakeJob(table=_Table(["x"], [{"x": i} for i in range(5)])))
    svc = _svc(tap, {JOB: tap.job})
    svc.run_adql(TAP, "SELECT x FROM t", max_rows=5, mode="async")
    tap.job.phase = "COMPLETED"
    out = svc.job_results(JOB, max_rows=50)
    assert out["truncated"] is True and any("row cap (5)" in w for w in out["warnings"])
    assert out["provenance"]["query"] == "SELECT x FROM t"


def test_results_before_completion_is_a_clear_failure():
    svc = _svc(FakeTap(), {JOB: FakeJob(phase="QUEUED")})
    _submitted(svc)
    out = svc.job_results(JOB)
    assert out["success"] is False and "QUEUED" in out["error"]
    svc = _svc(FakeTap(), {JOB: FakeJob(phase="ERROR", error="syntax")})
    assert "failed: syntax" in svc.job_results(JOB)["error"]


def test_results_refused_for_unknown_or_foreign_jobs():
    table = _Table(["x"], [{"x": 1}])
    svc = _svc(FakeTap(), {JOB: FakeJob(phase="COMPLETED", table=table)})
    out = svc.job_results(JOB)
    assert out["success"] is False and "submitted here" in out["error"]
    _submitted(svc, owner="alice")
    out = svc.job_results(JOB, owner="bob")
    assert out["success"] is False and "another user" in out["error"]
    assert svc.job_results(JOB, owner="alice")["success"] is True


def test_gone_job_says_resubmit():
    class Gone(Exception):
        pass

    def factory(url):
        err = Gone("404")
        err.response = type("R", (), {"status_code": 404})()
        raise err

    svc = VoRegistryService(tap_factory=lambda u: FakeTap(), job_factory=factory)
    out = svc.job_status(JOB)
    assert out["success"] is False and "Resubmit" in out["error"]


def test_job_url_is_ssrf_guarded():
    svc = _svc(FakeTap(), {})
    out = svc.job_status("http://169.254.169.254/latest")
    assert out["success"] is False and "public network address" in out["error"]


def test_abort_only_for_jobs_submitted_here_by_the_same_user():
    job = FakeJob()
    svc = _svc(FakeTap(job=job), {JOB: job})
    out = svc.job_abort(JOB)
    assert out["success"] is False and "submitted here" in out["error"] and not job.aborted

    svc.run_adql(TAP, "SELECT 1", mode="async", owner="alice")
    assert "another user" in svc.job_abort(JOB, owner="bob")["error"] and not job.aborted
    out = svc.job_abort(JOB, owner="alice")
    # UWS PHASE=ABORT, never DELETE; the book keeps the job for later status.
    assert out["success"] and out["phase"] == "ABORTED" and job.aborted and not job.deleted
    assert JOB in vr._SUBMITTED_JOBS


def test_auto_with_little_budget_runs_sync_instead(monkeypatch):
    """CX-18: below the job threshold a fast query still answers, synchronously."""
    tap = FakeTap()
    monkeypatch.setattr(vr, "_budget_remaining", lambda: 12.0)
    out = _svc(tap).run_adql(TAP, "SELECT a FROM t", mode="auto")
    assert out["success"] and out["rows"] == [{"a": 1}]
    assert tap.submitted == [] and len(tap.sync_calls) == 1
    assert "ran synchronously" in out["warnings"][0]


def test_auto_status_failure_is_not_reported_as_running(monkeypatch):
    """CX-19: a failed poll after submission is a failure carrying the job_url."""
    monkeypatch.setenv("VO_AUTO_SYNC_TIMEOUT", "30")
    job = FakeJob()
    svc = _svc(FakeTap(job=job), {JOB: job})
    monkeypatch.setattr(svc, "job_status",
                        lambda *a, **k: {"success": False, "error": "job expired"})
    out = svc.run_adql(TAP, "SELECT 1", mode="auto")
    assert out["success"] is False and out["job_url"] == JOB
    assert out["status"] == "job_submitted_status_unknown" and "job expired" in out["error"]


class _PhasedJob(FakeJob):
    """A job whose phase advances on each (long-)poll."""

    def __init__(self, phases, table=None, error=None):
        super().__init__(phase="PENDING", table=table, error=error)
        self._phases = list(phases)
        self.waits = []

    def run(self):
        self.ran = True

    def _update(self, wait_for_statechange=False, timeout=None):
        self.waits.append(timeout)
        if self._phases:
            self.phase = self._phases.pop(0)


def test_auto_is_async_first_and_returns_rows_when_done(monkeypatch):
    """CX-07: auto never runs the query twice; it submits once, long-polls,
    and returns rows when the job finishes inside the wait."""
    monkeypatch.setenv("VO_AUTO_SYNC_TIMEOUT", "30")
    job = _PhasedJob(["EXECUTING", "COMPLETED"], table=_Table(["x"], [{"x": 1}, {"x": 2}]))
    tap = FakeTap(job=job)
    out = _svc(tap, {JOB: job}).run_adql(TAP, "SELECT x FROM t", max_rows=50, mode="auto", owner="u")
    assert out["success"] and out["rows"] == [{"x": 1}, {"x": 2}] and out["job_url"] == JOB
    assert tap.sync_calls == [] and len(tap.submitted) == 1
    assert "finished within the wait" in out["warnings"][0]
    assert all(w <= 20 for w in job.waits)


def test_auto_returns_the_handle_when_still_running(monkeypatch):
    monkeypatch.setenv("VO_AUTO_SYNC_TIMEOUT", "1")
    job = _PhasedJob(["EXECUTING"] * 10)
    tap = FakeTap(job=job)
    out = _svc(tap, {JOB: job}).run_adql(TAP, "SELECT slow FROM t", mode="auto")
    assert out["success"] and out["job_url"] == JOB and "rows" not in out
    assert out["phase"] == "EXECUTING" and "vo_tap_job" in out["warnings"][0]
    assert tap.sync_calls == [] and len(tap.submitted) == 1


def test_auto_surfaces_a_server_side_query_error(monkeypatch):
    monkeypatch.setenv("VO_AUTO_SYNC_TIMEOUT", "30")
    job = _PhasedJob(["ERROR"], error="column foo not found")
    out = _svc(FakeTap(job=job), {JOB: job}).run_adql(TAP, "SELECT foo FROM t", mode="auto")
    assert out["success"] is False and "column foo not found" in out["error"]


def test_budget_exhaustion_is_not_a_timeout():
    from services.tool_budgets import BudgetExhausted

    tap = FakeTap(sync_exc=BudgetExhausted("VO tap.example.org", 0.2, 1.0))
    out = _svc(tap).run_adql(TAP, "SELECT slow FROM t")
    assert out["success"] is False and tap.submitted == []
    assert "mode='auto'" not in out["error"]


def test_sync_timeout_error_hints_auto_mode():
    tap = FakeTap(sync_exc=requests.exceptions.ReadTimeout("Read timed out."))
    out = _svc(tap).run_adql(TAP, "SELECT slow FROM t")
    assert out["success"] is False and "mode='auto'" in out["error"]
    assert tap.submitted == []


def test_status_wait_uses_uws_long_poll():
    class WaitJob(FakeJob):
        def _update(self, wait_for_statechange=False, timeout=None):
            self.waited = (wait_for_statechange, timeout)
            self.phase = "COMPLETED"

    job = WaitJob(phase="EXECUTING")
    out = _svc(FakeTap(), {JOB: job}).job_status(JOB, wait_seconds=99)
    assert job.waited == (True, 20.0)  # clamped to MAX_STATUS_WAIT_S
    assert out["phase"] == "COMPLETED"


def test_failed_run_keeps_the_created_job_addressable():
    """CX-06: submit succeeded, run() failed -> job_url returned and remembered."""
    class BadRunJob(FakeJob):
        def run(self):
            raise RuntimeError("phase POST timed out")

    job = BadRunJob()
    svc = _svc(FakeTap(job=job), {JOB: job})
    out = svc.run_adql(TAP, "SELECT 1", mode="async", owner="alice")
    assert out["success"] is False and out["job_url"] == JOB
    assert out["status"] == "job_created_start_unknown" and "may or may not be running" in out["error"]
    assert svc.job_abort(JOB, owner="alice")["success"] is True


def test_status_hides_another_users_query():
    """CX-02: status works for anyone, but only the owner sees the ADQL."""
    job = FakeJob(phase="EXECUTING")
    svc = _svc(FakeTap(job=job), {JOB: job})
    svc.run_adql(TAP, "SELECT secret FROM t", mode="async", owner="alice")
    mine = svc.job_status(JOB, owner="alice")
    theirs = svc.job_status(JOB, owner="bob")
    assert mine["provenance"]["query"] == "SELECT secret FROM t"
    assert "query" not in theirs["provenance"] and "adql" not in theirs["provenance"]


def test_job_book_is_bounded():
    book = vr._JobBook(limit=3)
    for i in range(5):
        book.add(f"https://x.example/{i}", {})
    assert "https://x.example/0" not in book and "https://x.example/4" in book


# ── capability layer ────────────────────────────────────────────────────────
class _Recorder:
    def __init__(self):
        self.kwargs = None

    def __call__(self, rows, **kwargs):
        self.kwargs = dict(kwargs, rows=rows)
        return {"success": True, "total_results": len(rows), "tool_name": kwargs["tool_name"]}


def _ctx(service):
    rec = _Recorder()
    return CallContext(services={"get_vo_registry_service": lambda: service,
                                 "external_catalog_table_result": rec}), rec


def test_capability_async_handle_passes_through_without_table_card():
    tap = FakeTap()
    ctx, rec = _ctx(_svc(tap))
    out = VoAdqlQuery().run(VoAdqlQuery.InputModel(access_url=TAP, adql="SELECT 1",
                                                   mode="async"), ctx).to_native()
    assert out["success"] and out["job_url"] == JOB and rec.kwargs is None


def test_capability_tap_job_results_builds_table_card():
    table = _Table(["x"], [{"x": 1}])
    svc = _svc(FakeTap(), {JOB: FakeJob(phase="COMPLETED", table=table)})
    _submitted(svc)
    ctx, rec = _ctx(svc)
    out = VoTapJob().run(VoTapJob.InputModel(job_url=JOB, action="results"), ctx).to_native()
    assert out["success"] and rec.kwargs["tool_name"] == "vo_tap_job"
    assert rec.kwargs["rows"] == [{"x": 1}]


def test_capability_auto_rows_get_a_table_card():
    class Svc:
        def run_adql(self, *a, **k):
            return {"success": True, "job_url": JOB, "rows": [{"x": 1}], "columns": ["x"],
                    "warnings": [], "provenance": {}}

    ctx, rec = _ctx(Svc())
    out = VoAdqlQuery().run(VoAdqlQuery.InputModel(access_url=TAP, adql="SELECT x FROM t",
                                                   mode="auto"), ctx).to_native()
    assert out["success"] and rec.kwargs is not None and rec.kwargs["rows"] == [{"x": 1}]


def test_capability_tap_job_bad_action():
    ctx, _ = _ctx(_svc(FakeTap()))
    out = VoTapJob().run(VoTapJob.InputModel(job_url=JOB, action="explode"), ctx).to_native()
    assert out["success"] is False and "action must be" in out["error"]
