"""Offline tests for audited archive-profile pitfalls: the PitfallAudit
contract, whole-token error triggers, endpoint -> profile recognition
(mirrors included), error hints on failed vo_* calls, and the live audit
runner's classification (scripts/audit_archive_profiles.py) with a fake
TAP runner."""

from __future__ import annotations

import importlib.util
import pathlib
import sys

import pytest
from pydantic import ValidationError

from capabilities.base import CallContext
from capabilities.vo import VoAdqlQuery, VoTapJob
from services import archive_profiles
from services.archive_profiles.schema import Pitfall, PitfallAudit

REPO = pathlib.Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("audit_archive_profiles",
                                               REPO / "scripts" / "audit_archive_profiles.py")
audit_mod = importlib.util.module_from_spec(_spec)
# dataclasses resolve annotations through sys.modules[cls.__module__].
sys.modules.setdefault("audit_archive_profiles", audit_mod)
_spec.loader.exec_module(audit_mod)


# ── PitfallAudit contract ───────────────────────────────────────────────────
@pytest.mark.parametrize("kwargs", [
    {"expect": "manual"},                                              # no reason
    {"expect": "manual", "reason": "x", "adql": "SELECT 1"},           # manual with probe
    {"expect": "ok", "adql": "SELECT 1"},                              # no endpoint
    {"expect": "ok", "endpoint_id": "t"},                              # no adql
    {"expect": "error", "endpoint_id": "t", "adql": "SELECT 1"},      # no error_contains
    {"expect": "ok", "endpoint_id": "t", "adql": "SELECT 1", "error_contains": "x"},
    {"expect": "columns_present", "endpoint_id": "t", "table": "a.b"},  # no columns
    {"expect": "columns_absent", "endpoint_id": "t", "table": "a.b", "columns": ["c"],
     "adql": "SELECT 1"},
    {"expect": "columns_present", "endpoint_id": "t", "table": "a.b'; --", "columns": ["c"]},
])
def test_invalid_audits_rejected(kwargs):
    with pytest.raises(ValidationError):
        PitfallAudit(**kwargs)


def test_audit_endpoint_must_be_a_tap_endpoint_of_the_profile():
    data = archive_profiles.get_profile("gaia").model_dump(mode="json", exclude_none=True)
    data["pitfalls"][0]["audit"]["endpoint_id"] = "nope"
    with pytest.raises(ValidationError, match="not a TAP endpoint"):
        type(archive_profiles.get_profile("gaia")).model_validate(data)


# ── triggers ────────────────────────────────────────────────────────────────
def _pitfall(*triggers):
    return Pitfall(id="p", summary="s", applies_to=[{"kind": "archive", "ref": "vo"}],
                   error_triggers=triggers)


def test_triggers_match_whole_tokens_only():
    p = _pitfall("TOP", "LOWER(")
    assert p.fires_on("SELECT TOP 5 a FROM t")
    assert not p.fires_on("SELECT desktop FROM t")
    assert p.fires_on("where lower(name) = 'x'")
    assert not _pitfall("gaia_dr3.").fires_on("FROM gaiadr3.gaia_source")
    assert _pitfall("gaia_dr3.").fires_on("FROM gaia_dr3.gaia_source")


def test_blank_trigger_rejected():
    with pytest.raises(ValidationError):
        _pitfall("  ")


# ── endpoint recognition + hints ────────────────────────────────────────────
@pytest.mark.parametrize("url,slug", [
    ("https://almascience.nrao.edu/tap", "alma"),
    ("https://almascience.org/tap/sync", "alma"),            # mirror host
    ("https://almascience.eso.org/sia2", "alma"),
    ("https://datalab.noirlab.edu/tap", "datalab"),
    ("https://gea.esac.esa.int/tap-server/tap/async/123", "gaia"),
    ("https://archive.eso.org/tap_obs", "eso"),
    ("https://ws.cadc-ccda.hia-iha.nrc-cnrc.gc.ca/argus", "cadc"),
    ("https://ws.cadc-ccda.hia-iha.nrc-cnrc.gc.ca/sia", "cadc"),
])
def test_profile_for_endpoint(url, slug):
    assert archive_profiles.profile_for_endpoint(url).archive == slug


@pytest.mark.parametrize("url", [
    "https://example.org/tap",
    "https://almascience.nrao.edu/tapx",       # path prefix must end at a segment
    "https://evil-almascience.org/tap",
    "not a url",
])
def test_unknown_endpoints_match_nothing(url):
    assert archive_profiles.profile_for_endpoint(url) is None


def test_error_hints_fire_for_matching_archive_only():
    dl = "https://datalab.noirlab.edu/tap"
    adql = "SELECT ra FROM t WHERE 1=CONTAINS(POINT('ICRS',ra,dec),CIRCLE('ICRS',1,2,3))"
    assert any("q3c_radial_query" in h for h in archive_profiles.error_hints(dl, adql, ""))
    # Same ADQL on an archive where geometry is fine: no hint.
    assert archive_profiles.error_hints("https://gea.esac.esa.int/tap-server/tap", adql, "") == []
    assert archive_profiles.error_hints("https://example.org/tap", adql, "") == []


class _FailingService:
    def __init__(self, out):
        self.out = out

    def run_adql(self, *a, **k):
        return dict(self.out)

    def job_status(self, *a, **k):
        return dict(self.out)


def _ctx(service):
    return CallContext(services={"get_vo_registry_service": lambda: service,
                                 "external_catalog_table_result": lambda rows, **k: {}})


def test_failed_adql_carries_hint_and_keeps_existing_hint():
    svc = _FailingService({"success": False, "error": "unresolved identifiers: gaia_source",
                           "hint": "server hint"})
    cap = VoAdqlQuery()
    out = cap.run(cap.InputModel(access_url="https://gea.esac.esa.int/tap-server/tap",
                                 adql="SELECT TOP 1 source_id FROM gaia_dr3.gaia_source"),
                  _ctx(svc)).to_native()
    assert out["success"] is False
    assert out["hint"].startswith("server hint | ") and "gaiadr3.gaia_source" in out["hint"]


def test_successful_adql_never_gets_hints():
    svc = _FailingService({"success": True, "rows": [], "columns": []})
    cap = VoAdqlQuery()
    out = cap.run(cap.InputModel(access_url="https://datalab.noirlab.edu/tap",
                                 adql="SELECT 1 WHERE 1=CONTAINS(POINT('ICRS',1,2),CIRCLE('ICRS',1,2,3))"),
                  CallContext(services={"get_vo_registry_service": lambda: svc,
                                        "external_catalog_table_result": lambda rows, **k: {"success": True}})
                  ).to_native()
    assert "hint" not in out


def test_failed_async_job_status_carries_hint():
    svc = _FailingService({"success": True, "phase": "ERROR",
                           "job_error": "function point(unknown, double precision) does not exist",
                           "provenance": {"endpoint": "https://datalab.noirlab.edu/tap",
                                          "query": "SELECT 1"}})
    cap = VoTapJob()
    out = cap.run(cap.InputModel(job_url="https://datalab.noirlab.edu/tap/async/1"),
                  _ctx(svc)).to_native()
    assert "q3c_radial_query" in out.get("hint", "")


# ── audit runner classification (fake TAP) ──────────────────────────────────
class _QueryError(Exception):
    pass


class _HttpError(Exception):
    def __init__(self, status):
        super().__init__(f"HTTP {status}")
        self.response = type("R", (), {"status_code": status})()


def _audit(profile_slug, pitfall_id, run):
    profile = archive_profiles.get_profile(profile_slug)
    pitfall = next(p for p in profile.pitfalls if p.id == pitfall_id)
    return audit_mod.audit_pitfall(profile, pitfall, run,
                                   is_query_error=lambda e: isinstance(e, _QueryError))


def test_runner_error_expect():
    def run(url, adql):
        raise _QueryError("2 unresolved identifiers: gaia_source")

    assert _audit("gaia", "esa_vs_datalab_schema_names", run).status == audit_mod.STILL_TRUE

    def run_wrong(url, adql):
        raise _QueryError("syntax error at TOP")

    assert _audit("gaia", "esa_vs_datalab_schema_names", run_wrong).status == audit_mod.STALE
    assert _audit("gaia", "esa_vs_datalab_schema_names",
                  lambda u, a: [{"source_id": 1}]).status == audit_mod.STALE


def test_runner_network_failure_is_unreachable_not_pass():
    def run(url, adql):
        raise ConnectionError("connection reset")

    assert _audit("gaia", "esa_vs_datalab_schema_names", run).status == audit_mod.UNREACHABLE
    assert _audit("gaia", "units_and_epoch", run).status == audit_mod.UNREACHABLE


def test_runner_dead_endpoint():
    def run(url, adql):
        raise _HttpError(404)

    assert _audit("cadc", "tap_lives_at_argus", run).status == audit_mod.DEAD


def test_runner_nonempty_and_columns():
    assert _audit("gaia", "units_and_epoch", lambda u, a: [{"x": 1}]).status == audit_mod.STILL_TRUE
    assert _audit("gaia", "units_and_epoch", lambda u, a: []).status == audit_mod.STALE

    seen = {}

    def cols(url, adql):
        seen["adql"] = adql
        return [{"column_name": "ruwe"}, {"column_name": "parallax"}, {"column_name": "pmra"}]

    row = _audit("gaia", "ruwe_in_dr3_main_table", cols)
    assert row.status == audit_mod.STALE and "pmdec" in row.detail
    assert "tap_schema.columns" in seen["adql"] and "'gaiadr3.gaia_source'" in seen["adql"]
    def absent(table_listed, cols):
        def run(url, adql):
            if "tap_schema.tables" in adql:
                return [{"table_name": "gaiadr2.gaia_source"}] if table_listed else []
            return cols
        return run

    assert _audit("gaia", "dr2_has_no_ruwe_column", absent(True, [])).status == audit_mod.STILL_TRUE
    assert _audit("gaia", "dr2_has_no_ruwe_column",
                  absent(True, [{"column_name": "ruwe"}])).status == audit_mod.STALE
    # A vanished table must not read as "column absent" (and the column
    # query itself is targeted, so a wide table cannot truncate it: CX-16).
    gone = _audit("gaia", "dr2_has_no_ruwe_column", absent(False, []))
    assert gone.status == audit_mod.STALE and "not found" in gone.detail


def test_runner_column_units():
    good = [{"column_name": c, "unit": u} for c, u in
            {"bandwidth": "Hz", "frequency": "GHz", "em_min": "m", "em_max": "m",
             "velocity_resolution": "m/s", "t_min": "d"}.items()]
    assert _audit("alma", "units_and_footprints", lambda u, a: good).status == audit_mod.STILL_TRUE
    bad = [dict(r, unit="MHz") if r["column_name"] == "frequency" else r for r in good]
    row = _audit("alma", "units_and_footprints", lambda u, a: bad)
    assert row.status == audit_mod.STALE and "frequency" in row.detail
    row = _audit("alma", "units_and_footprints", lambda u, a: good[:-1])
    assert row.status == audit_mod.STALE and "t_min" in row.detail


def test_column_units_audit_shape():
    with pytest.raises(ValidationError):
        PitfallAudit(expect="column_units", endpoint_id="t", table="a.b")
    with pytest.raises(ValidationError):
        PitfallAudit(expect="ok", endpoint_id="t", adql="SELECT 1", units={"a": "m"})


def test_runner_manual_is_listed_without_probe():
    def run(url, adql):
        raise AssertionError("manual audits never probe")

    assert _audit("cadc", "constrain_every_query", run).status == audit_mod.MANUAL


def test_main_exit_code(monkeypatch):
    def rows(*statuses):
        return lambda archives=None: [audit_mod.AuditRow("gaia", "p", "ok", st, "") for st in statuses]

    monkeypatch.setattr(audit_mod, "run_audits", rows(audit_mod.STILL_TRUE, audit_mod.MANUAL))
    assert audit_mod.main([]) == 0
    monkeypatch.setattr(audit_mod, "run_audits", rows(audit_mod.STILL_TRUE, audit_mod.STALE))
    assert audit_mod.main([]) == 1
    # Nothing verified is never green (CX-09).
    monkeypatch.setattr(audit_mod, "run_audits", rows(audit_mod.UNREACHABLE, audit_mod.MANUAL))
    assert audit_mod.main([]) == 2
    # A partial outage is incomplete too, never green (CX-09 reopen).
    monkeypatch.setattr(audit_mod, "run_audits", rows(audit_mod.STILL_TRUE, audit_mod.UNREACHABLE))
    assert audit_mod.main([]) == 2
    monkeypatch.setattr(audit_mod, "run_audits", rows())
    assert audit_mod.main([]) == 2


def test_main_rejects_unknown_archive(monkeypatch):
    monkeypatch.setattr(audit_mod, "run_audits", lambda archives=None: pytest.fail("must not run"))
    assert audit_mod.main(["--archive", "nope"]) == 2


def test_default_runner_bounds_a_trickling_probe(monkeypatch):
    """A probe that never finishes is abandoned at the wall deadline (CX-08)."""
    import threading

    import pyvo

    release = threading.Event()

    class Slow:
        def __init__(self, *a, **k):
            pass

        def run_sync(self, *a, **k):
            release.wait(10)
            raise RuntimeError("released")

    monkeypatch.setattr(pyvo.dal, "TAPService", Slow)
    run = audit_mod.default_runner(timeout=1, wall_seconds=0.3)
    with pytest.raises(TimeoutError) as err:
        run("https://tap.example.org/tap", "SELECT 1")
    assert "exceeded" in str(err.value)
    # The abandoned probe is a daemon thread, so it cannot block interpreter
    # exit (CX-08 reopen: executor workers are joined at exit).
    stuck = [t for t in threading.enumerate() if t.name == "audit-probe" and t.is_alive()]
    assert stuck and all(t.daemon for t in stuck)
    release.set()
