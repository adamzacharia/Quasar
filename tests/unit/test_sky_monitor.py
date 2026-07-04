"""Offline unit tests for F08 sky monitors (temp sqlite + fake ALeRCE)."""

from __future__ import annotations

import os

from services.sky_monitor import SkyMonitorService


class FakeAlerce:
    """Scriptable ALeRCE stand-in: returns queued responses per call."""

    def __init__(self):
        self.responses = []
        self.calls = []

    def push(self, rows=None, success=True, error=None):
        self.responses.append(
            {"success": success, "rows": rows or [], "error": error}
            if not success
            else {"success": True, "rows": rows or [], "count": len(rows or []),
                  "warnings": [], "provenance": {}}
        )

    def cone_objects(self, ra, dec, radius_arcsec=120, max_rows=200):
        self.calls.append({"ra": ra, "dec": dec, "radius_arcsec": radius_arcsec})
        if self.responses:
            return self.responses.pop(0)
        return {"success": True, "rows": [], "count": 0, "warnings": [], "provenance": {}}


def _svc(tmp_path, alerce=None):
    return SkyMonitorService(db_path=str(tmp_path / "monitor.db"),
                             alerce_client=alerce or FakeAlerce())


def _obj(oid, ndet=3, lastmjd=60000.5, cls="SN"):
    return {"oid": oid, "ndet": ndet, "lastmjd": lastmjd, "classalerce": cls}


def test_add_list_remove_roundtrip(tmp_path):
    svc = _svc(tmp_path)
    out = svc.add_target("SN test", 150.0, 2.5, radius_arcsec=100, note="follow-up")
    assert out["success"] is True
    tid = out["target"]["id"]

    listing = svc.list_targets()
    assert listing["success"] is True and listing["count"] == 1
    row = listing["rows"][0]
    assert row["name"] == "SN test" and row["hits"] == 0 and row["enabled"] == 1

    removed = svc.remove_target(tid)
    assert removed["success"] is True and removed["removed"]["name"] == "SN test"
    assert svc.list_targets()["count"] == 0

    assert svc.remove_target(999)["success"] is False


def test_duplicate_name_and_proximity_rejected(tmp_path):
    svc = _svc(tmp_path)
    assert svc.add_target("A", 10.0, 10.0)["success"] is True
    dup_name = svc.add_target("a", 200.0, -20.0)
    assert dup_name["success"] is False and "already exists" in dup_name["error"]
    # 3 arcsec away -> proximity duplicate
    near = svc.add_target("B", 10.0 + 3.0 / 3600.0 / 0.9848, 10.0)
    assert near["success"] is False and "arcsec" in near["error"]


def test_radius_clamp_and_coord_validation(tmp_path):
    svc = _svc(tmp_path)
    out = svc.add_target("C", 10.0, 10.0, radius_arcsec=5000)
    assert out["success"] is True
    assert out["target"]["radius_arcsec"] == 600.0
    assert any("clamped" in w for w in out["warnings"])
    assert svc.add_target("D", 400.0, 10.0)["success"] is False


def test_check_now_diffs_new_alerts(tmp_path):
    alerce = FakeAlerce()
    svc = _svc(tmp_path, alerce)
    svc.add_target("field", 255.0, 11.0, radius_arcsec=300)

    # first run: 3 oids -> 3 new
    alerce.push(rows=[_obj("ZTF1"), _obj("ZTF2"), _obj("ZTF3")])
    r1 = svc.check_now()
    assert r1["success"] is True and r1["n_new"] == 3
    assert {a["oid"] for a in r1["new_alerts"]} == {"ZTF1", "ZTF2", "ZTF3"}
    assert r1["n_targets_checked"] == 1

    # second run, same oids -> 0 new
    alerce.push(rows=[_obj("ZTF1"), _obj("ZTF2"), _obj("ZTF3")])
    r2 = svc.check_now()
    assert r2["n_new"] == 0

    # third run, one extra -> exactly 1 new
    alerce.push(rows=[_obj("ZTF1"), _obj("ZTF2"), _obj("ZTF3"), _obj("ZTF4", cls=None)])
    r3 = svc.check_now()
    assert r3["n_new"] == 1 and r3["new_alerts"][0]["oid"] == "ZTF4"

    # hit counts + last_checked_at persisted
    listing = svc.list_targets()
    assert listing["rows"][0]["hits"] == 4
    assert listing["rows"][0]["last_checked_at"]


def test_check_now_partial_failure_continues(tmp_path):
    alerce = FakeAlerce()
    svc = _svc(tmp_path, alerce)
    svc.add_target("good", 10.0, 10.0)
    svc.add_target("bad", 50.0, -30.0)

    alerce.push(rows=[_obj("ZTFA")])                 # good target
    alerce.push(success=False, error="ALeRCE 503")   # bad target
    out = svc.check_now()

    assert out["success"] is True
    assert out["n_new"] == 1
    assert out["n_targets_checked"] == 1
    assert any("bad" in w and "503" in w for w in out["warnings"])


def test_check_now_scoped_and_disabled(tmp_path):
    alerce = FakeAlerce()
    svc = _svc(tmp_path, alerce)
    t1 = svc.add_target("one", 10.0, 10.0)["target"]["id"]
    t2 = svc.add_target("two", 30.0, 30.0)["target"]["id"]

    # scoped to t2 only -> one alerce call at t2's coords
    alerce.push(rows=[_obj("ZTFX")])
    out = svc.check_now(target_id=t2)
    assert out["n_new"] == 1 and out["n_targets_checked"] == 1
    assert alerce.calls[-1]["ra"] == 30.0

    # disabled target skipped in full sweep
    assert svc.set_enabled(t1, False)["success"] is True
    alerce.push(rows=[])
    out2 = svc.check_now()
    assert out2["n_targets_checked"] == 1  # only t2
    assert alerce.calls[-1]["ra"] == 30.0

    # empty watchlist message
    svc.remove_target(t1)
    svc.remove_target(t2)
    out3 = svc.check_now()
    assert out3["success"] is True and out3["n_new"] == 0
    assert any("empty" in w for w in out3["warnings"])


def test_same_oid_counts_for_two_targets(tmp_path):
    alerce = FakeAlerce()
    svc = _svc(tmp_path, alerce)
    svc.add_target("f1", 10.0, 10.0)
    svc.add_target("f2", 10.1, 10.0)  # >5 arcsec apart, overlapping cones

    alerce.push(rows=[_obj("ZTFSHARED")])
    alerce.push(rows=[_obj("ZTFSHARED")])
    out = svc.check_now()
    assert out["n_new"] == 2
    assert {a["target_name"] for a in out["new_alerts"]} == {"f1", "f2"}


def test_background_thread_idempotent(tmp_path):
    svc = _svc(tmp_path)
    try:
        first = svc.start_background(interval_s=60)
        assert first["success"] is True and first["already_running"] is False
        second = svc.start_background(interval_s=60)
        assert second["already_running"] is True
    finally:
        svc.stop_background()
