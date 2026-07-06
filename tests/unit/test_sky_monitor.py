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


def test_nan_radius_and_insert_or_ignore_semantics(tmp_path):
    # guard review P3: NaN radius must not pass validation.
    svc = _svc(tmp_path)
    out = svc.add_target("nan-target", 10.0, 10.0, radius_arcsec=float("nan"))
    assert out["success"] is True
    assert out["target"]["radius_arcsec"] == 120.0
    assert any("finite" in w for w in out["warnings"])


def test_check_now_race_safe_when_hit_preinserted(tmp_path):
    # guard review P2: a hit inserted between cone query and INSERT (simulating
    # a concurrent checker) must NOT be double-counted or crash the check.
    alerce = FakeAlerce()
    svc = _svc(tmp_path, alerce)
    tid = svc.add_target("race", 10.0, 10.0)["target"]["id"]

    # pre-insert the hit exactly as a concurrent run would
    conn = svc._connect()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO sky_monitor_hits (target_id, oid, first_seen_at) VALUES (?, ?, ?)",
        (tid, "ZTFRACE", "2026-07-04T00:00:00+00:00"),
    )
    conn.commit(); conn.close()

    alerce.push(rows=[_obj("ZTFRACE")])
    out = svc.check_now()
    assert out["success"] is True
    assert out["n_new"] == 0  # INSERT OR IGNORE swallowed the duplicate


def test_legacy_db_with_duplicate_names_does_not_brick(tmp_path):
    # follow-on to guard P2 unique index: a pre-index DB can hold
    # case-insensitive duplicate names (disable + re-add). CREATE UNIQUE INDEX
    # then fails forever — every op, including the remove needed to fix it,
    # must still work (degraded to the app-level check).
    import sqlite3

    from services.sky_monitor import _HITS_DDL, _TARGETS_DDL

    db = tmp_path / "legacy.db"
    conn = sqlite3.connect(str(db))
    conn.execute(_TARGETS_DDL)
    conn.execute(_HITS_DDL)
    for name in ("SN dup", "sn DUP"):
        conn.execute(
            "INSERT INTO sky_monitor_targets (name, ra, dec, radius_arcsec, enabled, created_at) "
            "VALUES (?, 10.0, 10.0, 120, 1, '2026-01-01T00:00:00+00:00')",
            (name,),
        )
    conn.commit(); conn.close()

    svc = SkyMonitorService(db_path=str(db), alerce_client=FakeAlerce())
    listing = svc.list_targets()
    assert listing["success"] is True and listing["count"] == 2
    removed = svc.remove_target(listing["rows"][0]["id"])
    assert removed["success"] is True


def test_stop_then_restart_actually_restarts(tmp_path):
    # follow-on to guard review: stop set the event without joining, so an
    # immediate restart could see the dying thread, report already_running,
    # and leave monitoring silently off once the old loop exited.
    svc = _svc(tmp_path)
    try:
        assert svc.start_background(interval_s=60)["already_running"] is False
        assert svc.stop_background()["success"] is True
        restarted = svc.start_background(interval_s=60)
        assert restarted["success"] is True
        assert restarted["already_running"] is False
        assert svc._thread is not None and svc._thread.is_alive()
        assert not svc._thread_stop.is_set()
    finally:
        svc.stop_background()


def test_check_now_warns_at_page_cap(tmp_path):
    # cone_objects fetches a single 200-row page with no total count; a dense
    # field hitting the cap must warn instead of silently truncating.
    alerce = FakeAlerce()
    svc = _svc(tmp_path, alerce)
    svc.add_target("dense", 270.0, -30.0)

    alerce.push(rows=[_obj(f"ZTFD{i:03d}") for i in range(200)])
    out = svc.check_now()
    assert out["success"] is True and out["n_new"] == 200
    assert any("page cap" in w for w in out["warnings"])


def test_db_failure_on_one_target_keeps_other_alerts(tmp_path):
    # a DB write failure for one target must neither discard alerts already
    # committed for other targets nor mark its own alerts as seen.
    alerce = FakeAlerce()
    svc = _svc(tmp_path, alerce)
    svc.add_target("ok", 10.0, 10.0)
    svc.add_target("dbfail", 50.0, -30.0)

    real_connect = svc._connect
    calls = {"n": 0}

    def flaky_connect():
        # call 1 = target select, call 2 = 'ok' write, call 3 = 'dbfail' write
        calls["n"] += 1
        if calls["n"] == 3:
            raise RuntimeError("disk I/O error (simulated)")
        return real_connect()

    svc._connect = flaky_connect
    alerce.push(rows=[_obj("ZTFOK")])
    alerce.push(rows=[_obj("ZTFLOST")])
    out = svc.check_now()

    assert out["success"] is True
    assert out["n_new"] == 1 and out["new_alerts"][0]["oid"] == "ZTFOK"
    assert out["n_targets_checked"] == 1
    assert any("dbfail" in w and "re-detected" in w for w in out["warnings"])

    # the failed target's alert was NOT marked seen — it surfaces next check
    svc._connect = real_connect
    alerce.push(rows=[])            # 'ok' target: nothing new
    alerce.push(rows=[_obj("ZTFLOST")])
    out2 = svc.check_now()
    assert out2["n_new"] == 1 and out2["new_alerts"][0]["oid"] == "ZTFLOST"
