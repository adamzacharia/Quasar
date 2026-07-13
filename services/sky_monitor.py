"""Standing sky monitors: target watchlists checked against ALeRCE/ZTF alerts.

v1 is agent-tool-driven ("check now") with an OPTIONAL background thread that
is never auto-started. Storage goes through services.db.get_connection()
(Turso in prod, SQLite locally); a fresh connection is opened per operation
for sqlite thread-safety.
"""

from __future__ import annotations

import json
import math
import os
import random
import threading
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

MAX_RADIUS_ARCSEC = 600.0
DUPLICATE_PROXIMITY_ARCSEC = 5.0
DEFAULT_INTERVAL_S = 900

_TARGETS_DDL = """
CREATE TABLE IF NOT EXISTS sky_monitor_targets (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  name TEXT NOT NULL,
  ra REAL NOT NULL, dec REAL NOT NULL,
  radius_arcsec REAL NOT NULL DEFAULT 120,
  enabled INTEGER NOT NULL DEFAULT 1,
  created_at TEXT NOT NULL,
  last_checked_at TEXT,
  note TEXT
)
"""

_HITS_DDL = """
CREATE TABLE IF NOT EXISTS sky_monitor_hits (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  target_id INTEGER NOT NULL,
  oid TEXT NOT NULL,
  first_seen_at TEXT NOT NULL,
  last_mjd REAL, ndet INTEGER, class_name TEXT,
  UNIQUE(target_id, oid)
)
"""


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class SkyMonitorService:
    """Watchlist persistence + new-alert diffing against the ZTF alert stream."""

    def __init__(self, *, db_path: Optional[str] = None, alerce_client: Any = None):
        self.db_path = db_path
        self._alerce_client = alerce_client
        self._thread: Optional[threading.Thread] = None
        self._thread_stop = threading.Event()

    # ── storage plumbing ────────────────────────────────────────────────────
    def _connect(self):
        from services.db import get_connection  # lazy

        conn = (get_connection(local_db_path=self.db_path, autocommit=True)
                if self.db_path else get_connection(autocommit=True))
        cur = conn.cursor()
        cur.execute(_TARGETS_DDL)
        cur.execute(_HITS_DDL)
        # guard review P2: DB-enforced case-insensitive name uniqueness so a
        # concurrent add_target("same") pair cannot both insert.
        try:
            cur.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_sky_monitor_target_name "
                "ON sky_monitor_targets (LOWER(name))"
            )
        except Exception as idx_err:
            # A legacy DB can already hold case-insensitive duplicate names
            # (pre-index disable + re-add). Failing _connect() forever would
            # brick every operation including the remove_target needed to fix
            # it, so degrade to the app-level check alone for this database.
            if "unique" not in str(idx_err).lower():
                raise
        conn.commit()
        return conn

    def _get_alerce(self):
        if self._alerce_client is None:
            from services.alerce_client import AlerceClient  # lazy

            self._alerce_client = AlerceClient()
        return self._alerce_client

    # ── watchlist management ────────────────────────────────────────────────
    def add_target(self, name: Any, ra: Any, dec: Any, radius_arcsec: Any = 120,
                   note: Any = None) -> Dict[str, Any]:
        try:
            name_s = str(name or "").strip()
            if not name_s:
                return {"success": False, "error": "A target name is required."}
            ra_f, dec_f = _validate_coords(ra, dec)
            radius, warnings = _normalize_radius(radius_arcsec)

            conn = self._connect()
            try:
                cur = conn.cursor()
                cur.execute(
                    "SELECT id, name, ra, dec FROM sky_monitor_targets WHERE enabled = 1"
                )
                for row in cur.fetchall():
                    if str(row[1]).strip().lower() == name_s.lower():
                        return {"success": False,
                                "error": f"A watchlist entry named {row[1]!r} already exists (id {row[0]})."}
                    sep = _separation_arcsec(ra_f, dec_f, float(row[2]), float(row[3]))
                    if sep < DUPLICATE_PROXIMITY_ARCSEC:
                        return {"success": False,
                                "error": (f"Target {row[1]!r} (id {row[0]}) is already watched "
                                          f"{sep:.1f} arcsec from this position.")}
                created = _utc_now_iso()
                try:
                    cur.execute(
                        "INSERT INTO sky_monitor_targets (name, ra, dec, radius_arcsec, enabled, created_at, note) "
                        "VALUES (?, ?, ?, ?, 1, ?, ?)",
                        (name_s, ra_f, dec_f, radius, created, str(note) if note else None),
                    )
                except Exception as ins_err:  # unique-index race loser
                    if "unique" in str(ins_err).lower():
                        return {"success": False,
                                "error": (f"A watchlist entry named {name_s!r} already exists "
                                          f"(possibly disabled — list targets to see it).")}
                    raise
                conn.commit()
                target_id = cur.lastrowid
            finally:
                conn.close()
            return {
                "success": True,
                "target": {"id": target_id, "name": name_s, "ra": ra_f, "dec": dec_f,
                           "radius_arcsec": radius, "enabled": 1, "created_at": created,
                           "note": str(note) if note else None},
                "warnings": warnings,
                "provenance": {"service": "Quasar sky monitor"},
            }
        except Exception as exc:
            return {"success": False, "error": str(exc)}

    def list_targets(self) -> Dict[str, Any]:
        try:
            conn = self._connect()
            try:
                cur = conn.cursor()
                cur.execute(
                    "SELECT t.id, t.name, t.ra, t.dec, t.radius_arcsec, t.enabled, "
                    "t.created_at, t.last_checked_at, t.note, COUNT(h.id) "
                    "FROM sky_monitor_targets t "
                    "LEFT JOIN sky_monitor_hits h ON h.target_id = t.id "
                    "GROUP BY t.id, t.name, t.ra, t.dec, t.radius_arcsec, t.enabled, "
                    "t.created_at, t.last_checked_at, t.note ORDER BY t.id",
                )
                rows = [
                    {"id": r[0], "name": r[1], "ra": r[2], "dec": r[3],
                     "radius_arcsec": r[4], "enabled": int(r[5]),
                     "created_at": r[6], "last_checked_at": r[7], "note": r[8],
                     "hits": int(r[9] or 0)}
                    for r in cur.fetchall()
                ]
            finally:
                conn.close()
            return {"success": True, "rows": rows, "count": len(rows),
                    "warnings": [], "provenance": {"service": "Quasar sky monitor"}}
        except Exception as exc:
            return {"success": False, "error": str(exc)}

    def remove_target(self, target_id: Any) -> Dict[str, Any]:
        try:
            tid = int(target_id)
            conn = self._connect()
            try:
                cur = conn.cursor()
                cur.execute("SELECT name FROM sky_monitor_targets WHERE id = ?", (tid,))
                row = cur.fetchone()
                if row is None:
                    return {"success": False, "error": f"No watchlist entry with id {tid}."}
                cur.execute("DELETE FROM sky_monitor_hits WHERE target_id = ?", (tid,))
                cur.execute("DELETE FROM sky_monitor_targets WHERE id = ?", (tid,))
                conn.commit()
            finally:
                conn.close()
            return {"success": True, "removed": {"id": tid, "name": row[0]},
                    "warnings": [], "provenance": {"service": "Quasar sky monitor"}}
        except Exception as exc:
            return {"success": False, "error": str(exc)}

    def set_enabled(self, target_id: Any, enabled: Any) -> Dict[str, Any]:
        try:
            tid = int(target_id)
            flag = 1 if enabled else 0
            conn = self._connect()
            try:
                cur = conn.cursor()
                cur.execute("UPDATE sky_monitor_targets SET enabled = ? WHERE id = ?", (flag, tid))
                conn.commit()
                if getattr(cur, "rowcount", 0) == 0:
                    return {"success": False, "error": f"No watchlist entry with id {tid}."}
            finally:
                conn.close()
            return {"success": True, "id": tid, "enabled": flag, "warnings": [],
                    "provenance": {"service": "Quasar sky monitor"}}
        except Exception as exc:
            return {"success": False, "error": str(exc)}

    # ── the actual check ─────────────────────────────────────────────────────
    def check_now(self, target_id: Any = None) -> Dict[str, Any]:
        try:
            warnings: List[str] = []
            conn = self._connect()
            try:
                cur = conn.cursor()
                if target_id is not None:
                    cur.execute(
                        "SELECT id, name, ra, dec, radius_arcsec FROM sky_monitor_targets "
                        "WHERE id = ?", (int(target_id),),
                    )
                else:
                    cur.execute(
                        "SELECT id, name, ra, dec, radius_arcsec FROM sky_monitor_targets "
                        "WHERE enabled = 1 ORDER BY id",
                    )
                targets = cur.fetchall()
            finally:
                conn.close()

            if not targets:
                return {"success": True, "new_alerts": [], "n_new": 0,
                        "n_targets_checked": 0,
                        "warnings": ["The watchlist is empty (or the id is unknown)."],
                        "provenance": {"service": "Quasar sky monitor"}}

            new_alerts: List[Dict[str, Any]] = []
            checked = 0
            for tid, tname, ra, dec, radius in targets:
                result = self._get_alerce().cone_objects(ra, dec, radius_arcsec=radius, max_rows=200)
                if not result.get("success"):
                    warnings.append(f"ALeRCE check failed for {tname!r}: {result.get('error')}")
                    continue
                rows = result.get("rows") or []
                if len(rows) >= 200:
                    warnings.append(
                        f"{tname!r}: ALeRCE returned the 200-object page cap; results in this "
                        f"dense field may be truncated — consider a smaller radius."
                    )
                now_iso = _utc_now_iso()
                # A DB failure on ONE target must not discard alerts already
                # found for the others; alerts are reported only after their
                # commit lands, so an uncommitted batch re-surfaces next check
                # (duplicate-safe direction) instead of vanishing.
                target_alerts: List[Dict[str, Any]] = []
                try:
                    conn = self._connect()
                    try:
                        cur = conn.cursor()
                        for obj in rows:
                            oid = str(obj.get("oid") or "").strip()
                            if not oid:
                                continue
                            class_name = obj.get("classalerce") or obj.get("class") or obj.get("classification")
                            # guard review P2: INSERT OR IGNORE is race-safe under
                            # UNIQUE(target_id, oid) — "new" means the row actually
                            # landed (rowcount 1), not that a prior SELECT saw nothing.
                            cur.execute(
                                "INSERT OR IGNORE INTO sky_monitor_hits "
                                "(target_id, oid, first_seen_at, last_mjd, ndet, class_name) "
                                "VALUES (?, ?, ?, ?, ?, ?)",
                                (tid, oid, now_iso, _to_float(obj.get("lastmjd")),
                                 _to_int(obj.get("ndet")), str(class_name) if class_name else None),
                            )
                            if getattr(cur, "rowcount", 0) != 1:
                                continue
                            target_alerts.append({"target_name": tname, "oid": oid,
                                                  "ndet": _to_int(obj.get("ndet")),
                                                  "lastmjd": _to_float(obj.get("lastmjd")),
                                                  "class_name": str(class_name) if class_name else None})
                        cur.execute(
                            "UPDATE sky_monitor_targets SET last_checked_at = ? WHERE id = ?",
                            (now_iso, tid),
                        )
                        conn.commit()
                    finally:
                        conn.close()
                except Exception as db_err:
                    warnings.append(f"Database write failed for {tname!r}: {db_err}; "
                                    f"its alerts will be re-detected on the next check.")
                    continue
                checked += 1
                new_alerts.extend(target_alerts)

            return {"success": True, "new_alerts": new_alerts, "n_new": len(new_alerts),
                    "n_targets_checked": checked, "warnings": warnings,
                    "provenance": {"service": "Quasar sky monitor via ALeRCE ZTF"}}
        except Exception as exc:
            return {"success": False, "error": str(exc)}

    # ── optional background loop (never auto-started) ───────────────────────
    def start_background(self, interval_s: Any = DEFAULT_INTERVAL_S) -> Dict[str, Any]:
        try:
            interval = max(60, int(interval_s))
            if self._thread is not None and self._thread.is_alive():
                if not self._thread_stop.is_set():
                    return {"success": True, "already_running": True, "interval_s": interval}
                # A stop is pending: that loop is about to exit, so reporting
                # already_running would leave monitoring silently off. Let it
                # finish (it wakes immediately from its wait) and start fresh.
                self._thread.join(timeout=5.0)

            # Each loop owns a PRIVATE stop event captured in its closure: a
            # stale loop that outlives its join (e.g. mid network call) still
            # exits on its own event and can never be revived by a later clear.
            stop_event = threading.Event()
            self._thread_stop = stop_event

            def _loop():
                while not stop_event.is_set():
                    try:
                        self.check_now()
                    except Exception:
                        pass
                    # jitter so multiple instances never sync-hammer ALeRCE
                    stop_event.wait(interval + random.uniform(0, interval * 0.1))

            self._thread = threading.Thread(target=_loop, name="sky-monitor", daemon=True)
            self._thread.start()
            return {"success": True, "already_running": False, "interval_s": interval}
        except Exception as exc:
            return {"success": False, "error": str(exc)}

    def stop_background(self) -> Dict[str, Any]:
        self._thread_stop.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=5.0)
        return {"success": True}


# ── helpers ─────────────────────────────────────────────────────────────────
def _validate_coords(ra: Any, dec: Any) -> Tuple[float, float]:
    ra_f = float(ra)
    dec_f = float(dec)
    if not (0.0 <= ra_f < 360.0):
        raise ValueError(f"RA {ra_f} out of range [0, 360).")
    if not (-90.0 <= dec_f <= 90.0):
        raise ValueError(f"Dec {dec_f} out of range [-90, 90].")
    return ra_f, dec_f


def _normalize_radius(radius_arcsec: Any) -> Tuple[float, List[str]]:
    warnings: List[str] = []
    try:
        radius = float(radius_arcsec)
    except (TypeError, ValueError):
        return 120.0, ["radius_arcsec was not numeric; using 120."]
    if not math.isfinite(radius):
        return 120.0, ["radius_arcsec was not finite; using 120."]
    if radius <= 0:
        warnings.append(f"radius_arcsec {radius:g} is non-positive; using 120.")
        radius = 120.0
    elif radius > MAX_RADIUS_ARCSEC:
        warnings.append(f"radius_arcsec {radius:g} exceeds {MAX_RADIUS_ARCSEC:g}; clamped.")
        radius = MAX_RADIUS_ARCSEC
    return radius, warnings


def _separation_arcsec(ra1: float, dec1: float, ra2: float, dec2: float) -> float:
    """Great-circle separation (haversine, arcsec) — plenty for 5" dedupe."""
    r1, d1, r2, d2 = map(math.radians, (ra1, dec1, ra2, dec2))
    sin_dd = math.sin((d2 - d1) / 2.0)
    sin_dr = math.sin((r2 - r1) / 2.0)
    a = sin_dd ** 2 + math.cos(d1) * math.cos(d2) * sin_dr ** 2
    return math.degrees(2.0 * math.asin(min(1.0, math.sqrt(a)))) * 3600.0


def _to_float(value: Any) -> Optional[float]:
    try:
        f = float(value)
        return f if math.isfinite(f) else None
    except (TypeError, ValueError):
        return None


def _to_int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


__all__ = ["SkyMonitorService"]
