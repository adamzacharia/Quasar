# F08 — Standing sky monitors (target watchlists vs ZTF alerts)

Read `docs/plans/2026-07-feature-rollout/CONVENTIONS.md` and the exemplars it
names BEFORE coding.

## Objective

Users register sky targets; Quasar checks them against the ALeRCE/ZTF alert
stream and reports NEW alerts since the last check. v1 is agent-tool-driven
("check now") with an OPTIONAL background thread — no email/push, no
Celery deployment work.

## Files

- CREATE `services/sky_monitor.py`
- CREATE `tests/unit/test_sky_monitor.py`
- CREATE `scripts/smoke/smoke_f08_monitor.py`
- EDIT `core/agent.py` (4 anchors)

## Storage

Use `services.db.get_connection()` (Turso in prod, SQLite locally — the
wrapper quacks like sqlite3; parameter placeholders `?`). Create tables
lazily on first service use (CREATE TABLE IF NOT EXISTS):

```sql
CREATE TABLE IF NOT EXISTS sky_monitor_targets (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  name TEXT NOT NULL,
  ra REAL NOT NULL, dec REAL NOT NULL,
  radius_arcsec REAL NOT NULL DEFAULT 120,
  enabled INTEGER NOT NULL DEFAULT 1,
  created_at TEXT NOT NULL,       -- ISO UTC
  last_checked_at TEXT,
  note TEXT
);
CREATE TABLE IF NOT EXISTS sky_monitor_hits (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  target_id INTEGER NOT NULL,
  oid TEXT NOT NULL,
  first_seen_at TEXT NOT NULL,    -- ISO UTC (when WE first saw it)
  last_mjd REAL, ndet INTEGER, class_name TEXT,
  UNIQUE(target_id, oid)
);
```

For test isolation the service constructor accepts `db_path=None` passed to
`get_connection(local_db_path=db_path)` and `alerce_client=None` injectable.
Open a NEW connection per operation (sqlite thread-safety) and always close.

## Class and methods

```python
class SkyMonitorService:
    def __init__(self, *, db_path=None, alerce_client=None): ...

    def add_target(self, name, ra, dec, radius_arcsec=120, note=None) -> dict
        # validate coords; clamp radius <= 600 (warn); reject duplicate
        # (same name, or within 5" of an existing enabled target) with a
        # clear error; returns {"success", "target": {...}}
    def list_targets(self) -> dict
        # rows incl. hit counts:
        # SELECT t.*, COUNT(h.id) FROM targets t LEFT JOIN hits h ... GROUP BY
    def remove_target(self, target_id) -> dict
        # delete target + its hits; success False if id unknown
    def set_enabled(self, target_id, enabled) -> dict
    def check_now(self, target_id=None) -> dict
        # targets = one or all enabled
        # for each: alerce.cone_objects(ra, dec, radius_arcsec, max_rows=200)
        #   -> for each returned oid not already in hits: INSERT, collect as
        #      NEW with {"target_name", "oid", "ndet", "lastmjd",
        #                "class_name" (classalerce or class col)}
        #   update last_checked_at
        #   ALeRCE failure for one target -> warning, continue others
        # returns {"success": True, "new_alerts": [...], "n_new",
        #          "n_targets_checked", "warnings", "provenance"}
    def start_background(self, interval_s=900) -> dict
        # OPTIONAL helper (daemon threading.Thread calling check_now in a
        # loop with jitter; store thread handle; idempotent — second call
        # returns already-running). NOT auto-started anywhere. Gate any
        # future auto-start behind ENABLE_SKY_MONITOR env (do NOT wire into
        # the API in this task).
```

All timestamps `datetime.now(timezone.utc).isoformat()`.

## agent.py wiring

Tools (category `"analysis"`):

1. `monitor_add_target` — `_monitor_add_target(name, target_name=None,
   ra=None, dec=None, radius_arcsec=120, note=None)`; `name` required
   (label for the watchlist entry); position resolved via
   `_live_imagery_coordinates(target_name=target_name or name, ...)`.
   Returns the service dict directly.
   Description: "Add a sky position to the standing ZTF-alert watchlist;
   quasar remembers it across sessions and reports new alerts on each
   check."
2. `monitor_list_targets` — table card, columns `["id","name","ra","dec",
   "radius_arcsec","enabled","last_checked_at","hits"]`, source
   `"Quasar sky monitor"`.
   Description: "List the standing sky-monitor watchlist and hit counts."
3. `monitor_remove_target` — `_monitor_remove_target(target_id: int)`;
   returns service dict.
   Description: "Remove a watchlist entry (and its recorded alerts) by id."
4. `monitor_check_now` — `_monitor_check_now(target_id=None)`; if
   `new_alerts` non-empty return a table card of them (columns
   `["target_name","oid","ndet","lastmjd","class_name"]`, source
   `"ALeRCE ZTF via sky monitor"`), else return the dict with a clear
   `"note": "No new alerts since last check."`.
   Description: "Check the watchlist (or one target) against ALeRCE/ZTF NOW
   and report only alerts that are new since the previous check."

Status labels: `"monitor_add_target": "Adding sky-monitor target"`,
`"monitor_list_targets": "Listing sky-monitor watchlist"`,
`"monitor_remove_target": "Removing sky-monitor target"`,
`"monitor_check_now": "Checking watchlist for new ZTF alerts"`.

Prompt bullet: `When the user wants ongoing watching ("keep an eye on",
"alert me", "monitor"), use \`monitor_add_target\` then
\`monitor_check_now\`; report only NEW alerts.`

## Unit tests (offline — temp sqlite via tmp_path, fake alerce client)

1. add/list/remove/set_enabled round-trip; duplicate name rejected;
   5-arcsec-proximity duplicate rejected; radius clamp warns.
2. check_now first run: fake alerce returns 3 oids → 3 new, hits persisted,
   last_checked_at set. Second run same oids → 0 new. Third run 1 extra
   oid → exactly 1 new.
3. Two targets, alerce fails for one (success False) → other still checked,
   warning names the failed target, success True overall.
4. check_now(target_id) only touches that target. Disabled target skipped.
5. UNIQUE(target_id, oid): same oid for two different targets counts for
   both.

## Smoke expectations

Use a throwaway db_path under `test_results/`. Add target "smoke-galcenter"
at (255.0, 11.0, r=300"). First check_now → n_new ≥ 0 (busy ZTF field —
usually > 0; print). Second immediate check_now → n_new == 0 (MUST hold).
Remove target; list shows none. Exit 0 only if the second-check invariant
and cleanup hold.

## Chat acceptance question

"Keep an eye on SN 2026abc's field and tell me if anything new pops up" →
then later "any new alerts on my watchlist?"

## Acceptance checklist

- [ ] CONVENTIONS definition-of-done ticked
- [ ] Unit tests green; smoke passes; wiring complete (4 tools)
- [ ] No API/UI edits; no Celery; background thread NOT auto-started
