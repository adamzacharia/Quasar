"""Live smoke checks for F08 sky monitors (throwaway sqlite + real ALeRCE).

Run: .venv/Scripts/python.exe scripts/smoke/smoke_f08_monitor.py
"""

from __future__ import annotations

import os
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.sky_monitor import SkyMonitorService


def main() -> int:
    db_dir = ROOT / "test_results" / "sky_monitor"
    db_dir.mkdir(parents=True, exist_ok=True)
    db_path = str(db_dir / f"smoke_{uuid.uuid4().hex[:8]}.db")
    svc = SkyMonitorService(db_path=db_path)
    failures = 0

    added = svc.add_target("smoke-busy-field", 255.0, 11.0, radius_arcsec=300)
    if not added.get("success"):
        print(f"FAIL add_target: {added.get('error')}")
        return 1
    print(f"PASS add_target: id={added['target']['id']}")

    first = svc.check_now()
    if not first.get("success"):
        print(f"FAIL first check_now: {first.get('error')}")
        failures += 1
    elif first.get("warnings") and first.get("n_targets_checked") == 0:
        # ALeRCE outage: parsing/diffing is unit-proven, do not fail the gate
        print(f"SKIP first check_now: ALeRCE unreachable ({first['warnings'][:1]})")
        print("F08 smoke passed (with ALeRCE outage skip)")
        return 0
    else:
        print(f"PASS first check_now: n_new={first.get('n_new')} (busy ZTF field)")

    second = svc.check_now()
    if not second.get("success") or second.get("n_new") != 0:
        print(f"FAIL second check_now must find 0 new: {second.get('n_new')} ({second.get('error')})")
        failures += 1
    else:
        print("PASS second check_now: n_new=0 (diff invariant holds)")

    removed = svc.remove_target(added["target"]["id"])
    listing = svc.list_targets()
    if not (removed.get("success") and listing.get("count") == 0):
        print(f"FAIL cleanup: removed={removed.get('success')} remaining={listing.get('count')}")
        failures += 1
    else:
        print("PASS cleanup: watchlist empty")

    try:
        os.remove(db_path)
    except OSError:
        pass

    if failures:
        print(f"F08 smoke failed: {failures} check(s)")
        return 1
    print("F08 smoke passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
