"""Live smoke checks for F01 VO registry discovery.

Run: .venv/Scripts/python.exe scripts/smoke/smoke_f01_vo_registry.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.vo_registry import VoRegistryService

VIZIER_TAP = "https://tapvizier.cds.unistra.fr/TAPVizieR/tap"


def main() -> int:
    svc = VoRegistryService(timeout=60)
    failures = 0

    reg = svc.registry_search("GLEAM", service_type="tap")
    if reg.get("success") and reg.get("count", 0) >= 1 and any(
        any(k in (r.get("access_url") or "").lower() for k in ("vizier", "casda", "mwa"))
        for r in reg["rows"]
    ):
        print(f"PASS registry_search GLEAM/tap: {reg['count']} services, "
              f"first={reg['rows'][0]['access_url']}")
    else:
        print(f"FAIL registry_search: count={reg.get('count')} err={reg.get('error')}")
        failures += 1

    adql = svc.run_adql(VIZIER_TAP, 'SELECT TOP 5 * FROM "VIII/100/gleamegc"')
    if adql.get("success") and adql.get("count") == 5:
        print(f"PASS run_adql GLEAM top5: {adql['count']} rows, {len(adql['columns'])} cols")
    else:
        print(f"FAIL run_adql: count={adql.get('count')} err={adql.get('error')}")
        failures += 1

    guard = svc.run_adql(VIZIER_TAP, "DELETE FROM x")
    if not guard.get("success") and "SELECT" in guard.get("error", ""):
        print("PASS SELECT-only guard rejects DELETE")
    else:
        print(f"FAIL guard: {guard}")
        failures += 1

    tables = svc.list_tables(VIZIER_TAP, keyword="nvss", max_tables=20)
    if tables.get("success") and tables.get("count", 0) >= 1:
        print(f"PASS list_tables keyword=nvss: {tables['count']} tables "
              f"(e.g. {tables['rows'][0]['table_name']})")
    else:
        print(f"FAIL list_tables: count={tables.get('count')} err={tables.get('error')}")
        failures += 1

    if failures:
        print(f"F01 smoke failed: {failures} check(s)")
        return 1
    print("F01 smoke passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
