from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.dust_extinction import DustExtinctionService


def _compact(result):
    if not result.get("success"):
        return result
    out = {
        key: result.get(key)
        for key in ("success", "count", "ebv_sfd", "ebv_sf11", "ebv_sfd_mean", "ebv_sf11_mean", "warnings")
        if key in result
    }
    rows = result.get("rows") or []
    if rows:
        out["rows"] = rows[:5]
    return out


def main() -> int:
    svc = DustExtinctionService()
    ebv = svc.ebv(187.2779, 2.0524)
    table = svc.extinction_table(187.2779, 2.0524, bands=["V"])

    v_row = next((row for row in table.get("rows", []) if row.get("band") == "V"), None)
    ebv_ok = (
        ebv.get("success") is True
        and 0.005 <= float(ebv.get("ebv_sfd") or -1) <= 0.05
        and float(ebv.get("ebv_sf11") or 0) < float(ebv.get("ebv_sfd") or 0)
    )
    table_ok = (
        table.get("success") is True
        and v_row is not None
        and math.isclose(float(v_row.get("A_lambda")), 2.742 * float(table.get("ebv_sf11")), rel_tol=1e-9, abs_tol=1e-12)
    )

    checks = [
        ("3C 273 E(B-V)", ebv_ok, ebv),
        ("3C 273 V-band extinction", table_ok, table),
    ]
    all_ok = True
    for label, ok, result in checks:
        all_ok = all_ok and ok
        status = "PASS" if ok else "FAIL"
        print(f"{status} {label}: {json.dumps(_compact(result), sort_keys=True)}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
