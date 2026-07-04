"""Live smoke checks for F04 ATNF pulsar catalogue tools.

Run from the repository root with:
.venv/Scripts/python.exe scripts/smoke/smoke_f04_pulsars.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.pulsar_catalog import PulsarCatalogService


def _ok(label: str, detail: str) -> None:
    print(f"PASS {label}: {detail}")


def _fail(label: str, detail: str) -> None:
    print(f"FAIL {label}: {detail}")


def main() -> int:
    service = PulsarCatalogService()
    failures = 0

    lookup = service.pulsar_lookup("B0329+54")
    if not lookup.get("success"):
        _fail("B0329+54 lookup", lookup.get("error", "unknown error"))
        failures += 1
    elif not lookup.get("rows"):
        _fail("B0329+54 lookup", "no rows returned")
        failures += 1
    else:
        row = lookup["rows"][0]
        p0 = row.get("p0_s")
        dm = row.get("dm_pc_cm3")
        if p0 is not None and dm is not None and abs(float(p0) - 0.71452) <= 1e-3 and abs(float(dm) - 26.7) <= 0.5:
            _ok("B0329+54 lookup", f"P0={float(p0):.6f} s DM={float(dm):.2f}")
        else:
            _fail("B0329+54 lookup", f"unexpected P0={p0} DM={dm}")
            failures += 1

    crab = service.search_pulsars(83.63, 22.01, radius_deg=1.0, max_rows=25)
    if not crab.get("success"):
        _fail("Crab cone", crab.get("error", "unknown error"))
        failures += 1
    else:
        rows = crab.get("rows") or []
        crab_row = next((row for row in rows if row.get("jname") == "J0534+2200"), None)
        if crab_row:
            _ok("Crab cone", f"found J0534+2200 DM={crab_row.get('dm_pc_cm3')} sep={crab_row.get('sep_arcmin')} arcmin")
        else:
            names = ", ".join(str(row.get("jname")) for row in rows[:5])
            _fail("Crab cone", f"J0534+2200 not in first {len(rows)} row(s): {names}")
            failures += 1

    if failures:
        print(f"F04 smoke failed: {failures} check(s) failed")
        return 1
    print("F04 smoke passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())