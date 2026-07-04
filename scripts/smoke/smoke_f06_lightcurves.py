"""Live smoke checks for F06 light-curve tools.

Run from the repository root with:
.venv/Scripts/python.exe scripts/smoke/smoke_f06_lightcurves.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.alerce_client import AlerceClient
from services.lightcurve_suite import LightCurveSuite


def _ok(label: str, detail: str) -> None:
    print(f"PASS {label}: {detail}")


def _fail(label: str, detail: str) -> None:
    print(f"FAIL {label}: {detail}")


def main() -> int:
    service = LightCurveSuite()
    failures = 0

    search = service.search_space_lightcurves("Pi Mensae", mission="TESS", max_rows=20)
    if not search.get("success"):
        _fail("TESS search", search.get("error", "unknown error"))
        failures += 1
    elif int(search.get("count") or 0) < 5:
        _fail("TESS search", f"expected >=5 rows, got {search.get('count')}")
        failures += 1
    else:
        _ok("TESS search", f"count={search.get('count')} total={search.get('total_available')}")

    tess_period = service.period_search("tess", "Pi Mensae", min_period_d=0.05, max_period_d=30.0, index=0)
    if not tess_period.get("success"):
        _fail("TESS period_search", tess_period.get("error", "unknown error"))
        failures += 1
    elif int(tess_period.get("n_points") or 0) <= 1000:
        _fail("TESS period_search", f"expected >1000 points, got {tess_period.get('n_points')}")
        failures += 1
    else:
        _ok("TESS period_search", f"n={tess_period.get('n_points')} P={float(tess_period.get('best_period_d')):.5g} d")

    alerce = AlerceClient(timeout=60)
    cone = alerce.cone_objects(255.0, 11.0, radius_arcsec=300, max_rows=50)
    if not cone.get("success"):
        _fail("ZTF cone", cone.get("error", "unknown error"))
        failures += 1
    else:
        rows = cone.get("rows") or []
        oid = None
        if rows:
            rows = sorted(rows, key=lambda row: int(row.get("ndet") or 0), reverse=True)
            oid = rows[0].get("oid")
        if not oid:
            _fail("ZTF cone", "no oid returned near smoke position")
            failures += 1
        else:
            ztf = service.period_search("ztf", oid, min_period_d=0.05, max_period_d=30.0)
            if ztf.get("success"):
                _ok("ZTF period_search", f"oid={oid} n={ztf.get('n_points')} P={float(ztf.get('best_period_d')):.5g} d")
            else:
                err = str(ztf.get("error", "unknown error"))
                if "five finite" in err or "No finite ZTF" in err or "few" in err.lower():
                    _ok("ZTF period_search", f"oid={oid} clean no-period result: {err[:90]}")
                else:
                    _fail("ZTF period_search", err)
                    failures += 1

    if failures:
        print(f"F06 smoke failed: {failures} check(s) failed")
        return 1
    print("F06 smoke passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())