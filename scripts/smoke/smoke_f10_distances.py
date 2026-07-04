"""Live smoke checks for F10 distance tools.

Run from the repository root with:
.venv/Scripts/python.exe scripts/smoke/smoke_f10_distances.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.distance_service import DistanceService


def _ok(label: str, detail: str) -> None:
    print(f"PASS {label}: {detail}")


def _fail(label: str, detail: str) -> None:
    print(f"FAIL {label}: {detail}")


def main() -> int:
    service = DistanceService()
    failures = 0

    gaia = service.gaia_distances(269.44850, 4.73780, radius_arcsec=30, max_rows=10)
    if not gaia.get("success"):
        _fail("Gaia/Bailer-Jones", gaia.get("error", "unknown error"))
        failures += 1
    elif not gaia.get("rows"):
        _fail("Gaia/Bailer-Jones", "no rows returned for Barnard's Star field")
        failures += 1
    else:
        row = gaia["rows"][0]
        parallax = row.get("parallax_mas")
        r_geo = row.get("r_geo_pc")
        if parallax is not None and r_geo is not None and 546 <= parallax <= 548 and 1.80 <= r_geo <= 1.86:
            _ok("Gaia/Bailer-Jones", f"parallax={parallax:.3f} mas r_geo={r_geo:.3f} pc")
        else:
            _fail("Gaia/Bailer-Jones", f"unexpected nearest row parallax={parallax} r_geo={r_geo}")
            failures += 1

    # NED-D endpoint (nDistance) is genuinely slow/flaky: a clean timeout is a
    # SKIP (soft pass), but when it responds NGC 253 must give a sane distance.
    # Retry once before giving up.
    ned = service.ned_distances("NGC 253")
    if not ned.get("success"):
        ned = service.ned_distances("NGC 253")
    if not ned.get("success"):
        err = str(ned.get("error", "unknown error"))
        if "timeout" in err.lower() or "timed out" in err.lower() or "unavailable" in err.lower():
            print(f"SKIP NED-D: endpoint unreachable ({err[:80]}) — not counted as failure")
        else:
            _fail("NED-D", err)
            failures += 1
    else:
        summary = ned.get("summary") or {}
        n = int(summary.get("n") or 0)
        median = summary.get("median_mpc")
        if median is not None and n >= 1 and 2.0 <= float(median) <= 6.0:
            _ok("NED-D", f"n={n} median={float(median):.3f} Mpc")
        else:
            _fail("NED-D", f"unexpected summary n={n} median={median}")
            failures += 1

    # M87 (Virgo) sits ~27 deg from the CMB apex, so its CMB-frame velocity is a
    # robust POSITIVE check: v_CMB should be ~1611 km/s (matches NED). NGC 253
    # was a poor target — near the anti-apex it gives a tiny/negative v_CMB.
    velocity = service.velocity_frames(187.70593, 12.39112, v_helio_kms=1284)
    if not velocity.get("success"):
        _fail("velocity frames", velocity.get("error", "unknown error"))
        failures += 1
    else:
        cmb = next((row for row in velocity.get("rows", []) if row.get("frame") == "CMB"), {})
        v_cmb = cmb.get("v_kms")
        if v_cmb is not None and 1560 <= float(v_cmb) <= 1660:
            _ok("velocity frames", f"M87 v_CMB={float(v_cmb):.3f} km/s (expect ~1611)")
        else:
            _fail("velocity frames", f"unexpected v_CMB={v_cmb}")
            failures += 1

    if failures:
        print(f"F10 smoke failed: {failures} check(s) failed")
        return 1
    print("F10 smoke passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())