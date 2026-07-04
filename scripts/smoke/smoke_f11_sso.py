"""Live smoke checks for F11 solar-system tools (Horizons + SkyBoT).

Run: .venv/Scripts/python.exe scripts/smoke/smoke_f11_sso.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.solar_system import SolarSystemService


def main() -> int:
    svc = SolarSystemService(timeout=60)
    failures = 0

    eph = svc.horizons_ephemeris("Ceres", "2026-07-03", "2026-07-08")
    if not eph.get("success"):
        print(f"FAIL Horizons Ceres: {eph.get('error')}")
        failures += 1
    else:
        rows = eph.get("rows", [])
        vmags = [r["v_mag"] for r in rows if r.get("v_mag") is not None]
        deltas = [r["delta_au"] for r in rows if r.get("delta_au") is not None]
        ok = len(rows) >= 4 and vmags and 5.0 <= vmags[0] <= 10.0 and deltas and 1.0 <= deltas[0] <= 5.0
        print(f"{'PASS' if ok else 'FAIL'} Horizons Ceres: rows={len(rows)} v_mag={vmags[:1]} delta_au={deltas[:1]}")
        failures += 0 if ok else 1

    # SkyBoT's ephemeris backend is intermittently broken server-side
    # (calceph/numIntAstr errors observed 2026-07-03/04 across epochs), so try
    # several epochs and treat a full outage as SKIP, not FAIL — the record
    # parsing is pinned by offline unit tests against verified live shapes.
    from astropy.time import Time

    now_jd = Time.now().jd
    sky_ok = None
    for dback in (0, 3, 30):
        sky = svc.skybot_cone(180.0, 0.0, radius_deg=5.0, epoch=now_jd - dback)
        if sky.get("success"):
            sky_ok = (dback, sky)
            if sky.get("count", 0) > 0:
                break
    if sky_ok is None:
        print(f"SKIP SkyBoT: endpoint failing at all tried epochs ({sky.get('error')}) — not counted as failure")
    else:
        dback, sky = sky_ok
        n = sky.get("count", 0)
        names = [r.get("name") for r in sky.get("rows", [])[:3]]
        note = "" if n > 0 else f" (0 rows; server warnings={sky.get('warnings')})"
        print(f"PASS SkyBoT cone (180,0,5deg, epoch now-{dback}d): count={n} sample={names}{note}")

    if failures:
        print(f"F11 smoke failed: {failures} check(s)")
        return 1
    print("F11 smoke passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
