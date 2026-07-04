"""Live smoke test for F03 radio continuum SED."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.radio_sed import RadioSedService


RA_3C273 = 187.27792
DEC_3C273 = 2.05239
RADIUS_ARCSEC = 30.0


def main() -> int:
    svc = RadioSedService(timeout=20)
    compile_out = svc.compile_sed(RA_3C273, DEC_3C273, radius_arcsec=RADIUS_ARCSEC)
    if not compile_out.get("success"):
        print(f"FAIL compile_sed: {compile_out.get('error')}")
        return 1

    points = compile_out.get("points") or []
    print(f"compile_sed returned {len(points)} point(s)")
    for point in points:
        print(
            "  {survey:5s} {freq_mhz:7.1f} MHz {flux_mjy:10.1f} mJy "
            "sep={sep} arcsec beam={beam:g} epoch={epoch}".format(
                survey=str(point.get("survey", "")),
                freq_mhz=float(point.get("freq_mhz") or 0.0),
                flux_mjy=float(point.get("flux_mjy") or 0.0),
                sep=("n/a" if point.get("sep_arcsec") is None else f"{float(point['sep_arcsec']):.2f}"),
                beam=float(point.get("resolution_arcsec") or 0.0),
                epoch=point.get("epoch"),
            )
        )

    statuses = compile_out.get("provenance", {}).get("survey_status", [])
    skips = [row for row in statuses if row.get("status") != "matched"]
    if skips:
        print("per-survey skips/status:")
        for row in skips:
            print(f"  {row.get('survey')}: {row.get('status')} {row.get('error', '')}")

    for warning in compile_out.get("warnings") or []:
        print(f"warning: {warning}")

    if len(points) < 4:
        print("FAIL fewer than 4 of the 5 v1 registry catalogs returned 3C 273 data")
        return 1

    by_survey = {point.get("survey"): point for point in points}
    if "NVSS" not in by_survey or float(by_survey["NVSS"].get("flux_mjy") or 0) <= 20000:
        print("FAIL NVSS flux sanity check failed")
        return 1
    if "TGSS" in by_survey:
        tgss_flux = float(by_survey["TGSS"].get("flux_mjy") or 0)
        if not 80000 <= tgss_flux <= 140000:
            print(f"FAIL TGSS flux sanity check failed: {tgss_flux:.1f} mJy")
            return 1

    fit = svc.fit_spectral_index(points)
    if not fit.get("success"):
        print(f"FAIL fit_spectral_index: {fit.get('error')}")
        return 1
    alpha = float(fit.get("alpha"))
    print(f"fit alpha={alpha:.3f} alpha_err={fit.get('alpha_err')} chi2_red={fit.get('chi2_red')}")
    for flag in fit.get("flags") or []:
        print(f"flag: {flag}")
    if not -1.2 <= alpha <= 0.3:
        print("FAIL alpha outside expected 3C 273 smoke range")
        return 1
    if not fit.get("flags"):
        print("FAIL expected at least one honesty flag")
        return 1

    plot = svc.plot_sed(points, fit=fit, title="Radio SED: 3C 273")
    if not plot.get("success") or not str(plot.get("path", "")).endswith(".png"):
        print(f"FAIL plot_sed: {plot.get('error')}")
        return 1
    print(f"PASS radio_sed smoke plot={plot.get('path')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
