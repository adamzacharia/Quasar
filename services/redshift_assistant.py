# services/redshift_assistant.py
"""
Spectral Redshift Assistant (FEATURES.md O3).

Given a set of observed spectral features — emission/absorption line
frequencies (GHz, radio/mm) or wavelengths (Angstrom/micron, optical/IR) —
identify consistent line assignments and estimate the source redshift.

Algorithm (pure computation, no network):
  1. For every (observed feature, catalog line) pair compute the implied
     redshift z = (rest / observed) - 1 (frequency) or
     z = (observed / rest) - 1 (wavelength).
  2. For each implied z within [z_min, z_max], count how many OTHER observed
     features match a catalog line at that same z within `tolerance`.
  3. Rank candidate redshifts by (number of matched lines, then total
     fractional residual). Return the top candidates with assignments.

This mirrors how an astronomer cross-matches a line list against templates
and is robust to one or two unidentified features.
"""

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ── Rest-frame line catalogs ────────────────────────────────────────────

# Radio / (sub)mm lines in GHz
RADIO_MM_LINES_GHZ: Dict[str, float] = {
    "HI 21cm": 1.420406,
    "CO(1-0)": 115.27120,
    "CO(2-1)": 230.53800,
    "CO(3-2)": 345.79599,
    "CO(4-3)": 461.04077,
    "CO(5-4)": 576.26793,
    "CO(6-5)": 691.47308,
    "CO(7-6)": 806.65181,
    "CO(8-7)": 921.79970,
    "13CO(1-0)": 110.20135,
    "13CO(2-1)": 220.39868,
    "13CO(3-2)": 330.58797,
    "C18O(1-0)": 109.78217,
    "C18O(2-1)": 219.56035,
    "[CI](1-0)": 492.16065,
    "[CI](2-1)": 809.34197,
    "[CII] 158um": 1900.5369,
    "[NII] 205um": 1461.13141,
    "[NII] 122um": 2459.38010,
    "[OIII] 88um": 3393.00624,
    "HCN(1-0)": 88.63160,
    "HCN(3-2)": 265.88643,
    "HCO+(1-0)": 89.18852,
    "HCO+(3-2)": 267.55763,
    "CS(2-1)": 97.98095,
    "N2H+(1-0)": 93.17340,
    "H2O 22GHz": 22.23508,
    "SiO(2-1)": 86.84696,
}

# Optical / UV / NIR lines in Angstrom (vacuum)
OPTICAL_LINES_ANGSTROM: Dict[str, float] = {
    "Ly-alpha": 1215.67,
    "NV 1240": 1240.81,
    "CIV 1549": 1549.48,
    "HeII 1640": 1640.40,
    "CIII] 1909": 1908.73,
    "MgII 2798": 2798.75,
    "[OII] 3727": 3727.09,
    "[NeIII] 3869": 3869.86,
    "CaII K 3934": 3934.78,
    "CaII H 3969": 3969.59,
    "H-delta": 4102.89,
    "H-gamma": 4341.68,
    "H-beta": 4862.68,
    "[OIII] 4959": 4960.30,
    "[OIII] 5007": 5008.24,
    "MgI b 5175": 5176.70,
    "NaI D 5893": 5895.60,
    "[OI] 6300": 6302.05,
    "H-alpha": 6564.61,
    "[NII] 6583": 6585.27,
    "[SII] 6716": 6718.29,
    "[SII] 6731": 6732.67,
    "CaII 8542": 8544.44,
    "Pa-beta": 12821.6,
    "Pa-alpha": 18756.1,
}


def _candidate_redshifts(
    observed: List[float],
    catalog: Dict[str, float],
    mode: str,
    z_min: float,
    z_max: float,
) -> List[float]:
    """All implied redshifts from every (feature, line) pairing within range."""
    zs = []
    for obs in observed:
        if obs <= 0:
            continue
        for rest in catalog.values():
            if mode == "frequency":
                z = (rest / obs) - 1.0
            else:  # wavelength
                z = (obs / rest) - 1.0
            if z_min - 1e-9 <= z <= z_max + 1e-9:
                zs.append(z)
    return zs


def _match_at_z(
    observed: List[float],
    catalog: Dict[str, float],
    mode: str,
    z: float,
    tolerance: float,
) -> List[Dict[str, Any]]:
    """Best catalog assignment for each observed feature at redshift z."""
    assignments = []
    for obs in observed:
        best = None
        for name, rest in catalog.items():
            expected = rest / (1.0 + z) if mode == "frequency" else rest * (1.0 + z)
            frac = abs(obs - expected) / expected
            if frac <= tolerance and (best is None or frac < best["fractional_error"]):
                best = {
                    "observed": obs,
                    "line": name,
                    "rest_value": rest,
                    "expected_at_z": round(expected, 6),
                    "fractional_error": frac,
                }
        if best:
            best["fractional_error"] = round(best["fractional_error"], 6)
            assignments.append(best)
    return assignments


def estimate_redshift(
    observed_frequencies_ghz: Optional[List[float]] = None,
    observed_wavelengths_angstrom: Optional[List[float]] = None,
    observed_wavelengths_micron: Optional[List[float]] = None,
    z_min: float = 0.0,
    z_max: float = 12.0,
    tolerance: float = 0.002,
    max_candidates: int = 5,
) -> Dict[str, Any]:
    """
    Estimate source redshift from observed spectral features.

    Provide ONE of:
      observed_frequencies_ghz       — radio/mm features (GHz)
      observed_wavelengths_angstrom  — optical/UV features (Angstrom)
      observed_wavelengths_micron    — IR features (micron)

    Args:
        z_min / z_max: redshift search range
        tolerance: max fractional offset for a line match (0.002 = 600 km/s)
        max_candidates: number of candidate redshifts to return

    Returns:
        {"success": True, "candidates": [
            {"z": ..., "n_lines_matched": ..., "assignments": [...]}, ...],
         "best": {...} | None, "mode": "frequency"|"wavelength", ...}
    """
    try:
        # ── Select input mode and catalog ─────────────────────────────
        if observed_frequencies_ghz:
            observed = [float(v) for v in observed_frequencies_ghz if float(v) > 0]
            catalog, mode, unit = RADIO_MM_LINES_GHZ, "frequency", "GHz"
        elif observed_wavelengths_angstrom:
            observed = [float(v) for v in observed_wavelengths_angstrom if float(v) > 0]
            catalog, mode, unit = OPTICAL_LINES_ANGSTROM, "wavelength", "Angstrom"
        elif observed_wavelengths_micron:
            observed = [float(v) * 1e4 for v in observed_wavelengths_micron if float(v) > 0]
            catalog, mode, unit = OPTICAL_LINES_ANGSTROM, "wavelength", "Angstrom"
        else:
            return {"success": False,
                    "error": "Provide observed_frequencies_ghz, observed_wavelengths_angstrom, "
                             "or observed_wavelengths_micron"}

        if not observed:
            return {"success": False, "error": "No valid (positive) observed values supplied"}
        if z_max < z_min:
            return {"success": False, "error": f"z_max ({z_max}) < z_min ({z_min})"}

        observed = sorted(set(observed))

        # ── Generate and score candidate redshifts ────────────────────
        raw_zs = _candidate_redshifts(observed, catalog, mode, z_min, z_max)
        if not raw_zs:
            return {
                "success": True,
                "candidates": [],
                "best": None,
                "mode": mode,
                "unit": unit,
                "n_features": len(observed),
                "note": "No catalog line pairing yields a redshift in the requested range.",
            }

        scored: List[Dict[str, Any]] = []
        seen: List[float] = []
        for z in sorted(raw_zs):
            # Merge near-duplicate candidates (within tolerance in (1+z))
            if any(abs(z - s) / (1.0 + s) < tolerance for s in seen):
                continue
            seen.append(z)

            assignments = _match_at_z(observed, catalog, mode, z, tolerance)
            if not assignments:
                continue

            # Refine z with the mean implied redshift of all matched lines
            implied = []
            for a in assignments:
                if mode == "frequency":
                    implied.append(a["rest_value"] / a["observed"] - 1.0)
                else:
                    implied.append(a["observed"] / a["rest_value"] - 1.0)
            z_refined = float(sum(implied) / len(implied))
            residual = float(sum(a["fractional_error"] for a in assignments) / len(assignments))

            scored.append({
                "z": round(z_refined, 5),
                "n_lines_matched": len(assignments),
                "n_features": len(observed),
                "mean_fractional_residual": round(residual, 6),
                "assignments": assignments,
            })

        scored.sort(key=lambda c: (-c["n_lines_matched"], c["mean_fractional_residual"]))

        # Deduplicate refined candidates that converged to the same z
        deduped: List[Dict[str, Any]] = []
        for c in scored:
            if any(abs(c["z"] - d["z"]) / (1.0 + d["z"]) < tolerance for d in deduped):
                continue
            deduped.append(c)
        candidates = deduped[:max_candidates]

        best = candidates[0] if candidates else None
        confidence = "low"
        if best:
            if best["n_lines_matched"] >= 3:
                confidence = "high"
            elif best["n_lines_matched"] == 2:
                confidence = "medium"

        return {
            "success": True,
            "mode": mode,
            "unit": unit,
            "n_features": len(observed),
            "observed": observed,
            "tolerance": tolerance,
            "candidates": candidates,
            "best": best,
            "confidence": confidence,
            "note": (
                "Single-line identifications are degenerate — multiple redshifts are "
                "equally valid. Provide 2+ features for a reliable estimate."
                if best and best["n_lines_matched"] < 2 else ""
            ),
        }

    except Exception as e:
        logger.error(f"[REDSHIFT] estimate_redshift failed: {e}")
        return {"success": False, "error": str(e)}
