"""Optical/UV spectral-line lists in VACUUM wavelengths, keyed by SPARCL spectype.

Replaces the hardcoded 17-entry ``COMMON_LINES`` in services/sparcl_spectra.py,
whose values were internally MIXED air/vacuum: the Balmer/[O III]/[N II]/[S II]
entries were SDSS vacuum values, but Ca K 3933.66 / Ca H 3968.47 were air values
and [O II] 3727.4 was the air-doublet mean. SPARCL wavelength grids (DESI, SDSS)
are vacuum Angstroms, so every wavelength here is VACUUM (Angstrom), taken from
the SDSS idlspec2d spZline list (Morton 1991 air→vacuum convention) — the same
values DESI adopts. Air-only literature values were converted with the standard
IAU (Morton) refraction formula; the three corrected entries shift by ~+1.1 A:
  Ca K  3933.66 (air)  -> 3934.78 (vac)
  Ca H  3968.47 (air)  -> 3969.59 (vac)
  [O II] 3727.4 (air mean) -> 3728.48 (vacuum doublet midpoint of 3727.09/3729.88)

Lines carry a ``kind`` (emission / absorption / both) so plots can style them,
and lists deliberately remain ORDERED SEQUENCES of records — labels repeat
([O III], [S II], Ca II), so a name-keyed dict would silently drop lines.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

EMISSION = "emission"
ABSORPTION = "absorption"
BOTH = "both"


def _line(label: str, wavelength_vac: float, kind: str) -> Dict[str, Any]:
    return {"label": label, "wavelength_vac": float(wavelength_vac), "kind": kind}


# Master records (vacuum Angstrom). Sources: SDSS idlspec2d spZline vacuum
# wavelengths for the nebular/QSO lines; NIST air values converted to vacuum
# for the stellar absorption features (Ca H&K, G band, Mg b, Na D, Ca II IRT).
_LY_ALPHA = _line("Ly alpha", 1215.67, EMISSION)
_N_V = _line("N V", 1240.81, EMISSION)
_SI_IV = _line("Si IV+O IV]", 1399.8, EMISSION)     # blended QSO feature
_C_IV = _line("C IV", 1549.48, EMISSION)
_HE_II_UV = _line("He II", 1640.42, EMISSION)
_C_III = _line("C III]", 1908.73, EMISSION)
_MG_II = _line("Mg II", 2799.12, EMISSION)          # SDSS adopted vacuum value for the unresolved doublet (2796.35/2803.53)
_O_II = _line("[O II]", 3728.48, EMISSION)          # doublet midpoint (3727.09/3729.88 vac)
_NE_III = _line("[Ne III]", 3869.86, EMISSION)
_CA_K = _line("Ca K", 3934.78, ABSORPTION)          # air 3933.663 -> vacuum
_CA_H = _line("Ca H", 3969.59, ABSORPTION)          # air 3968.468 -> vacuum
_H_EPSILON = _line("H epsilon", 3971.20, BOTH)      # blends with Ca H in low-res data
_H_DELTA = _line("H delta", 4102.89, BOTH)
_G_BAND = _line("G band", 4305.61, ABSORPTION)      # CH band head (air 4304.4 -> vacuum)
_H_GAMMA = _line("H gamma", 4341.68, BOTH)
_O_III_4364 = _line("[O III]", 4364.44, EMISSION)
_HE_II_4686 = _line("He II", 4686.99, EMISSION)
_H_BETA = _line("H beta", 4862.68, BOTH)
_O_III_4960 = _line("[O III]", 4960.30, EMISSION)
_O_III_5008 = _line("[O III]", 5008.24, EMISSION)
_MG_B = _line("Mg b", 5176.80, ABSORPTION)          # triplet centroid (air 5175.36 -> vacuum)
_NA_D = _line("Na D", 5894.57, ABSORPTION)          # doublet mean (air 5889.95/5895.92 -> vacuum)
_N_II_6550 = _line("[N II]", 6549.86, EMISSION)
_H_ALPHA = _line("H alpha", 6564.61, BOTH)
_N_II_6585 = _line("[N II]", 6585.27, EMISSION)
_S_II_6718 = _line("[S II]", 6718.29, EMISSION)
_S_II_6733 = _line("[S II]", 6732.68, EMISSION)
_CA_II_8500 = _line("Ca II", 8500.36, ABSORPTION)   # IR triplet (air 8498.02 -> vacuum)
_CA_II_8544 = _line("Ca II", 8544.44, ABSORPTION)   # air 8542.09 -> vacuum
_CA_II_8664 = _line("Ca II", 8664.52, ABSORPTION)   # air 8662.14 -> vacuum


# Per-spectype sets (SPARCL vocabulary: GALAXY | STAR | QSO), wavelength-ordered.
SPECTYPE_LINES: Dict[str, List[Dict[str, Any]]] = {
    "GALAXY": [
        _O_II, _NE_III, _CA_K, _CA_H, _H_DELTA, _G_BAND, _H_GAMMA, _H_BETA,
        _O_III_4960, _O_III_5008, _MG_B, _NA_D, _N_II_6550, _H_ALPHA,
        _N_II_6585, _S_II_6718, _S_II_6733, _CA_II_8500, _CA_II_8544, _CA_II_8664,
    ],
    "QSO": [
        _LY_ALPHA, _N_V, _SI_IV, _C_IV, _HE_II_UV, _C_III, _MG_II, _O_II,
        _NE_III, _H_DELTA, _H_GAMMA, _O_III_4364, _HE_II_4686, _H_BETA,
        _O_III_4960, _O_III_5008, _H_ALPHA, _N_II_6585,
    ],
    "STAR": [
        _CA_K, _CA_H, _H_EPSILON, _H_DELTA, _G_BAND, _H_GAMMA, _H_BETA,
        _MG_B, _NA_D, _H_ALPHA, _CA_II_8500, _CA_II_8544, _CA_II_8664,
    ],
}


def lines_for_spectype(spectype: Optional[str]) -> List[Dict[str, Any]]:
    """Line records for a SPARCL spectype; unknown/None -> the union set.

    Returns copies so callers can annotate without mutating the module tables.
    """
    key = str(spectype or "").strip().upper()
    if key in SPECTYPE_LINES:
        return [dict(entry) for entry in SPECTYPE_LINES[key]]
    seen: set[Tuple[str, float]] = set()
    union: List[Dict[str, Any]] = []
    for entries in SPECTYPE_LINES.values():
        for entry in entries:
            marker = (entry["label"], entry["wavelength_vac"])
            if marker not in seen:
                seen.add(marker)
                union.append(dict(entry))
    union.sort(key=lambda entry: entry["wavelength_vac"])
    return union


def common_lines(spectype: Optional[str] = None) -> List[Tuple[str, float]]:
    """Back-compat shape for the legacy COMMON_LINES consumers: ordered
    (label, vacuum wavelength) tuples — labels repeat, so never dict this."""
    return [(entry["label"], entry["wavelength_vac"]) for entry in lines_for_spectype(spectype)]


def observed_lines(
    spectype: Optional[str],
    redshift: Optional[float],
    *,
    wavelength_min: Optional[float] = None,
    wavelength_max: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """Redshifted line positions, optionally clipped to an observed range.

    Keeps the legacy None-z guard: no redshift -> no markers (a STAR record
    with z~0 still gets markers because 0.0 is a valid float)."""
    if redshift is None:
        return []
    shift = 1.0 + float(redshift)
    out: List[Dict[str, Any]] = []
    for entry in lines_for_spectype(spectype):
        observed = entry["wavelength_vac"] * shift
        if wavelength_min is not None and observed < float(wavelength_min):
            continue
        if wavelength_max is not None and observed > float(wavelength_max):
            continue
        out.append({**entry, "wavelength_observed": observed})
    return out


__all__ = [
    "ABSORPTION",
    "BOTH",
    "EMISSION",
    "SPECTYPE_LINES",
    "common_lines",
    "lines_for_spectype",
    "observed_lines",
]
