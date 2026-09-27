"""MMDC (Markarian Multiwavelength Data Center, mmdc.am) access for Quasar.

Wraps the official ``astro-mmdc`` SDK (imported lazily, so a missing package
degrades to a clear tool error instead of breaking app import). Behaviour was
fixed by a live probe on 2026-09-26 (tmp/mmdc-integration-2026-09-26/PHASE0.md):

* ``sed.get_data`` JSON carries no MJDs; only the CSV is time-resolved, so the
  SED tool always downloads the FULL CSV and applies its own window rule.
* The server's ``mjd_start``/``mjd_end`` filter keeps rows that OVERLAP the
  window (a Fermi bin ending in 2022 survives a window ending in 2020). The
  default here is strict containment; both counts are always reported.
* Undated archival points (VOU-Blazars catalogs: NVSS, 2MASS, 4FGL, ...) carry
  the placeholder ``MJD_start == MJD_end == 55000.0``. They form a separate
  "archival, undated" group and are excluded from any time window unless the
  caller asks for them.
* The ``flag`` column is NaN / blank / ``Det`` / ``UL`` depending on the source
  pipeline; ``UL`` (case- and space-insensitive) is the only upper-limit mark.
* Membership in MMDC's blazar list is positional: the known-source list uses
  catalog names (``5BZBJ1959+6508``) and MMDC happily returns data for a
  non-blazar position it once cached (M82), so job status says nothing.

Light-curve contract (``LC_COLUMNS``; consumed by services/variability.py):
one row per measurement with

    time_mjd      float  MJD of the measurement (mjd_mid)
    date          str    ISO date (UTC) of time_mjd
    value         float  flux (unit in ``unit``)
    error         float  1-sigma flux error (NaN when unknown)
    band          str    catalog, plus ":<filter>" when the catalog has filters
    catalog       str    MMDC catalog code (MMDCGR, MMDCXRT, MMDCOUV, ZTF, ...)
    is_upper_limit bool
    spectral_index      float  photon index where MMDC provides one (NaN otherwise)
    spectral_index_err  float
    unit          str    flux unit; MMDC does not state light-curve units, so it
                         is marked "(inferred)"
    obsid         str    telescope observation id where present

Every MMDC-backed result carries ``attribution()``.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import io
import math
import os
import re
import tempfile
import threading
import time
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# ── constants ──────────────────────────────────────────────────────────────
MMDC_BASE_URL = os.getenv("MMDC_BASE_URL", "https://mmdc.am").rstrip("/")
MMDC_HOST = "mmdc.am"
MMDC_SDK_REQUIREMENT = "astro-mmdc==0.2.7"

ACKNOWLEDGMENT = "Some of the data used in this work were obtained from the MMDC."
MMDC_CITATION = "Sahakyan, Vardanyan, Giommi, et al. 2024, AJ 168, 289"
MMDC_BIBCODE = "2024AJ....168..289S"
MODELING_CITATIONS: Tuple[Tuple[str, str], ...] = (
    ("Bégué, Sahakyan, Dereli Bégué, et al. 2024, ApJ 963, 71", "2024ApJ...963...71B"),
    ("Sahakyan, Bégué, Casotto, et al. 2024, ApJ 971, 70", "2024ApJ...971...70S"),
)

UNDATED_MJD = 55000.0
MATCH_RADIUS_ARCSEC = 5.0
PLOTLY_POINT_CAP = 5000
H_EV_S = 4.135667696e-15  # Planck constant [eV s]


def _env_f(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    try:
        return float(raw) if raw else float(default)
    except ValueError:
        return float(default)


# Inner timeouts (seconds). services/tool_budgets.py reads these LIVE, so an
# env override that would break the budget hierarchy is caught by its test.
HTTP_TIMEOUT_S = _env_f("MMDC_HTTP_TIMEOUT_SECONDS", 45.0)
KNOWN_SOURCES_TIMEOUT_S = _env_f("MMDC_KNOWN_SOURCES_TIMEOUT_SECONDS", 30.0)  # 1.3 MB; >15 s once live
PREPARE_POLL_S = _env_f("MMDC_PREPARE_POLL_SECONDS", 50.0)
INFO_TIMEOUT_S = _env_f("MMDC_INFO_TIMEOUT_SECONDS", 10.0)
CSV_TIMEOUT_S = _env_f("MMDC_CSV_TIMEOUT_SECONDS", 90.0)  # full Mkn 421 CSV: 14 to 50 s live
LC_TIMEOUT_S = _env_f("MMDC_LC_TIMEOUT_SECONDS", 45.0)
KNOWN_SOURCES_TTL_S = _env_f("MMDC_KNOWN_SOURCES_TTL_SECONDS", 6 * 3600.0)
FRAME_CACHE_TTL_S = 900.0
FRAME_CACHE_MAX = 4

# Frequency ranges used for include_ranges and the per-band summary [Hz].
RANGE_EDGES: Tuple[Tuple[str, float, float], ...] = (
    ("radio", 0.0, 3.0e11),          # radio + mm/microwave (up to ~1 mm)
    ("infrared", 3.0e11, 3.9e14),    # ~1 mm to ~770 nm
    ("optical", 3.9e14, 1.0e15),     # ~770 nm to ~300 nm
    ("uv", 1.0e15, 2.4e16),          # ~300 nm to ~0.1 keV
    ("xray", 2.4e16, 2.4e20),        # ~0.1 keV to ~1 MeV
    ("gamma", 2.4e20, float("inf")),  # > ~1 MeV
)
RANGE_ALIASES = {
    "radio": "radio", "radio_waves": "radio", "microwave": "radio", "microwaves": "radio", "mm": "radio",
    "submm": "radio", "ir": "infrared", "infrared": "infrared", "optical": "optical", "visible": "optical",
    "uv": "uv", "ultraviolet": "uv", "xray": "xray", "x-ray": "xray", "x_rays": "xray", "xrays": "xray",
    "x-rays": "xray", "gamma": "gamma", "gamma-ray": "gamma", "gamma_rays": "gamma", "gammaray": "gamma",
    "gamma-rays": "gamma",
}

CATALOG_LABELS = {
    "MMDCGR": "Fermi-LAT (MMDCGR)",
    "MMDCXRT": "Swift-XRT (MMDCXRT)",
    "MMDCXRT_ORBIT": "Swift-XRT orbit (MMDCXRT_ORBIT)",
    "MMDCNuX": "NuSTAR (MMDCNuX)",
    "MMDCOUV": "Swift-UVOT (MMDCOUV)",
}
LC_CATALOGS = ("MMDCGR", "MMDCXRT", "MMDCXRT_ORBIT", "MMDCNuX", "MMDCOUV", "ASAS-SN", "ZTF", "PanSTARRS-LC", "SMARTS")
_LC_UNITS = {"MMDCGR": "ph cm^-2 s^-1 (inferred)"}
_LC_DEFAULT_UNIT = "erg cm^-2 s^-1 (inferred)"
LC_COLUMNS = ["time_mjd", "date", "value", "error", "band", "catalog", "is_upper_limit",
              "spectral_index", "spectral_index_err", "unit", "obsid"]
# Light-curve panels, highest energy first (shared MJD axis).
_LC_PANELS: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("gamma-ray", ("MMDCGR",)),
    ("X-ray", ("MMDCXRT", "MMDCXRT_ORBIT", "MMDCNuX")),
    ("UV", ("MMDCOUV:W1", "MMDCOUV:M2", "MMDCOUV:W2")),
    ("optical", ("MMDCOUV:U", "MMDCOUV:B", "MMDCOUV:V", "ASAS-SN", "ZTF", "PanSTARRS-LC", "SMARTS")),
)

WINDOW_RULE_TEXT = {
    "contained": "contained: a row is kept only if MJD_start >= window start AND MJD_end <= window end",
    "overlap": "overlap: a row is kept if its [MJD_start, MJD_end] interval intersects the window "
               "(bins may extend outside it)",
    "none": "no time window requested: every dated row is kept",
}


class MmdcUnavailable(RuntimeError):
    """The astro-mmdc SDK is not installed (or failed to import)."""


def import_sdk():
    """Import the SDK lazily; raise MmdcUnavailable with an install hint."""
    try:
        import astro_mmdc  # noqa: F401
        from astro_mmdc import MMDC
    except Exception as exc:  # ImportError, or a broken dependency chain
        raise MmdcUnavailable(
            f"MMDC support needs the astro-mmdc package ({MMDC_SDK_REQUIREMENT}); it could not be imported: {exc}"
        ) from exc
    return MMDC


def attribution(*, modeling: bool = False) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "acknowledgment": ACKNOWLEDGMENT,
        "citation": MMDC_CITATION,
        "bibcode": MMDC_BIBCODE,
    }
    if modeling:
        out["modeling_citations"] = [{"citation": c, "bibcode": b} for c, b in MODELING_CITATIONS]
    return out


# ── small pure helpers ─────────────────────────────────────────────────────
def mjd_to_date(mjd: Any) -> Optional[str]:
    from services.archive_catalogs import mjd_to_iso

    iso = mjd_to_iso(mjd)
    return iso[:10] if iso else None


def mjd_to_decimal_year(mjd: float) -> float:
    d = _dt.datetime(1858, 11, 17) + _dt.timedelta(days=float(mjd))
    start = _dt.datetime(d.year, 1, 1)
    return d.year + (d - start).total_seconds() / ((_dt.datetime(d.year + 1, 1, 1) - start).total_seconds())


def year_to_mjd(year: int) -> float:
    return (_dt.datetime(int(year), 1, 1) - _dt.datetime(1858, 11, 17)).days * 1.0


def angular_sep_arcsec(ra1: float, dec1: float, ra2: float, dec2: float) -> float:
    r1, d1, r2, d2 = map(math.radians, (ra1, dec1, ra2, dec2))
    # haversine: stable at small separations
    s = math.sin((d2 - d1) / 2) ** 2 + math.cos(d1) * math.cos(d2) * math.sin((r2 - r1) / 2) ** 2
    return math.degrees(2 * math.asin(min(1.0, math.sqrt(s)))) * 3600.0


def normalize_name(name: Any) -> str:
    s = str(name or "").strip().lower()
    s = re.sub(r"\bmarkarian\b", "mkn", s)
    s = re.sub(r"\bmrk\b", "mkn", s)
    return re.sub(r"[\s_]+", "", s)


def classify_range(freq_hz: float) -> str:
    try:
        f = float(freq_hz)
    except (TypeError, ValueError):
        return "unknown"
    for name, lo, hi in RANGE_EDGES:
        if lo <= f < hi:
            return name
    return "unknown"


def parse_window_bound(value: Any, *, is_end: bool) -> Tuple[Optional[float], str]:
    """Parse a window bound given as MJD, ISO date/datetime, 'YYYY-MM' or 'YYYY'.

    Returns (mjd, note). An END given only to day / month / year precision is
    expanded to the end of that period (the first instant of the next day /
    month / year), so "2020" as an end keeps everything observed in 2020.
    """
    if value is None or (isinstance(value, str) and not value.strip()):
        return None, ""
    from services.archive_catalogs import iso_to_mjd

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        v = float(value)
        if not math.isfinite(v):
            raise ValueError(f"window bound {value!r} is not finite")
        if 1800 <= v <= 2200 and float(v).is_integer():
            return _year_bound(int(v), is_end)
        if 10000 <= v <= 100000:
            return v, "MJD as given"
        raise ValueError(f"window bound {value!r} is neither a year (1800-2200) nor an MJD (10000-100000)")
    raw = str(value).strip()
    if re.fullmatch(r"\d{4}", raw):
        return _year_bound(int(raw), is_end)
    if re.fullmatch(r"\d{5}(\.\d+)?", raw):
        return float(raw), "MJD as given"
    m = re.fullmatch(r"(\d{4})-(\d{1,2})", raw)
    if m:
        y, mo = int(m.group(1)), int(m.group(2))
        if not 1 <= mo <= 12:
            raise ValueError(f"bad month in {raw!r}")
        start = _dt.datetime(y, mo, 1)
        if not is_end:
            return (start - _dt.datetime(1858, 11, 17)).days * 1.0, f"{raw} starts {start.date().isoformat()}"
        nxt = _dt.datetime(y + (mo == 12), 1 if mo == 12 else mo + 1, 1)
        return (nxt - _dt.datetime(1858, 11, 17)).days * 1.0, (
            f"end {raw} expanded to the end of that month ({nxt.date().isoformat()} 00:00 UTC)")
    mjd = iso_to_mjd(raw)
    if mjd is None:
        raise ValueError(f"unrecognised date {raw!r} (use YYYY, YYYY-MM, YYYY-MM-DD or an MJD)")
    if is_end and re.fullmatch(r"\d{4}-\d{1,2}-\d{1,2}", raw):
        return mjd + 1.0, f"end {raw} expanded to the end of that day"
    return mjd, "ISO date as given"


def _year_bound(year: int, is_end: bool) -> Tuple[float, str]:
    if is_end:
        return year_to_mjd(year + 1), f"end year {year} expanded to the end of {year} ({year + 1}-01-01 00:00 UTC)"
    return year_to_mjd(year), f"start year {year} = {year}-01-01"


def _is_ul(flag: Any) -> bool:
    return str(flag if flag is not None else "").strip().upper() == "UL"


def load_result(result_id: str, owner_id: Optional[str]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """A stored table by result_id, refused (as unknown) for another user's result."""
    from services.datalab_result_store import default_result_store

    frame, meta, status = default_result_store().lookup(str(result_id or "").strip())
    if status != "ok" or frame is None:
        why = {"expired": "has expired (results live about an hour); rerun the tool that produced it",
               "gone": "is no longer available; rerun the tool that produced it",
               "unknown": "is not a stored result id"}.get(status, status)
        raise KeyError(f"result_id {result_id!r} {why}")
    owner = str(meta.get("owner_id") or "").strip()
    caller = str(owner_id or "").strip()
    # An ownerless result is loadable only by an ownerless caller (scripts, tests):
    # a signed-in user must never read a result nobody owns (guard CX-03).
    if owner != caller:
        raise KeyError(f"result_id {result_id!r} is not a stored result id")
    return frame, meta


# ── SED table: normalize, window, summarize ────────────────────────────────
SED_COLUMNS = ["frequency_Hz", "energy_eV", "nuFnu_erg_cm2_s", "nuFnu_err_erg_cm2_s", "MJD_start", "MJD_end",
               "MJD_mid", "date_start", "date_end", "upper_limit", "undated", "catalog", "range", "reference"]


def normalize_sed_csv(source: Any) -> pd.DataFrame:
    """MMDC SED CSV (text, bytes, path or DataFrame) -> the normalized table."""
    if isinstance(source, pd.DataFrame):
        raw = source.copy()
    else:
        if isinstance(source, bytes):
            source = source.decode("utf-8", errors="replace")
        if isinstance(source, str) and "\n" in source:
            raw = pd.read_csv(io.StringIO(source), dtype={"flag": str, "catalog": str, "reference": str})
        else:
            raw = pd.read_csv(source, dtype={"flag": str, "catalog": str, "reference": str})
    required = {"frequency", "flux", "flux_err", "MJD_start", "MJD_end", "flag", "catalog"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"MMDC SED CSV is missing columns {sorted(missing)} (got {list(raw.columns)})")
    freq = pd.to_numeric(raw["frequency"], errors="coerce")
    flux = pd.to_numeric(raw["flux"], errors="coerce")
    err = pd.to_numeric(raw["flux_err"], errors="coerce")
    ms = pd.to_numeric(raw["MJD_start"], errors="coerce")
    me = pd.to_numeric(raw["MJD_end"], errors="coerce")
    undated = (ms == UNDATED_MJD) & (me == UNDATED_MJD) | ms.isna() | me.isna()
    ul = raw["flag"].map(_is_ul).astype(bool)
    df = pd.DataFrame({
        "frequency_Hz": freq,
        "energy_eV": freq * H_EV_S,
        "nuFnu_erg_cm2_s": flux,
        "nuFnu_err_erg_cm2_s": err,
        "MJD_start": ms.where(~undated),
        "MJD_end": me.where(~undated),
    })
    df["MJD_mid"] = (df["MJD_start"] + df["MJD_end"]) / 2.0
    df["date_start"] = df["MJD_start"].map(lambda v: mjd_to_date(v) if pd.notna(v) else "")
    df["date_end"] = df["MJD_end"].map(lambda v: mjd_to_date(v) if pd.notna(v) else "")
    df["upper_limit"] = ul.values
    df["undated"] = undated.values
    df["catalog"] = raw["catalog"].astype(str).str.strip()
    df["range"] = df["frequency_Hz"].map(classify_range)
    ref = raw["reference"] if "reference" in raw.columns else pd.Series([""] * len(raw))
    df["reference"] = ref.fillna("").astype(str).str.strip().str.rstrip("!").str.strip()
    df = df[df["frequency_Hz"].notna() & df["nuFnu_erg_cm2_s"].notna()].reset_index(drop=True)
    return df[SED_COLUMNS]


def _straddle_text(by_cat: Dict[str, Dict[str, Any]], key: str) -> str:
    if key == "rows":
        return ", ".join("%s: %d" % (c, v["rows"]) for c, v in by_cat.items())
    return ", ".join("%s up to %s" % (c, v[key]) for c, v in by_cat.items())


def apply_window(
    df: pd.DataFrame,
    start_mjd: Optional[float],
    end_mjd: Optional[float],
    mode: str = "contained",
    include_undated: bool = False,
    end_exclusive: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Apply the time-window rule. Returns (kept rows, report with both counts).

    ``end_exclusive`` (set when a year / month / day end was expanded to the
    first instant of the next period): that instant itself is outside the
    window, so the end comparison is strict (guard CX-02)."""
    mode = str(mode or "contained").strip().lower()
    if mode not in ("contained", "overlap"):
        raise ValueError("window_mode must be 'contained' or 'overlap'")
    if start_mjd is not None and end_mjd is not None and end_mjd <= start_mjd:
        raise ValueError(f"window end (MJD {end_mjd}) must be after its start (MJD {start_mjd})")
    dated = df[~df["undated"]]
    undated = df[df["undated"]]
    windowed = start_mjd is not None or end_mjd is not None
    lo = -np.inf if start_mjd is None else float(start_mjd)
    hi = np.inf if end_mjd is None else float(end_mjd)
    if end_exclusive and end_mjd is not None:
        contained_mask = (dated["MJD_start"] >= lo) & (dated["MJD_end"] < hi)
        overlap_mask = (dated["MJD_start"] < hi) & (dated["MJD_end"] >= lo)
    else:
        contained_mask = (dated["MJD_start"] >= lo) & (dated["MJD_end"] <= hi)
        overlap_mask = (dated["MJD_start"] <= hi) & (dated["MJD_end"] >= lo)
    straddle = dated[overlap_mask & ~contained_mask]
    straddle_by_cat: Dict[str, Dict[str, Any]] = {}
    for cat, g in straddle.groupby("catalog"):
        straddle_by_cat[str(cat)] = {
            "rows": int(len(g)),
            "earliest_MJD_start": round(float(g["MJD_start"].min()), 3),
            "latest_MJD_end": round(float(g["MJD_end"].max()), 3),
            "latest_date_end": mjd_to_date(g["MJD_end"].max()),
        }
    if not windowed:
        kept_dated = dated
        rule = "none"
    else:
        kept_dated = dated[contained_mask if mode == "contained" else overlap_mask]
        rule = mode
    include_undated_eff = bool(include_undated) or not windowed
    kept = pd.concat([kept_dated, undated]) if include_undated_eff else kept_dated
    report: Dict[str, Any] = {
        "rule_applied": rule,
        "rule_text": WINDOW_RULE_TEXT[rule] + (" (the end instant itself is excluded: the end was expanded to the "
                                               "first instant of the next period)" if end_exclusive and rule != "none" else ""),
        "window_end_exclusive": bool(end_exclusive),
        "window_start_mjd": None if start_mjd is None else round(float(start_mjd), 4),
        "window_end_mjd": None if end_mjd is None else round(float(end_mjd), 4),
        "window_start_date": mjd_to_date(start_mjd) if start_mjd is not None else None,
        "window_end_date": mjd_to_date(end_mjd) if end_mjd is not None else None,
        "dated_rows_available": int(len(dated)),
        "rows_kept_by_contained_rule": int(contained_mask.sum()),
        "rows_kept_by_overlap_rule": int(overlap_mask.sum()),
        "straddling_rows": int(len(straddle)),
        "straddling_by_catalog": straddle_by_cat,
        "dated_rows_outside_window": int(len(dated) - int(overlap_mask.sum())),
        "undated_archival_rows": int(len(undated)),
        "undated_included": bool(include_undated_eff and len(undated) > 0),
        "undated_catalogs": sorted({str(c) for c in undated["catalog"].unique()}),
        "undated_catalog_count": int(undated["catalog"].nunique()),
    }
    if len(kept_dated):
        report["kept_min_MJD_start"] = round(float(kept_dated["MJD_start"].min()), 3)
        report["kept_max_MJD_end"] = round(float(kept_dated["MJD_end"].max()), 3)
        report["kept_date_range"] = [mjd_to_date(kept_dated["MJD_start"].min()), mjd_to_date(kept_dated["MJD_end"].max())]
    else:
        report["kept_min_MJD_start"] = report["kept_max_MJD_end"] = None
        report["kept_date_range"] = None
    if windowed and not include_undated and len(undated):
        report["undated_note"] = (
            f"{len(undated)} undated archival points ({', '.join(report['undated_catalogs'][:8])}"
            f"{'...' if len(report['undated_catalogs']) > 8 else ''}) have no observation date in MMDC "
            "(placeholder MJD 55000) and were excluded from the window; pass include_undated=true to show them "
            "as a separate 'archival, undated' group.")
    elif windowed and include_undated and len(undated):
        report["undated_note"] = (
            f"{len(undated)} undated archival points are included as a separate 'archival, undated' group; "
            "they carry no observation date, so they are NOT evidence of emission inside the window.")
    if windowed and mode == "contained" and len(straddle):
        report["straddling_note"] = (
            f"{len(straddle)} rows overlap the window but extend outside it and were excluded by the contained rule "
            f"({_straddle_text(straddle_by_cat, 'rows')}); window_mode='overlap' would add them.")
    elif windowed and mode == "overlap" and len(straddle):
        report["straddling_note"] = (
            f"{len(straddle)} kept rows extend outside the window (overlap rule): "
            + _straddle_text(straddle_by_cat, "latest_date_end") + ".")
    return kept.reset_index(drop=True), report


def filter_catalogs_and_ranges(
    df: pd.DataFrame, exclude_catalogs: Optional[Sequence[str]], include_ranges: Optional[Sequence[str]]
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    info: Dict[str, Any] = {}
    out = df
    if exclude_catalogs:
        wanted = {str(c).strip().lower() for c in exclude_catalogs if str(c).strip()}
        mask = out["catalog"].str.lower().isin(wanted)
        info["excluded_catalogs"] = sorted({c for c in out.loc[mask, "catalog"]})
        info["rows_removed_by_exclude_catalogs"] = int(mask.sum())
        unknown = sorted(wanted - {c.lower() for c in df["catalog"].unique()})
        if unknown:
            info["exclude_catalogs_not_present"] = unknown
        out = out[~mask]
    if include_ranges:
        ranges = set()
        bad = []
        for r in include_ranges:
            key = RANGE_ALIASES.get(str(r).strip().lower().replace(" ", ""))
            (ranges.add(key) if key else bad.append(str(r)))
        if bad:
            raise ValueError(f"unknown include_ranges {bad}; use radio, infrared, optical, uv, xray, gamma")
        mask = out["range"].isin(ranges)
        info["included_ranges"] = sorted(ranges)
        info["rows_removed_by_include_ranges"] = int((~mask).sum())
        out = out[mask]
    return out.reset_index(drop=True), info


def _g(v: float) -> float:
    return float(f"{float(v):.4g}")


def summarize_sed(df: pd.DataFrame, max_groups: int = 30) -> Dict[str, Any]:
    groups: List[Dict[str, Any]] = []
    for (cat, rng), g in df.groupby(["catalog", "range"], sort=False):
        det = g[~g["upper_limit"]]
        dated = g[~g["undated"]]
        row: Dict[str, Any] = {
            "catalog": str(cat),
            "range": str(rng),
            "n": int(len(g)),
            "n_upper_limits": int(g["upper_limit"].sum()),
            "undated": bool(g["undated"].all()),
        }
        if len(dated):
            row["MJD_range"] = [round(float(dated["MJD_start"].min()), 2), round(float(dated["MJD_end"].max()), 2)]
            row["date_range"] = [mjd_to_date(dated["MJD_start"].min()), mjd_to_date(dated["MJD_end"].max())]
        if len(det):
            row["nuFnu_range_erg_cm2_s"] = [_g(det["nuFnu_erg_cm2_s"].min()), _g(det["nuFnu_erg_cm2_s"].max())]
        row["frequency_range_Hz"] = [_g(g["frequency_Hz"].min()), _g(g["frequency_Hz"].max())]
        groups.append(row)
    groups.sort(key=lambda r: -r["n"])
    shown = groups[:max_groups]
    rest = groups[max_groups:]
    dated_all = df[~df["undated"]]
    totals: Dict[str, Any] = {
        "rows": int(len(df)),
        "detections": int((~df["upper_limit"]).sum()),
        "upper_limits": int(df["upper_limit"].sum()),
        "undated_rows": int(df["undated"].sum()),
        "catalogs": int(df["catalog"].nunique()),
        "frequency_range_Hz": [_g(df["frequency_Hz"].min()), _g(df["frequency_Hz"].max())] if len(df) else None,
        "frequency_span_decades": round(math.log10(df["frequency_Hz"].max() / df["frequency_Hz"].min()), 1) if len(df) else None,
    }
    if len(dated_all):
        totals["MJD_range"] = [round(float(dated_all["MJD_start"].min()), 2), round(float(dated_all["MJD_end"].max()), 2)]
        totals["date_range"] = [mjd_to_date(dated_all["MJD_start"].min()), mjd_to_date(dated_all["MJD_end"].max())]
    out = {"by_catalog_and_range": shown, "totals": totals}
    if rest:
        out["other_groups"] = {"groups": len(rest), "rows": int(sum(r["n"] for r in rest))}
    return out


def downsample_sed(df: pd.DataFrame, cap: int = PLOTLY_POINT_CAP) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Subsample detections for plotting. Keeps EVERY upper limit and every
    undated point, plus per (catalog, range) extremes in flux, frequency and
    time; the rest is a uniform stride. The stored table is never touched."""
    n = len(df)
    cap = min(PLOTLY_POINT_CAP, max(100, int(cap or PLOTLY_POINT_CAP)))  # guard CX-07: never above the 5000 cap
    if n <= cap:
        return df, {"downsampled": False, "plotted_points": n, "table_rows": n}
    keep = set(df.index[df["upper_limit"] | df["undated"]])
    det = df[~df["upper_limit"] & ~df["undated"]]
    for _, g in det.groupby(["catalog", "range"]):
        for col in ("nuFnu_erg_cm2_s", "frequency_Hz", "MJD_mid"):
            s = g[col].dropna()
            if len(s):
                keep.add(s.idxmin())
                keep.add(s.idxmax())
    remaining = cap - len(keep)
    pool = det.index.difference(pd.Index(sorted(keep)))
    if remaining > 0 and len(pool):
        order = det.loc[pool].sort_values("MJD_mid").index
        idx = np.linspace(0, len(order) - 1, min(remaining, len(order))).round().astype(int)
        keep.update(order[np.unique(idx)])
    sub = df.loc[sorted(keep)]
    return sub, {
        "downsampled": True,
        "plotted_points": int(len(sub)),
        "table_rows": n,
        "note": (f"The plot shows {len(sub)} of {n} points (every upper limit, every undated point and each "
                 "catalog's flux/frequency/time extremes are kept; the rest is a uniform subsample). "
                 "The table card and its CSV export hold all rows."),
    }


def _year_ticks(lo_mjd: float, hi_mjd: float, max_ticks: int = 10) -> Tuple[List[float], List[str]]:
    y0 = int(math.floor(mjd_to_decimal_year(lo_mjd)))
    y1 = int(math.ceil(mjd_to_decimal_year(hi_mjd)))
    span = max(1, y1 - y0)
    step = max(1, int(math.ceil(span / max_ticks)))
    years = [y for y in range(y0, y1 + 1, step)]
    vals, text = [], []
    for y in years:
        m = year_to_mjd(y)
        if lo_mjd - 1 <= m <= hi_mjd + 1:
            vals.append(m)
            text.append(str(y))
    if not vals:
        mid = (lo_mjd + hi_mjd) / 2.0
        vals, text = [mid], [f"{mjd_to_decimal_year(mid):.2f}"]
    return vals, text


def _fmt_mjd(v: Any) -> str:
    return f"{float(v):.2f}" if v is not None and pd.notna(v) else "n/a"


def _sed_customdata(g: pd.DataFrame) -> List[List[Any]]:
    out = []
    for r in g.itertuples(index=False):
        out.append([
            r.catalog, r.range,
            "n/a" if pd.isna(r.nuFnu_err_erg_cm2_s) else f"{float(r.nuFnu_err_erg_cm2_s):.2e}",
            "undated" if r.undated else _fmt_mjd(r.MJD_start),
            "" if r.undated else _fmt_mjd(r.MJD_end),
            "archival, no date" if r.undated else f"{r.date_start} to {r.date_end}",
            "UPPER LIMIT" if r.upper_limit else "detection",
        ])
    return out


_SED_HOVER = ("%{customdata[0]} (%{customdata[1]})<br>ν = %{x:.3e} Hz<br>νFν = %{y:.3e} ± %{customdata[2]} "
              "erg cm⁻² s⁻¹<br>MJD %{customdata[3]} to %{customdata[4]}<br>%{customdata[5]}<br>"
              "%{customdata[6]}<extra></extra>")
_CATEGORICAL = ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f", "#edc948", "#b07aa1", "#ff9da7",
                "#9c755f", "#bab0ac", "#1f77b4", "#d62728"]


def sed_plotly_spec(df: pd.DataFrame, title: str, *, color_by: str = "epoch", energy_axis: bool = True,
                    model_curves: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """log-log nuFnu vs nu. color_by='epoch' colours by MJD mid (colorbar in
    calendar years); 'catalog' uses one colour per catalog. Upper limits are
    downward triangles with a short downward stem, never detection markers."""
    traces: List[Dict[str, Any]] = []
    dated = df[~df["undated"]]
    has_epoch = color_by == "epoch" and len(dated) > 0
    if has_epoch:
        cmin, cmax = float(dated["MJD_mid"].min()), float(dated["MJD_mid"].max())
        if cmax - cmin < 1:
            cmin, cmax = cmin - 1, cmax + 1
        tickvals, ticktext = _year_ticks(cmin, cmax)

    def _ul_stem(y: pd.Series) -> Dict[str, Any]:
        # A short downward stem (to 60 % of the limit) turns the triangle into an arrow.
        return {"type": "data", "symmetric": False, "array": [0.0] * len(y),
                "arrayminus": [float(v) * 0.4 for v in y], "thickness": 1.2, "width": 0, "visible": True}

    def _err(g: pd.DataFrame) -> Dict[str, Any]:
        e = g["nuFnu_err_erg_cm2_s"].fillna(0).clip(lower=0)
        # On log axes a bar reaching <= 0 is dropped by Plotly; clip the lower arm.
        lower = np.minimum(e.values, g["nuFnu_erg_cm2_s"].values * 0.95)
        return {"type": "data", "symmetric": False, "array": [float(v) for v in e.values],
                "arrayminus": [float(v) for v in lower], "thickness": 0.8, "width": 0, "visible": True,
                "color": "rgba(120,120,120,0.55)"}

    if has_epoch:
        det = dated[~dated["upper_limit"]]
        ul = dated[dated["upper_limit"]]
        marker_common = {"colorscale": "Viridis", "cmin": cmin, "cmax": cmax}
        if len(det):
            traces.append({
                "type": "scatter", "mode": "markers", "name": "detections (colour = epoch)",
                "x": det["frequency_Hz"].tolist(), "y": det["nuFnu_erg_cm2_s"].tolist(), "error_y": _err(det),
                "marker": {**marker_common, "color": det["MJD_mid"].round(3).tolist(), "size": 6, "opacity": 0.85,
                           "showscale": True,
                           "colorbar": {"title": {"text": "epoch (year)"}, "tickvals": tickvals, "ticktext": ticktext,
                                        "thickness": 14}},
                "customdata": _sed_customdata(det), "hovertemplate": _SED_HOVER,
            })
        if len(ul):
            traces.append({
                "type": "scatter", "mode": "markers", "name": "upper limits",
                "x": ul["frequency_Hz"].tolist(), "y": ul["nuFnu_erg_cm2_s"].tolist(), "error_y": _ul_stem(ul["nuFnu_erg_cm2_s"]),
                "marker": {**marker_common, "color": ul["MJD_mid"].round(3).tolist(), "symbol": "triangle-down",
                           "size": 9, "showscale": False, "line": {"width": 0.5, "color": "#333"}},
                "customdata": _sed_customdata(ul), "hovertemplate": _SED_HOVER,
            })
    else:
        cats = [c for c, _ in dated["catalog"].value_counts().items()]
        top = cats[:11]
        for i, cat in enumerate(top + (["other"] if len(cats) > 11 else [])):
            g = dated[dated["catalog"].isin(cats[11:])] if cat == "other" else dated[dated["catalog"] == cat]
            color = _CATEGORICAL[i % len(_CATEGORICAL)]
            det, ul = g[~g["upper_limit"]], g[g["upper_limit"]]
            label = CATALOG_LABELS.get(cat, cat)
            if len(det):
                traces.append({"type": "scatter", "mode": "markers", "name": label,
                               "legendgroup": cat, "x": det["frequency_Hz"].tolist(), "y": det["nuFnu_erg_cm2_s"].tolist(),
                               "error_y": _err(det), "marker": {"color": color, "size": 6, "opacity": 0.85},
                               "customdata": _sed_customdata(det), "hovertemplate": _SED_HOVER})
            if len(ul):
                traces.append({"type": "scatter", "mode": "markers", "name": f"{label} upper limits", "legendgroup": cat,
                               "x": ul["frequency_Hz"].tolist(), "y": ul["nuFnu_erg_cm2_s"].tolist(),
                               "error_y": {**_ul_stem(ul["nuFnu_erg_cm2_s"]), "color": color},
                               "marker": {"color": color, "symbol": "triangle-down", "size": 9},
                               "customdata": _sed_customdata(ul), "hovertemplate": _SED_HOVER})
    und = df[df["undated"]]
    if len(und):
        for kind, g in (("", und[~und["upper_limit"]]), (" upper limits", und[und["upper_limit"]])):
            if not len(g):
                continue
            is_ul = bool(kind)
            traces.append({
                "type": "scatter", "mode": "markers", "name": f"archival, undated{kind}",
                "x": g["frequency_Hz"].tolist(), "y": g["nuFnu_erg_cm2_s"].tolist(),
                **({"error_y": {**_ul_stem(g["nuFnu_erg_cm2_s"]), "color": "#888"}} if is_ul else {"error_y": _err(g)}),
                "marker": {"color": "rgba(0,0,0,0)", "symbol": "triangle-down-open" if is_ul else "circle-open",
                           "size": 9 if is_ul else 7, "line": {"color": "#888", "width": 1.2}},
                "customdata": _sed_customdata(g), "hovertemplate": _SED_HOVER,
            })
    for i, curve in enumerate(model_curves or []):
        traces.append({"type": "scatter", "mode": "lines", "name": curve.get("name") or f"model {i + 1}",
                       "x": list(curve["nu"]), "y": list(curve["nuFnu"]),
                       "line": {"color": curve.get("color") or ["#d62728", "#1f77b4", "#2ca02c"][i % 3],
                                "width": curve.get("width", 2), "dash": curve.get("dash", "solid")},
                       "opacity": curve.get("opacity", 1.0),
                       "showlegend": curve.get("showlegend", True),
                       "hovertemplate": (curve.get("name") or "model") + "<br>ν = %{x:.3e} Hz<br>νFν = %{y:.3e}<extra></extra>"})
    layout: Dict[str, Any] = {
        "title": {"text": title, "y": 0.985, "yanchor": "top"},
        "xaxis": {"type": "log", "title": {"text": "ν [Hz]"}, "exponentformat": "power"},
        "yaxis": {"type": "log", "title": {"text": "νFν [erg cm⁻² s⁻¹]"}, "exponentformat": "power"},
        "height": 520,
        "hovermode": "closest",
        "legend": {"orientation": "h", "y": -0.18},
        "margin": {"t": 125 if energy_axis else 60},
    }
    xs = [x for t in traces for x in t.get("x", []) if x and x > 0]
    if xs:
        lo, hi = math.log10(min(xs)) - 0.2, math.log10(max(xs)) + 0.2
        layout["xaxis"]["range"] = [lo, hi]
        if energy_axis:
            # Secondary top axis in eV. Plotly draws an axis only if a trace
            # references it, so an invisible two-point trace anchors it.
            layout["xaxis2"] = {"type": "log", "overlaying": "x", "side": "top", "title": {"text": "E [eV]"},
                                "range": [lo + math.log10(H_EV_S), hi + math.log10(H_EV_S)], "showgrid": False,
                                "exponentformat": "power"}
            ys = [y for t in traces for y in t.get("y", []) if y and y > 0]
            traces.append({"type": "scatter", "mode": "markers", "x": [10 ** lo * H_EV_S, 10 ** hi * H_EV_S],
                           "y": [min(ys), min(ys)] if ys else [1, 1], "xaxis": "x2", "marker": {"opacity": 0},
                           "hoverinfo": "skip", "showlegend": False, "name": "energy axis"})
    return {"data": traces, "layout": layout}


def ul_arrows(ax: Any, x: Any, y: Any, *, color: Any = "k", frac: float = 0.6, label: Optional[str] = None,
              hollow: bool = False) -> None:
    """Upper limits as arrows in matplotlib figures: a cap at the limit, a stem down to
    ``frac`` x limit and a downward triangle at its tip (guard CX-05: PNG fallbacks drew
    bare triangles). Drawn explicitly because errorbar(uplims=...) caret direction is
    easy to get backwards."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if not len(x):
        return
    tip = y * frac
    ax.vlines(x, tip, y, colors=color, linewidths=0.9)
    ax.scatter(x, y, marker="_", s=40, color=color)
    kw = {"facecolors": "none", "edgecolors": color} if hollow else {"color": color}
    ax.scatter(x, tip, marker="v", s=22, label=label, **kw)


def sed_png(df: pd.DataFrame, title: str, plotting_service: Any, *, color_by: str = "epoch",
            model_curves: Optional[List[Dict[str, Any]]] = None, name: str = "mmdc_sed") -> Dict[str, Any]:
    """Static PNG fallback (PlottingService) mirroring the plotly figure."""
    import uuid

    plt = plotting_service._apply_style(dark=False)
    fig, ax = plt.subplots(figsize=(8.5, 5.6))
    dated = df[~df["undated"]]
    det, ul = dated[~dated["upper_limit"]], dated[dated["upper_limit"]]
    if color_by == "epoch" and len(dated):
        vmin, vmax = float(dated["MJD_mid"].min()), float(dated["MJD_mid"].max() + 1e-6)
        sc = ax.scatter(det["frequency_Hz"], det["nuFnu_erg_cm2_s"], c=det["MJD_mid"], cmap="viridis", vmin=vmin,
                        vmax=vmax, s=10, alpha=0.8, label="detections")
        if len(ul):
            import matplotlib.cm as _cm
            import matplotlib.colors as _mc

            colors = _cm.viridis(_mc.Normalize(vmin=vmin, vmax=vmax)(ul["MJD_mid"].to_numpy()))
            ul_arrows(ax, ul["frequency_Hz"], ul["nuFnu_erg_cm2_s"], color=colors, label="upper limits")
        cb = fig.colorbar(sc, ax=ax, pad=0.01)
        tv, tt = _year_ticks(vmin, vmax)
        cb.set_ticks(tv)
        cb.set_ticklabels(tt)
        cb.set_label("epoch (year)")
    else:
        for i, (cat, g) in enumerate(dated.groupby("catalog")):
            color = _CATEGORICAL[i % len(_CATEGORICAL)]
            d, u = g[~g["upper_limit"]], g[g["upper_limit"]]
            ax.scatter(d["frequency_Hz"], d["nuFnu_erg_cm2_s"], s=10, color=color, label=cat)
            if len(u):
                ul_arrows(ax, u["frequency_Hz"], u["nuFnu_erg_cm2_s"], color=color)
    und = df[df["undated"]]
    if len(und):
        d, u = und[~und["upper_limit"]], und[und["upper_limit"]]
        ax.scatter(d["frequency_Hz"], d["nuFnu_erg_cm2_s"], s=22, facecolors="none", edgecolors="0.45",
                   label="archival, undated")
        if len(u):
            ul_arrows(ax, u["frequency_Hz"], u["nuFnu_erg_cm2_s"], color="0.45", hollow=True)
    for curve in model_curves or []:
        ax.plot(curve["nu"], curve["nuFnu"], lw=curve.get("width", 2), alpha=curve.get("opacity", 1.0),
                color=curve.get("color") or "#d62728", ls="--" if curve.get("dash") == "dash" else "-",
                label=curve.get("name") if curve.get("showlegend", True) else None)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("ν [Hz]")
    ax.set_ylabel("νFν [erg cm$^{-2}$ s$^{-1}$]")
    ax.set_title(title, fontsize=10)
    if len(ax.get_legend_handles_labels()[0]) <= 14:
        ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    render = plotting_service._save_and_encode(fig, f"{name}_{uuid.uuid4().hex[:10]}")
    return {"image_base64": render.get("base64_png"), "path": render.get("web_url"),
            "png_path": render.get("png_path"), "pdf_path": render.get("pdf_path")}


# ── light curves ───────────────────────────────────────────────────────────
def normalize_lightcurve_rows(rows: Iterable[Any]) -> pd.DataFrame:
    """SDK Observation objects (or dicts) -> the LC_COLUMNS contract."""
    recs = []
    for r in rows:
        d = r.model_dump() if hasattr(r, "model_dump") else dict(r)
        if not d.get("is_lightcurve", True):
            continue
        t = d.get("mjd_mid")
        if t is None and d.get("mjd_start") is not None and d.get("mjd_end") is not None:
            t = (float(d["mjd_start"]) + float(d["mjd_end"])) / 2.0
        if t is None or d.get("flux") is None:
            continue
        cat = str(d.get("catalog") or "")
        fb = d.get("filter_band")
        recs.append({
            "time_mjd": float(t),
            "value": float(d["flux"]),
            "error": float(d["flux_err"]) if d.get("flux_err") is not None else np.nan,
            "band": f"{cat}:{fb}" if fb else cat,
            "catalog": cat,
            "is_upper_limit": bool(d.get("is_upper_limit")),
            "spectral_index": float(d["spectral_index"]) if d.get("spectral_index") is not None else np.nan,
            "spectral_index_err": float(d["spectral_index_err"]) if d.get("spectral_index_err") is not None else np.nan,
            "unit": _LC_UNITS.get(cat, _LC_DEFAULT_UNIT),
            "obsid": str(d.get("obsid") or ""),
        })
    df = pd.DataFrame(recs, columns=[c for c in LC_COLUMNS if c != "date"])
    if len(df):
        df = df.sort_values(["band", "time_mjd"]).reset_index(drop=True)
    df.insert(1, "date", df["time_mjd"].map(mjd_to_date) if len(df) else pd.Series(dtype=str))
    return df[LC_COLUMNS]


def summarize_lightcurve(df: pd.DataFrame) -> List[Dict[str, Any]]:
    out = []
    for band, g in df.groupby("band", sort=False):
        det = g[~g["is_upper_limit"]]
        row = {
            "band": str(band),
            "n": int(len(g)),
            "n_upper_limits": int(g["is_upper_limit"].sum()),
            "MJD_range": [round(float(g["time_mjd"].min()), 2), round(float(g["time_mjd"].max()), 2)],
            "date_range": [mjd_to_date(g["time_mjd"].min()), mjd_to_date(g["time_mjd"].max())],
            "unit": str(g["unit"].iloc[0]),
            "n_with_spectral_index": int(g["spectral_index"].notna().sum()),
            "duplicate_epochs": int(g["time_mjd"].round(4).duplicated().sum()),
        }
        if len(det):
            row["flux_range"] = [_g(det["value"].min()), _g(det["value"].max())]
        out.append(row)
    out.sort(key=lambda r: -r["n"])
    return out


def _panel_of(band: str) -> int:
    for i, (_, members) in enumerate(_LC_PANELS):
        if band in members or band.split(":")[0] in members:
            return i
    return len(_LC_PANELS) - 1


def downsample_trace(g: pd.DataFrame, cap: int, value_col: str = "value", time_col: str = "time_mjd") -> pd.DataFrame:
    """Uniform stride that always keeps upper limits and the value/time extremes."""
    if len(g) <= cap:
        return g
    keep = set(g.index[g["is_upper_limit"]]) if "is_upper_limit" in g else set()
    for col in (value_col, time_col):
        s = g[col].dropna()
        if len(s):
            keep.add(s.idxmin())
            keep.add(s.idxmax())
    pool = g.index.difference(pd.Index(sorted(keep)))
    room = max(0, cap - len(keep))
    if room and len(pool):
        idx = np.linspace(0, len(pool) - 1, min(room, len(pool))).round().astype(int)
        keep.update(pool[np.unique(idx)])
    return g.loc[sorted(keep)]


def lightcurve_plotly_spec(df: pd.DataFrame, title: str, cap: int = PLOTLY_POINT_CAP) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Stacked panels (gamma / X-ray / UV / optical) on a shared MJD axis with
    a calendar-year top axis; upper limits as downward triangles."""
    bands = list(df["band"].unique())
    used_panels = sorted({_panel_of(b) for b in bands})
    n_pan = max(1, len(used_panels))
    gap = 0.04
    h = (1.0 - gap * (n_pan - 1)) / n_pan
    layout: Dict[str, Any] = {"title": {"text": title}, "height": max(360, 230 * n_pan + 80), "hovermode": "closest",
                              "legend": {"orientation": "h", "y": -0.12}, "margin": {"t": 80}}
    panel_axis = {}
    for k, p in enumerate(used_panels):
        top = 1.0 - k * (h + gap)
        name = "yaxis" if k == 0 else f"yaxis{k + 1}"
        unit = _LC_UNITS.get(_LC_PANELS[p][1][0].split(":")[0], _LC_DEFAULT_UNIT).replace(" (inferred)", "")
        layout[name] = {"domain": [max(0.0, top - h), top], "title": {"text": f"{_LC_PANELS[p][0]}<br>[{unit}]"},
                        "anchor": "x", "exponentformat": "power"}
        panel_axis[p] = "y" if k == 0 else f"y{k + 1}"
    layout["xaxis"] = {"title": {"text": "MJD"}, "anchor": panel_axis[used_panels[-1]]}
    per_trace = max(50, cap // max(1, len(bands)))
    traces: List[Dict[str, Any]] = []
    truncated = False
    for i, band in enumerate(bands):
        g = df[df["band"] == band]
        g2 = downsample_trace(g, per_trace)
        truncated = truncated or len(g2) < len(g)
        color = _CATEGORICAL[i % len(_CATEGORICAL)]
        yaxis = panel_axis[_panel_of(band)]
        label = CATALOG_LABELS.get(band, band)
        for is_ul, sub in ((False, g2[~g2["is_upper_limit"]]), (True, g2[g2["is_upper_limit"]])):
            if not len(sub):
                continue
            tr = {"type": "scatter", "mode": "markers",
                  "name": f"{label}{' limits' if is_ul else ''}", "legendgroup": band,
                  "x": sub["time_mjd"].round(5).tolist(), "y": sub["value"].tolist(), "yaxis": yaxis,
                  "marker": {"color": color, "size": 7 if is_ul else 5,
                             **({"symbol": "triangle-down", "opacity": 0.5} if is_ul else {})},
                  "customdata": [[d, band, "UPPER LIMIT" if is_ul else "detection"] for d in sub["date"]],
                  "hovertemplate": "%{customdata[1]}<br>MJD %{x:.3f} (%{customdata[0]})<br>flux %{y:.3e}<br>"
                                   "%{customdata[2]}<extra></extra>"}
            if not is_ul and sub["error"].notna().any():
                tr["error_y"] = {"type": "data", "array": sub["error"].fillna(0).tolist(), "visible": True,
                                 "thickness": 0.8, "width": 0}
            if is_ul:
                # a short downward stem turns the triangle into an arrow (guard CX-05)
                tr["error_y"] = {"type": "data", "symmetric": False, "array": [0.0] * len(sub),
                                 "arrayminus": [abs(float(v)) * 0.3 for v in sub["value"]], "visible": True,
                                 "thickness": 1.0, "width": 0, "color": color}
            traces.append(tr)
    if len(df):
        lo, hi = float(df["time_mjd"].min()), float(df["time_mjd"].max())
        pad = max(1.0, 0.02 * (hi - lo))
        layout["xaxis"]["range"] = [lo - pad, hi + pad]
        tv, tt = _year_ticks(lo, hi)
        layout["xaxis2"] = {"overlaying": "x", "side": "top", "range": [lo - pad, hi + pad], "tickvals": tv,
                            "ticktext": tt, "showgrid": False, "title": {"text": "year"}}
        traces.append({"type": "scatter", "mode": "markers", "x": [lo, hi], "y": [df["value"].iloc[0]] * 2,
                       "xaxis": "x2", "yaxis": panel_axis[used_panels[0]], "marker": {"opacity": 0},
                       "hoverinfo": "skip", "showlegend": False, "name": "year axis"})
    if truncated:
        layout["title"]["text"] = title + " (plot subsampled; table complete)"
    return {"data": traces, "layout": layout}, {"downsampled": truncated}


# ── the service ────────────────────────────────────────────────────────────
class MmdcService:
    """Stateful MMDC access: lazy SDK client, known-source cache, CSV cache."""

    def __init__(self, *, plotting_service: Any = None, client_factory: Optional[Callable[[], Any]] = None,
                 known_sources_loader: Optional[Callable[[], Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]]] = None):
        self._plotting_service = plotting_service
        self._client_factory = client_factory
        self._known_sources_loader = known_sources_loader
        self._client_obj = None
        self._lock = threading.Lock()
        self._known: Optional[Tuple[float, List[Dict[str, Any]], List[Dict[str, Any]]]] = None
        self._frames: Dict[str, Tuple[float, pd.DataFrame]] = {}

    # plumbing ------------------------------------------------------------
    @property
    def plotting_service(self):
        if self._plotting_service is None:
            from services.plotting import PlottingService

            self._plotting_service = PlottingService()
        return self._plotting_service

    def client(self, user_id: Optional[str] = None):
        with self._lock:
            if self._client_obj is None:
                if self._client_factory is not None:
                    self._client_obj = self._client_factory()
                else:
                    MMDC = import_sdk()
                    self._client_obj = MMDC(base_url=MMDC_BASE_URL, timeout=HTTP_TIMEOUT_S, app="quasar",
                                            api_key=os.getenv("MMDC_API_KEY") or None)
            base = self._client_obj
        if user_id and os.getenv("MMDC_API_KEY") and hasattr(base, "for_user"):
            # Opaque per-user id (never an email): MMDC records it only with a key.
            return base.for_user("q" + hashlib.sha256(str(user_id).encode()).hexdigest()[:24])
        return base

    def call(self, label: str, fn: Callable[[], Any], seconds: float) -> Any:
        """One MMDC round trip under the host breaker and the tool deadline."""
        from services.host_breaker import HostBreaker
        from services.tool_budgets import bounded_timeout, call_bounded

        url = f"{MMDC_BASE_URL}/api/"
        HostBreaker.check(url)
        wall = bounded_timeout(seconds, minimum=2.0, label=f"MMDC {label}")
        try:
            out = call_bounded(fn, wall, label=f"MMDC {label}", thread_name="quasar-mmdc")
        except Exception as exc:
            status = getattr(exc, "status_code", None)
            HostBreaker.record_failure(url, exc, status=status if isinstance(status, int) else None)
            raise
        HostBreaker.record_success(url)
        return out

    # known sources -------------------------------------------------------
    def _disk_cache_path(self) -> str:
        base = os.getenv("MMDC_CACHE_DIR") or os.path.join(os.path.dirname(__file__), "..", "ui-pro", "cache", "mmdc")
        return os.path.join(base, "known_sources.json")

    def _read_disk_cache(self) -> Optional[Tuple[float, list, list]]:
        try:
            import json

            with open(self._disk_cache_path(), encoding="utf-8") as fh:
                d = json.load(fh)
            return float(d["fetched_at"]), list(d["known"]), list(d["gamma"])
        except Exception:  # noqa: BLE001 - no cache yet / unreadable
            return None

    def _write_disk_cache(self, known: list, gamma: list) -> None:
        try:
            import json

            path = self._disk_cache_path()
            os.makedirs(os.path.dirname(path), exist_ok=True)
            tmp = path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump({"fetched_at": time.time(), "known": known, "gamma": gamma}, fh)
            os.replace(tmp, path)
        except Exception:  # noqa: BLE001 - the cache is an optimization
            pass

    def known_sources(self) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """(known_sources, gamma_ray_sources) from mmdc.am, cached in memory and on
        disk with a TTL (the lists are 1.3 MB and change rarely). When mmdc.am is
        slow, a stale disk copy is used rather than failing the whole tool."""
        with self._lock:
            if self._known and time.time() - self._known[0] < KNOWN_SOURCES_TTL_S:
                return self._known[1], self._known[2]
        disk = None if self._known_sources_loader is not None else self._read_disk_cache()
        if disk and time.time() - disk[0] < KNOWN_SOURCES_TTL_S:
            with self._lock:
                self._known = disk
            return disk[1], disk[2]
        if self._known_sources_loader is not None:
            known, gamma = self._known_sources_loader()
        else:
            import httpx

            def _load():
                with httpx.Client(base_url=MMDC_BASE_URL, timeout=KNOWN_SOURCES_TIMEOUT_S,
                                  headers={"X-MMDC-Client": "quasar"}) as c:
                    a = c.get("/api/known_sources/")
                    a.raise_for_status()
                    b = c.get("/api/gamma_ray_sources/")
                    b.raise_for_status()
                    return a.json(), b.json()

            try:
                known, gamma = self.call("known-source list", _load, KNOWN_SOURCES_TIMEOUT_S)
                self._write_disk_cache(known, gamma)
            except Exception:
                if disk:  # stale beats failing: membership lists change rarely
                    with self._lock:
                        self._known = (time.time() - KNOWN_SOURCES_TTL_S + 600.0, disk[1], disk[2])
                    return disk[1], disk[2]
                raise
        known = [s for s in known if isinstance(s, dict) and s.get("ra") is not None]
        gamma = [s for s in gamma if isinstance(s, dict) and s.get("ra") is not None]
        with self._lock:
            self._known = (time.time(), known, gamma)
        return known, gamma

    def match_known_source(self, ra: float, dec: float, radius_arcsec: float = MATCH_RADIUS_ARCSEC) -> Optional[Dict[str, Any]]:
        known, gamma = self.known_sources()
        best: Optional[Dict[str, Any]] = None
        cosd = max(1e-6, math.cos(math.radians(dec)))
        box = radius_arcsec / 3600.0 * 1.5
        for listname, lst in (("gamma_ray_sources", gamma), ("known_sources", known)):
            for s in lst:
                try:
                    sra, sdec = float(s["ra"]), float(s["dec"])
                except (TypeError, ValueError, KeyError):
                    continue
                if abs(sdec - dec) > box or abs(((sra - ra + 180) % 360) - 180) * cosd > box:
                    continue
                sep = angular_sep_arcsec(ra, dec, sra, sdec)
                if sep <= radius_arcsec and (best is None or sep < best["separation_arcsec"] - 1e-9):
                    best = {"name": s.get("name"), "ra": sra, "dec": sdec, "separation_arcsec": round(sep, 3),
                            "list": listname}
            if best is not None:
                break  # gamma_ray_sources carries common names; prefer it
        return best

    def name_lookup(self, target_name: str) -> Optional[Dict[str, Any]]:
        key = normalize_name(target_name)
        if not key:
            return None
        known, gamma = self.known_sources()
        for listname, lst in (("gamma_ray_sources", gamma), ("known_sources", known)):
            for s in lst:
                if normalize_name(s.get("name")) == key:
                    return {"name": s.get("name"), "ra": float(s["ra"]), "dec": float(s["dec"]), "list": listname}
        return None

    def resolve_position(self, target_name: Optional[str], ra: Optional[float], dec: Optional[float],
                         resolver: Optional[Callable[[str], Dict[str, Any]]]) -> Dict[str, Any]:
        """Order: explicit ra/dec, then MMDC's own source list by name, then the SIMBAD/NED resolver."""
        if ra is not None and dec is not None:
            ra_f, dec_f = float(ra), float(dec)
            if not (math.isfinite(ra_f) and math.isfinite(dec_f) and 0 <= ra_f < 360 and -90 <= dec_f <= 90):
                raise ValueError("ra/dec must be ICRS degrees with 0 <= ra < 360 and -90 <= dec <= 90")
            return {"ra": ra_f, "dec": dec_f, "label": target_name or f"RA={ra_f:.5f}, Dec={dec_f:.5f}",
                    "method": "explicit ra/dec"}
        if not target_name:
            raise ValueError("Provide target_name or both ra and dec.")
        hit = self.name_lookup(target_name)
        if hit:
            return {"ra": hit["ra"], "dec": hit["dec"], "label": str(target_name),
                    "method": f"MMDC {hit['list']} name match ({hit['name']})"}
        if resolver is None:
            raise ValueError(f"Could not resolve {target_name!r}: not in the MMDC source list and no resolver available.")
        res = resolver(str(target_name)) or {}
        if not res.get("success"):
            raise ValueError(res.get("error") or f"Could not resolve target {target_name!r}")
        return {"ra": float(res["ra_deg"]), "dec": float(res["dec_deg"]), "label": str(target_name),
                "method": f"{res.get('resolver') or 'name resolver'} name resolution"}

    # SED job + data ------------------------------------------------------
    def prepare(self, ra: float, dec: float, label: str, *, job_uuid: Optional[str] = None,
                user_id: Optional[str] = None, poll_seconds: Optional[float] = None) -> Dict[str, Any]:
        """Submit (or resume) MMDC's SED preparation and poll within budget."""
        from services.tool_budgets import remaining_seconds

        client = self.client(user_id)
        t0 = time.monotonic()
        if job_uuid:
            job = self.call("SED job status", lambda: client.sed.get_status(job_uuid), INFO_TIMEOUT_S)
        else:
            db = re.sub(r"[^A-Za-z0-9+._-]", "", str(label).replace(" ", "")) or f"pos_{ra:.4f}_{dec:+.4f}"
            job = self.call("SED prepare", lambda: client.sed.prepare(ra=ra, dec=dec, database_name=db[:64],
                                                                      source_name=str(label)[:120]), INFO_TIMEOUT_S)
        limit = PREPARE_POLL_S if poll_seconds is None else float(poll_seconds)
        delay = 2.0
        while str(job.status) not in ("done", "no_data", "error"):
            left = limit - (time.monotonic() - t0)
            rem = remaining_seconds()
            if rem is not None:
                # leave room for the CSV download and the result assembly
                left = min(left, rem - CSV_TIMEOUT_S - INFO_TIMEOUT_S - 5)
            if left <= delay:
                return {"status": str(job.status), "uuid": job.uuid, "pending": True,
                        "waited_s": round(time.monotonic() - t0, 1)}
            time.sleep(delay)
            delay = min(delay * 1.5, 8.0)
            job = self.call("SED job status", lambda: client.sed.get_status(job.uuid), INFO_TIMEOUT_S)
        return {"status": str(job.status), "uuid": job.uuid, "pending": False, "logs": job.logs,
                "waited_s": round(time.monotonic() - t0, 1)}

    def sed_frame(self, job_uuid: str, *, user_id: Optional[str] = None,
                  window: Optional[Tuple[Optional[float], Optional[float]]] = None) -> Tuple[pd.DataFrame, str]:
        """The SED table for a finished job, cached briefly. Returns (table, scope).

        With no window the FULL CSV is downloaded. With a window, the server's
        own filter is used as a superset: it keeps rows that OVERLAP
        [start - 1 d, end + 1 d], which contains every row our contained and
        overlap rules can keep, so both counts stay exact; the undated
        placeholder rows (MJD 55000) are fetched by a one-day probe when the
        window does not already cover that date. A full Mkn 421 CSV is 7.6 MB
        and took 14 to 50 s live; a one-month window is a few kB.
        """
        lo, hi = window if window else (None, None)
        windowed = lo is not None or hi is not None
        key = f"{job_uuid}|{lo}|{hi}"
        now = time.time()
        with self._lock:
            hit = self._frames.get(key)
            if hit and now - hit[0] < FRAME_CACHE_TTL_S:
                return hit[1]
        client = self.client(user_id)

        def _download(**kw):
            fd, path = tempfile.mkstemp(prefix="mmdc_sed_", suffix=".csv")
            os.close(fd)
            try:
                client.sed.download_csv(job_uuid, path, **kw)
                with open(path, "rb") as fh:
                    return fh.read()
            finally:
                try:
                    os.remove(path)
                except OSError:
                    pass

        if not windowed:
            df = normalize_sed_csv(self.call("SED CSV", _download, CSV_TIMEOUT_S))
            scope = "full MMDC SED table downloaded; window applied client-side"
        else:
            kw = {}
            if lo is not None:
                kw["mjd_start"] = float(lo) - 1.0
            if hi is not None:
                kw["mjd_end"] = float(hi) + 1.0
            df = normalize_sed_csv(self.call("SED CSV (window)", lambda: _download(**kw), CSV_TIMEOUT_S))
            covers = (lo is None or lo - 1.0 <= UNDATED_MJD) and (hi is None or UNDATED_MJD <= hi + 1.0)
            if not covers:
                probe = normalize_sed_csv(self.call(
                    "SED CSV (undated probe)", lambda: _download(mjd_start=UNDATED_MJD, mjd_end=UNDATED_MJD), CSV_TIMEOUT_S))
                df = pd.concat([df, probe[probe["undated"]]], ignore_index=True)
            scope = ("server-side superset: rows overlapping the window padded by 1 day, plus the undated placeholder "
                     "rows; the contained/overlap rule is applied client-side, so both counts are exact")
        out = (df, scope)
        with self._lock:
            self._frames[key] = (now, out)
            for k in sorted(self._frames, key=lambda k: self._frames[k][0])[:-FRAME_CACHE_MAX]:
                self._frames.pop(k, None)
        return out

    def source_info(self, job_uuid: str, *, user_id: Optional[str] = None) -> Dict[str, Any]:
        client = self.client(user_id)
        try:
            info = self.call("source info", lambda: client.sed.get_info(job_uuid), INFO_TIMEOUT_S)
        except Exception as exc:  # info is an enhancement; the data stand without it
            return {"error": f"MMDC source info unavailable: {exc}"}
        d = info.model_dump() if hasattr(info, "model_dump") else dict(info)
        return {k: d.get(k) for k in ("source_name", "ra", "dec", "redshift", "gal_lat", "gal_long")}

    def not_member_result(self, pos: Dict[str, Any], nearest_note: str = "") -> Dict[str, Any]:
        return {
            "success": False,
            "not_mmdc_source": True,
            "fallback_tool": "ned_sed_plot",
            "target": pos.get("label"),
            "resolved_position": {"ra": round(pos["ra"], 6), "dec": round(pos["dec"], 6), "method": pos.get("method")},
            "error": (
                f"{pos.get('label')} is not an MMDC blazar: no source in MMDC's known-source or gamma-ray source lists lies "
                f"within {MATCH_RADIUS_ARCSEC:g} arcsec of RA={pos['ra']:.5f}, Dec={pos['dec']:.5f}.{nearest_note} "
                "Call ned_sed_plot (literature SED from NED) for this target instead (radio_sed for a radio-only SED). "
                "Do not describe any MMDC data for it."),
        }

    def sed(
        self,
        *,
        target_name: Optional[str] = None,
        ra: Optional[float] = None,
        dec: Optional[float] = None,
        start_date: Any = None,
        end_date: Any = None,
        window_mode: str = "contained",
        include_undated: bool = False,
        exclude_catalogs: Optional[List[str]] = None,
        include_ranges: Optional[List[str]] = None,
        color_by: str = "epoch",
        max_points: int = PLOTLY_POINT_CAP,
        job_uuid: Optional[str] = None,
        require_known_source: bool = True,
        resolver: Optional[Callable[[str], Dict[str, Any]]] = None,
        user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Tool body for mmdc_sed. Heavy payloads ride in private keys
        ``_table`` (DataFrame) and ``_figure`` for the caller to turn into cards."""
        color_by = str(color_by or "epoch").strip().lower()
        if color_by not in ("epoch", "catalog"):
            raise ValueError("color_by must be 'epoch' or 'catalog'")
        start_mjd, start_note = parse_window_bound(start_date, is_end=False)
        end_mjd, end_note = parse_window_bound(end_date, is_end=True)
        pos = self.resolve_position(target_name, ra, dec, resolver)
        match = self.match_known_source(pos["ra"], pos["dec"])
        warnings: List[str] = []
        if match is None:
            if require_known_source:
                return self.not_member_result(pos)
            warnings.append(f"{pos['label']} is not in MMDC's blazar source lists; MMDC data at this position are "
                            "shown because require_known_source=false.")
        job = self.prepare(pos["ra"], pos["dec"], (match or {}).get("name") or pos["label"], job_uuid=job_uuid,
                           user_id=user_id)
        base = {"target": pos["label"], "resolved_position": {"ra": round(pos["ra"], 6), "dec": round(pos["dec"], 6),
                                                              "method": pos["method"]},
                "mmdc_source_match": match, "mmdc_job_uuid": job["uuid"], "attribution": attribution()}
        if job["pending"]:
            return {**base, "success": False, "pending": True, "job_status": job["status"],
                    "retry_after_s": 30,
                    "error": (f"MMDC is still assembling the SED for {pos['label']} (status '{job['status']}' after "
                              f"{job['waited_s']} s; a cold source takes about 30 s). Call mmdc_sed again with the same "
                              f"arguments plus job_uuid='{job['uuid']}' to collect it. This is not a failure.")}
        if job["status"] == "error":
            tail = str(job.get("logs") or "")[-300:]
            return {**base, "success": False, "error": f"MMDC SED preparation failed (status 'error'). {tail}".strip()}
        if job["status"] == "no_data":
            return {**base, "success": False, "no_data": True, "fallback_tool": "ned_sed_plot",
                    "error": f"MMDC has no SED data at this position (status 'no_data'). Try ned_sed_plot for literature photometry."}
        full, scope = self.sed_frame(job["uuid"], user_id=user_id, window=(start_mjd, end_mjd))
        info = self.source_info(job["uuid"], user_id=user_id)
        if info.get("error"):
            warnings.append(f"{info['error']}; the redshift is unknown for this call (not the same as MMDC having none).")
        kept, window = apply_window(full, start_mjd, end_mjd, window_mode, include_undated,
                                    end_exclusive="expanded" in end_note)
        window["fetch_scope"] = scope
        if start_mjd is not None or end_mjd is not None:
            # Only the window's superset was fetched: the all-epochs totals are not known here.
            window["dated_rows_available"] = None
            window["dated_rows_outside_window"] = None
        if start_note:
            window["start_input"] = {"value": start_date, "interpreted": start_note}
        if end_note:
            window["end_input"] = {"value": end_date, "interpreted": end_note}
            if "expanded" in end_note:
                # Show the inclusive last day; window_end_mjd stays the exact (exclusive) bound.
                window["window_end_date"] = mjd_to_date(end_mjd - 1e-6)
                window["window_end_note"] = (f"window ends at MJD {end_mjd:g} = {mjd_to_date(end_mjd)} 00:00 UTC, "
                                             f"so {window['window_end_date']} is the last day included")
        kept, filt = filter_catalogs_and_ranges(kept, exclude_catalogs, include_ranges)
        summary = summarize_sed(kept)
        redshift = info.get("redshift")
        window_label = ("all epochs" if window["rule_applied"] == "none" else
                        f"{window['window_start_date'] or '...'} to {window['window_end_date'] or '...'}, {window['rule_applied']}")
        title = f"{info.get('source_name') or pos['label']}: MMDC SED ({window_label})"
        result: Dict[str, Any] = {
            **base,
            "success": True,
            "source_name_mmdc": info.get("source_name"),
            "redshift": redshift,
            "redshift_source": ("MMDC source info (get_info)" if redshift is not None else
                                ("MMDC source-info lookup FAILED (" + str(info["error"])[:120] + "); redshift unknown"
                                 if info.get("error") else "not provided by MMDC")),
            "window": window,
            "filters": filt,
            "summary": summary,
            "rows_fetched": int(len(full)),
            "warnings": warnings,
            "answer_requirements": [
                f"State the window rule that was applied ({window['rule_applied']}) and the window bounds in MJD and dates.",
                "Report straddling and undated rows as the tool counted them; never describe data outside the window as in-window.",
                "Upper limits are marked as such; never quote an upper limit as a detection.",
                f"Include the MMDC acknowledgment and cite {MMDC_CITATION} ({MMDC_BIBCODE}).",
            ],
        }
        if window.get("straddling_note"):
            warnings.append(window["straddling_note"])
        if window.get("undated_note"):
            warnings.append(window["undated_note"])
        if kept.empty:
            result["note"] = ("No MMDC rows satisfy the window and filters. Say so plainly; do not substitute data "
                              "from other epochs.")
            return result
        plot_df, ds = downsample_sed(kept, max_points)
        result["plot"] = ds
        fig = sed_plotly_spec(plot_df, title, color_by=color_by)
        png = sed_png(plot_df, title, self.plotting_service, color_by=color_by)
        result["_table"] = (kept, f"MMDC SED {pos['label']} [{window_label}]")
        result["_figure"] = {**png, "plotly_spec": fig, "caption": title}
        return result

    # light curves ----------------------------------------------------------
    def lightcurve(
        self,
        *,
        target_name: Optional[str] = None,
        ra: Optional[float] = None,
        dec: Optional[float] = None,
        start_date: Any = None,
        end_date: Any = None,
        catalogs: Optional[List[str]] = None,
        bands: Optional[List[str]] = None,
        radius_arcsec: float = MATCH_RADIUS_ARCSEC,
        require_known_source: bool = True,
        resolver: Optional[Callable[[str], Dict[str, Any]]] = None,
        user_id: Optional[str] = None,
        make_plot: bool = True,
    ) -> Dict[str, Any]:
        """Tool body for mmdc_lightcurve (see the module docstring for the table contract)."""
        start_mjd, start_note = parse_window_bound(start_date, is_end=False)
        end_mjd, end_note = parse_window_bound(end_date, is_end=True)
        if start_mjd is not None and end_mjd is not None and end_mjd <= start_mjd:
            raise ValueError("end_date must be after start_date")
        radius = float(radius_arcsec or MATCH_RADIUS_ARCSEC)
        if not 0 < radius <= 60:
            raise ValueError("radius_arcsec must be in (0, 60]")
        pos = self.resolve_position(target_name, ra, dec, resolver)
        match = self.match_known_source(pos["ra"], pos["dec"])
        if match is None and require_known_source:
            out = self.not_member_result(pos)
            out["fallback_tool"] = "ztf_light_curve or search_space_lightcurves"
            out["error"] = out["error"].replace(
                "Call ned_sed_plot (literature SED from NED) for this target instead (radio_sed for a radio-only SED).",
                "Use search_ztf_alerts + ztf_light_curve, search_space_lightcurves, or datalab_star_lightcurve instead.")
            return out
        cats = [c.strip() for c in (catalogs or []) if str(c).strip()]
        bad = [c for c in cats if c not in LC_CATALOGS]
        if bad:
            raise ValueError(f"unknown MMDC light-curve catalogs {bad}; choose from {list(LC_CATALOGS)}")
        client = self.client(user_id)
        rows: List[Any] = []
        for cat in (cats or [None]):
            kw = {"radius_arcsec": radius, "is_lightcurve": True}
            if cat:
                kw["catalog"] = cat
            if start_mjd is not None:
                kw["mjd_min"] = start_mjd
            if end_mjd is not None:
                kw["mjd_max"] = end_mjd
            rows.extend(self.call(f"light curve {cat or 'all'}",
                                  lambda kw=kw: client.observations.cone_search(pos["ra"], pos["dec"], **kw), LC_TIMEOUT_S))
        df = normalize_lightcurve_rows(rows)
        # The server bounds mjd_mid; re-apply so the contract holds even if it changes.
        if start_mjd is not None:
            df = df[df["time_mjd"] >= start_mjd]
        if end_mjd is not None:
            # an expanded (year/month/day) end is exclusive: guard CX-02
            df = df[(df["time_mjd"] < end_mjd) if "expanded" in end_note else (df["time_mjd"] <= end_mjd)]
        if bands:
            want = {str(b).strip().lower() for b in bands}
            df = df[df["band"].str.lower().isin(want) | df["band"].str.split(":").str[-1].str.lower().isin(want)
                    | df["catalog"].str.lower().isin(want)]
        df = df.reset_index(drop=True)
        label = pos["label"]
        window = {"start_mjd": start_mjd, "end_mjd": end_mjd,
                  "start_date": mjd_to_date(start_mjd) if start_mjd is not None else None,
                  "end_date": mjd_to_date(end_mjd) if end_mjd is not None else None,
                  "rule": "a measurement is kept when its mid-time (mjd_mid) lies inside the window"}
        if start_note:
            window["start_input"] = start_note
        if end_note:
            window["end_input"] = end_note
        result: Dict[str, Any] = {
            "success": True,
            "target": label,
            "resolved_position": {"ra": round(pos["ra"], 6), "dec": round(pos["dec"], 6), "method": pos["method"]},
            "mmdc_source_match": match,
            "cone_radius_arcsec": radius,
            "window": window,
            "rows": int(len(df)),
            "bands": summarize_lightcurve(df),
            "units_note": "MMDC does not state light-curve flux units; MMDCGR values match Fermi-LAT photon flux "
                          "(ph cm^-2 s^-1) and the others energy flux (erg cm^-2 s^-1). Treat units as inferred.",
            "attribution": attribution(),
            "next_step": "Pass result_id to variability_analysis for Fvar, flares, lags and index-flux trends.",
        }
        if df.empty:
            result["note"] = "No MMDC light-curve rows in this cone and window."
            return result
        title = f"{label}: MMDC multiwavelength light curve"
        if make_plot:
            spec, ds = lightcurve_plotly_spec(df, title)
            result["plot"] = ds
            png = self._lightcurve_png(df, title)
            result["_figure"] = {**png, "plotly_spec": spec, "caption": title}
        result["_table"] = (df, f"MMDC light curve {label}")
        return result

    def _lightcurve_png(self, df: pd.DataFrame, title: str) -> Dict[str, Any]:
        import uuid

        plt = self.plotting_service._apply_style(dark=False)
        panels = sorted({_panel_of(b) for b in df["band"].unique()})
        fig, axes = plt.subplots(len(panels), 1, figsize=(9, 2.3 * len(panels) + 0.8), sharex=True, squeeze=False)
        for k, p in enumerate(panels):
            ax = axes[k][0]
            for i, (band, g) in enumerate(df[df["band"].map(_panel_of) == p].groupby("band")):
                d, u = g[~g["is_upper_limit"]], g[g["is_upper_limit"]]
                ax.errorbar(d["time_mjd"], d["value"], yerr=d["error"].fillna(0), fmt="o", ms=2.5, lw=0.6,
                            color=_CATEGORICAL[i % len(_CATEGORICAL)], label=band)
                if len(u):
                    ul_arrows(ax, u["time_mjd"], u["value"], color=_CATEGORICAL[i % len(_CATEGORICAL)], frac=0.7)
            ax.set_ylabel(_LC_PANELS[p][0], fontsize=8)
            ax.legend(fontsize=6, loc="upper right", ncol=3)
        axes[-1][0].set_xlabel("MJD")
        axes[0][0].set_title(title, fontsize=10)
        fig.tight_layout()
        render = self.plotting_service._save_and_encode(fig, f"mmdc_lc_{uuid.uuid4().hex[:10]}")
        return {"image_base64": render.get("base64_png"), "path": render.get("web_url"),
                "png_path": render.get("png_path"), "pdf_path": render.get("pdf_path")}


_DEFAULT: Optional[MmdcService] = None
_DEFAULT_LOCK = threading.Lock()


def default_mmdc_service() -> MmdcService:
    global _DEFAULT
    with _DEFAULT_LOCK:
        if _DEFAULT is None:
            _DEFAULT = MmdcService()
        return _DEFAULT
