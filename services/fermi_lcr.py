"""Fermi LAT Light Curve Repository (LCR) light curves: gamma-ray coverage that
does not depend on MMDC (Phase 5 of the MMDC campaign, 2026-09-26).

Endpoint verified live 2026-09-26 (tmp/mmdc-integration-2026-09-26/PHASE5-probe.md):
``queryDB.php?typeOfRequest=lightCurveData&source_name=4FGL ...&cadence=...&flux_type=...&index_type=...&ts_min=...``
returns JSON (served as text/html) with series of ``[MET, value]`` pairs:
``flux``, ``flux_upper_limits``, ``flux_error`` ``[MET, lo, hi]``, ``photon_index``,
``photon_index_interval`` ``[MET, a, b]``, ``ts``. Cadence ``daily`` returns ~3-day
bins (2113 bins over 17 yr for Mkn 421), plus ``weekly`` and ``monthly``;
``3-day`` / ``3day`` return nothing. The monitored-source list is
``typeOfRequest=SourceList&catalog=4FGL`` (2.6 MB, ~24 s; cached on disk).

Output follows the light-curve contract of services/mmdc_service.py (LC_COLUMNS),
so services/variability.py analyses it directly.
"""

from __future__ import annotations

import json
import math
import os
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

LCR_BASE = "https://fermi.gsfc.nasa.gov/ssc/data/access/lat/LightCurveRepository/queryDB.php"
LCR_HOST = "fermi.gsfc.nasa.gov"
MET_EPOCH_MJD = 51910.0  # Fermi MET 0 = 2001-01-01 00:00:00 UTC (leap seconds ignored: < 1e-4 d)
CADENCES = {"daily": "daily (the repository's shortest cadence; bins are ~3 days)", "weekly": "weekly", "monthly": "monthly"}
FLUX_UNITS = {"photon": "ph cm^-2 s^-1 (0.1-100 GeV)", "energy": "MeV cm^-2 s^-1 (0.1-100 GeV)"}
MATCH_RADIUS_DEG = 0.2
ATTRIBUTION = {
    "source": "Fermi LAT Light Curve Repository (NASA FSSC)",
    "citation": "Abdollahi et al. 2023, ApJS 265, 31 (2023ApJS..265...31A)",
    "acknowledgment": "This work used data from the Fermi LAT Light Curve Repository, provided by the Fermi Science Support Center.",
}


def _env_f(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    try:
        return float(raw) if raw else float(default)
    except ValueError:
        return float(default)


SOURCE_LIST_TIMEOUT_S = _env_f("FERMI_LCR_SOURCE_LIST_TIMEOUT_SECONDS", 45.0)
LC_TIMEOUT_S = _env_f("FERMI_LCR_LC_TIMEOUT_SECONDS", 30.0)
SOURCE_LIST_TTL_S = 7 * 86400.0


def met_to_mjd(met: float) -> float:
    return MET_EPOCH_MJD + float(met) / 86400.0


def _json_body(text: str) -> Any:
    i, j = text.find("["), text.find("{")
    starts = [k for k in (i, j) if k >= 0]
    if not starts:
        raise ValueError("Fermi LCR returned no JSON")
    k = min(starts)
    end = text.rfind("]" if text[k] == "[" else "}")
    return json.loads(text[k:end + 1])


def normalize_lcr(d: Dict[str, Any], band: str = "FermiLAT-LCR", unit: str = FLUX_UNITS["photon"]) -> Tuple[pd.DataFrame, int]:
    """LCR JSON -> (LC_COLUMNS rows, number of detections dropped for having no positive error).

    The repository publishes some unphysical bins with a zero-width error interval
    (live 2026-09-26, Mkn 421 weekly: flux 1.04e-3 with lo = hi, against a median
    of 1.6e-7); they are dropped and counted, never analysed."""
    from services.mmdc_service import LC_COLUMNS, mjd_to_date

    err = {int(r[0]): (float(r[1]), float(r[2])) for r in d.get("flux_error") or [] if len(r) >= 3}
    idx = {int(r[0]): float(r[1]) for r in d.get("photon_index") or [] if len(r) >= 2}
    idxi = {int(r[0]): (float(r[1]), float(r[2])) for r in d.get("photon_index_interval") or [] if len(r) >= 3}
    recs = []
    dropped = 0
    for met, val in (r for r in d.get("flux") or [] if len(r) >= 2):
        met = int(met)
        lo, hi = err.get(met, (math.nan, math.nan))
        e = (hi - lo) / 2.0 if math.isfinite(lo) and math.isfinite(hi) else math.nan
        if not (math.isfinite(e) and e > 0):
            dropped += 1
            continue
        ia, ib = idxi.get(met, (math.nan, math.nan))
        ie = abs(ia - ib) / 2.0 if math.isfinite(ia) and math.isfinite(ib) else math.nan
        recs.append({"time_mjd": met_to_mjd(met), "value": float(val), "error": e, "band": band, "catalog": "FermiLCR",
                     "is_upper_limit": False, "spectral_index": idx.get(met, math.nan),
                     "spectral_index_err": ie if ie > 0 else math.nan, "unit": unit, "obsid": ""})
    for met, val in (r for r in d.get("flux_upper_limits") or [] if len(r) >= 2):
        recs.append({"time_mjd": met_to_mjd(met), "value": float(val), "error": math.nan, "band": band,
                     "catalog": "FermiLCR", "is_upper_limit": True, "spectral_index": math.nan,
                     "spectral_index_err": math.nan, "unit": unit, "obsid": ""})
    df = pd.DataFrame(recs, columns=[c for c in LC_COLUMNS if c != "date"])
    if len(df):
        df = df.sort_values("time_mjd").reset_index(drop=True)
    df.insert(1, "date", df["time_mjd"].map(mjd_to_date) if len(df) else pd.Series(dtype=str))
    return df[LC_COLUMNS], dropped


class FermiLcrService:
    def __init__(self, *, fetch: Optional[Callable[[Dict[str, Any], float], str]] = None, plotting_service: Any = None):
        self._fetch_fn = fetch
        self._plotting = plotting_service
        self._lock = threading.Lock()
        self._sources: Optional[Tuple[float, List[Dict[str, Any]]]] = None

    @property
    def plotting(self):
        if self._plotting is None:
            from services.plotting import PlottingService

            self._plotting = PlottingService()
        return self._plotting

    def _get(self, params: Dict[str, Any], seconds: float) -> str:
        if self._fetch_fn is not None:
            return self._fetch_fn(params, seconds)

        from services.host_breaker import guarded_request
        from services.tool_budgets import bounded_timeout

        r = guarded_request("GET", LCR_BASE, params=params, timeout=bounded_timeout(seconds, label="Fermi LCR"))
        r.raise_for_status()
        return r.text

    def _cache_path(self) -> str:
        base = os.getenv("MMDC_CACHE_DIR") or os.path.join(os.path.dirname(__file__), "..", "ui-pro", "cache", "mmdc")
        return os.path.join(base, "fermi_lcr_sources.json")

    def sources(self) -> List[Dict[str, Any]]:
        with self._lock:
            if self._sources and time.time() - self._sources[0] < SOURCE_LIST_TTL_S:
                return self._sources[1]
        path = self._cache_path()
        stale = None
        try:
            with open(path, encoding="utf-8") as fh:
                d = json.load(fh)
            if time.time() - float(d["fetched_at"]) < SOURCE_LIST_TTL_S:
                with self._lock:
                    self._sources = (float(d["fetched_at"]), d["sources"])
                return d["sources"]
            stale = d["sources"]
        except Exception:  # noqa: BLE001 - no cache yet
            pass
        try:
            rows = _json_body(self._get({"typeOfRequest": "SourceList", "catalog": "4FGL", "magicWord": ""},
                                        SOURCE_LIST_TIMEOUT_S))
        except Exception:
            if stale:
                # memoize the stale copy for ~10 min so a slow repository is not re-hit on
                # every call (review CX-B13)
                with self._lock:
                    self._sources = (time.time() - SOURCE_LIST_TTL_S + 600.0, stale)
                return stale
            raise
        keep = [{"name": r.get("Source_Name"), "assoc": r.get("ASSOC1") or "", "ra": float(r["RAJ2000"]),
                 "dec": float(r["DEJ2000"]), "class": r.get("CLASS1") or ""} for r in rows
                if r.get("RAJ2000") not in (None, "") and r.get("DEJ2000") not in (None, "")]
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path + ".tmp", "w", encoding="utf-8") as fh:
                json.dump({"fetched_at": time.time(), "sources": keep}, fh)
            os.replace(path + ".tmp", path)
        except Exception:  # noqa: BLE001
            pass
        with self._lock:
            self._sources = (time.time(), keep)
        return keep

    def match_name(self, name: str) -> Optional[Dict[str, Any]]:
        from services.mmdc_service import normalize_name

        key = normalize_name(name)
        if not key:
            return None
        for s in self.sources():
            if key in (normalize_name(s["assoc"]), normalize_name(s["name"]),
                       normalize_name(str(s["name"]).replace("4FGL", ""))):
                return {**s, "separation_deg": None, "matched_by": "4FGL name or association"}
        return None

    def match(self, ra: float, dec: float) -> Optional[Dict[str, Any]]:
        from services.mmdc_service import angular_sep_arcsec

        srcs = self.sources()
        best = None
        for s in srcs:
            if abs(s["dec"] - dec) > MATCH_RADIUS_DEG * 1.5:
                continue
            sep = angular_sep_arcsec(ra, dec, s["ra"], s["dec"]) / 3600.0
            if sep <= MATCH_RADIUS_DEG and (best is None or sep < best["separation_deg"]):
                best = {**s, "separation_deg": round(sep, 4), "matched_by": "position"}
        return best

    def lightcurve(self, *, target_name: Optional[str] = None, ra: Optional[float] = None, dec: Optional[float] = None,
                   source_name: Optional[str] = None, cadence: str = "weekly", flux_type: str = "photon",
                   index_type: str = "fixed", ts_min: float = 4.0, start_date: Any = None, end_date: Any = None,
                   resolver: Optional[Callable[[str], Dict[str, Any]]] = None) -> Dict[str, Any]:
        from services.mmdc_service import mjd_to_date, parse_window_bound

        cadence = str(cadence or "weekly").strip().lower()
        if cadence not in CADENCES:
            raise ValueError(f"cadence must be one of {list(CADENCES)}")
        flux_type = str(flux_type or "photon").strip().lower()
        if flux_type not in FLUX_UNITS:
            raise ValueError("flux_type must be 'photon' or 'energy'")
        index_type = str(index_type or "fixed").strip().lower()
        if index_type not in ("fixed", "free"):
            raise ValueError("index_type must be 'fixed' or 'free'")
        ts_min = float(ts_min)
        if not 0 < ts_min <= 100:
            raise ValueError("ts_min must lie in (0, 100]")
        s_mjd, _ = parse_window_bound(start_date, is_end=False)
        e_mjd, e_note = parse_window_bound(end_date, is_end=True)
        end_exclusive = e_mjd is not None and "expanded" in e_note  # same rule as mmdc_service (CX-02 / review B05)
        src: Optional[Dict[str, Any]] = None
        if source_name:
            src = {"name": str(source_name).strip(), "assoc": "", "matched_by": "4FGL name given", "separation_deg": None}
        else:
            if ra is None or dec is None:
                if not target_name:
                    raise ValueError("Provide source_name (4FGL name), target_name, or ra and dec.")
                src = self.match_name(str(target_name))
                if src is None:
                    res = resolver(str(target_name)) if resolver else {}
                    if not (res or {}).get("success"):
                        raise ValueError((res or {}).get("error") or f"Could not resolve {target_name!r}")
                    ra, dec = float(res["ra_deg"]), float(res["dec_deg"])
            if src is None:
                src = self.match(float(ra), float(dec))
            if src is None:
                return {"success": False, "not_in_lcr": True,
                        "error": (f"No 4FGL source monitored by the Fermi LAT Light Curve Repository lies within "
                                  f"{MATCH_RADIUS_DEG} deg of {target_name or 'the position'}. The repository covers "
                                  "variable 4FGL sources only; do not claim a Fermi light curve for it.")}
        params = {"typeOfRequest": "lightCurveData", "source_name": src["name"], "cadence": cadence,
                  "flux_type": flux_type, "index_type": index_type, "ts_min": ts_min}
        body = _json_body(self._get(params, LC_TIMEOUT_S))
        if not isinstance(body, dict):
            raise ValueError(f"the Fermi LAT Light Curve Repository returned an unexpected body for {src['name']} "
                             f"({type(body).__name__}, not a light-curve object)")
        if not (body.get("flux") or body.get("flux_upper_limits")):
            return {"success": False, "not_in_lcr": True, "lcr_source": {k: src.get(k) for k in ("name", "assoc", "class")},
                    "error": (f"{src['name']} ({src.get('assoc') or 'no association'}) is a 4FGL source, but the Fermi LAT "
                              "Light Curve Repository has no light curve for it: it monitors only variable 4FGL sources. "
                              "Do not claim a Fermi light curve for it.")}
        df, n_bad = normalize_lcr(body, unit=FLUX_UNITS[flux_type])
        if s_mjd is not None:
            df = df[df["time_mjd"] >= s_mjd]
        if e_mjd is not None:
            df = df[(df["time_mjd"] < e_mjd) if end_exclusive else (df["time_mjd"] <= e_mjd)]
        df = df.reset_index(drop=True)
        label = target_name or src.get("assoc") or src["name"]
        det = df[~df["is_upper_limit"]]
        out: Dict[str, Any] = {
            "success": True, "target": label, "lcr_source": {k: src.get(k) for k in ("name", "assoc", "class", "matched_by", "separation_deg")},
            "request": {k: params[k] for k in ("cadence", "flux_type", "index_type", "ts_min")},
            "cadence_note": CADENCES[cadence], "flux_unit": FLUX_UNITS[flux_type],
            "rows": int(len(df)), "detections": int(len(det)), "upper_limits": int(df["is_upper_limit"].sum()),
            "n_bins_dropped_without_error": int(n_bad),
            "upper_limit_rule": f"bins with TS < {ts_min:g} are reported by the repository as upper limits",
            "window": {"start_mjd": s_mjd, "end_mjd": e_mjd, "end_exclusive": bool(end_exclusive),
                       "rule": "a bin is kept when its start time lies in the window"},
            "attribution": ATTRIBUTION,
            "next_step": "Pass result_id to variability_analysis (Fvar, Bayesian-block flares, lags against other bands).",
        }
        if len(df):
            out["MJD_range"] = [round(float(df["time_mjd"].min()), 3), round(float(df["time_mjd"].max()), 3)]
            out["date_range"] = [mjd_to_date(df["time_mjd"].min()), mjd_to_date(df["time_mjd"].max())]
            if len(det) > 1:
                out["median_bin_spacing_d"] = round(float(np.median(np.diff(det["time_mjd"].to_numpy()))), 3)
                out["flux_range"] = [float(f"{det['value'].min():.4g}"), float(f"{det['value'].max():.4g}")]
            if index_type == "free" and det["spectral_index"].notna().any():
                out["photon_index_range"] = [round(float(det["spectral_index"].min()), 3), round(float(det["spectral_index"].max()), 3)]
            title = f"{label}: Fermi-LAT LCR light curve ({cadence}, {flux_type} flux, TS >= {ts_min:g})"
            out["_figure"] = {**self._png(df, title), "plotly_spec": self._spec(df, title, FLUX_UNITS[flux_type]), "caption": title}
            out["_table"] = (df, f"Fermi LCR {src['name']} [{cadence}]")
        else:
            windowed = s_mjd is not None or e_mjd is not None
            out["note"] = (f"No usable bins{' in this window' if windowed else ''}"
                           + (f": {n_bad} bin(s) were dropped for a zero-width error interval" if n_bad else "")
                           + ".")  # review CX-B15
        return out

    @staticmethod
    def _spec(df: pd.DataFrame, title: str, unit: str) -> Dict[str, Any]:
        from services.mmdc_service import downsample_trace

        det, ul = df[~df["is_upper_limit"]], df[df["is_upper_limit"]]
        det = downsample_trace(det, 4500)
        traces = [{"type": "scatter", "mode": "markers", "name": "detections", "x": det["time_mjd"].round(4).tolist(),
                   "y": det["value"].tolist(), "error_y": {"type": "data", "array": det["error"].fillna(0).tolist(),
                                                           "visible": True, "thickness": 0.8, "width": 0},
                   "marker": {"size": 4, "color": "#e15759"}, "customdata": det["date"].tolist(),
                   "hovertemplate": "MJD %{x:.2f} (%{customdata})<br>flux %{y:.3e}<extra></extra>"}]
        if len(ul):
            traces.append({"type": "scatter", "mode": "markers", "name": "upper limits (TS below threshold)",
                           "x": ul["time_mjd"].round(4).tolist(), "y": ul["value"].tolist(),
                           "error_y": {"type": "data", "symmetric": False, "array": [0.0] * len(ul),
                                       "arrayminus": [abs(float(v)) * 0.3 for v in ul["value"]], "visible": True,
                                       "thickness": 1.0, "width": 0, "color": "#999"},
                           "marker": {"symbol": "triangle-down", "size": 6, "color": "#999", "opacity": 0.6}})
        return {"data": traces, "layout": {"title": {"text": title}, "height": 420, "hovermode": "closest",
                                           "xaxis": {"title": {"text": "MJD"}},
                                           "yaxis": {"title": {"text": f"flux [{unit}]"}, "exponentformat": "power"},
                                           "legend": {"orientation": "h"}}}

    def _png(self, df: pd.DataFrame, title: str) -> Dict[str, Any]:
        import uuid

        plt = self.plotting._apply_style(dark=False)
        fig, ax = plt.subplots(figsize=(9, 3.8))
        det, ul = df[~df["is_upper_limit"]], df[df["is_upper_limit"]]
        ax.errorbar(det["time_mjd"], det["value"], yerr=det["error"].fillna(0), fmt="o", ms=2, lw=0.5, color="#e15759")
        if len(ul):
            from services.mmdc_service import ul_arrows

            ul_arrows(ax, ul["time_mjd"], ul["value"], color="0.6", frac=0.7)
        ax.set_xlabel("MJD")
        ax.set_ylabel("flux")
        ax.set_title(title, fontsize=9)
        fig.tight_layout()
        r = self.plotting._save_and_encode(fig, f"fermi_lcr_{uuid.uuid4().hex[:10]}")
        return {"image_base64": r.get("base64_png"), "path": r.get("web_url")}
