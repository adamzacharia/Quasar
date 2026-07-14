"""Light-curve search, plotting, and Lomb-Scargle period tools."""

from __future__ import annotations

import math
import os
import re
import uuid
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from services.plotting import PlottingService


LIGHTCURVE_DEFAULT_BASE_URL = "MAST via lightkurve"
LIGHTCURVE_DEFAULT_TIMEOUT = 120.0
FEW_POINTS_WARNING = "few points; period unreliable"


class LightCurveSuite:
    """TESS/Kepler light curves and generic Lomb-Scargle period searches."""

    def __init__(
        self,
        *,
        base_url: Optional[str] = None,
        timeout: Optional[float] = None,
        plotting_service: Optional[PlottingService] = None,
        alerce_client: Optional[Any] = None,
        search_fn: Optional[Any] = None,
        cache_dir: Optional[Any] = None,
    ):
        self.base_url = str(base_url or os.getenv("LIGHTCURVE_BASE_URL") or LIGHTCURVE_DEFAULT_BASE_URL)
        self.timeout = float(timeout if timeout is not None else _env_float("LIGHTCURVE_TIMEOUT", LIGHTCURVE_DEFAULT_TIMEOUT))
        self.plotting_service = plotting_service or PlottingService()
        self.alerce_client = alerce_client
        self.search_fn = search_fn
        root = Path(__file__).resolve().parents[1]
        self.cache_dir = Path(cache_dir) if cache_dir is not None else root / "data" / "lightkurve_cache"

    def search_space_lightcurves(self, target: Any, mission: Optional[Any] = None, max_rows: Any = 20) -> Dict[str, Any]:
        try:
            target_s = _require_text(target, "target")
            mission_s, warnings = _normalize_mission(mission)
            row_limit, limit_warnings = _normalize_positive_int(max_rows, default=20, cap=100, label="max_rows")
            warnings.extend(limit_warnings)
            result = self._run_space_search(target_s, mission_s)
            table = getattr(result, "table", None)
            total = _table_len(table, fallback=_result_len(result))
            rows = [_normalize_search_row(_table_row(table, idx), idx) for idx in range(min(total, row_limit))]
            if total == 0:
                warnings.append("No TESS/Kepler/K2 light curves found; try name variants or a precise catalog identifier.")
            elif total > row_limit:
                warnings.append(f"Returned first {row_limit} of {total} available light curves.")
            return {
                "success": True,
                "rows": rows,
                "count": len(rows),
                "total_available": int(total),
                "warnings": warnings,
                "provenance": {
                    "service": "MAST light curves via lightkurve",
                    "base_url": self.base_url,
                    "target": target_s,
                    "mission": mission_s,
                    "max_rows": row_limit,
                    "timeout_s": self.timeout,
                },
            }
        except Exception as exc:
            return {"success": False, "error": str(exc)}

    def fetch_lightcurve_arrays(self, target: Any, mission: Optional[Any] = None, index: Any = 0):
        target_s = _require_text(target, "target")
        mission_s, warnings = _normalize_mission(mission)
        index_i = _nonnegative_int(index, "index")
        result = self._run_space_search(target_s, mission_s)
        table = getattr(result, "table", None)
        total = _result_len(result)
        if index_i >= total:
            raise IndexError(f"index {index_i} is out of range for {total} available light curve(s).")
        lc = result[index_i].download()
        if hasattr(lc, "remove_nans"):
            lc = lc.remove_nans()
        if hasattr(lc, "normalize"):
            lc = lc.normalize()
        time, value = _finite_arrays(_array_values(getattr(lc, "time", None)), _array_values(getattr(lc, "flux", None)))
        if len(time) == 0:
            raise ValueError("No finite flux points found after removing NaN/inf values.")
        meta = _lightcurve_meta(_table_row(table, index_i), getattr(lc, "meta", None), target_s, index_i)
        if warnings:
            meta["warnings"] = warnings
        return time, value, "flux", meta

    def plot_space_lightcurve(self, target: Any, mission: Optional[Any] = None, index: Any = 0) -> Dict[str, Any]:
        try:
            target_s = _require_text(target, "target")
            time, flux, value_kind, meta = self.fetch_lightcurve_arrays(target_s, mission=mission, index=index)
            warnings = list(meta.get("warnings") or [])
            plt = self.plotting_service._apply_style(dark=False)
            fig, ax = plt.subplots(figsize=(8.0, 4.2))
            ax.plot(time, flux, ".", ms=2.5, alpha=0.8, color="#0072B2")
            ax.set_xlabel("Time (days)")
            ax.set_ylabel("Normalized flux")
            ax.set_title(f"{target_s} ({_meta_label(meta)})")
            ax.grid(True, alpha=0.25)
            fig.tight_layout()
            render = self.plotting_service._save_and_encode(fig, f"space_lightcurve_{uuid.uuid4().hex[:10]}")
            return _image_result(render, {
                "plotly_spec": _lightcurve_plotly_spec(time, flux, value_kind, f"{target_s} ({_meta_label(meta)})"),
                "n_points": int(len(time)),
                "time_span_days": _time_span(time),
                "value_kind": value_kind,
                "warnings": warnings,
                "provenance": {
                    "service": "MAST light curves via lightkurve",
                    "base_url": self.base_url,
                    "target": target_s,
                    "mission": meta.get("mission"),
                    "index": _nonnegative_int(index, "index"),
                    "timeout_s": self.timeout,
                    "meta": meta,
                },
            })
        except Exception as exc:
            return {"success": False, "error": str(exc)}

    def period_search(
        self,
        source: Any,
        identifier: Any,
        mission: Optional[Any] = None,
        min_period_d: Any = 0.05,
        max_period_d: Any = 30.0,
        fid: Optional[Any] = None,
        index: Any = 0,
    ) -> Dict[str, Any]:
        try:
            source_s = _normalize_source(source)
            identifier_s = _require_text(identifier, "identifier")
            warnings: List[str] = []
            if source_s in {"tess", "kepler"}:
                mission_s, mission_warnings = _mission_for_source(source_s, mission)
                warnings.extend(mission_warnings)
                time, value, value_kind, meta = self.fetch_lightcurve_arrays(identifier_s, mission=mission_s, index=index)
                warnings.extend(meta.get("warnings") or [])
                provenance = {
                    "service": "MAST light curves via lightkurve",
                    "base_url": self.base_url,
                    "source": source_s,
                    "identifier": identifier_s,
                    "mission": mission_s,
                    "index": _nonnegative_int(index, "index"),
                    "timeout_s": self.timeout,
                    "meta": meta,
                }
            else:
                time, value, value_kind, meta, ztf_warnings = self._ztf_arrays(identifier_s, fid=fid)
                warnings.extend(ztf_warnings)
                provenance = {
                    "service": "ALeRCE ZTF lightcurve",
                    "source": source_s,
                    "identifier": identifier_s,
                    "fid": meta.get("fid"),
                    "alerce": meta.get("provenance", {}),
                }

            bounds, bound_warnings = _period_bounds(time, min_period_d, max_period_d)
            warnings.extend(bound_warnings)
            period = _run_lomb_scargle(time, value, value_kind=value_kind, min_period_d=bounds[0], max_period_d=bounds[1], warnings=warnings)
            if not period.get("success"):
                return period
            plot = self._period_plot(
                period["frequency_per_d"],
                period["power"],
                period["time_days"],
                period["value"],
                value_kind,
                identifier_s,
                period["best_period_d"],
                period["best_frequency_per_d"],
            )
            return _image_result(plot, {
                "plotly_spec": _period_plotly_spec(period, value_kind, identifier_s),
                "best_period_d": period["best_period_d"],
                "best_frequency_per_d": period["best_frequency_per_d"],
                "period_unc_d": period["period_unc_d"],
                "fap": period["fap"],
                "top_periods": period["top_periods"],
                "n_points": period["n_points"],
                "time_span_days": period["time_span_days"],
                "value_kind": value_kind,
                "warnings": warnings,
                "provenance": provenance,
            })
        except Exception as exc:
            return {"success": False, "error": str(exc)}

    def _run_space_search(self, target: str, mission: Optional[str]) -> Any:
        search_fn = self.search_fn or self._lightkurve_search_fn()
        return search_fn(target, mission=mission)

    def _lightkurve_search_fn(self) -> Any:
        os.environ.setdefault("LIGHTKURVE_CACHE_DIR", str(self.cache_dir))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        import lightkurve as lk

        return lk.search_lightcurve

    def _get_alerce_client(self) -> Any:
        if self.alerce_client is None:
            from services.alerce_client import AlerceClient

            self.alerce_client = AlerceClient(timeout=self.timeout)
        return self.alerce_client

    def _ztf_arrays(self, oid: str, fid: Optional[Any] = None) -> Tuple[Any, Any, str, Dict[str, Any], List[str]]:
        out = self._get_alerce_client().light_curve(oid)
        if not out.get("success"):
            raise RuntimeError(out.get("error") or f"ALeRCE light curve failed for {oid}.")
        rows = []
        for row in out.get("detections") or []:
            mjd = _float_or_none(row.get("mjd"))
            mag = _float_or_none(row.get("magpsf"))
            row_fid = _int_or_none(row.get("fid"))
            if mjd is not None and mag is not None and row_fid is not None:
                rows.append({"mjd": mjd, "magpsf": mag, "fid": row_fid})
        if not rows:
            raise ValueError(f"No finite ZTF magpsf detections found for {oid}.")
        warnings: List[str] = []
        selected_fid = _int_or_none(fid) if fid is not None else None
        if fid is not None and selected_fid is None:
            raise ValueError("fid must be an integer ZTF filter id.")
        if selected_fid is None:
            counts = Counter(row["fid"] for row in rows)
            selected_fid, n_selected = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0]
            warnings.append(f"Using ZTF fid {selected_fid} with the most finite detections ({n_selected} points).")
        selected = [row for row in rows if row["fid"] == selected_fid]
        if not selected:
            raise ValueError(f"No finite ZTF magpsf detections found for {oid} with fid {selected_fid}.")
        time, value = _finite_arrays([row["mjd"] for row in selected], [row["magpsf"] for row in selected])
        meta = {"oid": oid, "fid": selected_fid, "provenance": out.get("provenance", {})}
        return time, value, "mag", meta, warnings

    def _period_plot(self, frequency: Any, power: Any, time: Any, value: Any, value_kind: str, identifier: str, best_period: float, best_frequency: float) -> Dict[str, Any]:
        import numpy as np

        phase = ((time - np.nanmin(time)) / best_period) % 1.0
        order = np.argsort(phase)
        plt = self.plotting_service._apply_style(dark=False)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.4, 3.6))
        ax1.plot(frequency, power, color="#0072B2", lw=1.1)
        ax1.axvline(best_frequency, color="#D55E00", ls="--", lw=1.0)
        ax1.set_xlabel("Frequency (1/day)")
        ax1.set_ylabel("Lomb-Scargle power")
        ax1.set_title(f"Best period = {best_period:.5g} d")
        ax2.scatter(phase[order], value[order], s=16, alpha=0.8, color="#0072B2", edgecolors="none")
        ax2.scatter(phase[order] + 1.0, value[order], s=16, alpha=0.8, color="#0072B2", edgecolors="none")
        ax2.set_xlabel("Phase")
        ax2.set_ylabel("Magnitude" if value_kind == "mag" else "Normalized flux")
        if value_kind == "mag":
            ax2.invert_yaxis()
        ax2.set_title(identifier)
        fig.tight_layout()
        return self.plotting_service._save_and_encode(fig, f"period_search_{uuid.uuid4().hex[:10]}")


PLOTLY_POINT_CAP = 5000


def _downsample_pair(time: Any, value: Any, cap: int = PLOTLY_POINT_CAP) -> Tuple[Any, Any, bool]:
    """Uniform-stride subsample so interactive specs stay light (SVG scatter)."""
    import numpy as np

    t = np.asarray(time, dtype=float)
    v = np.asarray(value, dtype=float)
    if t.size <= cap:
        return t, v, False
    idx = np.linspace(0, t.size - 1, cap).round().astype(int)
    return t[idx], v[idx], True


def _lightcurve_plotly_spec(time: Any, value: Any, value_kind: str, title: str) -> Dict[str, Any]:
    """Interactive light-curve spec (basic Plotly bundle: plain scatter)."""
    t, v, truncated = _downsample_pair(time, value)
    y_title = "Magnitude" if value_kind == "mag" else "Normalized flux"
    layout: Dict[str, Any] = {
        "title": {"text": title + (" (subsampled)" if truncated else "")},
        "xaxis": {"title": {"text": "Time (days)"}},
        "yaxis": {"title": {"text": y_title}},
        "height": 420,
        "hovermode": "closest",
    }
    if value_kind == "mag":
        layout["yaxis"]["autorange"] = "reversed"
    return {
        "data": [{
            "type": "scatter",
            "mode": "markers",
            "name": "light curve",
            "x": [round(float(x), 6) for x in t],
            "y": [round(float(y), 6) for y in v],
            "marker": {"color": "#22d3ee", "size": 3, "opacity": 0.75},
            "hovertemplate": "t = %{x:.4f} d<br>%{y:.5f}<extra></extra>",
        }],
        "layout": layout,
    }


def _period_plotly_spec(period: Dict[str, Any], value_kind: str, identifier: str) -> Dict[str, Any]:
    """Periodogram + phase-fold as a two-panel interactive spec.

    The raw (time, value) arrays, best period, and aliases ride in
    ``layout.meta`` so the frontend can re-fold client-side without a server
    round-trip (period-slider card).
    """
    import numpy as np

    freq = np.asarray(period["frequency_per_d"], dtype=float)
    power = np.asarray(period["power"], dtype=float)
    # The fold trace is drawn twice (phase, phase+1) for wrap continuity, so
    # cap the RAW observations at 2000 -> <=4000 rendered markers (CX-31).
    t, v, truncated = _downsample_pair(period["time_days"], period["value"], cap=2000)
    best_period = float(period["best_period_d"])
    phase = ((t - np.nanmin(t)) / best_period) % 1.0

    pf, pw, _ = _downsample_pair(freq, power, cap=4000)
    y_title = "Magnitude" if value_kind == "mag" else "Normalized flux"
    fold_yaxis: Dict[str, Any] = {"title": {"text": y_title}, "anchor": "x2"}
    if value_kind == "mag":
        fold_yaxis["autorange"] = "reversed"
    fap = period.get("fap")
    title_txt = f"{identifier}: best period {best_period:.5g} d"
    if fap is not None:
        title_txt += f" (FAP {fap:.2g})"
    return {
        "data": [
            {
                "type": "scatter",
                "mode": "lines",
                "name": "periodogram",
                "x": [round(float(x), 6) for x in pf],
                "y": [round(float(y), 6) for y in pw],
                "line": {"color": "#22d3ee", "width": 1.2},
                "xaxis": "x",
                "yaxis": "y",
                "hovertemplate": "f = %{x:.5f} /d<br>power %{y:.4f}<extra></extra>",
            },
            {
                "type": "scatter",
                "mode": "markers",
                "name": "phase fold",
                "x": [round(float(x), 5) for x in np.concatenate([phase, phase + 1.0])],
                "y": [round(float(y), 6) for y in np.concatenate([v, v])],
                "marker": {"color": "#f59e0b", "size": 4, "opacity": 0.75},
                "xaxis": "x2",
                "yaxis": "y2",
                "hovertemplate": "phase %{x:.3f}<br>%{y:.5f}<extra></extra>",
            },
        ],
        "layout": {
            "title": {"text": title_txt + (" (subsampled)" if truncated else "")},
            "grid": {"rows": 1, "columns": 2, "pattern": "independent"},
            "xaxis": {"title": {"text": "Frequency (1/day)"}},
            "yaxis": {"title": {"text": "Lomb-Scargle power"}},
            "xaxis2": {"title": {"text": "Phase"}},
            "yaxis2": fold_yaxis,
            "height": 400,
            "showlegend": False,
            "shapes": [{
                "type": "line",
                "xref": "x", "yref": "paper",
                "x0": float(period["best_frequency_per_d"]), "x1": float(period["best_frequency_per_d"]),
                "y0": 0, "y1": 1,
                "line": {"color": "#f59e0b", "width": 1, "dash": "dash"},
            }],
            "meta": {
                "kind": "period_fold",
                "time_days": [round(float(x), 6) for x in t],
                "value": [round(float(y), 6) for y in v],
                "value_kind": value_kind,
                "best_period_d": best_period,
                "period_unc_d": period.get("period_unc_d"),
                "fap": fap,
                "top_periods": period.get("top_periods"),
                "identifier": identifier,
            },
        },
    }


def _run_lomb_scargle(time: Any, value: Any, *, value_kind: str, min_period_d: float, max_period_d: float, warnings: List[str]) -> Dict[str, Any]:
    try:
        import numpy as np
        from astropy.timeseries import LombScargle

        t, y = _finite_arrays(time, value)
        if len(t) < 5:
            return {"success": False, "error": "At least five finite light-curve points are required for period_search."}
        if len(np.unique(t)) < 5:
            return {"success": False, "error": "At least five unique finite observation times are required for period_search."}
        if len(t) < 20:
            warnings.append(FEW_POINTS_WARNING)
        if not np.isfinite(np.nanstd(y)) or float(np.nanstd(y)) <= 0.0:
            return {"success": False, "error": "Light-curve values are constant or non-finite; no periodogram can be computed."}
        ls = LombScargle(t, y)
        freq, power = ls.autopower(minimum_frequency=1.0 / max_period_d, maximum_frequency=1.0 / min_period_d, samples_per_peak=20)
        freq = np.asarray(freq, dtype=float)
        power = np.asarray(power, dtype=float)
        finite_power = np.isfinite(freq) & np.isfinite(power)
        if not finite_power.any():
            return {"success": False, "error": "Lomb-Scargle periodogram returned no finite power values."}
        valid_indices = np.where(finite_power)[0]
        best_idx = int(valid_indices[np.nanargmax(power[finite_power])])
        best_frequency = float(freq[best_idx])
        best_period = float(1.0 / best_frequency)
        return {
            "success": True,
            "time_days": t,
            "value": y,
            "value_kind": value_kind,
            "frequency_per_d": freq,
            "power": power,
            "best_period_d": best_period,
            "best_frequency_per_d": best_frequency,
            "period_unc_d": _peak_half_width_period_uncertainty(freq, power, best_idx),
            "fap": _false_alarm_probability(ls, power[best_idx], warnings),
            "top_periods": _top_periods(freq, power, min_bin_sep=5, count=3),
            "n_points": int(len(t)),
            "time_span_days": _time_span(t),
        }
    except Exception as exc:
        return {"success": False, "error": str(exc)}


def _false_alarm_probability(ls: Any, peak_power: Any, warnings: List[str]) -> Optional[float]:
    import warnings as py_warnings

    try:
        with py_warnings.catch_warnings():
            py_warnings.simplefilter("ignore", RuntimeWarning)
            fap = float(ls.false_alarm_probability(peak_power))
    except Exception as exc:
        warnings.append(f"False-alarm probability could not be computed: {exc}")
        return None
    return fap if math.isfinite(fap) else None


def _top_periods(freq: Any, power: Any, *, min_bin_sep: int, count: int) -> List[Dict[str, float]]:
    import numpy as np

    order = [int(i) for i in np.argsort(power)[::-1] if np.isfinite(freq[i]) and np.isfinite(power[i]) and freq[i] > 0]
    chosen: List[int] = []
    for idx in order:
        if all(abs(idx - prev) > min_bin_sep for prev in chosen):
            chosen.append(idx)
        if len(chosen) >= count:
            break
    return [{"period_d": float(1.0 / freq[idx]), "frequency_per_d": float(freq[idx]), "power": float(power[idx])} for idx in chosen]


def _peak_half_width_period_uncertainty(freq: Any, power: Any, best_idx: int) -> Optional[float]:
    import numpy as np

    peak = float(power[best_idx])
    if not math.isfinite(peak) or peak <= 0:
        return None
    threshold = 0.5 * peak
    left = best_idx
    while left > 0 and np.isfinite(power[left]) and power[left] >= threshold:
        left -= 1
    right = best_idx
    last = len(power) - 1
    while right < last and np.isfinite(power[right]) and power[right] >= threshold:
        right += 1
    if left == 0 or right == last:
        return None
    p_left = 1.0 / float(freq[left])
    p_right = 1.0 / float(freq[right])
    if not math.isfinite(p_left) or not math.isfinite(p_right):
        return None
    return float(abs(p_left - p_right) / 2.0)


def _period_bounds(time: Any, min_period: Any, max_period: Any) -> Tuple[Tuple[float, float], List[str]]:
    import numpy as np

    min_p = _finite_positive_float(min_period, "min_period_d")
    max_p = _finite_positive_float(max_period, "max_period_d")
    if min_p >= max_p:
        raise ValueError("min_period_d must be smaller than max_period_d.")
    warnings: List[str] = []
    t = np.asarray(time, dtype=float)
    t = t[np.isfinite(t)]
    span = _time_span(t)
    if span is not None and span > 0 and max_p > span:
        warnings.append(f"max_period_d {max_p:g} exceeds the finite time span ({span:g} d); long periods are unreliable.")
    if len(t) >= 3:
        diffs = np.diff(np.sort(np.unique(t)))
        diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
        if len(diffs):
            cadence = float(np.median(diffs))
            if min_p < 2.0 * cadence:
                warnings.append(f"min_period_d {min_p:g} is shorter than twice the median cadence ({2.0 * cadence:g} d); short-period aliases are likely.")
    return (min_p, max_p), warnings


def _image_result(render: Dict[str, Any], extra: Dict[str, Any]) -> Dict[str, Any]:
    out = {
        "success": True,
        "image_base64": render.get("base64_png"),
        "path": render.get("web_url"),
        "png_path": render.get("png_path"),
        "pdf_path": render.get("pdf_path"),
    }
    out.update(extra)
    return out


def _normalize_search_row(row: Any, index: int) -> Dict[str, Any]:
    return {
        "index": int(index),
        "mission": _str_or_none(_row_get(row, ("mission",))),
        "year": _int_or_none(_row_get(row, ("year",))),
        "author": _str_or_none(_row_get(row, ("author",))),
        "exptime_s": _float_or_none(_row_get(row, ("exptime", "t_exptime"))),
        "target_name": _str_or_none(_row_get(row, ("target_name", "target", "object"))),
        "distance_arcsec": _float_or_none(_row_get(row, ("distance", "distance_arcsec"))),
    }


def _lightcurve_meta(row: Any, lc_meta: Any, target: str, index: int) -> Dict[str, Any]:
    meta = dict(lc_meta or {}) if isinstance(lc_meta, dict) else {}
    mission = _str_or_none(_row_get(row, ("mission",))) or _str_or_none(meta.get("MISSION"))
    author = _str_or_none(_row_get(row, ("author",))) or _str_or_none(meta.get("AUTHOR"))
    exptime = _float_or_none(_row_get(row, ("exptime", "t_exptime"))) or _float_or_none(meta.get("EXPTIME"))
    label = mission or target
    return {
        "mission": mission,
        "author": author,
        "exptime_s": exptime,
        "label": label,
        "sector_or_quarter": _sector_or_quarter(mission),
        "target": target,
        "index": int(index),
    }


def _sector_or_quarter(mission: Optional[str]) -> Optional[str]:
    if not mission:
        return None
    match = re.search(r"\b(Sector|Quarter|Campaign)\s+([0-9]+)", mission, flags=re.IGNORECASE)
    return f"{match.group(1).title()} {match.group(2)}" if match else None


def _meta_label(meta: Dict[str, Any]) -> str:
    parts = [part for part in (meta.get("mission"), meta.get("author"), meta.get("sector_or_quarter")) if part]
    return ", ".join(str(part) for part in parts) or str(meta.get("label") or "light curve")


def _mission_for_source(source: str, mission: Optional[Any]) -> Tuple[str, List[str]]:
    mission_s, warnings = _normalize_mission(mission)
    if source == "tess":
        if mission_s and mission_s != "TESS":
            raise ValueError("source='tess' requires mission='TESS' or mission omitted.")
        return "TESS", warnings
    if mission_s and mission_s not in {"Kepler", "K2"}:
        raise ValueError("source='kepler' requires mission='Kepler', mission='K2', or mission omitted.")
    return mission_s or "Kepler", warnings


def _normalize_mission(mission: Optional[Any]) -> Tuple[Optional[str], List[str]]:
    if mission is None or str(mission).strip() == "":
        return None, []
    raw = str(mission).strip()
    mapping = {"tess": "TESS", "kepler": "Kepler", "k2": "K2"}
    if raw.lower() in mapping:
        return mapping[raw.lower()], []
    return None, [f"Unsupported mission {raw!r}; searching all TESS/Kepler/K2 light curves."]


def _normalize_source(source: Any) -> str:
    value = str(source or "").strip().lower()
    if value not in {"tess", "kepler", "ztf"}:
        raise ValueError("source must be one of: tess, kepler, ztf.")
    return value


def _result_len(result: Any) -> int:
    try:
        return int(len(result))
    except Exception:
        return _table_len(getattr(result, "table", None))


def _table_len(table: Any, fallback: int = 0) -> int:
    if table is None:
        return int(fallback)
    try:
        return int(len(table))
    except Exception:
        return int(fallback)


def _table_row(table: Any, index: int) -> Any:
    if table is None:
        return {}
    try:
        return table[index]
    except Exception:
        try:
            rows = list(table)
        except Exception:
            rows = []
        return rows[index] if 0 <= index < len(rows) else {}


def _row_get(row: Any, names: Sequence[str]) -> Any:
    if row is None:
        return None
    if isinstance(row, dict):
        keys = list(row.keys())
    else:
        try:
            keys = list(row.keys())
        except Exception:
            keys = []
    lower_map = {str(key).lower(): key for key in keys}
    for name in names:
        key = lower_map.get(name.lower())
        if key is not None:
            try:
                return row[key]
            except Exception:
                pass
    for name in names:
        if hasattr(row, name):
            return getattr(row, name)
    return None


def _array_values(value: Any) -> Any:
    if value is None:
        return []
    if hasattr(value, "to_value"):
        try:
            return value.to_value()
        except Exception:
            pass
    if hasattr(value, "value"):
        try:
            return value.value
        except Exception:
            pass
    return value


def _finite_arrays(time: Any, value: Any) -> Tuple[Any, Any]:
    import numpy as np

    t = np.asarray(_array_values(time), dtype=float)
    y = np.asarray(_array_values(value), dtype=float)
    if t.shape != y.shape:
        raise ValueError("time and value arrays must have the same shape.")
    mask = np.isfinite(t) & np.isfinite(y)
    return t[mask], y[mask]


def _time_span(time: Any) -> Optional[float]:
    import numpy as np

    t = np.asarray(time, dtype=float)
    t = t[np.isfinite(t)]
    if len(t) < 2:
        return None
    return float(np.nanmax(t) - np.nanmin(t))


def _require_text(value: Any, name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{name} is required.")
    return text


def _normalize_positive_int(value: Any, *, default: int, cap: int, label: str) -> Tuple[int, List[str]]:
    warnings: List[str] = []
    try:
        out = int(value)
    except (TypeError, ValueError):
        out = default
        warnings.append(f"{label} was not an integer; using {default}.")
    if out <= 0:
        warnings.append(f"{label} must be positive; using {default}.")
        out = default
    elif out > cap:
        warnings.append(f"{label} {out} exceeds {cap}; clamped to {cap}.")
        out = cap
    return out, warnings


def _nonnegative_int(value: Any, label: str) -> int:
    try:
        out = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be an integer.") from exc
    if out < 0:
        raise ValueError(f"{label} must be non-negative.")
    return out


def _finite_positive_float(value: Any, label: str) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric.") from exc
    if not math.isfinite(out) or out <= 0:
        raise ValueError(f"{label} must be finite and positive.")
    return out


def _env_float(name: str, default: float) -> float:
    try:
        value = float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) and value > 0 else default


def _float_or_none(value: Any) -> Optional[float]:
    if value is None or _is_masked(value):
        return None
    try:
        if hasattr(value, "to_value"):
            value = value.to_value()
        if hasattr(value, "item") and not isinstance(value, (str, bytes)):
            value = value.item()
        out = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return out if math.isfinite(out) else None


def _int_or_none(value: Any) -> Optional[int]:
    if value is None or _is_masked(value):
        return None
    try:
        if hasattr(value, "item") and not isinstance(value, (str, bytes)):
            value = value.item()
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return None


def _str_or_none(value: Any) -> Optional[str]:
    if value is None or _is_masked(value):
        return None
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    text = str(value).strip()
    return text if text and text.lower() != "nan" else None


def _is_masked(value: Any) -> bool:
    try:
        import numpy as np

        return bool(np.ma.is_masked(value))
    except Exception:
        return False


__all__ = ["LightCurveSuite"]