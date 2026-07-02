"""NOIRLab SparCL spectrum search and plotting helpers."""

from __future__ import annotations

import importlib
import math
import uuid
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from services.plotting import PlottingService


DEFAULT_RELEASES = ["DESI-EDR"]
OUTFIELDS = ["sparcl_id", "ra", "dec", "redshift", "spectype", "data_release"]
RETRIEVE_FIELDS = ["wavelength", "flux", "model", "redshift", "spectype"]
COMMON_LINES = [
    ("Ly alpha", 1215.67),
    ("C IV", 1549.06),
    ("Mg II", 2798.75),
    ("[O II]", 3727.4),
    ("Ca K", 3933.66),
    ("Ca H", 3968.47),
    ("H delta", 4102.9),
    ("H gamma", 4341.7),
    ("H beta", 4862.7),
    ("[O III]", 4960.3),
    ("[O III]", 5008.2),
    ("Mg b", 5176.7),
    ("Na D", 5895.6),
    ("H alpha", 6564.6),
    ("[N II]", 6585.3),
    ("[S II]", 6718.3),
    ("[S II]", 6732.7),
]


class SparclSpectraService:
    """Lazy cached SparCL client wrapper for source search and spectrum plots."""

    def __init__(self, *, plotting_service: Optional[PlottingService] = None):
        self.plotting_service = plotting_service or PlottingService()
        self._client = None

    def _get_client(self):
        if self._client is not None:
            return self._client
        try:
            module = importlib.import_module("sparcl.client")
            client_cls = getattr(module, "SparclClient")
        except Exception as exc:
            raise RuntimeError(f"SparCL unavailable: pip install sparclclient. Underlying error: {exc}") from exc
        try:
            self._client = client_cls()
        except Exception as exc:
            raise RuntimeError(f"SparCL client initialization failed: {exc}") from exc
        return self._client

    def find_spectra(
        self,
        ra: Any,
        dec: Any,
        radius_arcsec: Any = 60,
        data_release: Optional[Sequence[str] | str] = None,
        limit: Any = 20,
    ) -> Dict[str, Any]:
        try:
            ra_f, dec_f = _validate_coords(ra, dec)
            radius = _positive_float(radius_arcsec, 60.0)
            if radius > 3600.0:
                radius = 3600.0
                warnings = ["radius_arcsec exceeds 3600; clamped to 3600 arcsec."]
            else:
                warnings = []
            limit_i = _positive_int(limit, 20, 200)
            releases = _normalize_releases(data_release)
            client = self._get_client()
            ranges, range_warnings = _search_ranges(ra_f, dec_f, radius)
            warnings.extend(range_warnings)
            records: List[Dict[str, Any]] = []
            for ra_min, ra_max, dec_min, dec_max in ranges:
                result = client.find(
                    outfields=list(OUTFIELDS),
                    constraints={"data_release": releases, "ra": [ra_min, ra_max], "dec": [dec_min, dec_max]},
                    limit=limit_i,
                )
                records.extend(_records_list(result))

            rows: List[Dict[str, Any]] = []
            seen = set()
            for record in records:
                row = _normalize_record(record)
                if row is None:
                    continue
                distance = _angular_distance_arcsec(ra_f, dec_f, row["ra"], row["dec"])
                if distance > radius:
                    continue
                sid = str(row.get("sparcl_id") or "")
                if sid and sid in seen:
                    continue
                if sid:
                    seen.add(sid)
                row["distance_arcsec"] = distance
                rows.append(row)
            rows.sort(key=lambda row: row.get("distance_arcsec", float("inf")))
            rows = rows[:limit_i]
            return {
                "success": True,
                "rows": rows,
                "count": len(rows),
                "warnings": warnings,
                "provenance": {
                    "service": "NOIRLab SparCL",
                    "data_release": releases,
                    "ra": ra_f,
                    "dec": dec_f,
                    "radius_arcsec": radius,
                    "limit": limit_i,
                    "query_boxes": [list(rng) for rng in ranges],
                },
            }
        except Exception as exc:
            return {"success": False, "error": str(exc)}

    def plot_spectrum(self, sparcl_id: str, mark_lines: bool = True, smooth: Any = 0) -> Dict[str, Any]:
        sid = str(sparcl_id or "").strip()
        if not sid:
            return {"success": False, "error": "sparcl_id is required."}
        try:
            client = self._get_client()
            result = client.retrieve(uuid_list=[sid], include=list(RETRIEVE_FIELDS))
            records = _records_list(result)
            if not records:
                return {"success": False, "error": f"No SparCL spectrum returned for {sid}."}
            record = records[0]
            wavelength_raw = record.get("wavelength")
            flux_raw = record.get("flux")
            wavelength = np.asarray(wavelength_raw if wavelength_raw is not None else [], dtype=float)
            flux = np.asarray(flux_raw if flux_raw is not None else [], dtype=float)
            if wavelength.size == 0 or flux.size == 0 or wavelength.size != flux.size:
                return {"success": False, "error": f"SparCL spectrum {sid} has invalid wavelength/flux arrays."}
            finite = np.isfinite(wavelength) & np.isfinite(flux)
            wavelength = wavelength[finite]
            flux = flux[finite]
            if wavelength.size == 0:
                return {"success": False, "error": f"SparCL spectrum {sid} has no finite wavelength/flux points."}
            warnings: List[str] = []
            smooth_i = _positive_int(smooth, 0, 1000, allow_zero=True)
            plot_flux = flux
            if smooth_i > 1:
                if smooth_i >= flux.size:
                    warnings.append("smooth exceeds spectrum length; smoothing skipped.")
                else:
                    kernel = np.ones(smooth_i, dtype=float) / float(smooth_i)
                    plot_flux = np.convolve(flux, kernel, mode="same")

            model = record.get("model")
            model_arr = None
            if model is not None:
                candidate = np.asarray(model, dtype=float)
                if candidate.size == finite.size:
                    model_arr = candidate[finite]
                elif candidate.size == wavelength.size:
                    model_arr = candidate

            redshift = _float_or_none(record.get("redshift"))
            spectype = str(record.get("spectype") or "Spectrum")
            plt = self.plotting_service._apply_style(dark=False)
            fig, ax = plt.subplots(figsize=(9.0, 4.8))
            ax.plot(wavelength, plot_flux, lw=0.8, alpha=0.65, label="flux")
            if model_arr is not None and model_arr.size == wavelength.size:
                ax.plot(wavelength, model_arr, lw=1.0, alpha=0.85, label="model")
            if mark_lines and redshift is not None:
                self._mark_lines(ax, wavelength, redshift)
            ax.set_xlabel("Wavelength [A]")
            ax.set_ylabel("Flux")
            z_label = "unknown" if redshift is None else f"{redshift:.5g}"
            ax.set_title(f"{spectype} z={z_label} - {sid[:8]}")
            ax.legend(loc="best", fontsize=8)
            fig.tight_layout()
            render = self.plotting_service._save_and_encode(fig, f"sparcl_spectrum_{uuid.uuid4().hex[:10]}")
            return {
                "success": True,
                "image_base64": render.get("base64_png"),
                "path": render.get("web_url"),
                "png_path": render.get("png_path"),
                "pdf_path": render.get("pdf_path"),
                "sparcl_id": sid,
                "n_points": int(wavelength.size),
                "redshift": redshift,
                "spectype": spectype,
                "warnings": warnings,
                "provenance": {"service": "NOIRLab SparCL", "include": list(RETRIEVE_FIELDS)},
            }
        except Exception as exc:
            return {"success": False, "error": str(exc)}

    @staticmethod
    def _mark_lines(ax: Any, wavelength: np.ndarray, redshift: float) -> None:
        lo = float(np.nanmin(wavelength))
        hi = float(np.nanmax(wavelength))
        ymax = ax.get_ylim()[1]
        for label, rest in COMMON_LINES:
            observed = rest * (1.0 + redshift)
            if lo <= observed <= hi:
                ax.axvline(observed, color="0.4", linestyle="--", linewidth=0.7, alpha=0.45)
                ax.text(observed, ymax, label, rotation=90, va="top", ha="right", fontsize=7, alpha=0.7)


def _validate_coords(ra: Any, dec: Any) -> Tuple[float, float]:
    try:
        ra_f = float(ra)
        dec_f = float(dec)
    except (TypeError, ValueError) as exc:
        raise ValueError("ra and dec must be numeric ICRS degrees.") from exc
    if not math.isfinite(ra_f) or not math.isfinite(dec_f):
        raise ValueError("ra and dec must be finite ICRS degrees.")
    if not 0.0 <= ra_f < 360.0:
        raise ValueError("ra must satisfy 0 <= ra < 360 degrees.")
    if not -90.0 <= dec_f <= 90.0:
        raise ValueError("dec must satisfy -90 <= dec <= 90 degrees.")
    return ra_f, dec_f


def _normalize_releases(value: Optional[Sequence[str] | str]) -> List[str]:
    if value is None:
        return list(DEFAULT_RELEASES)
    if isinstance(value, str):
        releases = [part.strip() for part in value.split(",") if part.strip()]
    else:
        releases = [str(part).strip() for part in value if str(part).strip()]
    return releases or list(DEFAULT_RELEASES)


def _positive_float(value: Any, default: float) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) and out > 0 else default


def _positive_int(value: Any, default: int, cap: int, *, allow_zero: bool = False) -> int:
    try:
        out = int(value)
    except (TypeError, ValueError):
        out = default
    if allow_zero and out <= 0:
        return 0
    if out <= 0:
        out = default
    return min(out, cap)


def _float_or_none(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _search_ranges(ra: float, dec: float, radius_arcsec: float) -> Tuple[List[Tuple[float, float, float, float]], List[str]]:
    radius_deg = radius_arcsec / 3600.0
    cos_dec = abs(math.cos(math.radians(dec)))
    dra = 180.0 if cos_dec < 1e-8 else min(180.0, radius_deg / cos_dec)
    dec_min = max(-90.0, dec - radius_deg)
    dec_max = min(90.0, dec + radius_deg)
    ra_min = ra - dra
    ra_max = ra + dra
    warnings: List[str] = []
    if dra >= 180.0:
        return [(0.0, 360.0, dec_min, dec_max)], ["RA search box spans all right ascensions near the pole."]
    if ra_min < 0.0:
        warnings.append("RA search box crosses 0 deg; split into two SparCL queries.")
        return [(0.0, ra_max, dec_min, dec_max), (360.0 + ra_min, 360.0, dec_min, dec_max)], warnings
    if ra_max > 360.0:
        warnings.append("RA search box crosses 360 deg; split into two SparCL queries.")
        return [(ra_min, 360.0, dec_min, dec_max), (0.0, ra_max - 360.0, dec_min, dec_max)], warnings
    return [(ra_min, ra_max, dec_min, dec_max)], warnings


def _records_list(result: Any) -> List[Dict[str, Any]]:
    records = getattr(result, "records", result)
    if records is None:
        return []
    return [dict(record) for record in records]


def _normalize_record(record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    ra = _float_or_none(record.get("ra"))
    dec = _float_or_none(record.get("dec"))
    sid = record.get("sparcl_id")
    if sid is None or ra is None or dec is None:
        return None
    return {
        "sparcl_id": str(sid),
        "ra": ra,
        "dec": dec,
        "redshift": _float_or_none(record.get("redshift")),
        "spectype": record.get("spectype"),
        "data_release": record.get("data_release"),
    }


def _angular_distance_arcsec(ra1: float, dec1: float, ra2: float, dec2: float) -> float:
    r1 = math.radians(ra1)
    d1 = math.radians(dec1)
    r2 = math.radians(ra2)
    d2 = math.radians(dec2)
    sd = math.sin((d2 - d1) / 2.0)
    sr = math.sin((r2 - r1) / 2.0)
    a = sd * sd + math.cos(d1) * math.cos(d2) * sr * sr
    angle = 2.0 * math.asin(min(1.0, math.sqrt(max(0.0, a))))
    return math.degrees(angle) * 3600.0


__all__ = ["SparclSpectraService"]
