"""ALeRCE/ZTF alert broker client and plotting helpers."""

from __future__ import annotations

import io
import math
import os
import uuid
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests

from services.plotting import PlottingService


ALERCE_DEFAULT_BASE_URL = "https://api.alerce.online/ztf/v1"
ALERCE_DEFAULT_STAMP_URL = "https://avro.alerce.online/get_stamp"
STAMP_TYPES = ("science", "template", "difference")
FID_STYLE = {
    1: ("g", "#2ca02c"),
    2: ("r", "#d62728"),
    3: ("i", "#9467bd"),
}


class AlerceClient:
    """Small timeout-bound ALeRCE client for ZTF object, light-curve, and stamp tools."""

    def __init__(
        self,
        *,
        base_url: Optional[str] = None,
        stamp_url: Optional[str] = None,
        timeout: Optional[float] = None,
        plotting_service: Optional[PlottingService] = None,
    ):
        self.base_url = str(base_url or os.getenv("ALERCE_BASE_URL") or ALERCE_DEFAULT_BASE_URL).rstrip("/")
        self.stamp_url = str(stamp_url or os.getenv("ALERCE_STAMP_URL") or ALERCE_DEFAULT_STAMP_URL)
        self.timeout = float(timeout if timeout is not None else _env_float("ALERCE_TIMEOUT", 30.0))
        self.plotting_service = plotting_service or PlottingService()

    def cone_objects(self, ra: Any, dec: Any, radius_arcsec: Any = 120, max_rows: Any = 25) -> Dict[str, Any]:
        try:
            ra_f, dec_f = _validate_coords(ra, dec)
            radius, warnings = _normalize_radius(radius_arcsec)
            page_size = _normalize_positive_int(max_rows, 25, 500)
            payload = self._get_json(
                f"{self.base_url}/objects",
                params={"ra": ra_f, "dec": dec_f, "radius": radius, "page_size": page_size},
            )
            items = payload.get("items") if isinstance(payload, dict) else []
            rows = [self._normalize_object(item) for item in (items or [])]
            return {
                "success": True,
                "rows": rows,
                "count": len(rows),
                "warnings": warnings,
                "provenance": {
                    "service": "ALeRCE ZTF objects",
                    "base_url": self.base_url,
                    "ra": ra_f,
                    "dec": dec_f,
                    "radius_arcsec": radius,
                    "page_size": page_size,
                },
            }
        except Exception as exc:
            return {"success": False, "error": str(exc)}

    def light_curve(self, oid: str) -> Dict[str, Any]:
        try:
            oid_s = _require_oid(oid)
            payload = self._get_json(f"{self.base_url}/objects/{oid_s}/lightcurve")
            detections = [self._normalize_detection(row) for row in (payload.get("detections") or [])]
            non_detections = [self._normalize_non_detection(row) for row in (payload.get("non_detections") or [])]
            return {
                "success": True,
                "oid": oid_s,
                "detections": detections,
                "non_detections": non_detections,
                "n_detections": len(detections),
                "n_non_detections": len(non_detections),
                "provenance": {"service": "ALeRCE ZTF lightcurve", "base_url": self.base_url},
            }
        except Exception as exc:
            return {"success": False, "error": str(exc)}

    def plot_light_curve(self, oid: str) -> Dict[str, Any]:
        lc = self.light_curve(oid)
        if not lc.get("success"):
            return lc
        if not lc.get("detections") and not lc.get("non_detections"):
            return {"success": False, "error": f"No ZTF light-curve points found for {oid}."}
        try:
            plt = self.plotting_service._apply_style(dark=False)
            fig, ax = plt.subplots(figsize=(8, 4.8))
            plotted = False
            for fid, (label, color) in FID_STYLE.items():
                dets = [row for row in lc["detections"] if row.get("fid") == fid and row.get("mjd") is not None and row.get("magpsf") is not None]
                if dets:
                    ax.errorbar(
                        [row["mjd"] for row in dets],
                        [row["magpsf"] for row in dets],
                        yerr=[row.get("sigmapsf") or 0 for row in dets],
                        fmt="o",
                        markersize=4,
                        color=color,
                        ecolor=color,
                        alpha=0.85,
                        label=f"{label} detections",
                    )
                    plotted = True
                nondets = [row for row in lc["non_detections"] if row.get("fid") == fid and row.get("mjd") is not None and row.get("diffmaglim") is not None]
                if nondets:
                    ax.scatter(
                        [row["mjd"] for row in nondets],
                        [row["diffmaglim"] for row in nondets],
                        marker="v",
                        color=color,
                        alpha=0.3,
                        label=f"{label} limits",
                    )
                    plotted = True
            if not plotted:
                plt.close(fig)
                return {"success": False, "error": f"No plottable ZTF light-curve points found for {oid}."}
            ax.invert_yaxis()
            ax.set_xlabel("MJD")
            ax.set_ylabel("PSF magnitude")
            ax.set_title(str(lc["oid"]))
            ax.legend(loc="best", fontsize=8)
            fig.tight_layout()
            render = self.plotting_service._save_and_encode(fig, f"ztf_light_curve_{uuid.uuid4().hex[:10]}")
            return {
                "success": True,
                "image_base64": render.get("base64_png"),
                "path": render.get("web_url"),
                "png_path": render.get("png_path"),
                "pdf_path": render.get("pdf_path"),
                "oid": lc["oid"],
                "n_detections": lc["n_detections"],
                "n_non_detections": lc["n_non_detections"],
                "provenance": lc.get("provenance", {}),
            }
        except Exception as exc:
            return {"success": False, "error": str(exc)}

    def stamp_triplet(self, oid: str, candid: Optional[Any] = None) -> Dict[str, Any]:
        try:
            oid_s = _require_oid(oid)
            candid_s = str(candid).strip() if candid is not None and str(candid).strip() else None
            if candid_s is None:
                lc = self.light_curve(oid_s)
                if not lc.get("success"):
                    return lc
                stamped = [row for row in lc.get("detections", []) if row.get("has_stamp") and row.get("candid")]
                if not stamped:
                    return {"success": False, "error": f"No ZTF detections with stamps found for {oid_s}."}
                stamped.sort(key=lambda row: _float_or_none(row.get("mjd")) or float("-inf"), reverse=True)
                candid_s = str(stamped[0]["candid"])

            images: Dict[str, bytes] = {}
            for stamp_type in STAMP_TYPES:
                images[stamp_type] = self._fetch_stamp(oid_s, candid_s, stamp_type)

            plt = self.plotting_service._apply_style(dark=False)
            fig, axes = plt.subplots(1, 3, figsize=(9, 3.2), squeeze=False)
            for ax, stamp_type, title in zip(axes[0], STAMP_TYPES, ("Science", "Template", "Difference")):
                ax.imshow(plt.imread(io.BytesIO(images[stamp_type])), cmap="gray")
                ax.set_title(title)
                ax.axis("off")
            fig.suptitle(f"ZTF stamps {oid_s} / {candid_s}")
            fig.tight_layout()
            render = self.plotting_service._save_and_encode(fig, f"ztf_stamps_{uuid.uuid4().hex[:10]}")
            return {
                "success": True,
                "image_base64": render.get("base64_png"),
                "path": render.get("web_url"),
                "png_path": render.get("png_path"),
                "pdf_path": render.get("pdf_path"),
                "oid": oid_s,
                "candid": candid_s,
                "stamp_types": list(STAMP_TYPES),
                "provenance": {"service": "ALeRCE ZTF stamps", "stamp_url": self.stamp_url},
            }
        except Exception as exc:
            return {"success": False, "error": str(exc)}

    def _get_json(self, url: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        try:
            response = requests.get(url, params=params, timeout=self.timeout)
        except requests.RequestException as exc:
            raise RuntimeError(f"ALeRCE request failed: {exc}") from exc
        status = int(getattr(response, "status_code", 0) or 0)
        if status != 200:
            text = str(getattr(response, "text", "") or "")[:200]
            raise RuntimeError(f"ALeRCE returned HTTP {status}: {text}")
        try:
            payload = response.json()
        except Exception as exc:
            raise RuntimeError("ALeRCE returned invalid JSON.") from exc
        if not isinstance(payload, dict):
            raise RuntimeError("ALeRCE returned an unexpected JSON payload.")
        return payload

    def _fetch_stamp(self, oid: str, candid: str, stamp_type: str) -> bytes:
        params = {"oid": oid, "candid": candid, "type": stamp_type, "format": "png"}
        try:
            response = requests.get(self.stamp_url, params=params, timeout=self.timeout)
        except requests.RequestException as exc:
            raise RuntimeError(f"ALeRCE stamp request failed: {exc}") from exc
        status = int(getattr(response, "status_code", 0) or 0)
        if status != 200:
            text = str(getattr(response, "text", "") or "")[:200]
            raise RuntimeError(f"ALeRCE stamp {stamp_type} returned HTTP {status}: {text}")
        content = bytes(getattr(response, "content", b"") or b"")
        if not content.startswith(b"\x89PNG\r\n\x1a\n"):
            raise RuntimeError(f"ALeRCE stamp {stamp_type} did not return a PNG image.")
        return content

    @staticmethod
    def _normalize_object(item: Dict[str, Any]) -> Dict[str, Any]:
        row = {key: item.get(key) for key in ("oid", "ndet", "meanra", "meandec", "firstmjd", "lastmjd")}
        for key in ("classalerce", "probability", "classifier", "class", "classification", "prob"):
            if key in item:
                row[key] = item.get(key)
        return row

    @staticmethod
    def _normalize_detection(row: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "mjd": _float_or_none(row.get("mjd")),
            "magpsf": _float_or_none(row.get("magpsf")),
            "sigmapsf": _float_or_none(row.get("sigmapsf")),
            "fid": _int_or_none(row.get("fid")),
            "candid": row.get("candid"),
            "has_stamp": bool(row.get("has_stamp")),
            "diffmaglim": _float_or_none(row.get("diffmaglim")),
        }

    @staticmethod
    def _normalize_non_detection(row: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "mjd": _float_or_none(row.get("mjd")),
            "fid": _int_or_none(row.get("fid")),
            "diffmaglim": _float_or_none(row.get("diffmaglim")),
        }


def _env_float(name: str, default: float) -> float:
    try:
        value = float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) and value > 0 else default


def _require_oid(oid: Any) -> str:
    value = str(oid or "").strip()
    if not value:
        raise ValueError("oid is required.")
    return value


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


def _normalize_radius(radius_arcsec: Any) -> Tuple[float, List[str]]:
    warnings: List[str] = []
    try:
        radius = float(radius_arcsec)
    except (TypeError, ValueError):
        radius = 120.0
        warnings.append("radius_arcsec was not numeric; using 120 arcsec.")
    if not math.isfinite(radius) or radius <= 0:
        radius = 120.0
        warnings.append("radius_arcsec must be positive; using 120 arcsec.")
    elif radius > 3600.0:
        warnings.append(f"radius_arcsec {radius:g} exceeds 3600; clamped to 3600 arcsec.")
        radius = 3600.0
    return radius, warnings


def _normalize_positive_int(value: Any, default: int, cap: int) -> int:
    try:
        out = int(value)
    except (TypeError, ValueError):
        out = default
    if out <= 0:
        out = default
    return min(out, cap)


def _float_or_none(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _int_or_none(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


__all__ = ["AlerceClient"]
