"""HiPS/hips2fits image cutouts for Quasar live imagery tools."""

from __future__ import annotations

import base64
import io
import math
import os
import uuid
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests

from services import plotting


HIPS2FITS_DEFAULT_BASE_URL = "https://alasky.cds.unistra.fr/hips-image-services/hips2fits"
VLASS_HIPS_ID = "NRAO/P/VLASS-Quicklook-MedianStack"

SURVEY_ALIASES: Dict[str, str] = {
    "optical": "CDS/P/DSS2/color",
    "dss": "CDS/P/DSS2/color",
    "dss2": "CDS/P/DSS2/color",
    "dss2_red": "CDS/P/DSS2/red",
    "sdss": "CDS/P/SDSS9/color",
    "2mass": "CDS/P/2MASS/color",
    "nir": "CDS/P/2MASS/color",
    "wise": "CDS/P/allWISE/color",
    "mir": "CDS/P/allWISE/color",
    "galex": "CDS/P/GALEXGR6/AIS/color",
    "uv": "CDS/P/GALEXGR6/AIS/color",
    "xray": "CDS/P/RASS",
    "rosat": "CDS/P/RASS",
    "gamma": "CDS/P/Fermi/color",
    "fermi": "CDS/P/Fermi/color",
    "vlass": VLASS_HIPS_ID,
    "radio": VLASS_HIPS_ID,
}


class HipsImageError(ValueError):
    """Raised for user-correctable HiPS image failures."""


def resolve_survey(survey: Optional[str]) -> str:
    """Resolve a friendly survey alias to a HiPS ID, preserving raw HiPS IDs."""
    raw = str(survey or "optical").strip()
    if not raw:
        raw = "optical"
    key = raw.lower()
    if key in SURVEY_ALIASES:
        return SURVEY_ALIASES[key]
    if "/" in raw:
        return raw
    raise HipsImageError(
        f"Unknown HiPS survey alias {raw!r}. Use one of {sorted(SURVEY_ALIASES)} or pass a raw HiPS ID."
    )


def _env_float(name: str, default: float) -> float:
    try:
        value = float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) and value > 0 else default


def _validate_coords(ra: Any, dec: Any) -> Tuple[float, float]:
    try:
        ra_f = float(ra)
        dec_f = float(dec)
    except (TypeError, ValueError) as exc:
        raise HipsImageError("ra and dec must be numeric ICRS degrees.") from exc
    if not math.isfinite(ra_f) or not math.isfinite(dec_f):
        raise HipsImageError("ra and dec must be finite ICRS degrees.")
    if not 0.0 <= ra_f < 360.0:
        raise HipsImageError("ra must satisfy 0 <= ra < 360 degrees.")
    if not -90.0 <= dec_f <= 90.0:
        raise HipsImageError("dec must satisfy -90 <= dec <= 90 degrees.")
    return ra_f, dec_f


def _normalize_fov_width(fov_deg: Any, width: Any) -> Tuple[float, int, List[str]]:
    warnings: List[str] = []
    try:
        fov = float(fov_deg)
    except (TypeError, ValueError):
        fov = 0.25
        warnings.append("fov_deg was not numeric; using 0.25 deg.")
    if not math.isfinite(fov) or fov <= 0:
        warnings.append("fov_deg must be positive; using 0.25 deg.")
        fov = 0.25
    elif fov > 10.0:
        warnings.append(f"fov_deg {fov:g} exceeds 10 deg; clamped to 10 deg.")
        fov = 10.0

    try:
        width_i = int(width)
    except (TypeError, ValueError):
        width_i = 512
        warnings.append("width was not an integer; using 512 px.")
    if width_i < 64:
        warnings.append(f"width {width_i} is below 64 px; clamped to 64 px.")
        width_i = 64
    elif width_i > 2048:
        warnings.append(f"width {width_i} exceeds 2048 px; clamped to 2048 px.")
        width_i = 2048
    return fov, width_i, warnings


def _png_magic(data: bytes) -> bool:
    return isinstance(data, (bytes, bytearray)) and bytes(data).startswith(b"\x89PNG\r\n\x1a\n")


class HipsImageService:
    """Fetch and render public HiPS imagery through CDS hips2fits."""

    def __init__(
        self,
        *,
        base_url: Optional[str] = None,
        timeout: Optional[float] = None,
        plotting_service: Optional[plotting.PlottingService] = None,
    ):
        self.base_url = str(base_url or os.getenv("HIPS2FITS_BASE_URL") or HIPS2FITS_DEFAULT_BASE_URL)
        self.timeout = float(timeout if timeout is not None else _env_float("HIPS2FITS_TIMEOUT", 45.0))
        self.plotting_service = plotting_service or plotting.PlottingService()

    def cutout(
        self,
        ra: Any,
        dec: Any,
        fov_deg: Any = 0.25,
        survey: str = "optical",
        width: Any = 512,
        stretch: Optional[str] = None,
        title: Optional[str] = None,
    ) -> Dict[str, Any]:
        try:
            ra_f, dec_f = _validate_coords(ra, dec)
            survey_id = resolve_survey(survey)
            fov, width_i, warnings = _normalize_fov_width(fov_deg, width)
            png_bytes = self._fetch_png(survey_id, ra_f, dec_f, fov, width_i, stretch=stretch)
            os.makedirs(plotting.PLOT_OUTPUT_DIR, exist_ok=True)
            name = f"hips_{uuid.uuid4().hex[:10]}"
            png_path = os.path.join(plotting.PLOT_OUTPUT_DIR, f"{name}.png")
            with open(png_path, "wb") as handle:
                handle.write(png_bytes)
            image_base64 = base64.b64encode(png_bytes).decode("ascii")
            return {
                "success": True,
                "image_base64": image_base64,
                "path": f"/plots/{name}.png",
                "png_path": png_path,
                "survey": survey,
                "survey_id": survey_id,
                "ra": ra_f,
                "dec": dec_f,
                "fov_deg": fov,
                "width": width_i,
                "warnings": warnings,
                "provenance": {
                    "service": "CDS hips2fits",
                    "base_url": self.base_url,
                    "params": self._params(survey_id, ra_f, dec_f, fov, width_i, stretch=stretch),
                    "title": title,
                    "coverage_note": "hips2fits can return a blank PNG outside a survey footprint; blankness is not detected in v1.",
                },
            }
        except Exception as exc:
            return {"success": False, "error": str(exc)}

    def multiband_panel(
        self,
        ra: Any,
        dec: Any,
        fov_deg: Any = 0.25,
        surveys: Iterable[str] = ("optical", "2mass", "wise"),
        width: Any = 300,
        title: Optional[str] = None,
    ) -> Dict[str, Any]:
        try:
            ra_f, dec_f = _validate_coords(ra, dec)
            survey_list = list(surveys or ("optical", "2mass", "wise"))
            if not survey_list:
                return {"success": False, "error": "Provide at least one survey for hips_multiband_panel."}
            fov, width_i, warnings = _normalize_fov_width(fov_deg, width)
            plt = self.plotting_service._apply_style(dark=False)
            cols = min(3, len(survey_list))
            rows = int(math.ceil(len(survey_list) / cols))
            fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.0 * rows), squeeze=False)
            panels: List[Dict[str, Any]] = []
            any_image = False
            for idx, survey in enumerate(survey_list):
                ax = axes[idx // cols][idx % cols]
                ax.axis("off")
                label = str(survey)
                panel = {"survey": label}
                try:
                    survey_id = resolve_survey(label)
                    png_bytes = self._fetch_png(survey_id, ra_f, dec_f, fov, width_i, stretch=None)
                    image = plt.imread(io.BytesIO(png_bytes))
                    ax.imshow(image)
                    panel.update({"success": True, "survey_id": survey_id})
                    any_image = True
                except Exception as exc:
                    panel.update({"success": False, "error": str(exc)})
                    warnings.append(f"{label}: {exc}")
                    ax.text(0.5, 0.5, "unavailable", ha="center", va="center", transform=ax.transAxes)
                ax.set_title(label, fontsize=9)
                panels.append(panel)
            for idx in range(len(survey_list), rows * cols):
                axes[idx // cols][idx % cols].axis("off")
            fig.suptitle(title or f"HiPS multiband panel RA={ra_f:.5f}, Dec={dec_f:.5f}")
            fig.tight_layout()
            if not any_image:
                plt.close(fig)
                return {"success": False, "error": "No requested HiPS panels were available.", "warnings": warnings, "panels": panels}
            render = self.plotting_service._save_and_encode(fig, f"hips_panel_{uuid.uuid4().hex[:10]}")
            return {
                "success": True,
                "image_base64": render.get("base64_png"),
                "path": render.get("web_url"),
                "png_path": render.get("png_path"),
                "pdf_path": render.get("pdf_path"),
                "ra": ra_f,
                "dec": dec_f,
                "fov_deg": fov,
                "width": width_i,
                "surveys": survey_list,
                "warnings": warnings,
                "panels": panels,
                "provenance": {
                    "service": "CDS hips2fits",
                    "base_url": self.base_url,
                    "coverage_note": "hips2fits can return blank panels outside a survey footprint; blankness is not detected in v1.",
                },
            }
        except Exception as exc:
            return {"success": False, "error": str(exc)}

    def vlass_cutout(self, ra: Any, dec: Any, fov_deg: Any = 0.1, width: Any = 512) -> Dict[str, Any]:
        return self.cutout(ra, dec, fov_deg=fov_deg, survey="vlass", width=width, stretch="asinh")

    def _fetch_png(
        self,
        survey_id: str,
        ra: float,
        dec: float,
        fov_deg: float,
        width: int,
        *,
        stretch: Optional[str] = None,
    ) -> bytes:
        if survey_id == VLASS_HIPS_ID and dec < -40.0:
            raise HipsImageError("VLASS covers Dec > -40 deg only; choose a northern target or another radio survey.")
        params = self._params(survey_id, ra, dec, fov_deg, width, stretch=stretch)
        try:
            response = requests.get(self.base_url, params=params, timeout=self.timeout)
        except requests.RequestException as exc:
            raise HipsImageError(f"hips2fits request failed: {exc}") from exc
        status = int(getattr(response, "status_code", 0) or 0)
        if status != 200:
            text = str(getattr(response, "text", "") or "")[:200]
            raise HipsImageError(f"hips2fits returned HTTP {status}: {text}")
        content = bytes(getattr(response, "content", b"") or b"")
        if not _png_magic(content):
            raise HipsImageError("hips2fits did not return a PNG image.")
        return content

    @staticmethod
    def _params(
        survey_id: str,
        ra: float,
        dec: float,
        fov_deg: float,
        width: int,
        *,
        stretch: Optional[str] = None,
    ) -> Dict[str, Any]:
        params: Dict[str, Any] = {
            "hips": survey_id,
            "ra": ra,
            "dec": dec,
            "fov": fov_deg,
            "width": width,
            "height": width,
            "projection": "TAN",
            "format": "png",
        }
        if stretch:
            params["stretch"] = stretch
        return params


__all__ = ["HipsImageError", "HipsImageService", "SURVEY_ALIASES", "resolve_survey"]
