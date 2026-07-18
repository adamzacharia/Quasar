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
# A 2048x2048 float64 layer is ~34 MB; cap well above that but far below the
# 150 MB archive-FITS limit so a runaway response can't balloon memory.
MAX_FITS_LAYER_MB = 64

SURVEY_ALIASES: Dict[str, str] = {
    "optical": "CDS/P/DSS2/color",
    "dss": "CDS/P/DSS2/color",
    "dss2": "CDS/P/DSS2/color",
    "dss2_red": "CDS/P/DSS2/red",
    "dss2_blue": "CDS/P/DSS2/blue",
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
    # Single-band aliases — use these as RGB channels (color aliases above are
    # 3-plane JPEG-style HiPS and make poor RGB inputs).
    "2mass_j": "CDS/P/2MASS/J",
    "2mass_h": "CDS/P/2MASS/H",
    "2mass_k": "CDS/P/2MASS/K",
    "sdss_g": "CDS/P/SDSS9/g",
    "sdss_r": "CDS/P/SDSS9/r",
    "sdss_i": "CDS/P/SDSS9/i",
    "sdss_z": "CDS/P/SDSS9/z",
    "wise_w1": "CDS/P/allWISE/W1",
    "wise_w2": "CDS/P/allWISE/W2",
    "wise_w3": "CDS/P/allWISE/W3",
    "wise_w4": "CDS/P/allWISE/W4",
    "galex_nuv": "CDS/P/GALEXGR6/AIS/NUV",
    "galex_fuv": "CDS/P/GALEXGR6/AIS/FUV",
    # Herschel FIR maps (R5 HiPS photometry; IDs verified against the CDS
    # MocServer 2026-07-18). PACS100 is on the photometry known-bad list —
    # see services/image_analysis.py HIPS_PHOTOMETRY_VALIDATION.
    "pacs70": "ESAVO/P/HERSCHEL/PACS70",
    "pacs100": "ESAVO/P/HERSCHEL/PACS100",
    "pacs160": "ESAVO/P/HERSCHEL/PACS160",
    "spire250": "ESAVO/P/HERSCHEL/SPIRE-250",
    "spire350": "ESAVO/P/HERSCHEL/SPIRE-350",
    "spire500": "ESAVO/P/HERSCHEL/SPIRE-500",
}

# Color HiPS aliases (multi-plane JPEG surfaces). Poor RGB channels because a
# single plane is not a real band; rgb_composite warns when one is used.
COLOR_HIPS_ALIASES = frozenset({"optical", "dss", "dss2", "sdss", "2mass", "nir",
                                "wise", "mir", "galex", "uv", "gamma", "fermi"})


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

    def fits_url(
        self,
        ra: Any,
        dec: Any,
        fov_deg: Any = 0.25,
        survey: str = "optical",
        width: Any = 512,
    ) -> str:
        """Build a hips2fits format=fits URL — calibrated pixels for any HiPS survey.

        The URL is a plain GET target, so it can be handed to
        services.fits_service._download_fits, offered as a card download, or
        fed to the image-analysis tools.
        """
        from urllib.parse import urlencode

        ra_f, dec_f = _validate_coords(ra, dec)
        survey_id = resolve_survey(survey)
        fov, width_i, _ = _normalize_fov_width(fov_deg, width)
        if survey_id == VLASS_HIPS_ID and dec_f <= -40.0:
            raise HipsImageError("VLASS covers Dec > -40 deg only; choose a northern target or another radio survey.")
        params = self._params(survey_id, ra_f, dec_f, fov, width_i)
        params["format"] = "fits"
        return f"{self.base_url}?{urlencode(params)}"

    def rgb_composite(
        self,
        ra: Any,
        dec: Any,
        surveys: Iterable[str],
        fov_deg: Any = 0.25,
        width: Any = 512,
        stretch: Any = 5.0,
        q: Any = 8.0,
        title: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Lupton RGB composite from three single-band HiPS surveys (R, G, B order).

        hips2fits serves every layer at identical WCS geometry (same ra/dec/fov/
        width/projection), so no reprojection is needed — the three arrays are
        pixel-aligned by construction. Blank layers are rejected with a clear
        error instead of silently rendering an empty channel.
        """
        import numpy as np

        try:
            ra_f, dec_f = _validate_coords(ra, dec)
            survey_list = [str(s) for s in (surveys or [])]
            if len(survey_list) != 3:
                raise HipsImageError("rgb_composite needs exactly three surveys (R, G, B order).")
            fov, width_i, warnings = _normalize_fov_width(fov_deg, width)

            layers: List[np.ndarray] = []
            survey_ids: List[str] = []
            for band in survey_list:
                if band.strip().lower() in COLOR_HIPS_ALIASES:
                    raise HipsImageError(
                        f"{band!r} is a color (multi-plane) HiPS and cannot be an RGB channel. "
                        "Use single-band aliases: 2mass_j/h/k, sdss_g/r/i/z, wise_w1..w4, "
                        "galex_nuv/fuv, dss2_red/dss2_blue (or a raw single-band HiPS ID)."
                    )
                survey_id = resolve_survey(band)
                survey_ids.append(survey_id)
                data = self._fetch_fits_layer(survey_id, ra_f, dec_f, fov, width_i, reject_multiplane=True)
                finite = data[np.isfinite(data)]
                if finite.size == 0 or float(np.nanmax(data)) == float(np.nanmin(data)):
                    raise HipsImageError(
                        f"Survey {band!r} returned a blank tile here (no coverage). "
                        "Pick another survey for that channel."
                    )
                if finite.size < 0.05 * data.size:
                    warnings.append(f"{band}: <5% of pixels have coverage; channel is mostly blank.")
                layers.append(data)  # keep NaNs so percentiles ignore blank pixels

            from astropy.visualization import make_lupton_rgb

            # Normalize each channel to its robust [p1, p99.5] range before Lupton.
            # Percentiles are taken over the TRUE finite pixels (NaN-excluded) so
            # blank edges don't drag the channel's black point to zero (CX-26);
            # NaNs are zeroed only after scaling.
            scaled = []
            for data in layers:
                finite = data[np.isfinite(data)]
                lo, hi = np.percentile(finite, [1.0, 99.5])
                span = (hi - lo) if hi > lo else 1.0
                scaled.append(np.clip((np.nan_to_num(data, nan=lo) - lo) / span, 0, None))
            try:
                stretch_f = float(stretch)
            except (TypeError, ValueError):
                stretch_f = 5.0
            try:
                q_f = float(q)
            except (TypeError, ValueError):
                q_f = 8.0
            rgb = make_lupton_rgb(scaled[0], scaled[1], scaled[2], stretch=stretch_f, Q=q_f)
            rgb = rgb[::-1]  # FITS arrays are bottom-up; PNG rows are top-down

            from PIL import Image

            os.makedirs(plotting.PLOT_OUTPUT_DIR, exist_ok=True)
            name = f"hips_rgb_{uuid.uuid4().hex[:10]}"
            png_path = os.path.join(plotting.PLOT_OUTPUT_DIR, f"{name}.png")
            Image.fromarray(rgb).save(png_path)
            with open(png_path, "rb") as handle:
                image_base64 = base64.b64encode(handle.read()).decode("ascii")
            return {
                "success": True,
                "image_base64": image_base64,
                "path": f"/plots/{name}.png",
                "png_path": png_path,
                "ra": ra_f,
                "dec": dec_f,
                "fov_deg": fov,
                "width": width_i,
                "surveys": survey_list,
                "survey_ids": survey_ids,
                "channels": {"red": survey_list[0], "green": survey_list[1], "blue": survey_list[2]},
                "warnings": warnings,
                "provenance": {
                    "service": "CDS hips2fits (format=fits) + astropy make_lupton_rgb",
                    "base_url": self.base_url,
                    "title": title,
                },
            }
        except Exception as exc:
            return {"success": False, "error": str(exc)}

    def _fetch_fits_layer(self, survey_id: str, ra: float, dec: float, fov_deg: float, width: int,
                          reject_multiplane: bool = False):
        """Fetch one hips2fits layer as a 2D float array (format=fits)."""
        import io as _io

        import numpy as np
        from astropy.io import fits as afits

        if survey_id == VLASS_HIPS_ID and dec <= -40.0:
            raise HipsImageError("VLASS covers Dec > -40 deg only.")
        params = self._params(survey_id, ra, dec, fov_deg, width)
        params["format"] = "fits"
        # Stream so the body is capped BEFORE it is fully buffered (CX-27):
        # width is <=2048 px (~34 MB real max), so anything past the cap is
        # anomalous and must not balloon memory.
        max_bytes = MAX_FITS_LAYER_MB * 1024 * 1024
        try:
            response = requests.get(self.base_url, params=params, timeout=self.timeout, stream=True)
        except requests.RequestException as exc:
            raise HipsImageError(f"hips2fits FITS request failed: {exc}") from exc
        try:
            status = int(getattr(response, "status_code", 0) or 0)
            if status != 200:
                # Read a bounded slice of the error body — never buffer a large
                # error response via response.text on a streamed request (CX-27).
                snippet = ""
                try:
                    raw_iter = response.iter_content(chunk_size=512) if hasattr(response, "iter_content") else None
                    if raw_iter is not None:
                        for piece in raw_iter:
                            snippet = (piece or b"").decode("utf-8", "replace")[:200]
                            break
                    else:
                        snippet = str(getattr(response, "text", "") or "")[:200]
                except Exception:
                    snippet = ""
                raise HipsImageError(f"hips2fits returned HTTP {status}: {snippet}")
            clen = response.headers.get("Content-Length") if hasattr(response, "headers") else None
            if clen and int(clen) > max_bytes:
                raise HipsImageError(f"hips2fits FITS layer is {int(clen) / 1e6:.0f} MB, exceeds {MAX_FITS_LAYER_MB} MB.")
            chunks = []
            total = 0
            iterator = response.iter_content(chunk_size=1024 * 256) if hasattr(response, "iter_content") else None
            if iterator is not None:
                for chunk in iterator:
                    if not chunk:
                        continue
                    total += len(chunk)
                    if total > max_bytes:
                        raise HipsImageError(f"hips2fits FITS layer exceeds {MAX_FITS_LAYER_MB} MB.")
                    chunks.append(chunk)
                content = b"".join(chunks)
            else:  # test stubs without streaming
                content = bytes(getattr(response, "content", b"") or b"")
                if len(content) > max_bytes:
                    raise HipsImageError(f"hips2fits FITS layer exceeds {MAX_FITS_LAYER_MB} MB.")
        finally:
            close = getattr(response, "close", None)
            if callable(close):
                close()
        try:
            with afits.open(_io.BytesIO(content), memmap=False) as hdul:
                data = None
                for hdu in hdul:
                    if hdu.data is not None and hdu.data.ndim >= 2:
                        raw = np.asarray(hdu.data, dtype=float)
                        # A color HiPS returns a 3/4-plane cube. Silently taking
                        # plane 0 as a "band" is wrong — reject it for RGB use so
                        # a raw color HiPS ID can't slip past the alias check (CX-35).
                        if reject_multiplane and raw.ndim >= 3 and raw.shape[0] in (2, 3, 4):
                            raise HipsImageError(
                                f"Survey {survey_id!r} returned a {raw.shape[0]}-plane color image, "
                                "not a single band — pick a single-band survey for this RGB channel."
                            )
                        data = raw
                        while data.ndim > 2:
                            data = data[0]
                        break
                if data is None:
                    raise ValueError("no image HDU")
                return data
        except Exception as exc:
            raise HipsImageError(f"hips2fits did not return a readable FITS image: {exc}") from exc

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
        if survey_id == VLASS_HIPS_ID and dec <= -40.0:
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
