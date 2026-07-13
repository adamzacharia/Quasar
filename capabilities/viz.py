"""
capabilities/viz.py — the visualization / FITS-rendering family as
transport-pure capabilities (P1 family migration #6).

Logic relocated VERBATIM from ``core/agent.py`` (``_get_sky_image`` /
``_render_fits_image`` / ``_overlay_fits_images`` / ``_overlay_archive_images``
/ ``_compute_moment_map`` / ``_extract_spectrum`` / ``_fit_spectral_line`` /
``_generate_finding_chart`` plus the self-free helpers
``_overlay_region_coordinates`` / ``_mast_product_access_url`` /
``_pick_mast_fits_product`` / ``_find_alma_overlay_product`` — the last takes
its two services as parameters). Structural changes only:

  * clients arrive via ``CallContext.services`` (``skyview_client``,
    ``mast_client``, ``search_service``, ``datalink_client``), fetched with
    ``ctx.services.get(...)`` at the legacy read position (inside the try for
    everything the legacy body read there);
  * ``last_run_result`` image cards are written through the injected
    ``set_last_run_result`` accessor. These tools set ``image_url`` to a
    server-relative ``/plots/...`` path produced by services.fits_service /
    skyview — no base64 stripping is needed (unlike the Data Lab plot tools),
    so there is no image-wrapper at the adapter boundary; byte-parity with the
    inline methods which did exactly this;
  * the fits_service imports stay lazy inside the try, exactly as inline;
  * every path returns a :class:`ToolResult` whose ``native`` payload is the
    exact legacy output dict (byte-parity), including returning the service
    result dict itself (render/overlay/moment/spectrum/fit/chart pass the
    service dict through verbatim).

None of these methods carried ``@log_tool``.
"""

from __future__ import annotations

import logging
import os
import re
import uuid
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from pydantic import BaseModel, ConfigDict

from capabilities.base import BaseCapability, ToolResult
from services.data_product_triage import is_fits_product, product_rank, unique_values

logger = logging.getLogger(__name__)


def _native(out: Dict[str, Any]) -> ToolResult:
    """Wrap a legacy output dict as a byte-parity ToolResult."""
    ok = bool(isinstance(out, dict) and out.get("success"))
    err = out.get("error") if isinstance(out, dict) else None
    return ToolResult(
        success=ok,
        error=(str(err) if (err is not None and not ok) else None),
        native=out,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers (relocated verbatim; self-free)
# ─────────────────────────────────────────────────────────────────────────────
def overlay_region_coordinates(
    region: str,
    ra_deg: float = None,
    dec_deg: float = None,
) -> Optional[Tuple[float, float, str]]:
    if ra_deg is not None and dec_deg is not None:
        try:
            ra = float(ra_deg)
            dec = float(dec_deg)
            if 0.0 <= ra < 360.0 and -90.0 <= dec <= 90.0:
                return ra, dec, str(region or f"RA {ra:.5f} Dec {dec:.5f}").strip()
        except (TypeError, ValueError):
            return None

    text = str(region or "").strip()
    coord_match = re.search(
        r"(?:ra\s*[=:]?\s*)?(\d+(?:\.\d+)?)\s*[, ]+\s*(?:dec\s*[=:]?\s*)?([+-]?\d+(?:\.\d+)?)",
        text,
        flags=re.IGNORECASE,
    )
    if coord_match:
        try:
            ra = float(coord_match.group(1))
            dec = float(coord_match.group(2))
            if 0.0 <= ra < 360.0 and -90.0 <= dec <= 90.0:
                return ra, dec, text
        except (TypeError, ValueError):
            pass

    key = str(region or "").strip().lower().replace(" ", "")
    if key in {"hudf", "hubbleultradeepfield", "ultradeepfield"}:
        return 53.1625, -27.7914, "HUDF"
    if not text:
        return None

    try:
        from integrations.alminer_client import _resolve_simbad_cached
        resolved_ra, resolved_dec = _resolve_simbad_cached(text)
        if resolved_ra is not None and resolved_dec is not None:
            return float(resolved_ra), float(resolved_dec), text
    except Exception:
        pass

    try:
        from astropy.coordinates import SkyCoord
        coord = SkyCoord.from_name(text)
        return float(coord.ra.deg), float(coord.dec.deg), text
    except Exception:
        return None
    return None


def mast_product_access_url(row: Dict[str, Any]) -> str:
    import urllib.parse

    for key in ("access_url", "dataURL", "data_url", "url"):
        value = str(row.get(key) or "").strip()
        if value.startswith("http"):
            return value
    data_uri = str(row.get("dataURI") or row.get("data_uri") or "").strip()
    if data_uri:
        return "https://mast.stsci.edu/api/v0.1/Download/file?uri=" + urllib.parse.quote(data_uri, safe="")
    return ""


def pick_mast_fits_product(products: pd.DataFrame, max_product_mb: float) -> Optional[Dict[str, Any]]:
    if products is None or products.empty:
        return None
    candidates = products.copy()
    filename_col = next((c for c in ("productFilename", "filename", "File") if c in candidates.columns), None)
    if filename_col:
        candidates = candidates[candidates[filename_col].astype(str).str.contains(r"\.fits?(\.gz)?$", case=False, regex=True, na=False)]
    type_col = next((c for c in ("productType", "product_type") if c in candidates.columns), None)
    if type_col:
        science = candidates[candidates[type_col].astype(str).str.upper().str.contains("SCIENCE", na=False)]
        if not science.empty:
            candidates = science
    size_col = next((c for c in ("size_mb", "Size (MB)", "productSize", "size") if c in candidates.columns), None)
    if size_col:
        sizes = pd.to_numeric(candidates[size_col], errors="coerce")
        if size_col not in {"size_mb", "Size (MB)"}:
            sizes = sizes / (1024 * 1024)
        under = candidates[(sizes.isna()) | (sizes <= float(max_product_mb))]
        if not under.empty:
            candidates = under
    for _, product in candidates.iterrows():
        row = product.to_dict()
        url = mast_product_access_url(row)
        if url:
            row["access_url"] = url
            return row
    return None


def find_alma_overlay_product(
    ra_deg: float,
    dec_deg: float,
    radius_arcmin: float,
    max_product_mb: float,
    *,
    search_service: Any,
    datalink_client: Any,
) -> Optional[Dict[str, Any]]:
    radius_deg = max(float(radius_arcmin or 1.0), 0.1) / 60.0
    query = f"""
SELECT TOP 100
       target_name, proposal_id, member_ous_uid, s_ra, s_dec, band_list,
       dataproduct_type, s_resolution, frequency, bandwidth
FROM ivoa.obscore
WHERE CONTAINS(POINT('ICRS', s_ra, s_dec), CIRCLE('ICRS', {ra_deg:.8f}, {dec_deg:.8f}, {radius_deg:.8f})) = 1
ORDER BY s_resolution
"""
    service = search_service.alminer_client._get_tap_service()
    result = service.search(query)
    alma_df = result.to_table().to_pandas()
    if hasattr(search_service.alminer_client, "_standardize_columns"):
        alma_df = search_service.alminer_client._standardize_columns(alma_df)
    for mous_uid in unique_values(alma_df, ["member_ous_uid"], limit=8):
        listing = datalink_client.list_files(mous_uid=mous_uid)
        if not listing.get("success"):
            continue
        files = sorted(listing.get("files", []), key=product_rank)
        for file_info in files:
            size_mb = float(file_info.get("size_mb") or 0)
            if size_mb and size_mb > float(max_product_mb):
                continue
            if is_fits_product(file_info) and file_info.get("access_url"):
                selected = dict(file_info)
                selected["member_ous_uid"] = mous_uid
                return selected
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Capabilities
# ─────────────────────────────────────────────────────────────────────────────
class _In(BaseModel):
    model_config = ConfigDict(extra="ignore")


class GetSkyImageInput(_In):
    target_name: Optional[str] = None
    survey: Optional[str] = "dss2"
    radius_arcmin: Optional[float] = 5.0
    ra: Optional[float] = None
    dec: Optional[float] = None


class GetSkyImage(BaseCapability):
    name = "get_sky_image"
    description = (
        "Fetch a sky survey cutout image for a target. Returns a FITS file "
        "and PNG preview from surveys like DSS2 (optical), 2MASS (near-IR), "
        "SDSS (optical), WISE (mid-IR), NVSS/FIRST (radio). "
        "Use this when users ask for 'an image of', 'show me', 'DSS image', "
        "or 'what does X look like'."
    )
    category = "general"
    InputModel = GetSkyImageInput
    annotations = {"read_only": False, "cost": "network"}

    def run(self, inp, ctx) -> ToolResult:
        set_lrr = ctx.service("set_last_run_result")
        target_name, survey, radius_arcmin = inp.target_name, inp.survey, inp.radius_arcmin
        ra, dec = inp.ra, inp.dec
        try:
            skyview_client = ctx.services.get("skyview_client")
            result = skyview_client.get_image(
                target=target_name, survey=survey,
                ra=ra, dec=dec,
                radius_arcmin=radius_arcmin, save=True
            )

            if not result.get("success"):
                return _native(result)

            # If we have a preview PNG, set it as the run result for UI display.
            # NB: the SSE layer emits image cards from the `image_url` key ONLY, and the
            # preview lives in ~/quasar_data (not web-served) — so copy it into the served
            # /plots dir and reference that URL (base64 data-URI as a fallback). The old
            # `image_path` key was silently ignored and SkyView images never displayed.
            if result.get("preview_path"):
                image_url = None
                try:
                    import shutil
                    from services.plotting import PLOT_OUTPUT_DIR
                    os.makedirs(PLOT_OUTPUT_DIR, exist_ok=True)
                    dest_name = f"skyview_{uuid.uuid4().hex[:10]}.png"
                    shutil.copyfile(result["preview_path"], os.path.join(PLOT_OUTPUT_DIR, dest_name))
                    image_url = f"/plots/{dest_name}"
                except Exception:
                    try:
                        import base64 as _b64
                        with open(result["preview_path"], "rb") as _f:
                            image_url = "data:image/png;base64," + _b64.b64encode(_f.read()).decode()
                    except Exception:
                        image_url = None
                if image_url:
                    set_lrr({
                        "type": "image",
                        "image_url": image_url,
                        "caption": f"SkyView {result.get('survey', 'DSS2')}: {target_name or f'RA={ra}, Dec={dec}'}",
                        "source": f"SkyView ({result.get('survey', 'DSS2')})",
                        "tool_name": "get_sky_image"
                    })

            label = target_name or f"RA={ra:.3f}, Dec={dec:.3f}"
            return _native({
                "success": True,
                "target": label,
                "survey": result.get("survey"),
                "fits_path": result.get("fits_path", ""),
                "preview_path": result.get("preview_path", ""),
                "image_shape": result.get("image_shape"),
                "note": (
                    f"Fetched {result.get('survey')} image for {label}. "
                    f"FITS saved to: {result.get('fits_path', 'N/A')}. "
                    f"Preview shown in UI."
                )
            })
        except Exception as e:
            return _native({"success": False, "error": f"Sky image fetch failed: {str(e)}"})


class RenderFitsImageInput(_In):
    url: Optional[str]
    title: Optional[str] = ""
    colormap: Optional[str] = "inferno"
    stretch: Optional[str] = "sqrt"


class RenderFitsImage(BaseCapability):
    name = "render_fits_image"
    description = (
        "Download a FITS file from an archive URL and render it as a "
        "publication-quality image displayed inline in the chat. Use this "
        "when the user asks to SEE or VISUALIZE data. The URL should come "
        "from a prior archive search (access_url or datalink URL)."
    )
    category = "analysis"
    InputModel = RenderFitsImageInput
    annotations = {"read_only": False, "cost": "network"}

    def run(self, inp, ctx) -> ToolResult:
        set_lrr = ctx.service("set_last_run_result")
        url, title, colormap, stretch = inp.url, inp.title, inp.colormap, inp.stretch
        try:
            from services.fits_service import render_fits_image
            result = render_fits_image(url, title=title, colormap=colormap, stretch=stretch)
            if result.get("success"):
                set_lrr({
                    "type": "image",
                    "image_url": result["image_path"],
                    "caption": result.get("caption", ""),
                })
            return _native(result)
        except Exception as e:
            return _native({"success": False, "error": str(e)})


class OverlayFitsImagesInput(_In):
    base_url: Optional[str]
    contour_url: Optional[str]
    base_label: Optional[str] = "JWST"
    contour_label: Optional[str] = "ALMA"
    base_cmap: Optional[str] = "inferno"
    contour_levels: Optional[int] = 8


class OverlayFitsImages(BaseCapability):
    name = "overlay_fits_images"
    description = (
        "Download two FITS files and create an overlay composite: one rendered "
        "as a colorscale background, the other as contours on top. Uses WCS "
        "reprojection to align them. Perfect for showing ALMA contours on "
        "JWST/HST colorscale images."
    )
    category = "analysis"
    InputModel = OverlayFitsImagesInput
    annotations = {"read_only": False, "cost": "network"}

    def run(self, inp, ctx) -> ToolResult:
        set_lrr = ctx.service("set_last_run_result")
        base_url, contour_url = inp.base_url, inp.contour_url
        base_label, contour_label = inp.base_label, inp.contour_label
        base_cmap, contour_levels = inp.base_cmap, inp.contour_levels
        try:
            from services.fits_service import overlay_fits_images
            result = overlay_fits_images(
                base_url, contour_url,
                base_label=base_label, contour_label=contour_label,
                base_cmap=base_cmap, contour_levels=contour_levels,
            )
            if result.get("success"):
                set_lrr({
                    "type": "image",
                    "image_url": result["image_path"],
                    "caption": result.get("caption", ""),
                })
            return _native(result)
        except Exception as e:
            return _native({"success": False, "error": str(e)})


class OverlayArchiveImagesInput(_In):
    region: Optional[str]
    ra_deg: Optional[float] = None
    dec_deg: Optional[float] = None
    base_archive: Optional[str] = "MAST"
    base_collection: Optional[str] = "JWST"
    contour_archive: Optional[str] = "ALMA"
    radius_arcmin: Optional[float] = 1.0
    max_product_mb: Optional[float] = 150.0


class OverlayArchiveImages(BaseCapability):
    name = "overlay_archive_images"
    description = (
        "End-to-end archive image overlay workflow. Queries MAST/JWST for a "
        "background FITS image and ALMA/DataLink for contour FITS near a named "
        "region, WCS-aligns them, and renders a PNG. Use for requests like "
        "'Overlay ALMA contours on JWST image for HUDF'."
    )
    category = "analysis"
    InputModel = OverlayArchiveImagesInput
    annotations = {"read_only": False, "cost": "network"}

    def run(self, inp, ctx) -> ToolResult:
        set_lrr = ctx.service("set_last_run_result")
        region, ra_deg, dec_deg = inp.region, inp.ra_deg, inp.dec_deg
        base_archive, base_collection = inp.base_archive, inp.base_collection
        contour_archive = inp.contour_archive
        radius_arcmin, max_product_mb = inp.radius_arcmin, inp.max_product_mb

        coords = overlay_region_coordinates(region, ra_deg=ra_deg, dec_deg=dec_deg)
        if not coords:
            return _native({"success": False, "error": f"Could not resolve overlay region: {region}. Provide ra_deg and dec_deg for arbitrary regions."})
        if str(base_archive or "MAST").upper() != "MAST" or str(contour_archive or "ALMA").upper() != "ALMA":
            return _native({"success": False, "error": "overlay_archive_images currently supports MAST/JWST base images with ALMA contours."})

        ra_deg, dec_deg, label = coords
        try:
            mast_client = ctx.services.get("mast_client")
            mast_obs = mast_client.search_by_position(
                ra_deg,
                dec_deg,
                radius_arcmin=float(radius_arcmin or 1.0),
                mission=base_collection or "JWST",
                max_results=80,
            )
            if mast_obs is None or mast_obs.empty:
                return _native({"success": False, "error": f"No {base_collection or 'JWST'} MAST observations found near {label}."})
            mast_products = mast_client.get_product_list(mast_obs, productType="SCIENCE", extension="fits")
            base_product = pick_mast_fits_product(mast_products, max_product_mb=max_product_mb)
            if not base_product:
                return _native({"success": False, "error": f"No small science FITS product found in MAST near {label}."})

            contour_product = find_alma_overlay_product(
                ra_deg, dec_deg, float(radius_arcmin or 1.0), max_product_mb,
                search_service=ctx.services.get("search_service"),
                datalink_client=ctx.services.get("datalink_client"),
            )
            if not contour_product:
                return _native({"success": False, "error": f"No small public ALMA FITS product found near {label}."})

            from services.fits_service import overlay_fits_images
            result = overlay_fits_images(
                base_product["access_url"],
                contour_product["access_url"],
                base_label=f"{base_collection or 'JWST'} {label}",
                contour_label=f"ALMA {label}",
                base_cmap="inferno",
                contour_levels=8,
            )
            result["region"] = label
            result["selected_products"] = {
                "base": {
                    "filename": base_product.get("productFilename") or base_product.get("filename"),
                    "access_url": base_product.get("access_url"),
                },
                "contour": {
                    "filename": contour_product.get("filename"),
                    "access_url": contour_product.get("access_url"),
                    "member_ous_uid": contour_product.get("member_ous_uid"),
                },
            }
            if result.get("success"):
                set_lrr({
                    "type": "image",
                    "image_url": result["image_path"],
                    "caption": result.get("caption", ""),
                })
            return _native(result)
        except Exception as e:
            return _native({"success": False, "error": str(e), "region": region})


class ComputeMomentMapInput(_In):
    url: Optional[str]
    order: Optional[int] = 0
    title: Optional[str] = ""
    colormap: Optional[str] = "inferno"
    freq_min_ghz: Optional[float] = None
    freq_max_ghz: Optional[float] = None


class ComputeMomentMap(BaseCapability):
    name = "compute_moment_map"
    description = (
        "Download a FITS spectral cube and compute a moment map. "
        "Moment 0 = integrated intensity (total emission). "
        "Moment 1 = velocity field (mean velocity). "
        "Moment 2 = velocity dispersion (turbulence). "
        "Use this for ALMA cubes when the user asks about emission maps, "
        "velocity fields, or line intensity maps."
    )
    category = "analysis"
    InputModel = ComputeMomentMapInput
    annotations = {"read_only": False, "cost": "network"}

    def run(self, inp, ctx) -> ToolResult:
        set_lrr = ctx.service("set_last_run_result")
        url, order, title, colormap = inp.url, inp.order, inp.title, inp.colormap
        freq_min_ghz, freq_max_ghz = inp.freq_min_ghz, inp.freq_max_ghz
        try:
            from services.fits_service import compute_moment_map
            result = compute_moment_map(
                url, order=order, title=title, colormap=colormap,
                freq_min_ghz=freq_min_ghz, freq_max_ghz=freq_max_ghz,
            )
            if result.get("success"):
                set_lrr({
                    "type": "image",
                    "image_url": result["image_path"],
                    "caption": result.get("caption", ""),
                })
            return _native(result)
        except Exception as e:
            return _native({"success": False, "error": str(e)})


class ExtractSpectrumInput(_In):
    url: Optional[str]
    ra_deg: Optional[float] = None
    dec_deg: Optional[float] = None
    x_pixel: Optional[int] = None
    y_pixel: Optional[int] = None
    title: Optional[str] = ""


class ExtractSpectrum(BaseCapability):
    name = "extract_spectrum"
    description = (
        "Download a FITS spectral cube and extract a 1D spectrum at a given "
        "sky position (RA/Dec) or pixel coordinate. If no position is given, "
        "extracts at the peak emission pixel. The spectrum is plotted as "
        "flux vs frequency/velocity and displayed inline."
    )
    category = "analysis"
    InputModel = ExtractSpectrumInput
    annotations = {"read_only": False, "cost": "network"}

    def run(self, inp, ctx) -> ToolResult:
        set_lrr = ctx.service("set_last_run_result")
        url, ra_deg, dec_deg = inp.url, inp.ra_deg, inp.dec_deg
        x_pixel, y_pixel, title = inp.x_pixel, inp.y_pixel, inp.title
        try:
            from services.fits_service import extract_spectrum
            result = extract_spectrum(
                url, ra_deg=ra_deg, dec_deg=dec_deg,
                x_pixel=x_pixel, y_pixel=y_pixel, title=title,
            )
            if result.get("success"):
                set_lrr({
                    "type": "image",
                    "image_url": result["image_path"],
                    "caption": result.get("caption", ""),
                })
            return _native(result)
        except Exception as e:
            return _native({"success": False, "error": str(e)})


class FitSpectralLineInput(_In):
    url: Optional[str]
    ra_deg: Optional[float] = None
    dec_deg: Optional[float] = None
    x_pixel: Optional[int] = None
    y_pixel: Optional[int] = None
    title: Optional[str] = ""


class FitSpectralLine(BaseCapability):
    name = "fit_spectral_line"
    description = (
        "Download a FITS spectral cube, extract a 1D spectrum at a given "
        "position, and fit a Gaussian profile to the strongest line. "
        "Returns peak flux, FWHM (in frequency and velocity), center "
        "frequency, and integrated flux. The fit is overlaid on the "
        "spectrum plot."
    )
    category = "analysis"
    InputModel = FitSpectralLineInput
    annotations = {"read_only": False, "cost": "network"}

    def run(self, inp, ctx) -> ToolResult:
        set_lrr = ctx.service("set_last_run_result")
        url, ra_deg, dec_deg = inp.url, inp.ra_deg, inp.dec_deg
        x_pixel, y_pixel, title = inp.x_pixel, inp.y_pixel, inp.title
        try:
            from services.fits_service import fit_spectral_line
            result = fit_spectral_line(
                url, ra_deg=ra_deg, dec_deg=dec_deg,
                x_pixel=x_pixel, y_pixel=y_pixel, title=title,
            )
            if result.get("success"):
                set_lrr({
                    "type": "image",
                    "image_url": result["image_path"],
                    "caption": result.get("caption", ""),
                })
            return _native(result)
        except Exception as e:
            return _native({"success": False, "error": str(e)})


class GenerateFindingChartInput(_In):
    target: Optional[str] = None
    ra: Optional[float] = None
    dec: Optional[float] = None
    survey: Optional[str] = "DSS2 Red"
    fov_arcmin: Optional[float] = 5.0
    title: Optional[str] = None


class GenerateFindingChart(BaseCapability):
    name = "generate_finding_chart"
    description = (
        "Generate a publication-quality finding chart for a target. "
        "Creates a DSS2 or 2MASS image with WCS axes, a target "
        "crosshair marker, N/E compass arrows, and an angular scale "
        "bar. Use when the user needs a finding chart for observations "
        "or proposals."
    )
    category = "analysis"
    InputModel = GenerateFindingChartInput
    annotations = {"read_only": False, "cost": "network"}

    def run(self, inp, ctx) -> ToolResult:
        set_lrr = ctx.service("set_last_run_result")
        target, ra, dec = inp.target, inp.ra, inp.dec
        survey, fov_arcmin, title = inp.survey, inp.fov_arcmin, inp.title
        try:
            skyview_client = ctx.services.get("skyview_client")
            result = skyview_client.generate_finding_chart(
                target=target, ra=ra, dec=dec,
                survey=survey, fov_arcmin=fov_arcmin, title=title,
            )
            if result.get("success"):
                set_lrr({
                    "type": "image",
                    "image_url": result["image_path"],
                    "caption": result.get("caption", ""),
                })
            return _native(result)
        except Exception as e:
            return _native({"success": False, "error": str(e)})


CAPABILITIES: List[BaseCapability] = [
    GetSkyImage(),
    RenderFitsImage(),
    OverlayFitsImages(),
    OverlayArchiveImages(),
    ComputeMomentMap(),
    ExtractSpectrum(),
    FitSpectralLine(),
    GenerateFindingChart(),
]

__all__ = [
    "CAPABILITIES",
    "overlay_region_coordinates", "mast_product_access_url",
    "pick_mast_fits_product", "find_alma_overlay_product",
    "GetSkyImage", "RenderFitsImage", "OverlayFitsImages",
    "OverlayArchiveImages", "ComputeMomentMap", "ExtractSpectrum",
    "FitSpectralLine", "GenerateFindingChart",
]
