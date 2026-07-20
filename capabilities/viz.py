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
    radius_arcsec: Optional[float] = None


class ExtractSpectrum(BaseCapability):
    name = "extract_spectrum"
    description = (
        "Download a FITS spectral cube and extract a 1D spectrum at a given "
        "sky position (RA/Dec) or pixel coordinate. If no position is given, "
        "extracts at the peak emission pixel. Pass radius_arcsec for an "
        "APERTURE-INTEGRATED spectrum (converted Jy/beam→Jy when the cube "
        "has a beam) — single-pixel spectra underestimate resolved sources. "
        "The spectrum is plotted as flux vs frequency/velocity and displayed "
        "inline."
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
                radius_arcsec=inp.radius_arcsec,
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


# ─────────────────────────────────────────────────────────────────────────────
# Quantitative image analysis + hips2fits FITS-mode products (2026-07)
#
# These capabilities construct their (stateless, env-configured) services
# directly instead of threading them through the agent CallContext — the viz
# ctx provider stays untouched and tests can monkeypatch the service classes.
# ─────────────────────────────────────────────────────────────────────────────
def _resolve_capability_coords(
    target_name: Optional[str],
    ra: Optional[float],
    dec: Optional[float],
) -> Tuple[Optional[float], Optional[float], str]:
    """(ra, dec, label) from explicit coords or a resolvable target name."""
    if ra is not None and dec is not None:
        return float(ra), float(dec), f"RA={float(ra):.5f}, Dec={float(dec):.5f}"
    if target_name:
        coords = overlay_region_coordinates(str(target_name))
        if coords:
            return coords[0], coords[1], coords[2]
        raise ValueError(f"Could not resolve target {target_name!r}; provide ra/dec.")
    return None, None, ""


class _AnalysisInput(_In):
    url: Optional[str] = None
    survey: Optional[str] = None
    target_name: Optional[str] = None
    ra: Optional[float] = None
    dec: Optional[float] = None
    fov_deg: Optional[float] = None
    width: Optional[int] = None
    title: Optional[str] = ""


class _AnalysisCapability(BaseCapability):
    """Shared run() for image-analysis tools: resolve position, call the
    service function, attach the rendered card."""

    category = "analysis"
    annotations = {"read_only": False, "cost": "network"}
    service_fn_name = ""  # set per subclass
    extra_fields: Tuple[str, ...] = ()

    def run(self, inp, ctx) -> ToolResult:
        set_lrr = ctx.service("set_last_run_result")
        try:
            import services.image_analysis as image_analysis

            # A direct FITS url wins outright — never let an unresolvable
            # optional target_name fail a valid url request (CX-06). Only
            # resolve coordinates when there is no url.
            if inp.url:
                ra, dec, label = None, None, ""
            else:
                ra, dec, label = _resolve_capability_coords(inp.target_name, inp.ra, inp.dec)
                if ra is None or dec is None:
                    return _native({"success": False,
                                    "error": "Provide a FITS url, or a survey + target_name/ra/dec cutout position."})
            kwargs: Dict[str, Any] = {
                "url": inp.url,
                "survey": inp.survey or ("optical" if inp.url is None else None),
                "ra": ra,
                "dec": dec,
                "fov_deg": inp.fov_deg,
                "width": inp.width,
                "title": inp.title or (label if label else ""),
            }
            for field in self.extra_fields:
                kwargs[field] = getattr(inp, field)
            result = getattr(image_analysis, self.service_fn_name)(**kwargs)
            if result.get("success") and result.get("image_path"):
                set_lrr({
                    "type": "image",
                    "image_url": result["image_path"],
                    "caption": result.get("caption", ""),
                })
            return _native(result)
        except Exception as e:
            return _native({"success": False, "error": str(e)})


class ImageStatisticsInput(_AnalysisInput):
    pass


class ImageStatistics(_AnalysisCapability):
    name = "image_statistics"
    description = (
        "Measure sigma-clipped statistics, robust MAD noise, a 5-sigma "
        "point-source limit, coverage/blankness fractions, and a pixel-value "
        "histogram for any FITS image (archive URL) or hips2fits survey cutout "
        "(survey + position). Use before deeper analysis to check 'is my "
        "source detectable here' or 'is this cutout actually blank'."
    )
    InputModel = ImageStatisticsInput
    service_fn_name = "image_statistics"


class DetectSourcesInput(_AnalysisInput):
    threshold_sigma: Optional[float] = 5.0
    fwhm_arcsec: Optional[float] = None
    max_sources: Optional[int] = 100


class DetectSources(_AnalysisCapability):
    name = "detect_sources"
    description = (
        "Detect sources in a FITS image or survey cutout (photutils "
        "DAOStarFinder with segmentation fallback) and measure aperture "
        "photometry with local annulus background subtraction. Returns an "
        "annotated detection image plus a source list (positions, RA/Dec, "
        "peak, aperture flux, SNR; Jy for Jy/beam radio maps). Answers 'how "
        "many sources are in this field and how bright are they'."
    )
    InputModel = DetectSourcesInput
    service_fn_name = "detect_and_measure_sources"
    extra_fields = ("threshold_sigma", "fwhm_arcsec", "max_sources")


class MeasureRegionInput(_AnalysisInput):
    region: Optional[str] = None
    radius_arcsec: Optional[float] = None


class MeasureRegion(_AnalysisCapability):
    name = "measure_region"
    description = (
        "Measure statistics inside a sky region on a FITS image or survey "
        "cutout: sum, mean/median, MAD RMS, area in arcsec^2, and — for "
        "Jy/beam maps with a restoring beam — integrated flux density in Jy. "
        "Accepts a DS9 region string (circle/ellipse/box/polygon, e.g. "
        "'circle(150.1d, 2.2d, 30\")') or ra/dec + radius_arcsec for a cone. "
        "Renders the region outline on the image."
    )
    InputModel = MeasureRegionInput
    service_fn_name = "measure_region"
    extra_fields = ("region", "radius_arcsec")


class FitGaussianSourceInput(_AnalysisInput):
    x_pixel: Optional[float] = None
    y_pixel: Optional[float] = None
    box_arcsec: Optional[float] = None


class FitGaussianSource(_AnalysisCapability):
    name = "fit_gaussian_source"
    description = (
        "Fit a 2D Gaussian to a source in a FITS image (the CASA imfit "
        "workflow): peak, integrated flux (Jy for Jy/beam maps), fitted "
        "FWHM sizes and position angle, and — when the header carries a "
        "restoring beam — the beam-DECONVOLVED size or a 'consistent with "
        "point source' verdict. Renders a data/model/residual panel. "
        "Defaults to the peak pixel if no position is given."
    )
    InputModel = FitGaussianSourceInput
    service_fn_name = "fit_gaussian_source"
    extra_fields = ("x_pixel", "y_pixel", "box_arcsec")


class RadialProfileInput(_AnalysisInput):
    x_pixel: Optional[float] = None
    y_pixel: Optional[float] = None
    max_radius_arcsec: Optional[float] = None


class RadialProfileCap(_AnalysisCapability):
    name = "radial_profile"
    description = (
        "Compute an azimuthally averaged radial profile and curve of growth "
        "at a position in a FITS image or survey cutout: FWHM, half-light "
        "radius, asymptotic (total) flux, and a convergence check, with the "
        "restoring-beam HWHM marked for radio maps. The standard "
        "extended-vs-point-source and total-flux diagnostic."
    )
    InputModel = RadialProfileInput
    service_fn_name = "radial_profile"
    extra_fields = ("x_pixel", "y_pixel", "max_radius_arcsec")


class HipsAperturePhotometryInput(_In):
    target_name: Optional[str] = None
    ra: Optional[float] = None
    dec: Optional[float] = None
    bands: Optional[List[str]] = None
    radius_arcsec: Optional[float] = None
    fov_deg: Optional[float] = None
    width: Optional[int] = None
    include_known_bad: Optional[bool] = False
    title: Optional[str] = ""


class HipsAperturePhotometry(BaseCapability):
    name = "hips_aperture_photometry"
    description = (
        "Multi-band aperture photometry from HiPS survey cutouts (R5): place a "
        "circular aperture with a local background annulus on hips2fits FITS "
        "cutouts of one position across several bands (default GALEX FUV/NUV + "
        "SDSS g/r/i; also pacs160/spire250/spire350/spire500, or any HiPS ID). "
        "Sums are reported in native map units with a per-survey validation "
        "status (Giordano et al. 2025 validated 9 maps; PACS 100um is known-bad "
        "and skipped) and a mandatory ~10%-accuracy caveat. For quick multi-"
        "wavelength SED-shaped checks — NOT publication-grade photometry."
    )
    category = "analysis"
    InputModel = HipsAperturePhotometryInput
    annotations = {"read_only": False, "cost": "network"}

    # Minimal fallback if services.image_analysis itself fails to import —
    # the caveat must survive even that error path (verify CX-03).
    _CAVEAT_FALLBACK = (
        "HiPS aperture photometry is suitable ONLY for work that does not "
        "require better than ~10% flux accuracy."
    )

    @classmethod
    def _caveat(cls) -> str:
        try:
            from services.image_analysis import HIPS_PHOTOMETRY_CAVEAT
            return HIPS_PHOTOMETRY_CAVEAT
        except Exception:
            return cls._CAVEAT_FALLBACK

    def run(self, inp, ctx) -> ToolResult:
        set_lrr = ctx.service("set_last_run_result")
        # CX-03: the ~10% caveat is mandatory on EVERY response shape this
        # tool can produce — capability-level errors included.
        try:
            import services.image_analysis as image_analysis

            ra, dec, label = _resolve_capability_coords(inp.target_name, inp.ra, inp.dec)
            if ra is None or dec is None:
                return _native({"success": False,
                                "error": "Provide target_name or ra/dec for the aperture center.",
                                "accuracy_caveat": self._caveat()})
            result = image_analysis.hips_aperture_photometry(
                ra=ra, dec=dec,
                bands=inp.bands,
                radius_arcsec=inp.radius_arcsec,
                fov_deg=inp.fov_deg,
                width=inp.width,
                include_known_bad=bool(inp.include_known_bad),
                title=inp.title or (label if label else ""),
            )
            if result.get("success") and result.get("image_path"):
                set_lrr({
                    "type": "image",
                    "image_url": result["image_path"],
                    "caption": result.get("caption", ""),
                })
            return _native(result)
        except Exception as e:
            return _native({"success": False, "error": str(e),
                            "accuracy_caveat": self._caveat()})


class HipsContourOverlayInput(_In):
    base_survey: Optional[str] = "optical"
    contour_survey: Optional[str] = "vlass"
    target_name: Optional[str] = None
    ra: Optional[float] = None
    dec: Optional[float] = None
    fov_deg: Optional[float] = 0.25
    width: Optional[int] = 512
    contour_levels: Optional[int] = 8
    base_cmap: Optional[str] = "inferno"


class HipsContourOverlay(BaseCapability):
    name = "hips_contour_overlay"
    description = (
        "Overlay one survey as CONTOURS on another survey's image at any sky "
        "position via hips2fits calibrated FITS cutouts — e.g. VLASS radio "
        "contours on DSS2 optical, or WISE on SDSS. Works for any of the "
        "~1000 HiPS surveys (aliases: optical/dss2, sdss, 2mass, wise, galex, "
        "xray, vlass/radio, or raw HiPS IDs). The classic multiwavelength "
        "proposal figure, without needing archive FITS products."
    )
    category = "analysis"
    InputModel = HipsContourOverlayInput
    annotations = {"read_only": False, "cost": "network"}

    def run(self, inp, ctx) -> ToolResult:
        set_lrr = ctx.service("set_last_run_result")
        try:
            from services.fits_service import overlay_fits_images
            from services.hips_images import HipsImageService, resolve_survey

            ra, dec, label = _resolve_capability_coords(inp.target_name, inp.ra, inp.dec)
            if ra is None or dec is None:
                return _native({"success": False, "error": "Provide target_name or ra/dec."})
            svc = HipsImageService()
            # Clamp the FoV the same way fits_url does, so the interactive card's
            # meta advertises the FoV actually rendered, not the raw request (CX-08).
            from services.hips_images import _normalize_fov_width
            clamped_fov, _w, _fov_warn = _normalize_fov_width(inp.fov_deg, inp.width)
            base_url = svc.fits_url(ra, dec, fov_deg=clamped_fov, survey=inp.base_survey, width=inp.width)
            contour_url = svc.fits_url(ra, dec, fov_deg=clamped_fov, survey=inp.contour_survey, width=inp.width)
            result = overlay_fits_images(
                base_url, contour_url,
                base_label=f"{inp.base_survey} {label}".strip(),
                contour_label=str(inp.contour_survey),
                base_cmap=inp.base_cmap or "inferno",
                contour_levels=int(inp.contour_levels or 8),
            )
            if result.get("success"):
                result["surveys"] = {
                    "base": resolve_survey(inp.base_survey),
                    "contours": resolve_survey(inp.contour_survey),
                }
                result["ra"], result["dec"], result["fov_deg"] = ra, dec, clamped_fov
                if _fov_warn:
                    result["warnings"] = list(result.get("warnings") or []) + _fov_warn
                set_lrr({
                    "type": "image",
                    "image_url": result["image_path"],
                    "caption": result.get("caption", ""),
                    "meta": {"kind": "hips", "ra": ra, "dec": dec,
                             "fov_deg": clamped_fov, "survey": resolve_survey(inp.base_survey)},
                })
            return _native(result)
        except Exception as e:
            return _native({"success": False, "error": str(e)})


class ImageDifferenceInput(_In):
    url_a: Optional[str] = None
    url_b: Optional[str] = None
    survey_a: Optional[str] = None
    survey_b: Optional[str] = None
    target_name: Optional[str] = None
    ra: Optional[float] = None
    dec: Optional[float] = None
    fov_deg: Optional[float] = 0.25
    width: Optional[int] = 512
    label_a: Optional[str] = "Image A"
    label_b: Optional[str] = "Image B"
    scale_match: Optional[bool] = True
    title: Optional[str] = ""


class ImageDifference(BaseCapability):
    name = "image_difference"
    description = (
        "WCS-align two FITS images (reproject B onto A) with robust "
        "background/gain matching, subtract them, and render A, aligned B, "
        "and the A−B residual with residual statistics (MAD RMS, max "
        "|residual|, significant-pixel fraction). Accepts two FITS urls OR "
        "two survey aliases + one position (hips2fits FITS cutouts). Use for "
        "epoch-to-epoch transient checks and survey comparisons."
    )
    category = "analysis"
    InputModel = ImageDifferenceInput
    annotations = {"read_only": False, "cost": "network"}

    def run(self, inp, ctx) -> ToolResult:
        set_lrr = ctx.service("set_last_run_result")
        try:
            from services.fits_service import difference_image
            from services.hips_images import HipsImageService

            url_a, url_b = inp.url_a, inp.url_b
            label_a, label_b = inp.label_a or "Image A", inp.label_b or "Image B"
            # A single url is ambiguous — never silently replace the caller's
            # explicit image with a survey-derived one (CX-07).
            if bool(url_a) != bool(url_b):
                return _native({"success": False,
                                "error": "Provide BOTH url_a and url_b, or NEITHER (use survey_a+survey_b "
                                         "with a position instead). A single url is ambiguous."})
            if not (url_a and url_b):
                if not (inp.survey_a and inp.survey_b):
                    return _native({"success": False,
                                    "error": "Provide url_a+url_b, or survey_a+survey_b with a position."})
                from services.hips_images import resolve_survey

                # Two aliases that resolve to the SAME HiPS product (e.g. both
                # "vlass" -> the single median-stack surface) would difference an
                # image against itself and report a misleading ~zero residual.
                # Per-epoch VLASS is not in HiPS — steer to vlass_epoch_comparison.
                if resolve_survey(inp.survey_a) == resolve_survey(inp.survey_b):
                    hint = (" For VLASS epoch-to-epoch differences use vlass_epoch_comparison "
                            "(per-epoch VLASS is not available as a HiPS survey)."
                            if "vlass" in str(inp.survey_a).lower() else "")
                    return _native({"success": False,
                                    "error": f"survey_a and survey_b resolve to the same HiPS product "
                                             f"({resolve_survey(inp.survey_a)}); differencing it against "
                                             f"itself is a no-op.{hint}"})
                ra, dec, _ = _resolve_capability_coords(inp.target_name, inp.ra, inp.dec)
                if ra is None or dec is None:
                    return _native({"success": False, "error": "Provide target_name or ra/dec."})
                svc = HipsImageService()
                url_a = svc.fits_url(ra, dec, fov_deg=inp.fov_deg, survey=inp.survey_a, width=inp.width)
                url_b = svc.fits_url(ra, dec, fov_deg=inp.fov_deg, survey=inp.survey_b, width=inp.width)
                if inp.label_a == "Image A":
                    label_a = str(inp.survey_a)
                if inp.label_b == "Image B":
                    label_b = str(inp.survey_b)
            result = difference_image(
                url_a, url_b, label_a=label_a, label_b=label_b,
                scale_match=bool(inp.scale_match), title=inp.title or "",
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


class HipsRgbCompositeInput(_In):
    surveys: Optional[List[str]] = None
    target_name: Optional[str] = None
    ra: Optional[float] = None
    dec: Optional[float] = None
    fov_deg: Optional[float] = 0.25
    width: Optional[int] = 512
    stretch: Optional[float] = 5.0
    q: Optional[float] = 8.0
    title: Optional[str] = None


class HipsRgbComposite(BaseCapability):
    name = "hips_rgb_composite"
    description = (
        "Build a Lupton three-color RGB composite from any three SINGLE-BAND "
        "HiPS surveys (R, G, B order) at a sky position — e.g. "
        "['2mass_k', '2mass_h', '2mass_j'], ['wise_w3','wise_w2','wise_w1'], "
        "or raw HiPS IDs like 'CDS/P/SDSS9/i'. Use single-band aliases "
        "(2mass_j/h/k, sdss_g/r/i/z, wise_w1..w4, galex_nuv/fuv, "
        "dss2_red/dss2_blue), NOT the color aliases (wise/2mass/sdss), which "
        "are multi-plane. Layers arrive pixel-aligned from hips2fits. Blank "
        "layers are rejected with a clear error."
    )
    category = "analysis"
    InputModel = HipsRgbCompositeInput
    annotations = {"read_only": False, "cost": "network"}

    def run(self, inp, ctx) -> ToolResult:
        set_lrr = ctx.service("set_last_run_result")
        try:
            from services.hips_images import HipsImageService

            ra, dec, label = _resolve_capability_coords(inp.target_name, inp.ra, inp.dec)
            if ra is None or dec is None:
                return _native({"success": False, "error": "Provide target_name or ra/dec."})
            # Default to genuine single-band channels (near-IR J/H/K), not the
            # color HiPS aliases which are 3-plane JPEG surfaces (CX-09).
            surveys = inp.surveys or ["2mass_k", "2mass_h", "2mass_j"]
            result = HipsImageService().rgb_composite(
                ra, dec, surveys, fov_deg=inp.fov_deg, width=inp.width,
                stretch=inp.stretch, q=inp.q, title=inp.title,
            )
            if result.get("success"):
                caption = inp.title or (
                    f"RGB composite ({'/'.join(str(s) for s in surveys)})"
                    + (f": {label}" if label else "")
                )
                image_url = result.get("path") or (
                    f"data:image/png;base64,{result['image_base64']}" if result.get("image_base64") else None
                )
                if image_url:
                    set_lrr({"type": "image", "image_url": image_url, "caption": caption})
                result = {k: v for k, v in result.items() if k not in ("image_base64", "path", "png_path")}
                result["image_attached"] = True
            return _native(result)
        except Exception as e:
            return _native({"success": False, "error": str(e)})


class VlassEpochComparisonInput(_In):
    target_name: Optional[str] = None
    ra: Optional[float] = None
    dec: Optional[float] = None
    radius_arcsec: Optional[float] = 60.0
    max_epochs: Optional[int] = 6
    title: Optional[str] = None


class VlassEpochComparison(BaseCapability):
    name = "vlass_epoch_comparison"
    description = (
        "Compare VLASS 3 GHz radio epochs (2017→now) at a position: fetches "
        "per-epoch Quicklook cutouts from CADC (per-epoch VLASS is not in "
        "HiPS), renders a shared-stretch epoch panel with blinkable frames, "
        "and reports per-epoch peak flux, RMS, and a variability verdict "
        "(with the Quicklook ~15% systematic folded in). Use for radio "
        "transient/variability triage. Dec > -40 only."
    )
    category = "analysis"
    InputModel = VlassEpochComparisonInput
    annotations = {"read_only": False, "cost": "network"}

    def run(self, inp, ctx) -> ToolResult:
        set_lrr = ctx.service("set_last_run_result")
        try:
            from services.vlass_epochs import VlassEpochService

            ra, dec, label = _resolve_capability_coords(inp.target_name, inp.ra, inp.dec)
            if ra is None or dec is None:
                return _native({"success": False, "error": "Provide target_name or ra/dec."})
            result = VlassEpochService().epoch_comparison(
                ra, dec, radius_arcsec=inp.radius_arcsec,
                max_epochs=inp.max_epochs or 6,
                title=inp.title or (f"VLASS epochs: {label}" if label else None),
            )
            if result.get("success") and result.get("path"):
                card: Dict[str, Any] = {
                    "type": "image",
                    "image_url": result["path"],
                    "caption": inp.title or f"VLASS epoch comparison: {label}",
                }
                frames = result.get("frames") or []
                if len(frames) >= 2:
                    card["meta"] = {"kind": "blink", "ra": ra, "dec": dec, "frames": frames}
                    # Registration/coverage caveats must reach the CARD, not
                    # just the LLM-facing payload (CX-18): an unregistered
                    # frame with no on-card note reads as a real transient.
                    if result.get("warnings"):
                        card["meta"]["warnings"] = list(result["warnings"])
                set_lrr(card)
                result = {k: v for k, v in result.items() if k not in ("path", "png_path")}
                result["image_attached"] = True
            return _native(result)
        except Exception as e:
            return _native({"success": False, "error": str(e)})


class MocOperationsInput(_In):
    survey_ids: Optional[List[str]] = None
    operation: Optional[str] = "intersection"
    target_name: Optional[str] = None
    ra: Optional[float] = None
    dec: Optional[float] = None
    fov_deg: Optional[float] = 20.0
    survey: Optional[str] = "optical"
    order: Optional[int] = 8
    ra_list: Optional[List[float]] = None
    dec_list: Optional[List[float]] = None


# Aurora accent palette for MOC overlays (mirrors agent._MOC_OVERLAY_COLORS).
_MOC_COLORS = ["#22d3ee", "#fbbf24", "#34d399", "#a78bfa"]
_DERIVED_MOC_COLOR = "#f472b6"


class MocOperations(BaseCapability):
    name = "moc_operations"
    description = (
        "MOC coverage algebra: intersect/union/difference the sky footprints "
        "of surveys (MOCServer dataset IDs, e.g. 'CDS/P/DES-DR2/g' — get them "
        "from survey_coverage) and report the resulting area in deg^2. "
        "Optionally pass ra_list/dec_list to flag which of your targets fall "
        "inside the derived footprint. Renders the derived MOC (pink) with "
        "the inputs on an interactive sky view. Answers 'where do these "
        "surveys overlap?' and 'which candidates have joint coverage?'."
    )
    category = "archive"
    InputModel = MocOperationsInput
    annotations = {"read_only": False, "cost": "network"}

    def run(self, inp, ctx) -> ToolResult:
        set_lrr = ctx.service("set_last_run_result")
        try:
            from services.hips_images import HipsImageService
            from services.moc_coverage import MocCoverageService

            ids = [str(s) for s in (inp.survey_ids or []) if str(s).strip()]
            if not ids:
                return _native({"success": False,
                                "error": "survey_ids is required (MOCServer dataset IDs; find them with survey_coverage)."})
            moc_service = MocCoverageService()
            result = moc_service.moc_operation(
                ids, operation=inp.operation or "intersection", order=inp.order or 8,
                ra_list=inp.ra_list, dec_list=inp.dec_list,
            )
            if not result.get("success") or result.get("empty"):
                return _native(result)

            # Visualization is best-effort: the algebra result stands alone.
            try:
                ra, dec, label = _resolve_capability_coords(inp.target_name, inp.ra, inp.dec)
                mocs: List[Dict[str, Any]] = [{
                    "id": f"{result['operation']}({', '.join(ids)})",
                    "name": result["operation"],
                    "color": _DERIVED_MOC_COLOR,
                    "moc_json": result["moc_json"],
                }]
                inputs_geometry = moc_service.moc_geometry(ids, order=min(int(inp.order or 8), 7))
                for k, moc in enumerate(inputs_geometry.get("mocs") or []):
                    mocs.append({
                        "id": moc["id"],
                        "name": str(moc["id"]).rsplit("/", 1)[-1] or moc["id"],
                        "color": _MOC_COLORS[k % len(_MOC_COLORS)],
                        "moc_json": moc["moc_json"],
                    })
                if ra is None or dec is None:
                    # Center the view on the derived MOC's first (deepest-order) cell
                    import astropy_healpix as ah
                    import astropy.units as u

                    moc_json = result["moc_json"]
                    order_key = sorted(moc_json.keys(), key=int)[-1]
                    ipix = int(moc_json[order_key][0])
                    hp = ah.HEALPix(nside=2 ** int(order_key), order="nested")
                    lon, lat = hp.healpix_to_lonlat(ipix)
                    ra, dec = float(lon.to_value(u.deg)), float(lat.to_value(u.deg))
                    label = "derived footprint"
                cutout = HipsImageService().cutout(ra, dec, fov_deg=inp.fov_deg or 20.0,
                                                   survey=inp.survey or "optical", width=512)
                if cutout.get("success"):
                    image_url = cutout.get("path") or (
                        f"data:image/png;base64,{cutout['image_base64']}" if cutout.get("image_base64") else None
                    )
                    if image_url:
                        set_lrr({
                            "type": "image",
                            "image_url": image_url,
                            "caption": (f"{result['operation'].capitalize()} of {len(ids)} footprints "
                                        f"({result['area_deg2']:g} deg²) at {label}"),
                            "meta": {
                                "kind": "hips",
                                "ra": ra,
                                "dec": dec,
                                "fov_deg": cutout.get("fov_deg", inp.fov_deg or 20.0),
                                "survey": cutout.get("survey_id") or "CDS/P/DSS2/color",
                                "mocs": mocs,
                            },
                        })
                        result["note"] = ("Open the card's Interactive view to see the derived footprint "
                                          "(pink) and the input MOCs drawn on the sky.")
            except Exception as viz_err:
                logger.warning(f"[MOC-OPS] visualization skipped: {viz_err}")
                result.setdefault("warnings", []).append(f"Sky visualization skipped: {viz_err}")

            # The MOC JSON payload is heavy and UI-only; keep it out of the LLM result.
            result.pop("moc_json", None)
            return _native(result)
        except Exception as e:
            return _native({"success": False, "error": str(e)})


class PvSliceInput(_In):
    url: Optional[str] = None
    ra_start: Optional[float] = None
    dec_start: Optional[float] = None
    ra_end: Optional[float] = None
    dec_end: Optional[float] = None
    width_arcsec: Optional[float] = None
    title: Optional[str] = ""


class PvSlice(BaseCapability):
    name = "pv_slice"
    description = (
        "Extract a position-velocity (PV) diagram from a FITS spectral cube "
        "along an ARBITRARY sky path (pvextractor): give two endpoints "
        "(ra_start/dec_start → ra_end/dec_end, e.g. along a disk major axis "
        "from fit_gaussian_source) and an optional averaging width in arcsec. "
        "The canonical rotation/outflow/infall diagnostic for ALMA/VLA cubes."
    )
    category = "analysis"
    InputModel = PvSliceInput
    annotations = {"read_only": False, "cost": "network"}

    def run(self, inp, ctx) -> ToolResult:
        set_lrr = ctx.service("set_last_run_result")
        try:
            from services.fits_service import pv_slice
            result = pv_slice(
                inp.url,
                ra_start=inp.ra_start, dec_start=inp.dec_start,
                ra_end=inp.ra_end, dec_end=inp.dec_end,
                width_arcsec=inp.width_arcsec, title=inp.title or "",
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
    ImageStatistics(),
    DetectSources(),
    MeasureRegion(),
    FitGaussianSource(),
    RadialProfileCap(),
    HipsAperturePhotometry(),
    HipsContourOverlay(),
    ImageDifference(),
    HipsRgbComposite(),
    VlassEpochComparison(),
    MocOperations(),
    PvSlice(),
]

__all__ = [
    "CAPABILITIES",
    "overlay_region_coordinates", "mast_product_access_url",
    "pick_mast_fits_product", "find_alma_overlay_product",
    "GetSkyImage", "RenderFitsImage", "OverlayFitsImages",
    "OverlayArchiveImages", "ComputeMomentMap", "ExtractSpectrum",
    "FitSpectralLine", "GenerateFindingChart",
]
