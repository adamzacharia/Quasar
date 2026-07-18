"""
Quantitative image analysis on FITS data.

Turns any FITS image Quasar can fetch (archive access_url, hips2fits
format=fits cutout, local path) into measurements instead of pictures:

  - image_statistics: sigma-clipped stats, noise, blank detection, histogram
  - detect_and_measure_sources: photutils detection + aperture photometry
  - measure_region: DS9-region / cone statistics with Jy/beam -> Jy conversion
  - fit_gaussian_source: 2D Gaussian fit with restoring-beam deconvolution
  - radial_profile: azimuthally averaged profile + curve of growth

All functions follow the fits_service conventions: module-level, lazy heavy
imports, temp download with cleanup, and a
{"success": True, "image_path": "/api/images/...", ...} result shape.
"""

from __future__ import annotations

import gc
import logging
import math
import os
import uuid
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from services.fits_service import RENDERED_DIR, _download_fits, _pick_science_hdu

logger = logging.getLogger(__name__)

MAX_SOURCES_CAP = 500
DEFAULT_MAX_SOURCES = 100
FIGURE_FACECOLOR = "#0f172a"


class ImageAnalysisError(ValueError):
    """Raised for user-correctable image-analysis failures."""


# ─────────────────────────────────────────────────────────────────────────────
# Shared loading / geometry helpers
# ─────────────────────────────────────────────────────────────────────────────
def _resolve_input_url(
    url: Optional[str],
    survey: Optional[str],
    ra: Optional[float],
    dec: Optional[float],
    fov_deg: Optional[float],
    width: Optional[int],
) -> str:
    """Accept either a direct FITS url or (survey, ra, dec) hips2fits inputs."""
    if url:
        return str(url)
    if survey is not None and ra is not None and dec is not None:
        from services.hips_images import HipsImageService

        return HipsImageService().fits_url(
            ra, dec, fov_deg=(fov_deg if fov_deg is not None else 0.25),
            survey=survey, width=(width if width is not None else 512),
        )
    raise ImageAnalysisError(
        "Provide either a FITS url, or survey + ra + dec for a hips2fits cutout."
    )


def _load_fits_image(fits_path: str) -> Tuple[np.ndarray, Any, Any, Optional[str]]:
    """Open a downloaded FITS file -> (2D float array, WCS-or-None, header,
    cube-slice disclosure note or None).

    The note is set when the input was a 3D+ cube collapsed to one plane
    (IMG-11) — callers must surface it in their warnings so a single channel
    is never silently presented as "the image".
    """
    from astropy.io import fits as afits
    from astropy.wcs import WCS

    with afits.open(fits_path, memmap=True) as hdul:
        data_2d, hdu_idx, slice_info = _pick_science_hdu(hdul, return_slice_info=True)
        header = hdul[hdu_idx].header.copy()
        try:
            wcs = WCS(header, naxis=2)
            if not wcs.has_celestial:
                wcs = None
        except Exception:
            wcs = None
        data = np.asarray(data_2d, dtype=float)
    return data, wcs, header, (slice_info["note"] if slice_info else None)


def _pixel_scale_arcsec(wcs: Any) -> Optional[float]:
    """Mean absolute pixel scale in arcsec/pixel, or None without WCS."""
    if wcs is None:
        return None
    try:
        from astropy.wcs.utils import proj_plane_pixel_scales

        scales_deg = proj_plane_pixel_scales(wcs)
        return float(np.mean(np.abs(scales_deg[:2])) * 3600.0)
    except Exception:
        return None


def _pixel_area_arcsec2(wcs: Any, pix_arcsec: Optional[float]) -> Optional[float]:
    """True pixel solid angle in arcsec² from the WCS (handles non-square /
    rotated pixels correctly), falling back to mean-scale² without a usable
    proj_plane_pixel_area. Used for areas and beam→pixel conversions (CX-10)."""
    if wcs is not None:
        try:
            from astropy.wcs.utils import proj_plane_pixel_area

            return float(abs(proj_plane_pixel_area(wcs)) * 3600.0 * 3600.0)
        except Exception:
            pass
    return (pix_arcsec ** 2) if (pix_arcsec and pix_arcsec > 0) else None


def _nonsquare_pixel_warning(wcs: Any) -> Optional[str]:
    """Warn when the two axis scales differ enough that the single mean scale
    still used for aperture radii and radial profiles is a poor approximation
    (CX-10; fitted FWHM/PA are now measured on-sky and are exempt).

    hips2fits TAN cutouts are square by construction; this only fires for
    arbitrary archive FITS with anisotropic pixels.
    """
    if wcs is None:
        return None
    try:
        from astropy.wcs.utils import proj_plane_pixel_scales

        scales = np.abs(np.asarray(proj_plane_pixel_scales(wcs)[:2], dtype=float))
        if scales.min() <= 0:
            return None
        ratio = float(scales.max() / scales.min())
        if ratio > 1.02:
            return (f"Pixel scales differ by {(ratio - 1) * 100:.0f}% between axes; "
                    "fitted FWHM/PA are measured on-sky, but aperture radii and "
                    "radial profiles use the mean scale and are approximate for this image.")
    except Exception:
        return None
    return None


def _beam_info(header: Any, pixel_area_arcsec2: Optional[float]) -> Optional[Dict[str, float]]:
    """Restoring-beam geometry from BMAJ/BMIN/BPA, if present.

    ``pixel_area_arcsec2`` is the TRUE per-pixel solid angle (from the WCS), so
    the beam→pixel conversion is correct for non-square/rotated pixels (CX-10).
    """
    try:
        bmaj = header.get("BMAJ")
        bmin = header.get("BMIN")
        if bmaj is None or bmin is None:
            return None
        bmaj_arcsec = float(bmaj) * 3600.0
        bmin_arcsec = float(bmin) * 3600.0
        if not (math.isfinite(bmaj_arcsec) and bmaj_arcsec > 0 and bmin_arcsec > 0):
            return None
        info: Dict[str, float] = {
            "bmaj_arcsec": bmaj_arcsec,
            "bmin_arcsec": bmin_arcsec,
            "bpa_deg": float(header.get("BPA", 0.0) or 0.0),
            # Gaussian beam solid angle: pi/(4 ln 2) * bmaj * bmin
            "beam_area_arcsec2": math.pi / (4.0 * math.log(2.0)) * bmaj_arcsec * bmin_arcsec,
        }
        if pixel_area_arcsec2 and pixel_area_arcsec2 > 0:
            info["beam_area_pix"] = info["beam_area_arcsec2"] / pixel_area_arcsec2
        return info
    except Exception:
        return None


def _bunit(header: Any) -> str:
    return str(header.get("BUNIT", "") or "").strip()


def _is_jy_per_beam(bunit: str) -> bool:
    key = bunit.lower().replace(" ", "")
    return "jy/beam" in key or "jy.beam-1" in key or "jybeam-1" in key


def _sky_to_pixel(wcs: Any, ra: float, dec: float) -> Tuple[float, float]:
    from astropy.coordinates import SkyCoord
    import astropy.units as u

    coord = SkyCoord(ra=float(ra) * u.deg, dec=float(dec) * u.deg, frame="icrs")
    x, y = wcs.world_to_pixel(coord)
    return float(x), float(y)


def _pixel_to_sky(wcs: Any, x: float, y: float) -> Tuple[Optional[float], Optional[float]]:
    try:
        sky = wcs.pixel_to_world(float(x), float(y))
        return float(sky.ra.deg), float(sky.dec.deg)
    except Exception:
        return None, None


def _sky_position_angle(wcs: Any, x: float, y: float, major_angle_pix: float) -> Optional[float]:
    """Position angle (deg East of North, 0-180) of a pixel-frame direction.

    Steps a small amount along the major-axis direction in pixel space,
    converts both endpoints to sky, and takes the bearing — correct for
    rotated/skewed WCS, unlike assuming pixel-y == North (CX-10).
    """
    try:
        from astropy.coordinates import SkyCoord

        step = 2.0
        x2 = x + step * math.cos(major_angle_pix)
        y2 = y + step * math.sin(major_angle_pix)
        c0 = wcs.pixel_to_world(float(x), float(y))
        c1 = wcs.pixel_to_world(float(x2), float(y2))
        if not isinstance(c0, SkyCoord) or not isinstance(c1, SkyCoord):
            return None
        pa = c0.position_angle(c1).to_value("deg")
        return float(pa % 180.0)
    except Exception:
        return None


def _sky_length_arcsec(wcs: Any, x: float, y: float, angle_pix: float,
                       length_pix: float) -> Optional[float]:
    """On-sky length (arcsec) of a pixel-frame segment centred on (x, y).

    Steps ±length_pix/2 along ``angle_pix`` in pixel space, converts both
    endpoints to sky, and returns their separation — exact for anisotropic,
    rotated, or skewed WCS, where multiplying by ONE mean pixel scale
    mis-reports any size along an axis whose scale differs from the mean
    (CX-10: a source elongated along a 2″/pix axis of a 1″×2″ image had its
    FWHM under-reported by 25% by the mean scale).
    """
    try:
        if not (math.isfinite(length_pix) and length_pix > 0):
            return None
        dx = 0.5 * length_pix * math.cos(angle_pix)
        dy = 0.5 * length_pix * math.sin(angle_pix)
        c0 = wcs.pixel_to_world(float(x - dx), float(y - dy))
        c1 = wcs.pixel_to_world(float(x + dx), float(y + dy))
        sep = float(c0.separation(c1).to_value("arcsec"))
        if not math.isfinite(sep) or sep <= 0:
            return None
        return sep
    except Exception:
        return None


def _resolve_position(
    data: np.ndarray,
    wcs: Any,
    ra: Optional[float],
    dec: Optional[float],
    x_pixel: Optional[float],
    y_pixel: Optional[float],
) -> Tuple[float, float, str]:
    """Position -> (x, y, label). Defaults to the peak pixel."""
    if ra is not None and dec is not None:
        if wcs is None:
            raise ImageAnalysisError("Image has no celestial WCS; pass x_pixel/y_pixel instead of ra/dec.")
        x, y = _sky_to_pixel(wcs, ra, dec)
        ny, nx = data.shape
        if not (math.isfinite(x) and math.isfinite(y)) or not (-0.5 <= x < nx - 0.5 and -0.5 <= y < ny - 0.5):
            raise ImageAnalysisError("The requested RA/Dec falls outside this image.")
        return x, y, f"RA={float(ra):.5f}, Dec={float(dec):.5f}"
    if x_pixel is not None and y_pixel is not None:
        return float(x_pixel), float(y_pixel), f"pixel ({float(x_pixel):.1f}, {float(y_pixel):.1f})"
    filled = np.nan_to_num(data, nan=-np.inf, posinf=-np.inf, neginf=-np.inf)
    if not np.isfinite(filled).any() or np.all(filled == -np.inf):
        raise ImageAnalysisError("Image contains no finite pixels.")
    peak = np.unravel_index(int(np.argmax(filled)), filled.shape)
    return float(peak[1]), float(peak[0]), f"peak pixel ({peak[1]}, {peak[0]})"


def _finite_fraction(data: np.ndarray) -> float:
    if data.size == 0:
        return 0.0
    return float(np.isfinite(data).sum()) / float(data.size)


def _blankness(data: np.ndarray) -> Tuple[bool, List[str]]:
    """Degenerate-cutout detection (mirrors datalab_image_service heuristics)."""
    warnings: List[str] = []
    finite = data[np.isfinite(data)]
    frac = _finite_fraction(data)
    if finite.size == 0:
        return True, ["Image is entirely NaN/inf — likely outside the survey footprint."]
    if frac < 0.05:
        warnings.append(f"Only {frac * 100:.1f}% of pixels are finite — mostly outside coverage.")
    if float(np.nanmax(finite)) == float(np.nanmin(finite)):
        return True, warnings + ["Image is constant-valued — likely a blank/no-coverage tile."]
    return (frac < 0.05), warnings


def _save_figure(fig: Any, prefix: str) -> str:
    """Save a matplotlib figure into the served rendered-images dir."""
    import matplotlib.pyplot as plt

    img_name = f"{prefix}_{uuid.uuid4().hex[:10]}.png"
    img_path = os.path.join(RENDERED_DIR, img_name)
    fig.savefig(img_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return f"/api/images/{img_name}"


def _image_axes(fig: Any, wcs: Any):
    """Create the standard dark-theme image axes (WCS projected if possible)."""
    if wcs is not None:
        ax = fig.add_subplot(111, projection=wcs)
        ax.coords[0].set_axislabel("RA (J2000)", color="white", fontsize=10)
        ax.coords[1].set_axislabel("Dec (J2000)", color="white", fontsize=10)
        ax.coords[0].set_ticklabel(color="white", fontsize=8)
        ax.coords[1].set_ticklabel(color="white", fontsize=8)
    else:
        ax = fig.add_subplot(111)
        ax.tick_params(colors="white", labelsize=8)
    ax.set_facecolor("black")
    return ax


def _zscale_norm(data: np.ndarray):
    from astropy.visualization import ImageNormalize, SqrtStretch, ZScaleInterval

    finite = data[np.isfinite(data)]
    return ImageNormalize(finite, interval=ZScaleInterval(), stretch=SqrtStretch())


def _round(value: Any, digits: int = 6) -> Optional[float]:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(f):
        return None
    return round(f, digits)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Image statistics / noise / blankness
# ─────────────────────────────────────────────────────────────────────────────
def image_statistics(
    url: Optional[str] = None,
    survey: Optional[str] = None,
    ra: Optional[float] = None,
    dec: Optional[float] = None,
    fov_deg: Optional[float] = None,
    width: Optional[int] = None,
    title: str = "",
) -> Dict[str, Any]:
    """Sigma-clipped statistics, noise estimate, and a pixel-value histogram."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from astropy.stats import mad_std, sigma_clipped_stats

    fits_path = None
    try:
        resolved = _resolve_input_url(url, survey, ra, dec, fov_deg, width)
        fits_path = _download_fits(resolved, title or "statistics")
        data, wcs, header, cube_note = _load_fits_image(fits_path)

        blank, warnings = _blankness(data)
        if cube_note:
            warnings.append(cube_note)  # IMG-11: disclose the collapsed plane
        finite = data[np.isfinite(data)]
        if finite.size == 0:
            return {"success": False, "error": "Image contains no finite pixels (blank tile).",
                    "blank": True, "warnings": warnings}

        mean, median, std = sigma_clipped_stats(finite, sigma=3.0, maxiters=5)
        robust_rms = float(mad_std(finite))
        pix_arcsec = _pixel_scale_arcsec(wcs)
        _geom_warn = _nonsquare_pixel_warning(wcs)
        if _geom_warn:
            warnings.append(_geom_warn)
        pixel_area_arcsec2 = _pixel_area_arcsec2(wcs, pix_arcsec)
        beam = _beam_info(header, pixel_area_arcsec2)
        bunit = _bunit(header)

        stats = {
            "shape": [int(s) for s in data.shape],
            "bunit": bunit or None,
            "pixel_scale_arcsec": _round(pix_arcsec, 4),
            "finite_fraction": _round(_finite_fraction(data), 4),
            "zero_fraction": _round(float(np.sum(finite == 0.0)) / float(finite.size), 4),
            "min": _round(float(np.min(finite))),
            "max": _round(float(np.max(finite))),
            "mean_clipped": _round(mean),
            "median_clipped": _round(median),
            "std_clipped": _round(std),
            "mad_rms": _round(robust_rms),
            "percentiles": {
                "p01": _round(float(np.percentile(finite, 1))),
                "p50": _round(float(np.percentile(finite, 50))),
                "p99": _round(float(np.percentile(finite, 99))),
            },
        }
        if beam:
            stats["beam"] = {k: _round(v, 4) for k, v in beam.items()}
        # 5-sigma point-source detectability estimate in map units.
        stats["limiting_5sigma"] = _round(5.0 * robust_rms)

        # Histogram over a robust range
        lo, hi = np.percentile(finite, [0.5, 99.5])
        fig, ax = plt.subplots(figsize=(9, 4.5), facecolor=FIGURE_FACECOLOR)
        ax.set_facecolor(FIGURE_FACECOLOR)
        sample = finite if finite.size <= 2_000_000 else np.random.default_rng(0).choice(finite, 2_000_000, replace=False)
        ax.hist(sample[(sample >= lo) & (sample <= hi)], bins=120, color="#06b6d4", alpha=0.85)
        ax.set_yscale("log")
        ax.axvline(median, color="#f59e0b", linewidth=1.2, linestyle="--",
                   label=f"median = {median:.4g}")
        ax.axvline(median + 5 * robust_rms, color="#34d399", linewidth=1.2, linestyle=":",
                   label=f"median + 5×MAD-RMS = {median + 5 * robust_rms:.4g}")
        ax.set_xlabel(f"Pixel value{f' ({bunit})' if bunit else ''}", color="white", fontsize=10)
        ax.set_ylabel("Pixels", color="white", fontsize=10)
        ax.tick_params(colors="white", labelsize=8)
        for spine in ax.spines.values():
            spine.set_color("#334155")
        ax.legend(loc="upper right", fontsize=8, facecolor="#1e293b",
                  edgecolor="#334155", labelcolor="white")
        full_title = f"{title} — Image statistics" if title else "Image statistics"
        ax.set_title(full_title, color="white", fontsize=12, pad=10)

        image_path = _save_figure(fig, "imgstats")
        if blank:
            warnings.append("Image looks blank; treat any measurement from it as unreliable.")
        return {
            "success": True,
            "image_path": image_path,
            "caption": full_title,
            "blank": bool(blank),
            "statistics": stats,
            "warnings": warnings,
        }
    except Exception as e:
        logger.error(f"[IMG-ANALYSIS] image_statistics failed: {e}")
        return {"success": False, "error": str(e)}
    finally:
        gc.collect()  # release memmap handles before unlink (Windows)
        if fits_path and os.path.exists(fits_path):
            try:
                os.unlink(fits_path)
            except PermissionError:
                logger.warning(f"[IMG-ANALYSIS] Could not remove temp file still in use: {fits_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Source detection + aperture photometry
# ─────────────────────────────────────────────────────────────────────────────
def detect_and_measure_sources(
    url: Optional[str] = None,
    survey: Optional[str] = None,
    ra: Optional[float] = None,
    dec: Optional[float] = None,
    fov_deg: Optional[float] = None,
    width: Optional[int] = None,
    threshold_sigma: float = 5.0,
    fwhm_arcsec: Optional[float] = None,
    max_sources: int = DEFAULT_MAX_SOURCES,
    title: str = "",
) -> Dict[str, Any]:
    """Detect point-like sources and measure annulus-subtracted aperture photometry."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from astropy.stats import sigma_clipped_stats
    from photutils.aperture import ApertureStats, CircularAnnulus, CircularAperture
    from photutils.detection import DAOStarFinder
    from photutils.segmentation import SourceCatalog, detect_sources as seg_detect

    fits_path = None
    try:
        threshold_sigma = float(threshold_sigma) if threshold_sigma and float(threshold_sigma) > 0 else 5.0
        max_sources = int(max(1, min(MAX_SOURCES_CAP, int(max_sources or DEFAULT_MAX_SOURCES))))

        resolved = _resolve_input_url(url, survey, ra, dec, fov_deg, width)
        fits_path = _download_fits(resolved, title or "source detection")
        data, wcs, header, cube_note = _load_fits_image(fits_path)

        blank, warnings = _blankness(data)
        if cube_note:
            warnings.append(cube_note)  # IMG-11: disclose the collapsed plane
        if blank:
            return {"success": False, "blank": True,
                    "error": "Image looks blank (all-NaN or constant) — cannot detect sources.",
                    "warnings": warnings}

        pix_arcsec = _pixel_scale_arcsec(wcs)
        _geom_warn = _nonsquare_pixel_warning(wcs)
        if _geom_warn:
            warnings.append(_geom_warn)
        pixel_area_arcsec2 = _pixel_area_arcsec2(wcs, pix_arcsec)
        beam = _beam_info(header, pixel_area_arcsec2)
        bunit = _bunit(header)
        jy_per_beam = _is_jy_per_beam(bunit) and beam is not None and beam.get("beam_area_pix")

        finite = data[np.isfinite(data)]
        _, bkg_median, bkg_std = sigma_clipped_stats(finite, sigma=3.0, maxiters=5)
        if not (math.isfinite(bkg_std) and bkg_std > 0):
            return {"success": False, "error": "Could not estimate a positive background RMS."}

        # Detection kernel width: explicit arg > restoring beam > 3-pixel default
        if fwhm_arcsec and pix_arcsec:
            fwhm_pix = max(1.5, float(fwhm_arcsec) / pix_arcsec)
        elif beam and pix_arcsec:
            fwhm_pix = max(1.5, beam["bmaj_arcsec"] / pix_arcsec)
        else:
            fwhm_pix = 3.0

        work = np.where(np.isfinite(data), data - bkg_median, 0.0)
        method = "DAOStarFinder"
        positions: List[Tuple[float, float]] = []
        peaks: List[float] = []
        try:
            finder = DAOStarFinder(threshold=threshold_sigma * bkg_std, fwhm=fwhm_pix)
            table = finder(work)
        except Exception as exc:
            logger.warning(f"[IMG-ANALYSIS] DAOStarFinder failed ({exc}); using segmentation")
            table = None
        if table is not None and len(table) > 0:
            # photutils 2.x renamed xcentroid -> x_centroid; accept both
            xcol = "x_centroid" if "x_centroid" in table.colnames else "xcentroid"
            ycol = "y_centroid" if "y_centroid" in table.colnames else "ycentroid"
            for row in table:
                positions.append((float(row[xcol]), float(row[ycol])))
                peaks.append(float(row["peak"]))
        else:
            method = "segmentation"
            segm = seg_detect(work, threshold_sigma * bkg_std, npixels=5)
            if segm is None:
                return {
                    "success": True,
                    "n_sources": 0,
                    "sources": [],
                    "method": method,
                    "threshold_sigma": threshold_sigma,
                    "background_rms": _round(bkg_std),
                    "note": f"No sources above {threshold_sigma:g} sigma.",
                    "warnings": warnings,
                }
            catalog = SourceCatalog(work, segm)
            for obj in catalog:
                positions.append((float(obj.xcentroid), float(obj.ycentroid)))
                peaks.append(float(obj.max_value))

        # Aperture photometry with local annulus background
        r_ap = max(2.0, 1.5 * fwhm_pix)
        annulus = CircularAnnulus(positions, r_in=2.0 * r_ap, r_out=3.0 * r_ap)
        apertures = CircularAperture(positions, r=r_ap)
        nonfinite_mask = ~np.isfinite(data)
        ap_stats = ApertureStats(data, annulus, mask=nonfinite_mask, sigma_clip=None)
        local_bkg = np.nan_to_num(ap_stats.median, nan=bkg_median)
        # IMG-06: the sum and the background-subtraction area must cover the
        # SAME pixels. nan_to_num summed NaN pixels as 0 while local_bkg was
        # subtracted over the full geometric aperture area, biasing net fluxes
        # low near coverage gaps — mask non-finite pixels so both the sum and
        # the effective area exclude them.
        src_stats = ApertureStats(data, apertures, mask=nonfinite_mask, sigma_clip=None)
        ap_sums = np.nan_to_num(np.atleast_1d(np.asarray(src_stats.sum, dtype=float)), nan=0.0)
        eff_areas = np.atleast_1d(np.asarray(src_stats.sum_aper_area.value, dtype=float))
        geom_area = float(apertures.area)

        sources: List[Dict[str, Any]] = []
        for i, (x, y) in enumerate(positions):
            eff_area = float(eff_areas[i]) if math.isfinite(float(eff_areas[i])) else 0.0
            net = float(ap_sums[i]) - float(local_bkg[i]) * eff_area
            # peaks[] were measured on the background-subtracted `work` array,
            # so they are already net peaks — do not subtract the median again.
            snr = peaks[i] / bkg_std
            entry: Dict[str, Any] = {
                "id": i + 1,
                "x": _round(x, 2),
                "y": _round(y, 2),
                "peak": _round(peaks[i]),
                "aperture_sum": _round(net),
                "snr": _round(snr, 2),
            }
            if eff_area < geom_area - 1e-6:
                # IMG-06: per-source coverage flag when NaN/out-of-image
                # pixels were excluded from the aperture.
                entry["aperture_clipped"] = True
                entry["aperture_effective_area_pix"] = _round(eff_area, 2)
            if wcs is not None:
                sra, sdec = _pixel_to_sky(wcs, x, y)
                entry["ra"] = _round(sra, 6)
                entry["dec"] = _round(sdec, 6)
            if jy_per_beam:
                entry["flux_jy"] = _round(net / beam["beam_area_pix"])
            sources.append(entry)

        if any(src.get("aperture_clipped") for src in sources):
            warnings.append(
                "Some apertures include non-finite or out-of-image pixels; "
                "their photometry uses only the finite pixels and is flagged "
                "aperture_clipped (IMG-06)."
            )

        sources.sort(key=lambda s: (s.get("aperture_sum") if s.get("aperture_sum") is not None else -np.inf), reverse=True)
        truncated = len(sources) > max_sources
        if truncated:
            warnings.append(f"Detected {len(sources)} sources; returning the {max_sources} brightest.")
            sources = sources[:max_sources]
        for rank, src in enumerate(sources, start=1):
            src["id"] = rank

        # Annotated overlay
        fig = plt.figure(figsize=(10, 10), facecolor=FIGURE_FACECOLOR)
        ax = _image_axes(fig, wcs)
        ax.imshow(data, origin="lower", cmap="inferno", norm=_zscale_norm(data))
        for src in sources:
            circ = plt.Circle((src["x"], src["y"]), r_ap, fill=False,
                              edgecolor="#22d3ee", linewidth=0.9, alpha=0.9)
            ax.add_patch(circ)
        for src in sources[:30]:
            ax.annotate(str(src["id"]), (src["x"] + r_ap, src["y"] + r_ap),
                        color="#22d3ee", fontsize=8)
        flux_note = "flux_jy in Jy (Jy/beam map)" if jy_per_beam else f"aperture_sum in {bunit or 'map units'}·pix"
        full_title = (f"{title} — {len(sources)} sources ≥{threshold_sigma:g}σ"
                      if title else f"Source detection: {len(sources)} ≥{threshold_sigma:g}σ")
        ax.set_title(full_title, color="white", fontsize=13, pad=12)
        image_path = _save_figure(fig, "detect")

        return {
            "success": True,
            "image_path": image_path,
            "caption": full_title,
            "n_sources": len(sources),
            "truncated": truncated,
            "method": method,
            "threshold_sigma": threshold_sigma,
            "fwhm_pixels": _round(fwhm_pix, 2),
            "aperture_radius_pix": _round(r_ap, 2),
            "background_median": _round(bkg_median),
            "background_rms": _round(bkg_std),
            "bunit": bunit or None,
            "flux_note": flux_note,
            "sources": sources,
            "warnings": warnings,
        }
    except Exception as e:
        logger.error(f"[IMG-ANALYSIS] detect_and_measure_sources failed: {e}")
        return {"success": False, "error": str(e)}
    finally:
        gc.collect()  # release memmap handles before unlink (Windows)
        if fits_path and os.path.exists(fits_path):
            try:
                os.unlink(fits_path)
            except PermissionError:
                logger.warning(f"[IMG-ANALYSIS] Could not remove temp file still in use: {fits_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Region statistics (DS9 strings or simple cones)
# ─────────────────────────────────────────────────────────────────────────────
def measure_region(
    url: Optional[str] = None,
    survey: Optional[str] = None,
    ra: Optional[float] = None,
    dec: Optional[float] = None,
    fov_deg: Optional[float] = None,
    width: Optional[int] = None,
    region: Optional[str] = None,
    radius_arcsec: Optional[float] = None,
    title: str = "",
) -> Dict[str, Any]:
    """Statistics inside a DS9 region string or an (ra, dec, radius) cone.

    When the map is in Jy/beam and carries a restoring beam, also reports the
    integrated flux density in Jy (sum / beam area).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from astropy.stats import mad_std

    fits_path = None
    try:
        resolved = _resolve_input_url(url, survey, ra, dec, fov_deg, width)
        fits_path = _download_fits(resolved, title or "region statistics")
        data, wcs, header, cube_note = _load_fits_image(fits_path)

        pix_arcsec = _pixel_scale_arcsec(wcs)
        pixel_area_arcsec2 = _pixel_area_arcsec2(wcs, pix_arcsec)
        beam = _beam_info(header, pixel_area_arcsec2)
        bunit = _bunit(header)
        # `warnings` is defined further down (after region parsing); stash the
        # geometry note and append it there.
        _region_geom_warn = _nonsquare_pixel_warning(wcs)

        # Build the pixel region
        region_label = ""
        if region:
            from regions import Regions

            parsed = None
            region_s = str(region).strip()
            # A bare DS9 shape has no coordinate-frame line; default to ICRS.
            candidates = [region_s] if ";" in region_s.split("(")[0] else [f"icrs; {region_s}", region_s]
            for candidate in candidates:
                for fmt in ("ds9", "crtf"):
                    try:
                        attempt = Regions.parse(candidate, format=fmt)
                        if len(attempt) > 0:
                            parsed = attempt
                            break
                    except Exception:
                        continue
                if parsed is not None:
                    break
            if parsed is None or len(parsed) == 0:
                raise ImageAnalysisError(
                    "Could not parse the region string. Provide a DS9 region like "
                    "'circle(150.1, 2.2, 30\")' (ICRS degrees) or 'polygon(...)', "
                    "or pass ra/dec/radius_arcsec."
                )
            reg = parsed[0]
            if len(parsed) > 1:
                pass  # single-region v1; extra regions ignored with a warning below
            pixel_region = reg.to_pixel(wcs) if hasattr(reg, "to_pixel") else reg
            region_label = str(region).strip()
        else:
            if ra is None or dec is None or not radius_arcsec:
                raise ImageAnalysisError(
                    "Provide a DS9 `region` string, or ra + dec + radius_arcsec for a circular aperture."
                )
            if wcs is None:
                raise ImageAnalysisError("Image has no celestial WCS; a sky circle cannot be placed.")
            from astropy.coordinates import SkyCoord
            import astropy.units as u
            from regions import CircleSkyRegion

            sky = CircleSkyRegion(
                center=SkyCoord(ra=float(ra) * u.deg, dec=float(dec) * u.deg),
                radius=float(radius_arcsec) * u.arcsec,
            )
            pixel_region = sky.to_pixel(wcs)
            region_label = f"circle RA={float(ra):.5f}, Dec={float(dec):.5f}, r={float(radius_arcsec):g}\""

        mask = pixel_region.to_mask(mode="center")
        # IMG-01: the default cutout fill is 0.0, so a region overlapping the
        # image edge gained fabricated zero-valued pixels that pass the
        # isfinite filter and pollute every statistic — fill with NaN so
        # out-of-image pixels are excluded instead.
        cutout = mask.cutout(data, fill_value=np.nan)
        if cutout is None:
            raise ImageAnalysisError("The region does not overlap the image.")
        values = cutout[np.asarray(mask.data, dtype=bool)]
        values = values[np.isfinite(values)]
        if values.size == 0:
            raise ImageAnalysisError("The region contains no finite pixels.")

        region_clipped = False
        bbox = getattr(mask, "bbox", None)
        if bbox is not None:
            ny_img, nx_img = data.shape
            region_clipped = (bbox.ixmin < 0 or bbox.iymin < 0
                              or bbox.ixmax > nx_img or bbox.iymax > ny_img)

        warnings: List[str] = []
        if cube_note:
            warnings.append(cube_note)  # IMG-11: disclose the collapsed plane
        if _region_geom_warn:
            warnings.append(_region_geom_warn)
        if region_clipped:
            warnings.append(
                "Region extends past the image boundary; statistics cover "
                "only the in-image pixels (IMG-01)."
            )
        if region and "\n" in str(region).strip():
            warnings.append("Multiple regions supplied; only the first was measured.")

        npix = int(values.size)
        total = float(np.sum(values))
        stats: Dict[str, Any] = {
            "n_pixels": npix,
            "sum": _round(total),
            "mean": _round(float(np.mean(values))),
            "median": _round(float(np.median(values))),
            "std": _round(float(np.std(values))),
            "mad_rms": _round(float(mad_std(values))),
            "min": _round(float(np.min(values))),
            "max": _round(float(np.max(values))),
            "bunit": bunit or None,
        }
        if pix_arcsec:
            stats["area_arcsec2"] = _round(npix * (pixel_area_arcsec2 or pix_arcsec ** 2), 3)
        if _is_jy_per_beam(bunit) and beam and beam.get("beam_area_pix"):
            stats["n_beams"] = _round(npix / beam["beam_area_pix"], 3)
            stats["integrated_flux_jy"] = _round(total / beam["beam_area_pix"])
            stats["peak_jy_per_beam"] = _round(float(np.max(values)))
        elif _is_jy_per_beam(bunit):
            warnings.append("Map is Jy/beam but has no BMAJ/BMIN — integrated Jy not computable.")

        # Overlay plot
        fig = plt.figure(figsize=(10, 10), facecolor=FIGURE_FACECOLOR)
        ax = _image_axes(fig, wcs)
        ax.imshow(data, origin="lower", cmap="inferno", norm=_zscale_norm(data))
        try:
            pixel_region.plot(ax=ax, edgecolor="#22d3ee", linewidth=1.6, fill=False)
        except Exception:
            pass
        full_title = f"{title} — Region statistics" if title else "Region statistics"
        ax.set_title(full_title, color="white", fontsize=13, pad=12)
        image_path = _save_figure(fig, "region")

        return {
            "success": True,
            "image_path": image_path,
            "caption": full_title,
            "region": region_label,
            "statistics": stats,
            "warnings": warnings,
        }
    except Exception as e:
        logger.error(f"[IMG-ANALYSIS] measure_region failed: {e}")
        return {"success": False, "error": str(e)}
    finally:
        gc.collect()  # release memmap handles before unlink (Windows)
        if fits_path and os.path.exists(fits_path):
            try:
                os.unlink(fits_path)
            except PermissionError:
                logger.warning(f"[IMG-ANALYSIS] Could not remove temp file still in use: {fits_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 4. 2D Gaussian source fit with beam deconvolution
# ─────────────────────────────────────────────────────────────────────────────
def fit_gaussian_source(
    url: Optional[str] = None,
    survey: Optional[str] = None,
    ra: Optional[float] = None,
    dec: Optional[float] = None,
    fov_deg: Optional[float] = None,
    width: Optional[int] = None,
    x_pixel: Optional[float] = None,
    y_pixel: Optional[float] = None,
    box_arcsec: Optional[float] = None,
    title: str = "",
) -> Dict[str, Any]:
    """Fit a 2D Gaussian + constant offset to a source (the CASA imfit workflow).

    Reports peak, integrated flux, fitted FWHM sizes/PA, and — when the header
    carries a restoring beam — the beam-deconvolved size or a
    'consistent with point source' verdict.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from astropy.modeling import fitting, models
    from astropy.stats import sigma_clipped_stats

    FWHM = 2.0 * math.sqrt(2.0 * math.log(2.0))

    fits_path = None
    try:
        resolved = _resolve_input_url(url, survey, ra, dec, fov_deg, width)
        fits_path = _download_fits(resolved, title or "gaussian fit")
        data, wcs, header, cube_note = _load_fits_image(fits_path)

        blank, warnings = _blankness(data)
        if cube_note:
            warnings.append(cube_note)  # IMG-11: disclose the collapsed plane
        if blank:
            return {"success": False, "blank": True,
                    "error": "Image looks blank — nothing to fit.", "warnings": warnings}

        pix_arcsec = _pixel_scale_arcsec(wcs)
        _geom_warn = _nonsquare_pixel_warning(wcs)
        if _geom_warn:
            warnings.append(_geom_warn)
        pixel_area_arcsec2 = _pixel_area_arcsec2(wcs, pix_arcsec)
        beam = _beam_info(header, pixel_area_arcsec2)
        bunit = _bunit(header)

        x0, y0, pos_label = _resolve_position(data, wcs, ra, dec, x_pixel, y_pixel)

        # Fit box: explicit > 6x beam major > 41 px
        if box_arcsec and pix_arcsec:
            half = max(5, int(round(float(box_arcsec) / pix_arcsec / 2.0)))
        elif beam and pix_arcsec:
            half = max(8, int(round(3.0 * beam["bmaj_arcsec"] / pix_arcsec)))
        else:
            half = 20
        ny, nx = data.shape
        xlo, xhi = max(0, int(x0) - half), min(nx, int(x0) + half + 1)
        ylo, yhi = max(0, int(y0) - half), min(ny, int(y0) + half + 1)
        cut = data[ylo:yhi, xlo:xhi]
        if cut.shape[0] < 5 or cut.shape[1] < 5:
            raise ImageAnalysisError("Fit box is too small (position too close to the image edge).")
        yy, xx = np.mgrid[ylo:yhi, xlo:xhi]
        good = np.isfinite(cut)
        if good.sum() < 12:
            raise ImageAnalysisError("Too few finite pixels in the fit box.")

        _, bkg_median, bkg_std = sigma_clipped_stats(data[np.isfinite(data)], sigma=3.0, maxiters=5)
        peak_guess = float(np.nanmax(cut)) - bkg_median
        sigma_guess = (beam["bmaj_arcsec"] / pix_arcsec / FWHM) if (beam and pix_arcsec) else max(2.0, half / 5.0)

        model = models.Gaussian2D(
            amplitude=peak_guess, x_mean=x0, y_mean=y0,
            x_stddev=sigma_guess, y_stddev=sigma_guess, theta=0.0,
        ) + models.Const2D(amplitude=bkg_median)
        fitter = fitting.LevMarLSQFitter(calc_uncertainties=True)
        import warnings as _w
        with _w.catch_warnings():
            _w.simplefilter("ignore")
            fitted = fitter(model, xx[good], yy[good], cut[good], maxiter=500)

        g = fitted[0]
        amp = float(g.amplitude.value)
        sx, sy = abs(float(g.x_stddev.value)), abs(float(g.y_stddev.value))
        theta = float(g.theta.value)
        xc, yc = float(g.x_mean.value), float(g.y_mean.value)
        offset = float(fitted[1].amplitude.value)

        # Major/minor FWHM in pixels -> arcsec.
        maj_pix, min_pix = (sx, sy) if sx >= sy else (sy, sx)
        # Major-axis orientation in the PIXEL frame (radians).
        major_angle_pix = theta + (0.0 if sx >= sy else math.pi / 2.0)
        pa_pixel_deg = math.degrees(major_angle_pix) % 180.0
        # Sky position angle (deg E of N): step a little along the major axis in
        # pixel space, convert both ends to sky, and take the bearing — this is
        # correct even for rotated/skewed WCS, not just unrotated TAN (CX-10).
        pa_deg = pa_pixel_deg
        pa_frame = "pixel"
        if wcs is not None:
            sky_pa = _sky_position_angle(wcs, xc, yc, major_angle_pix)
            if sky_pa is not None:
                pa_deg = sky_pa
                pa_frame = "sky (E of N)"
        fit_out: Dict[str, Any] = {
            "peak": _round(amp),
            "peak_unit": bunit or "map units",
            "offset": _round(offset),
            "x": _round(xc, 2),
            "y": _round(yc, 2),
            "fwhm_major_pix": _round(maj_pix * FWHM, 3),
            "fwhm_minor_pix": _round(min_pix * FWHM, 3),
            "pa_deg": _round(pa_deg, 1),
            "pa_frame": pa_frame,
            "snr": _round(amp / bkg_std, 1) if bkg_std > 0 else None,
        }
        if wcs is not None:
            fra, fdec = _pixel_to_sky(wcs, xc, yc)
            fit_out["ra"] = _round(fra, 6)
            fit_out["dec"] = _round(fdec, 6)
        # Sky-true FWHM (CX-10): measure each axis on the sky along its own
        # direction instead of scaling pixels by the single MEAN pixel scale,
        # which mis-sizes anisotropic/rotated WCS images. Falls back to the
        # mean-scale product when the WCS conversion fails.
        fwhm_major_arcsec: Optional[float] = None
        fwhm_minor_arcsec: Optional[float] = None
        if pix_arcsec:
            if wcs is not None:
                fwhm_major_arcsec = _sky_length_arcsec(
                    wcs, xc, yc, major_angle_pix, maj_pix * FWHM)
                fwhm_minor_arcsec = _sky_length_arcsec(
                    wcs, xc, yc, major_angle_pix + math.pi / 2.0, min_pix * FWHM)
            if fwhm_major_arcsec is None:
                fwhm_major_arcsec = maj_pix * FWHM * pix_arcsec
            if fwhm_minor_arcsec is None:
                fwhm_minor_arcsec = min_pix * FWHM * pix_arcsec
            fit_out["fwhm_major_arcsec"] = _round(fwhm_major_arcsec, 3)
            fit_out["fwhm_minor_arcsec"] = _round(fwhm_minor_arcsec, 3)
        # Effective arcsec/pix along each fitted axis — used so the error bars
        # and the deconvolution beam stay consistent with the sky-true sizes.
        _scale_maj = (fwhm_major_arcsec / (maj_pix * FWHM)
                      if (fwhm_major_arcsec and maj_pix > 0) else pix_arcsec)
        _scale_min = (fwhm_minor_arcsec / (min_pix * FWHM)
                      if (fwhm_minor_arcsec and min_pix > 0) else pix_arcsec)

        # Parameter uncertainties when the fitter produced a covariance matrix:
        # peak, FWHM major/minor (σ_err × FWHM factor, in pix and arcsec), and PA.
        cov = getattr(fitter, "fit_info", {}).get("param_cov") if hasattr(fitter, "fit_info") else None
        if cov is not None:
            try:
                errs = np.sqrt(np.abs(np.diag(cov)))
                names = [n for n in fitted.param_names if not fitted.fixed[n]]
                err_map = dict(zip(names, errs))
                if "amplitude_0" in err_map:
                    fit_out["peak_err"] = _round(float(err_map["amplitude_0"]))
                # x/y stddev map to major/minor by which axis was larger.
                sx_err = float(err_map.get("x_stddev_0", float("nan")))
                sy_err = float(err_map.get("y_stddev_0", float("nan")))
                maj_err, min_err = (sx_err, sy_err) if sx >= sy else (sy_err, sx_err)
                if math.isfinite(maj_err):
                    fit_out["fwhm_major_pix_err"] = _round(maj_err * FWHM, 3)
                    if _scale_maj:
                        fit_out["fwhm_major_arcsec_err"] = _round(maj_err * FWHM * _scale_maj, 3)
                if math.isfinite(min_err):
                    fit_out["fwhm_minor_pix_err"] = _round(min_err * FWHM, 3)
                    if _scale_min:
                        fit_out["fwhm_minor_arcsec_err"] = _round(min_err * FWHM * _scale_min, 3)
                if "theta_0" in err_map and math.isfinite(err_map["theta_0"]):
                    fit_out["pa_deg_err"] = _round(math.degrees(float(err_map["theta_0"])), 1)
            except Exception:
                pass

        # Integrated flux: 2*pi*amp*sx*sy [unit*pix^2]; Jy for Jy/beam maps
        volume = 2.0 * math.pi * amp * sx * sy
        if _is_jy_per_beam(bunit) and beam and beam.get("beam_area_pix"):
            fit_out["integrated_flux_jy"] = _round(volume / beam["beam_area_pix"])
        else:
            fit_out["integrated_flux_map_units_pix2"] = _round(volume)

        # Beam deconvolution via radio-beam
        deconvolved: Optional[Dict[str, Any]] = None
        point_source = None
        if beam and pix_arcsec:
            try:
                import astropy.units as u
                from radio_beam import Beam

                fitted_beam = Beam(
                    # Sky-true sizes (CX-10) — consistent with BPA/BMAJ/BMIN,
                    # which are sky quantities.
                    major=(fwhm_major_arcsec or maj_pix * FWHM * pix_arcsec) * u.arcsec,
                    minor=(fwhm_minor_arcsec or min_pix * FWHM * pix_arcsec) * u.arcsec,
                    pa=pa_deg * u.deg,
                )
                restoring = Beam(
                    major=beam["bmaj_arcsec"] * u.arcsec,
                    minor=beam["bmin_arcsec"] * u.arcsec,
                    pa=beam["bpa_deg"] * u.deg,
                )
                try:
                    dec_beam = fitted_beam.deconvolve(restoring)
                    deconvolved = {
                        "major_arcsec": _round(dec_beam.major.to_value(u.arcsec), 3),
                        "minor_arcsec": _round(dec_beam.minor.to_value(u.arcsec), 3),
                        "pa_deg": _round(dec_beam.pa.to_value(u.deg) % 180.0, 1),
                    }
                    point_source = False
                except Exception as deconv_err:
                    # radio-beam raises BeamError/ValueError (version-dependent)
                    # when the fitted size is not larger than the beam.
                    if "deconvolve" not in str(deconv_err).lower():
                        raise
                    point_source = True  # fitted size <= beam: unresolved
            except Exception as exc:
                warnings.append(f"Beam deconvolution unavailable: {exc}")
        if deconvolved:
            fit_out["deconvolved"] = deconvolved
        if point_source is not None:
            fit_out["consistent_with_point_source"] = bool(point_source)

        # Data / model / residual triptych
        model_img = np.full_like(cut, np.nan)
        model_img[good] = fitted(xx[good], yy[good])
        residual = np.where(good, cut - model_img, np.nan)
        norm = _zscale_norm(cut)
        fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6), facecolor=FIGURE_FACECOLOR)
        for ax, img, label in zip(axes, (cut, model_img, residual), ("Data", "Model", "Residual")):
            ax.imshow(img, origin="lower", cmap="inferno", norm=norm)
            ax.set_title(label, color="white", fontsize=11)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_facecolor("black")
        size_txt = ""
        if fit_out.get("fwhm_major_arcsec"):
            size_txt = f" — {fit_out['fwhm_major_arcsec']}\" × {fit_out['fwhm_minor_arcsec']}\""
        full_title = (f"{title} — Gaussian fit at {pos_label}{size_txt}"
                      if title else f"Gaussian fit at {pos_label}{size_txt}")
        fig.suptitle(full_title, color="white", fontsize=12)
        image_path = _save_figure(fig, "gaussfit")

        residual_rms = float(np.sqrt(np.nanmean(residual ** 2)))
        return {
            "success": True,
            "image_path": image_path,
            "caption": full_title,
            "position": pos_label,
            "fit": fit_out,
            "residual_rms": _round(residual_rms),
            "background_rms": _round(bkg_std),
            "beam": ({k: _round(v, 4) for k, v in beam.items()} if beam else None),
            "warnings": warnings,
        }
    except Exception as e:
        logger.error(f"[IMG-ANALYSIS] fit_gaussian_source failed: {e}")
        return {"success": False, "error": str(e)}
    finally:
        gc.collect()  # release memmap handles before unlink (Windows)
        if fits_path and os.path.exists(fits_path):
            try:
                os.unlink(fits_path)
            except PermissionError:
                logger.warning(f"[IMG-ANALYSIS] Could not remove temp file still in use: {fits_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Radial profile + curve of growth
# ─────────────────────────────────────────────────────────────────────────────
def radial_profile(
    url: Optional[str] = None,
    survey: Optional[str] = None,
    ra: Optional[float] = None,
    dec: Optional[float] = None,
    fov_deg: Optional[float] = None,
    width: Optional[int] = None,
    x_pixel: Optional[float] = None,
    y_pixel: Optional[float] = None,
    max_radius_arcsec: Optional[float] = None,
    title: str = "",
) -> Dict[str, Any]:
    """Azimuthally averaged radial profile and curve of growth at a position."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from astropy.stats import sigma_clipped_stats
    from photutils.profiles import CurveOfGrowth, RadialProfile

    fits_path = None
    try:
        resolved = _resolve_input_url(url, survey, ra, dec, fov_deg, width)
        fits_path = _download_fits(resolved, title or "radial profile")
        data, wcs, header, cube_note = _load_fits_image(fits_path)

        blank, warnings = _blankness(data)
        if cube_note:
            warnings.append(cube_note)  # IMG-11: disclose the collapsed plane
        if blank:
            return {"success": False, "blank": True,
                    "error": "Image looks blank — no profile to measure.", "warnings": warnings}

        pix_arcsec = _pixel_scale_arcsec(wcs)
        _geom_warn = _nonsquare_pixel_warning(wcs)
        if _geom_warn:
            warnings.append(_geom_warn)
        pixel_area_arcsec2 = _pixel_area_arcsec2(wcs, pix_arcsec)
        beam = _beam_info(header, pixel_area_arcsec2)
        x0, y0, pos_label = _resolve_position(data, wcs, ra, dec, x_pixel, y_pixel)

        ny, nx = data.shape
        edge = float(min(x0, y0, nx - 1 - x0, ny - 1 - y0))
        if max_radius_arcsec and pix_arcsec:
            max_r = min(edge, float(max_radius_arcsec) / pix_arcsec)
        else:
            max_r = min(edge, min(nx, ny) / 4.0)
        if max_r < 3:
            raise ImageAnalysisError("Position is too close to the image edge for a profile.")

        _, bkg_median, bkg_std = sigma_clipped_stats(data[np.isfinite(data)], sigma=3.0, maxiters=5)
        work = np.nan_to_num(data - bkg_median, nan=0.0)

        n_bins = int(max(8, min(60, max_r)))
        edges = np.linspace(0.0, max_r, n_bins + 1)
        rp = RadialProfile(work, (x0, y0), edges)
        cog_radii = edges[1:]
        cog = CurveOfGrowth(work, (x0, y0), cog_radii)

        r_unit = "arcsec" if pix_arcsec else "pix"
        scale = pix_arcsec if pix_arcsec else 1.0
        prof_r = rp.radius * scale
        prof_v = rp.profile
        cog_r = cog.radius * scale
        cog_v = cog.profile

        # FWHM from the first half-max crossing of the (monotonic-ish) profile
        fwhm_val = None
        try:
            peak_val = float(prof_v[0])
            half = peak_val / 2.0
            below = np.where(prof_v <= half)[0]
            if peak_val > 0 and below.size:
                j = int(below[0])
                if j > 0:
                    r1, r2 = prof_r[j - 1], prof_r[j]
                    v1, v2 = prof_v[j - 1], prof_v[j]
                    frac = (v1 - half) / (v1 - v2) if v1 != v2 else 0.0
                    fwhm_val = 2.0 * float(r1 + frac * (r2 - r1))
        except Exception:
            pass

        # Half-light radius + asymptotic flux from the curve of growth
        total = float(cog_v[-1])
        half_light = None
        if total > 0:
            above = np.where(cog_v >= total / 2.0)[0]
            if above.size:
                half_light = float(cog_r[int(above[0])])
        # Flatness check: has the COG converged?
        converged = bool(total > 0 and cog_v.size >= 4 and
                         abs(cog_v[-1] - cog_v[-3]) <= 0.02 * abs(total))
        if not converged:
            warnings.append("Curve of growth has not flattened — total flux is a lower bound; "
                            "increase max_radius_arcsec.")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8), facecolor=FIGURE_FACECOLOR)
        for ax in (ax1, ax2):
            ax.set_facecolor(FIGURE_FACECOLOR)
            ax.tick_params(colors="white", labelsize=9)
            for spine in ax.spines.values():
                spine.set_color("#334155")
        ax1.plot(prof_r, prof_v, color="#06b6d4", linewidth=1.6)
        ax1.axhline(0, color="#475569", linewidth=0.6, linestyle="--")
        # The beam HWHM marker is in arcsec, so only draw it when the radius
        # axis is actually in arcsec (i.e. a usable pixel scale exists). (CX-13)
        if beam and r_unit == "arcsec":
            ax1.axvline(beam["bmaj_arcsec"] / 2.0,
                        color="#f59e0b", linewidth=1.2, linestyle=":",
                        label=f"beam HWHM ({beam['bmaj_arcsec'] / 2.0:.2f}\")")
            ax1.legend(loc="upper right", fontsize=8, facecolor="#1e293b",
                       edgecolor="#334155", labelcolor="white")
        ax1.set_xlabel(f"Radius ({r_unit})", color="white", fontsize=10)
        ax1.set_ylabel("Mean intensity (bkg-subtracted)", color="white", fontsize=10)
        ax1.set_title("Radial profile", color="white", fontsize=11)
        ax2.plot(cog_r, cog_v, color="#34d399", linewidth=1.6)
        if half_light is not None:
            ax2.axvline(half_light, color="#f59e0b", linewidth=1.2, linestyle=":",
                        label=f"half-light r = {half_light:.2f} {r_unit}")
            ax2.legend(loc="lower right", fontsize=8, facecolor="#1e293b",
                       edgecolor="#334155", labelcolor="white")
        ax2.set_xlabel(f"Aperture radius ({r_unit})", color="white", fontsize=10)
        ax2.set_ylabel("Enclosed flux (map units·pix)", color="white", fontsize=10)
        ax2.set_title("Curve of growth", color="white", fontsize=11)
        full_title = f"{title} — Radial profile at {pos_label}" if title else f"Radial profile at {pos_label}"
        fig.suptitle(full_title, color="white", fontsize=12)
        image_path = _save_figure(fig, "radprof")

        result: Dict[str, Any] = {
            "success": True,
            "image_path": image_path,
            "caption": full_title,
            "position": pos_label,
            "radius_unit": r_unit,
            "max_radius": _round(max_r * scale, 3),
            "fwhm": _round(fwhm_val, 3),
            "half_light_radius": _round(half_light, 3),
            "enclosed_flux_total": _round(total),
            "cog_converged": converged,
            "background_rms": _round(bkg_std),
            "warnings": warnings,
        }
        if beam:
            result["beam_fwhm_arcsec"] = _round(beam["bmaj_arcsec"], 3)
            if fwhm_val and r_unit == "arcsec" and fwhm_val <= beam["bmaj_arcsec"] * 1.1:
                result["note"] = "Profile FWHM is within ~10% of the beam — source is unresolved."
        return result
    except Exception as e:
        logger.error(f"[IMG-ANALYSIS] radial_profile failed: {e}")
        return {"success": False, "error": str(e)}
    finally:
        gc.collect()  # release memmap handles before unlink (Windows)
        if fits_path and os.path.exists(fits_path):
            try:
                os.unlink(fits_path)
            except PermissionError:
                logger.warning(f"[IMG-ANALYSIS] Could not remove temp file still in use: {fits_path}")


__all__ = [
    "ImageAnalysisError",
    "image_statistics",
    "detect_and_measure_sources",
    "measure_region",
    "fit_gaussian_source",
    "radial_profile",
]
