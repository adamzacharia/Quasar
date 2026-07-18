"""
FITS file download, rendering, and overlay service.

Handles:
  - Downloading FITS files from archive URLs (CADC, ALMA, ESO)
  - Rendering FITS images as PNG with astronomical colormaps
  - Creating overlay composites (e.g., ALMA contours on JWST colorscale)
  - WCS-aware reprojection for aligning images from different instruments
"""

import os
import gc
import uuid
import math
import tempfile
import logging
import shutil
from typing import Dict, Any, Optional, Tuple

import numpy as np
import requests as http_requests

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────
MAX_DOWNLOAD_MB = 150              # Reject files larger than this
DOWNLOAD_TIMEOUT_S = 180           # Timeout for HTTP download
HIPS2FITS_TIMEOUT_S = 45           # Shorter cap for hips2fits cutout URLs
RENDERED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "rendered_images")
os.makedirs(RENDERED_DIR, exist_ok=True)


def _pick_science_hdu(hdul, return_slice_info: bool = False):
    """Find the first HDU with 2D image data (skip tables, empty primary).

    3D+ cubes are collapsed to 2D by taking the middle plane of each extra
    axis. With ``return_slice_info=True`` a third element is returned: a dict
    ``{"plane": mid, "n_planes": n, "note": ...}`` describing the collapse
    (IMG-11), or None when the input was already 2D — callers must disclose
    which plane was used instead of presenting it as "the image".
    """
    from astropy.io import fits as afits  # noqa: F811

    for i, hdu in enumerate(hdul):
        if hdu.data is not None and hdu.data.ndim >= 2:
            # If 3D+ cube, collapse to 2D by taking the middle slice
            data = hdu.data
            slice_info = None
            while data.ndim > 2:
                n_planes = int(data.shape[0])
                mid = n_planes // 2
                if n_planes > 1:
                    # IMG-11: record the collapse so callers can caption it.
                    slice_info = {
                        "plane": mid,
                        "n_planes": n_planes,
                        "note": (
                            f"Input is a spectral cube; only the middle plane "
                            f"(channel {mid} of {n_planes}) was analyzed/rendered. "
                            "Use cube-aware tools (moment maps, spectra, channel "
                            "maps) for the full cube."
                        ),
                    }
                data = data[mid]  # take middle channel
            if return_slice_info:
                return data, i, slice_info
            return data, i
    raise ValueError("No 2D image data found in FITS file")


def _download_fits(url: str, label: str = "FITS") -> str:
    """Download a FITS file from URL to a temp file. Returns path."""
    logger.info(f"[FITS] Downloading {label} from {url}")

    if url.startswith("file://") or os.path.exists(url):
        src = url.replace("file://", "", 1)
        tmp = tempfile.NamedTemporaryFile(suffix=".fits", delete=False)
        tmp.close()
        shutil.copyfile(src, tmp.name)
        return tmp.name

    # hips2fits cutout URLs get the shorter 45 s cap (they return in seconds);
    # arbitrary archive FITS keep the generous 180 s. Match both the literal
    # service name AND a configured proxy base so a HIPS2FITS_BASE_URL override
    # is still recognized (CX-28).
    _hips_base = os.getenv("HIPS2FITS_BASE_URL", "").strip().lower()
    _low = url.lower()
    is_hips = "hips2fits" in _low or (_hips_base and _low.startswith(_hips_base))
    timeout_s = HIPS2FITS_TIMEOUT_S if is_hips else DOWNLOAD_TIMEOUT_S
    resp = http_requests.get(url, timeout=timeout_s, stream=True)
    resp.raise_for_status()

    # Check content-length if available
    content_length = resp.headers.get("Content-Length")
    if content_length and int(content_length) > MAX_DOWNLOAD_MB * 1024 * 1024:
        raise ValueError(
            f"FITS file is {int(content_length) / (1024*1024):.0f} MB, "
            f"exceeds {MAX_DOWNLOAD_MB} MB limit"
        )

    tmp = tempfile.NamedTemporaryFile(suffix=".fits", delete=False)
    size = 0
    try:
        for chunk in resp.iter_content(chunk_size=1024 * 256):
            size += len(chunk)
            if size > MAX_DOWNLOAD_MB * 1024 * 1024:
                tmp.close()
                os.unlink(tmp.name)
                raise ValueError(f"FITS file exceeds {MAX_DOWNLOAD_MB} MB limit")
            tmp.write(chunk)
        tmp.close()
    except Exception:
        tmp.close()
        if os.path.exists(tmp.name):
            os.unlink(tmp.name)
        raise

    logger.info(f"[FITS] Downloaded {size / (1024*1024):.1f} MB → {tmp.name}")
    return tmp.name


def render_fits_image(
    url: str,
    title: str = "",
    colormap: str = "inferno",
    stretch: str = "sqrt",
) -> Dict[str, Any]:
    """
    Download a FITS file and render it as a PNG image.

    Returns:
        {"success": True, "image_path": "/api/images/xxx.png", "caption": "..."}
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from astropy.io import fits as afits
    from astropy.wcs import WCS
    from astropy.visualization import (
        ZScaleInterval, ImageNormalize,
        SqrtStretch, LogStretch, LinearStretch, AsinhStretch,
    )

    fits_path = None
    try:
        fits_path = _download_fits(url, title or "image")

        with afits.open(fits_path, memmap=True) as hdul:
            # IMG-11: capture whether a cube was collapsed to one plane so the
            # caption/warnings disclose it instead of calling it "the image".
            data_2d, hdu_idx, slice_info = _pick_science_hdu(hdul, return_slice_info=True)
            try:
                wcs = WCS(hdul[hdu_idx].header, naxis=2)
            except Exception:
                wcs = None

            # Normalize with ZScale (standard astronomical stretch)
            stretch_map = {
                "sqrt": SqrtStretch(),
                "log": LogStretch(),
                "linear": LinearStretch(),
                "asinh": AsinhStretch(),
            }
            norm = ImageNormalize(
                data_2d,
                interval=ZScaleInterval(),
                stretch=stretch_map.get(stretch, SqrtStretch()),
            )

            # Render
            fig = plt.figure(figsize=(10, 10), facecolor="black")
            if wcs:
                ax = fig.add_subplot(111, projection=wcs)
                ax.coords[0].set_axislabel("RA (J2000)", color="white", fontsize=10)
                ax.coords[1].set_axislabel("Dec (J2000)", color="white", fontsize=10)
                ax.coords[0].set_ticklabel(color="white", fontsize=8)
                ax.coords[1].set_ticklabel(color="white", fontsize=8)
            else:
                ax = fig.add_subplot(111)

            ax.imshow(data_2d, origin="lower", cmap=colormap, norm=norm)
            if title:
                ax.set_title(title, color="white", fontsize=14, pad=12)

            ax.set_facecolor("black")
            fig.patch.set_facecolor("#0f172a")

            # Save PNG
            img_name = f"{uuid.uuid4().hex[:12]}.png"
            img_path = os.path.join(RENDERED_DIR, img_name)
            fig.savefig(img_path, dpi=150, bbox_inches="tight",
                        facecolor=fig.get_facecolor())
            plt.close(fig)

        caption = title or "FITS Image"
        result = {
            "success": True,
            "image_path": f"/api/images/{img_name}",
            "caption": caption,
        }
        if slice_info:
            # IMG-11: disclose the collapsed plane in caption and warnings.
            result["caption"] = (
                f"{caption} — middle channel {slice_info['plane']} of "
                f"{slice_info['n_planes']} (cube collapsed to one plane)"
            )
            result["cube_slice"] = {"plane": slice_info["plane"],
                                    "n_planes": slice_info["n_planes"]}
            result["warnings"] = [slice_info["note"]]
        return result

    except Exception as e:
        logger.error(f"[FITS] render_fits_image failed: {e}")
        return {"success": False, "error": str(e)}

    finally:
        gc.collect()  # release memmap handles before unlink (Windows)
        if fits_path and os.path.exists(fits_path):
            try:
                os.unlink(fits_path)
            except PermissionError:
                logger.warning(f"[FITS] Could not remove temp file still in use: {fits_path}")


def overlay_fits_images(
    base_url: str,
    contour_url: str,
    base_label: str = "JWST",
    contour_label: str = "ALMA",
    base_cmap: str = "inferno",
    contour_levels: int = 8,
) -> Dict[str, Any]:
    """
    Download two FITS files, reproject the contour image onto the base WCS,
    and render the base as a colorscale with the contour data overlaid.

    Returns:
        {"success": True, "image_path": "/api/images/xxx.png", "caption": "..."}
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from astropy.io import fits as afits
    from astropy.wcs import WCS
    from astropy.visualization import ZScaleInterval, ImageNormalize, SqrtStretch

    base_path = None
    contour_path = None
    try:
        base_path = _download_fits(base_url, f"{base_label} base")
        contour_path = _download_fits(contour_url, f"{contour_label} contour")

        with afits.open(base_path, memmap=False) as base_hdul, \
             afits.open(contour_path, memmap=False) as cont_hdul:

            base_data, base_idx = _pick_science_hdu(base_hdul)
            cont_data, cont_idx = _pick_science_hdu(cont_hdul)

            base_wcs = WCS(base_hdul[base_idx].header, naxis=2)
            cont_wcs = WCS(cont_hdul[cont_idx].header, naxis=2)

            # Reproject contour onto base WCS
            try:
                from reproject import reproject_interp
                cont_reproj, _ = reproject_interp(
                    (cont_data, cont_wcs),
                    base_wcs,
                    shape_out=base_data.shape,
                )
            except ImportError:
                logger.warning("[FITS] reproject not installed, skipping alignment")
                cont_reproj = cont_data

            # Render
            norm = ImageNormalize(
                base_data, interval=ZScaleInterval(), stretch=SqrtStretch()
            )

            fig = plt.figure(figsize=(10, 10), facecolor="black")
            ax = fig.add_subplot(111, projection=base_wcs)

            # Base colorscale
            ax.imshow(base_data, origin="lower", cmap=base_cmap, norm=norm)

            # Contour overlay
            valid = cont_reproj[np.isfinite(cont_reproj)]
            if len(valid) > 0:
                vmin, vmax = np.nanpercentile(valid, [30, 99.5])
                if np.isfinite(vmin) and np.isfinite(vmax) and vmax > vmin:
                    levels = np.linspace(vmin, vmax, contour_levels)
                    ax.contour(
                        cont_reproj, levels=levels, colors="cyan",
                        linewidths=0.8, alpha=0.85, origin="lower",
                    )

            ax.set_title(
                f"{base_label} (color) + {contour_label} (contours)",
                color="white", fontsize=14, pad=12,
            )
            ax.coords[0].set_axislabel("RA (J2000)", color="white", fontsize=10)
            ax.coords[1].set_axislabel("Dec (J2000)", color="white", fontsize=10)
            ax.coords[0].set_ticklabel(color="white", fontsize=8)
            ax.coords[1].set_ticklabel(color="white", fontsize=8)
            ax.set_facecolor("black")
            fig.patch.set_facecolor("#0f172a")

            img_name = f"overlay_{uuid.uuid4().hex[:10]}.png"
            img_path = os.path.join(RENDERED_DIR, img_name)
            fig.savefig(img_path, dpi=150, bbox_inches="tight",
                        facecolor=fig.get_facecolor())
            plt.close(fig)

        caption = f"{base_label} (color) + {contour_label} (contours)"
        return {
            "success": True,
            "image_path": f"/api/images/{img_name}",
            "caption": caption,
        }

    except Exception as e:
        logger.error(f"[FITS] overlay_fits_images failed: {e}")
        return {"success": False, "error": str(e)}

    finally:
        for p in (base_path, contour_path):
            if p and os.path.exists(p):
                try:
                    os.unlink(p)
                except PermissionError:
                    logger.warning(f"[FITS] Could not remove temp file still in use: {p}")
        gc.collect()


def difference_image(
    url_a: str,
    url_b: str,
    label_a: str = "Image A",
    label_b: str = "Image B",
    scale_match: bool = True,
    title: str = "",
) -> Dict[str, Any]:
    """Reproject image B onto image A's WCS and render A, aligned B, and A−B.

    With ``scale_match`` (default) B is background- and gain-matched to A via
    sigma-clipped statistics before subtracting, so epoch-to-epoch transient
    checks aren't swamped by calibration offsets. Returns residual statistics
    (MAD RMS, max |residual|, fraction of |residual| > 5σ pixels).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from astropy.stats import mad_std, sigma_clipped_stats
    from astropy.visualization import ImageNormalize, SqrtStretch, ZScaleInterval
    from astropy.wcs import WCS

    path_a = None
    path_b = None
    try:
        from reproject import reproject_interp  # hard requirement for alignment

        from astropy.io import fits as afits

        path_a = _download_fits(url_a, f"{label_a}")
        path_b = _download_fits(url_b, f"{label_b}")

        with afits.open(path_a, memmap=False) as hdul_a, \
             afits.open(path_b, memmap=False) as hdul_b:
            data_a, idx_a = _pick_science_hdu(hdul_a)
            data_b, idx_b = _pick_science_hdu(hdul_b)
            data_a = np.asarray(data_a, dtype=float)
            data_b = np.asarray(data_b, dtype=float)
            wcs_a = WCS(hdul_a[idx_a].header, naxis=2)
            wcs_b = WCS(hdul_b[idx_b].header, naxis=2)
            bunit = str(hdul_a[idx_a].header.get("BUNIT", "") or "").strip()

        for name, arr in ((label_a, data_a), (label_b, data_b)):
            finite = arr[np.isfinite(arr)]
            if finite.size == 0 or float(np.max(finite)) == float(np.min(finite)):
                return {"success": False,
                        "error": f"{name} is blank (all-NaN or constant) — nothing to difference."}

        data_b_aligned, footprint = reproject_interp((data_b, wcs_b), wcs_a, shape_out=data_a.shape)
        overlap = np.isfinite(data_a) & np.isfinite(data_b_aligned) & (footprint > 0)
        if overlap.sum() < 100:
            return {"success": False,
                    "error": "The two images do not overlap on the sky (fewer than 100 shared pixels)."}

        warnings: List[str] = [] if scale_match else ["scale_match disabled: raw subtraction."]
        b_matched = data_b_aligned
        scale_applied = None
        if scale_match:
            _, med_a, std_a = sigma_clipped_stats(data_a[overlap], sigma=3.0, maxiters=5)
            _, med_b, std_b = sigma_clipped_stats(data_b_aligned[overlap], sigma=3.0, maxiters=5)
            gain = (std_a / std_b) if (std_b and math.isfinite(std_b) and std_b > 0) else 1.0
            if not (0.05 <= gain <= 20.0):
                warnings.append(
                    f"Gain match {gain:.3g} is outside the sane range (0.05–20) — the images "
                    "are probably in different units; applied offset-only matching instead."
                )
                gain = 1.0
            b_matched = (data_b_aligned - med_b) * gain + med_a
            scale_applied = {"offset_b": float(med_b), "gain": float(gain), "offset_a": float(med_a)}

        diff = np.where(overlap, data_a - b_matched, np.nan)
        diff_vals = diff[np.isfinite(diff)]
        rms = float(mad_std(diff_vals))
        max_abs = float(np.max(np.abs(diff_vals)))
        sig_frac = float(np.mean(np.abs(diff_vals) > 5.0 * rms)) if rms > 0 else 0.0

        norm = ImageNormalize(data_a[np.isfinite(data_a)], interval=ZScaleInterval(), stretch=SqrtStretch())
        vmax = 5.0 * rms if rms > 0 else (max_abs or 1.0)
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), facecolor="#0f172a",
                                 subplot_kw={"projection": wcs_a})
        panels = ((data_a, label_a, "inferno", norm, None),
                  (b_matched, f"{label_b} (aligned)", "inferno", norm, None),
                  (diff, f"{label_a} − {label_b}", "RdBu_r", None, vmax))
        im = None
        for ax, (img, label, cmap, pnorm, pvmax) in zip(axes, panels):
            if pnorm is not None:
                im0 = ax.imshow(img, origin="lower", cmap=cmap, norm=pnorm)
            else:
                im0 = ax.imshow(img, origin="lower", cmap=cmap, vmin=-pvmax, vmax=pvmax)
                im = im0
            ax.set_title(label, color="white", fontsize=10)
            for coord in (0, 1):
                ax.coords[coord].set_ticklabel(color="white", fontsize=6)
                ax.coords[coord].set_axislabel(" ")
            ax.set_facecolor("black")
        if im is not None:
            cbar = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02)
            cbar.set_label(f"Residual{f' ({bunit})' if bunit else ''}", color="white", fontsize=9)
            cbar.ax.yaxis.set_tick_params(color="white")
            plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white", fontsize=7)
        full_title = f"{title} — Difference image" if title else f"Difference: {label_a} − {label_b}"
        fig.suptitle(full_title, color="white", fontsize=13)

        img_name = f"diff_{uuid.uuid4().hex[:10]}.png"
        img_path = os.path.join(RENDERED_DIR, img_name)
        fig.savefig(img_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)

        return {
            "success": True,
            "image_path": f"/api/images/{img_name}",
            "caption": full_title,
            "residual_mad_rms": round(rms, 8),
            "residual_max_abs": round(max_abs, 8),
            "significant_pixel_fraction": round(sig_frac, 6),
            "overlap_pixels": int(overlap.sum()),
            "bunit": bunit or None,
            "scale_match": scale_applied,
            "warnings": warnings,
        }
    except Exception as e:
        logger.error(f"[FITS] difference_image failed: {e}")
        return {"success": False, "error": str(e)}
    finally:
        for p in (path_a, path_b):
            if p and os.path.exists(p):
                try:
                    os.unlink(p)
                except PermissionError:
                    logger.warning(f"[FITS] Could not remove temp file still in use: {p}")
        gc.collect()


def compute_moment_map(
    url: str,
    order: int = 0,
    title: str = "",
    colormap: str = "inferno",
    freq_min_ghz: float = None,
    freq_max_ghz: float = None,
) -> Dict[str, Any]:
    """
    Download a FITS spectral cube and compute a moment map.

    Moment 0 = Integrated intensity (total emission)
    Moment 1 = Velocity field (mean velocity at each pixel)
    Moment 2 = Velocity dispersion (spread of velocities)

    Returns:
        {"success": True, "image_path": "/api/images/xxx.png", "caption": "..."}
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from astropy.io import fits as afits
    from astropy.wcs import WCS
    from astropy.visualization import ZScaleInterval, ImageNormalize, SqrtStretch
    import astropy.units as u

    fits_path = None
    try:
        fits_path = _download_fits(url, title or "cube")

        try:
            from spectral_cube import SpectralCube
            cube = SpectralCube.read(fits_path, memmap=True)
        except Exception as e:
            # Fallback: if spectral-cube can't read it, try opening as regular FITS
            logger.warning(f"[FITS] spectral-cube failed ({e}), trying astropy fallback")
            return render_fits_image(url, title=title or "FITS Image", colormap=colormap)

        # Optional frequency range slab
        if freq_min_ghz is not None or freq_max_ghz is not None:
            try:
                lo = (freq_min_ghz * u.GHz) if freq_min_ghz else cube.spectral_axis.min()
                hi = (freq_max_ghz * u.GHz) if freq_max_ghz else cube.spectral_axis.max()
                cube = cube.spectral_slab(lo, hi)
            except Exception as e:
                logger.warning(f"[FITS] spectral_slab failed: {e}")

        # IMG-05: moment 1/2 on a frequency axis are frequency moments, not a
        # velocity field. Convert to velocity (radio convention, cube rest
        # frequency) when possible; otherwise label the map honestly and drop
        # the approaching/receding diverging colormap, whose blue/red sense is
        # inverted for frequency.
        result_warnings = []
        axis_is_freq = cube.spectral_axis.unit.is_equivalent(u.Hz)
        if order in (1, 2) and axis_is_freq:
            try:
                cube = cube.with_spectral_unit(u.km / u.s, velocity_convention="radio")
                result_warnings.append(
                    "Spectral axis converted from frequency to velocity "
                    "(radio convention, cube rest frequency) for the moment map."
                )
            except Exception as conv_err:
                logger.warning(f"[FITS] frequency→velocity conversion unavailable: {conv_err}")

        velocity_axis = cube.spectral_axis.unit.is_equivalent(u.m / u.s)
        if velocity_axis or order == 0:
            moment_labels = {
                0: "Integrated Intensity (Moment 0)",
                1: "Velocity Field (Moment 1)",
                2: "Velocity Dispersion (Moment 2)",
            }
        else:
            axis_word = "Frequency" if axis_is_freq else f"Spectral ({cube.spectral_axis.unit})"
            moment_labels = {
                1: f"Mean {axis_word} (Moment 1)",
                2: f"{axis_word} Dispersion (Moment 2)",
            }
            result_warnings.append(
                "The cube has no usable rest frequency, so moment "
                f"{order} is in spectral-axis units ({cube.spectral_axis.unit}), not velocity."
            )

        # Compute moment — suppress NaN warnings
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            moment = cube.moment(order=order)

        data_2d = moment.value
        wcs_2d = moment.wcs

        # Render
        norm = ImageNormalize(
            data_2d[np.isfinite(data_2d)],
            interval=ZScaleInterval(),
            stretch=SqrtStretch(),
        )

        # Choose colormap based on moment type
        if order == 1 and velocity_axis:
            cmap = "RdBu_r"   # velocity field: blue = approaching, red = receding
        elif order == 1:
            cmap = "viridis"  # IMG-05: non-diverging for a mean-frequency map
        elif order == 2:
            cmap = "magma"    # dispersion: dark = calm, bright = turbulent
        else:
            cmap = colormap   # moment 0: user choice or default

        fig = plt.figure(figsize=(10, 10), facecolor="black")
        ax = fig.add_subplot(111, projection=wcs_2d)

        im = ax.imshow(data_2d, origin="lower", cmap=cmap, norm=norm)

        # Colorbar
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        unit_str = str(moment.unit) if hasattr(moment, 'unit') else ""
        cbar.set_label(unit_str, color="white", fontsize=10)
        cbar.ax.yaxis.set_tick_params(color="white")
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white", fontsize=8)

        label = moment_labels.get(order, f"Moment {order}")
        full_title = f"{title} — {label}" if title else label
        ax.set_title(full_title, color="white", fontsize=14, pad=12)

        ax.coords[0].set_axislabel("RA (J2000)", color="white", fontsize=10)
        ax.coords[1].set_axislabel("Dec (J2000)", color="white", fontsize=10)
        ax.coords[0].set_ticklabel(color="white", fontsize=8)
        ax.coords[1].set_ticklabel(color="white", fontsize=8)
        ax.set_facecolor("black")
        fig.patch.set_facecolor("#0f172a")

        img_name = f"moment{order}_{uuid.uuid4().hex[:10]}.png"
        img_path = os.path.join(RENDERED_DIR, img_name)
        fig.savefig(img_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)

        return {
            "success": True,
            "image_path": f"/api/images/{img_name}",
            "caption": full_title,
            "moment_unit": unit_str,  # IMG-05: spectral-axis unit is explicit
            "warnings": result_warnings,
        }

    except Exception as e:
        logger.error(f"[FITS] compute_moment_map failed: {e}")
        return {"success": False, "error": str(e)}

    finally:
        gc.collect()  # release memmap handles before unlink (Windows)
        if fits_path and os.path.exists(fits_path):
            try:
                os.unlink(fits_path)
            except PermissionError:
                logger.warning(f"[FITS] Could not remove temp file still in use: {fits_path}")


def generate_channel_maps(
    url: str,
    n_channels: int = 12,
    title: str = "",
    colormap: str = "inferno",
    freq_min_ghz: float = None,
    freq_max_ghz: float = None,
    vel_min_kms: float = None,
    vel_max_kms: float = None,
) -> Dict[str, Any]:
    """
    Download a FITS spectral cube and render a grid of velocity/frequency
    channel maps (the classic "channel map" figure of radio astronomy).

    Channels are selected evenly across the (optionally restricted)
    spectral axis and share a common intensity normalization so emission
    can be compared between panels.

    Args:
        url: FITS cube URL or local path
        n_channels: Number of panels to render (4-24, default 12)
        title: Figure title prefix
        colormap: Matplotlib colormap for the panels
        freq_min_ghz / freq_max_ghz: Optional frequency slab bounds
        vel_min_kms / vel_max_kms: Optional velocity slab bounds (used if
            the cube's spectral axis is in velocity, or convertible)

    Returns:
        {"success": True, "image_path": "/api/images/xxx.png", "caption": "..."}
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from astropy.visualization import ZScaleInterval, ImageNormalize, SqrtStretch
    import astropy.units as u

    fits_path = None
    try:
        n_channels = int(max(4, min(24, n_channels)))

        fits_path = _download_fits(url, title or "channel maps")

        try:
            from spectral_cube import SpectralCube
            cube = SpectralCube.read(fits_path, memmap=True)
        except Exception as e:
            logger.error(f"[FITS] spectral-cube can't read file: {e}")
            return {"success": False, "error": f"Not a spectral cube: {e}"}

        if cube.shape[0] < 2:
            return {"success": False,
                    "error": "Cube has fewer than 2 spectral channels — channel maps need a 3D cube"}

        # ── Optional spectral slab ────────────────────────────────────
        try:
            if freq_min_ghz is not None or freq_max_ghz is not None:
                lo = (freq_min_ghz * u.GHz) if freq_min_ghz else cube.spectral_axis.min()
                hi = (freq_max_ghz * u.GHz) if freq_max_ghz else cube.spectral_axis.max()
                cube = cube.spectral_slab(lo, hi)
            elif vel_min_kms is not None or vel_max_kms is not None:
                lo = (vel_min_kms * u.km / u.s) if vel_min_kms is not None else cube.spectral_axis.min()
                hi = (vel_max_kms * u.km / u.s) if vel_max_kms is not None else cube.spectral_axis.max()
                cube = cube.spectral_slab(lo, hi)
        except Exception as e:
            logger.warning(f"[FITS] spectral_slab failed (continuing with full cube): {e}")

        n_spec = cube.shape[0]
        n_channels = min(n_channels, n_spec)

        # Evenly spaced channel indices across the spectral axis
        idx = np.unique(np.linspace(0, n_spec - 1, n_channels).round().astype(int))

        # ── Per-channel labels in natural units ──────────────────────
        spec_axis = cube.spectral_axis
        if spec_axis.unit.is_equivalent(u.m / u.s):
            ch_vals = spec_axis.to(u.km / u.s).value
            ch_fmt = "{:.1f} km/s"
        elif spec_axis.unit.is_equivalent(u.Hz):
            ch_vals = spec_axis.to(u.GHz).value
            ch_fmt = "{:.4f} GHz"
        else:
            ch_vals = spec_axis.value
            ch_fmt = "{:.4g} " + str(spec_axis.unit)

        # ── Shared normalization across all selected channels ────────
        import warnings as _warnings
        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore")
            sample = np.concatenate([
                np.asarray(cube[i].value, dtype=float).ravel() for i in idx
            ])
        sample = sample[np.isfinite(sample)]
        if sample.size == 0:
            return {"success": False, "error": "All selected channels are empty (NaN)"}
        norm = ImageNormalize(sample, interval=ZScaleInterval(), stretch=SqrtStretch())

        # ── Grid layout ───────────────────────────────────────────────
        ncols = 4 if len(idx) > 9 else 3
        nrows = int(math.ceil(len(idx) / ncols))

        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(3.2 * ncols, 3.2 * nrows),
            facecolor="#0f172a",
            squeeze=False,
        )

        im = None
        for panel, ax in enumerate(axes.flat):
            if panel >= len(idx):
                ax.set_visible(False)
                continue
            ch = int(idx[panel])
            with _warnings.catch_warnings():
                _warnings.simplefilter("ignore")
                data = np.asarray(cube[ch].value, dtype=float)
            im = ax.imshow(data, origin="lower", cmap=colormap, norm=norm)
            ax.text(
                0.04, 0.94, ch_fmt.format(ch_vals[ch]),
                transform=ax.transAxes, color="white", fontsize=9,
                va="top", ha="left",
                bbox=dict(facecolor="black", alpha=0.55, edgecolor="none", pad=2),
            )
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_facecolor("black")

        if im is not None:
            cbar = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02)
            cbar.set_label(str(cube.unit), color="white", fontsize=10)
            cbar.ax.yaxis.set_tick_params(color="white")
            plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white", fontsize=8)

        full_title = f"{title} — Channel Maps" if title else "Channel Maps"
        fig.suptitle(full_title, color="white", fontsize=15, y=0.995)

        img_name = f"chanmap_{uuid.uuid4().hex[:10]}.png"
        img_path = os.path.join(RENDERED_DIR, img_name)
        fig.savefig(img_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)

        return {
            "success": True,
            "image_path": f"/api/images/{img_name}",
            "caption": full_title,
            "n_panels": int(len(idx)),
            "spectral_range": f"{ch_vals[0]:.4g} to {ch_vals[-1]:.4g} ({ch_fmt.split(' ', 1)[-1].strip('{}')})"
                              if len(ch_vals) else "",
        }

    except Exception as e:
        logger.error(f"[FITS] generate_channel_maps failed: {e}")
        return {"success": False, "error": str(e)}

    finally:
        gc.collect()  # release memmap handles before unlink (Windows)
        if fits_path and os.path.exists(fits_path):
            try:
                os.unlink(fits_path)
            except PermissionError:
                logger.warning(f"[FITS] Could not remove temp file still in use: {fits_path}")


def pv_slice(
    url: str,
    ra_start: float = None,
    dec_start: float = None,
    ra_end: float = None,
    dec_end: float = None,
    width_arcsec: float = None,
    title: str = "",
) -> Dict[str, Any]:
    """
    Extract a position-velocity diagram along an arbitrary sky path using
    pvextractor (the canonical rotation/outflow diagnostic for cubes).

    The path runs from (ra_start, dec_start) to (ra_end, dec_end); an optional
    width in arcsec averages perpendicular to the path. Unlike a fixed
    pixel-row slice, this handles inclined disks, outflows, and filaments at
    any position angle.

    Returns:
        {"success": True, "image_path": "/api/images/xxx.png", "caption": "...",
         "path_length_arcsec": ..., "position_angle_deg": ...}
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import astropy.units as u
    from astropy.coordinates import SkyCoord

    fits_path = None
    try:
        for name, value in (("ra_start", ra_start), ("dec_start", dec_start),
                            ("ra_end", ra_end), ("dec_end", dec_end)):
            if value is None:
                return {"success": False, "error": f"{name} is required (path endpoints in ICRS degrees)."}
        if width_arcsec is not None and not (math.isfinite(float(width_arcsec)) and float(width_arcsec) > 0):
            return {"success": False, "error": "width_arcsec must be a positive number of arcseconds."}

        fits_path = _download_fits(url, title or "pv slice")
        try:
            from spectral_cube import SpectralCube
            cube = SpectralCube.read(fits_path, memmap=True)
        except Exception as e:
            logger.error(f"[FITS] spectral-cube can't read file: {e}")
            return {"success": False, "error": f"Not a spectral cube: {e}"}

        warnings_list = []
        # Prefer a velocity axis when the cube has a rest frequency
        if cube.spectral_axis.unit.is_equivalent(u.Hz):
            try:
                cube = cube.with_spectral_unit(u.km / u.s, velocity_convention="radio")
            except Exception as exc:
                warnings_list.append(f"Could not convert to velocity ({exc}); using native spectral units.")

        from pvextractor import Path as PvPath, extract_pv_slice

        start = SkyCoord(ra=float(ra_start) * u.deg, dec=float(dec_start) * u.deg, frame="icrs")
        end = SkyCoord(ra=float(ra_end) * u.deg, dec=float(dec_end) * u.deg, frame="icrs")
        length_arcsec = float(start.separation(end).arcsec)
        pa_deg = float(start.position_angle(end).deg)
        if length_arcsec <= 0:
            return {"success": False, "error": "Path endpoints coincide — pick two distinct positions."}

        path_coords = SkyCoord([start, end])
        if width_arcsec and float(width_arcsec) > 0:
            path = PvPath(path_coords, width=float(width_arcsec) * u.arcsec)
        else:
            path = PvPath(path_coords)

        import warnings as _w
        with _w.catch_warnings():
            _w.simplefilter("ignore")
            pv = extract_pv_slice(cube, path)

        data = np.asarray(pv.data, dtype=float)
        if not np.isfinite(data).any():
            return {"success": False, "error": "The PV slice contains no finite pixels — "
                                               "check that the path crosses the cube's field."}

        # Axis extents from the PV header (axis 1 = offset, axis 2 = spectral)
        header = pv.header
        ny, nx = data.shape
        off0 = float(header.get("CRVAL1", 0.0)) + (1 - float(header.get("CRPIX1", 1))) * float(header.get("CDELT1", 1.0))
        off1 = off0 + (nx - 1) * float(header.get("CDELT1", 1.0))
        sp0 = float(header.get("CRVAL2", 0.0)) + (1 - float(header.get("CRPIX2", 1))) * float(header.get("CDELT2", 1.0))
        sp1 = sp0 + (ny - 1) * float(header.get("CDELT2", 1.0))
        cunit1 = str(header.get("CUNIT1", "deg") or "deg").strip().lower()
        cunit2 = str(header.get("CUNIT2", "") or "").strip().lower()
        if cunit1 in ("deg", "degree", "degrees"):
            off0, off1 = off0 * 3600.0, off1 * 3600.0
        elif cunit1 == "arcmin":
            off0, off1 = off0 * 60.0, off1 * 60.0
        if cunit2 in ("m/s", "m s-1", "m.s**-1"):
            sp0, sp1, sp_label = sp0 / 1e3, sp1 / 1e3, "Velocity (km/s)"
        elif cunit2 in ("km/s", "km s-1"):
            sp_label = "Velocity (km/s)"
        elif cunit2 in ("hz",):
            sp0, sp1, sp_label = sp0 / 1e9, sp1 / 1e9, "Frequency (GHz)"
        else:
            sp_label = f"Spectral axis ({cunit2 or 'native'})"

        from astropy.visualization import ZScaleInterval, ImageNormalize, SqrtStretch
        norm = ImageNormalize(data[np.isfinite(data)], interval=ZScaleInterval(), stretch=SqrtStretch())

        fig, ax = plt.subplots(figsize=(10.5, 6), facecolor="#0f172a")
        ax.set_facecolor("black")
        im = ax.imshow(data, origin="lower", cmap="inferno", norm=norm, aspect="auto",
                       extent=[off0, off1, sp0, sp1])
        ax.set_xlabel("Offset along path (arcsec)", color="white", fontsize=11)
        ax.set_ylabel(sp_label, color="white", fontsize=11)
        ax.tick_params(colors="white", labelsize=9)
        for spine in ax.spines.values():
            spine.set_color("#334155")
        cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.03)
        cbar.set_label(str(cube.unit), color="white", fontsize=9)
        cbar.ax.yaxis.set_tick_params(color="white")
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white", fontsize=8)
        width_txt = f", width {float(width_arcsec):g}\"" if width_arcsec else ""
        full_title = (f"{title} — PV slice (PA {pa_deg:.1f}°, {length_arcsec:.1f}\"{width_txt})"
                      if title else f"PV slice (PA {pa_deg:.1f}°, {length_arcsec:.1f}\"{width_txt})")
        ax.set_title(full_title, color="white", fontsize=12, pad=10)

        img_name = f"pvslice_{uuid.uuid4().hex[:10]}.png"
        img_path = os.path.join(RENDERED_DIR, img_name)
        fig.savefig(img_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)

        return {
            "success": True,
            "image_path": f"/api/images/{img_name}",
            "caption": full_title,
            "path_length_arcsec": round(length_arcsec, 2),
            "position_angle_deg": round(pa_deg, 2),
            "width_arcsec": (round(float(width_arcsec), 2) if width_arcsec else None),
            "spectral_axis": sp_label,
            "shape": [int(ny), int(nx)],
            "warnings": warnings_list,
        }

    except Exception as e:
        logger.error(f"[FITS] pv_slice failed: {e}")
        return {"success": False, "error": str(e)}

    finally:
        gc.collect()  # release memmap handles before unlink (Windows)
        if fits_path and os.path.exists(fits_path):
            try:
                os.unlink(fits_path)
            except PermissionError:
                logger.warning(f"[FITS] Could not remove temp file still in use: {fits_path}")


def _footprint_error(cube, ra_deg: float, dec_deg: float,
                     x_pixel: Optional[int], y_pixel: Optional[int],
                     nx: int, ny: int) -> str:
    """Typed out-of-footprint message naming requested vs covered sky (IMG-02)."""
    if x_pixel is not None and y_pixel is not None:
        where = f"falls at pixel ({x_pixel}, {y_pixel}), outside"
    else:
        where = "does not project onto the image plane and is outside"
    message = (
        f"Requested position RA={ra_deg:.5f}, Dec={dec_deg:.5f} {where} "
        f"the {nx}x{ny} pixel cube footprint."
    )
    try:
        lo = cube.wcs.celestial.pixel_to_world(0, 0)
        hi = cube.wcs.celestial.pixel_to_world(nx - 1, ny - 1)
        message += (
            f" The cube covers RA {min(lo.ra.deg, hi.ra.deg):.5f} to "
            f"{max(lo.ra.deg, hi.ra.deg):.5f}, Dec {min(lo.dec.deg, hi.dec.deg):.5f} "
            f"to {max(lo.dec.deg, hi.dec.deg):.5f} (deg)."
        )
    except Exception:
        pass
    return message


def extract_spectrum(
    url: str,
    ra_deg: float = None,
    dec_deg: float = None,
    x_pixel: int = None,
    y_pixel: int = None,
    title: str = "",
    radius_arcsec: float = None,
) -> Dict[str, Any]:
    """
    Download a FITS spectral cube and extract a 1D spectrum at a given
    position (RA/Dec or pixel coordinates).

    If no position is given, extracts the spectrum at the cube center
    (peak emission pixel of collapsed image).

    With ``radius_arcsec`` the spectrum is APERTURE-INTEGRATED (summed over a
    circular aperture); for Jy/beam cubes with a restoring beam the result is
    converted to Jy. Single-pixel spectra underestimate the flux of anything
    resolved — prefer an aperture for photometric statements.

    Returns:
        {"success": True, "image_path": "/api/images/xxx.png", "caption": "..."}
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import astropy.units as u

    fits_path = None
    try:
        fits_path = _download_fits(url, title or "spectrum")

        try:
            from spectral_cube import SpectralCube
            cube = SpectralCube.read(fits_path, memmap=True)
        except Exception as e:
            logger.error(f"[FITS] spectral-cube can't read file: {e}")
            return {"success": False, "error": f"Not a spectral cube: {e}"}

        # Determine pixel position
        ny, nx = cube.shape[1], cube.shape[2]
        if ra_deg is not None and dec_deg is not None:
            # Convert RA/Dec to pixel via WCS
            from astropy.coordinates import SkyCoord
            coord = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame="icrs")
            try:
                # astropy world_to_pixel returns (x, y) — assigning it as
                # (y, x) transposed every off-center RA/Dec position (CX-05).
                x_pix, y_pix = cube.wcs.celestial.world_to_pixel(coord)
                x_val, y_val = float(x_pix), float(y_pix)
            except Exception as wcs_err:
                # IMG-02: never silently fall back to the cube center while
                # the title still claims the requested RA/Dec — fail loudly.
                raise ValueError(
                    f"Could not convert RA={ra_deg:.5f}, Dec={dec_deg:.5f} to a "
                    f"pixel via the cube WCS: {wcs_err}"
                )
            if not (math.isfinite(x_val) and math.isfinite(y_val)):
                # IMG-02: unprojectable position == outside the footprint.
                raise ValueError(_footprint_error(cube, ra_deg, dec_deg, None, None, nx, ny))
            x_pixel = int(round(x_val))
            y_pixel = int(round(y_val))
            if not (0 <= x_pixel < nx and 0 <= y_pixel < ny):
                # IMG-02: out-of-footprint positions must error with the
                # requested vs covered region, never clamp to an edge pixel.
                raise ValueError(_footprint_error(cube, ra_deg, dec_deg, x_pixel, y_pixel, nx, ny))
        elif x_pixel is not None and y_pixel is not None:
            pass  # use provided pixel coords
        else:
            # Find peak pixel from moment 0
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                collapsed = cube.moment(order=0).value
            collapsed = np.nan_to_num(collapsed, nan=0)
            peak = np.unravel_index(np.argmax(collapsed), collapsed.shape)
            y_pixel, x_pixel = int(peak[0]), int(peak[1])

        # Bounds clamp for the derived-position paths (peak pixel is always
        # in-bounds; explicit pixel coords keep the legacy clamp) — the RA/Dec
        # path above errors instead of clamping (IMG-02).
        x_pixel = max(0, min(x_pixel, nx - 1))
        y_pixel = max(0, min(y_pixel, ny - 1))

        # Extract 1D spectrum: single pixel, or aperture-integrated
        aperture_info = None
        if radius_arcsec is not None and float(radius_arcsec) > 0:
            from astropy.wcs.utils import proj_plane_pixel_scales

            pix_deg = float(np.mean(np.abs(proj_plane_pixel_scales(cube.wcs.celestial))))
            r_pix = float(radius_arcsec) / 3600.0 / pix_deg if pix_deg > 0 else 0.0
            if r_pix < 1.0:
                r_pix = 1.0
            yy, xx = np.mgrid[0:ny, 0:nx]
            mask2d = (xx - x_pixel) ** 2 + (yy - y_pixel) ** 2 <= r_pix ** 2
            n_ap = int(mask2d.sum())
            import warnings as _w
            with _w.catch_warnings():
                _w.simplefilter("ignore")
                spectrum = cube.with_mask(mask2d[np.newaxis, :, :]).sum(axis=(1, 2))
            aperture_info = {
                "radius_arcsec": round(float(radius_arcsec), 3),
                "radius_pix": round(r_pix, 2),
                "n_pixels": n_ap,
            }
        else:
            spectrum = cube[:, y_pixel, x_pixel]
        spec_axis = cube.spectral_axis
        flux = spectrum.value

        # Determine spectral axis label
        if spec_axis.unit.is_equivalent(u.Hz):
            x_data = spec_axis.to(u.GHz).value
            x_label = "Frequency (GHz)"
        elif spec_axis.unit.is_equivalent(u.m / u.s):
            x_data = spec_axis.to(u.km / u.s).value
            x_label = "Velocity (km/s)"
        elif spec_axis.unit.is_equivalent(u.m):
            x_data = spec_axis.to(u.um).value
            x_label = "Wavelength (μm)"
        else:
            x_data = spec_axis.value
            x_label = f"Spectral Axis ({spec_axis.unit})"

        flux_unit = str(spectrum.unit) if hasattr(spectrum, 'unit') else "Flux"
        # Aperture sums of Jy/beam maps convert to Jy via the beam area
        if aperture_info is not None and "jy" in flux_unit.lower() and "beam" in flux_unit.lower():
            try:
                beam = getattr(cube, "beam", None)
                if beam is not None:
                    pix_area_sr = ((pix_deg * u.deg) ** 2).to(u.sr)
                    beam_pix = float((beam.sr / pix_area_sr).decompose().value)
                    if beam_pix > 0:
                        flux = flux / beam_pix
                        flux_unit = "Jy"
                        aperture_info["beam_area_pix"] = round(beam_pix, 3)
            except Exception as beam_err:
                logger.warning(f"[FITS] Jy/beam→Jy conversion skipped: {beam_err}")

        # Plot
        fig, ax = plt.subplots(figsize=(12, 5), facecolor="#0f172a")
        ax.set_facecolor("#0f172a")

        ax.plot(x_data, flux, color="#06b6d4", linewidth=0.8, alpha=0.9)
        ax.fill_between(x_data, flux, alpha=0.15, color="#06b6d4")

        ax.set_xlabel(x_label, color="white", fontsize=11)
        ax.set_ylabel(flux_unit, color="white", fontsize=11)
        ax.tick_params(colors="white", labelsize=9)
        for spine in ax.spines.values():
            spine.set_color("#334155")

        pos_label = f"pixel ({x_pixel}, {y_pixel})"
        if ra_deg is not None:
            pos_label = f"RA={ra_deg:.4f}°, Dec={dec_deg:.4f}°"
        if aperture_info is not None:
            pos_label += f" (aperture r={aperture_info['radius_arcsec']:g}\")"
        full_title = f"{title} — Spectrum at {pos_label}" if title else f"Spectrum at {pos_label}"
        ax.set_title(full_title, color="white", fontsize=13, pad=10)

        ax.axhline(0, color="#475569", linewidth=0.5, linestyle="--")

        img_name = f"spectrum_{uuid.uuid4().hex[:10]}.png"
        img_path = os.path.join(RENDERED_DIR, img_name)
        fig.savefig(img_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)

        result = {
            "success": True,
            "image_path": f"/api/images/{img_name}",
            "caption": full_title,
            "pixel_position": {"x": x_pixel, "y": y_pixel},
            "n_channels": len(flux),
            "flux_unit": flux_unit,
            "spectral_range": f"{x_data[0]:.4f} – {x_data[-1]:.4f} {x_label.split('(')[-1].rstrip(')')}",
        }
        if aperture_info is not None:
            result["aperture"] = aperture_info
            result["note"] = ("Aperture-integrated spectrum"
                              + (" converted to Jy via the beam area." if flux_unit == "Jy" else "."))
        return result

    except Exception as e:
        logger.error(f"[FITS] extract_spectrum failed: {e}")
        return {"success": False, "error": str(e)}

    finally:
        gc.collect()  # release memmap handles before unlink (Windows)
        if fits_path and os.path.exists(fits_path):
            try:
                os.unlink(fits_path)
            except PermissionError:
                logger.warning(f"[FITS] Could not remove temp file still in use: {fits_path}")


def fit_spectral_line(
    url: str,
    ra_deg: float = None,
    dec_deg: float = None,
    x_pixel: int = None,
    y_pixel: int = None,
    title: str = "",
    fit_gaussian: bool = True,
) -> Dict[str, Any]:
    """
    Extract a 1D spectrum from a FITS cube and optionally fit a Gaussian
    profile to the strongest spectral line.

    Returns:
        {
          "success": True,
          "image_path": "/api/images/xxx.png",
          "gaussian_fit": {
            "center_ghz": ..., "fwhm_km_s": ...,
            "peak_flux": ..., "integrated_flux": ...
          }
        }
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import astropy.units as u

    fits_path = None
    try:
        fits_path = _download_fits(url, title or "line_fit")

        try:
            from spectral_cube import SpectralCube
            cube = SpectralCube.read(fits_path, memmap=True)
        except Exception as e:
            logger.error(f"[FITS] spectral-cube can't read file: {e}")
            return {"success": False, "error": f"Not a spectral cube: {e}"}

        # Determine pixel position (same logic as extract_spectrum)
        ny, nx = cube.shape[1], cube.shape[2]
        if ra_deg is not None and dec_deg is not None:
            from astropy.coordinates import SkyCoord
            coord = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame="icrs")
            try:
                # astropy world_to_pixel returns (x, y) — assigning it as
                # (y, x) transposed every off-center RA/Dec position (CX-05).
                x_pix, y_pix = cube.wcs.celestial.world_to_pixel(coord)
                x_val, y_val = float(x_pix), float(y_pix)
            except Exception as wcs_err:
                # IMG-02: no silent center fallback under a title that still
                # claims the requested RA/Dec.
                raise ValueError(
                    f"Could not convert RA={ra_deg:.5f}, Dec={dec_deg:.5f} to a "
                    f"pixel via the cube WCS: {wcs_err}"
                )
            if not (math.isfinite(x_val) and math.isfinite(y_val)):
                # IMG-02: unprojectable position == outside the footprint.
                raise ValueError(_footprint_error(cube, ra_deg, dec_deg, None, None, nx, ny))
            x_pixel = int(round(x_val))
            y_pixel = int(round(y_val))
            if not (0 <= x_pixel < nx and 0 <= y_pixel < ny):
                # IMG-02: out-of-footprint positions error, never edge-clamp.
                raise ValueError(_footprint_error(cube, ra_deg, dec_deg, x_pixel, y_pixel, nx, ny))
        elif x_pixel is not None and y_pixel is not None:
            pass
        else:
            # Peak pixel from moment 0
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                collapsed = cube.moment(order=0).value
            collapsed = np.nan_to_num(collapsed, nan=0)
            peak = np.unravel_index(np.argmax(collapsed), collapsed.shape)
            y_pixel, x_pixel = int(peak[0]), int(peak[1])

        # Bounds clamp for the derived/explicit-pixel paths only (IMG-02).
        x_pixel = max(0, min(x_pixel, nx - 1))
        y_pixel = max(0, min(y_pixel, ny - 1))

        # Extract 1D spectrum
        spectrum = cube[:, y_pixel, x_pixel]
        spec_axis = cube.spectral_axis
        flux = spectrum.value.copy()

        # Determine spectral axis
        if spec_axis.unit.is_equivalent(u.Hz):
            x_data = spec_axis.to(u.GHz).value
            x_label = "Frequency (GHz)"
        elif spec_axis.unit.is_equivalent(u.m / u.s):
            x_data = spec_axis.to(u.km / u.s).value
            x_label = "Velocity (km/s)"
        elif spec_axis.unit.is_equivalent(u.m):
            x_data = spec_axis.to(u.um).value
            x_label = "Wavelength (um)"
        else:
            x_data = spec_axis.value
            x_label = f"Spectral Axis ({spec_axis.unit})"

        flux_unit = str(spectrum.unit) if hasattr(spectrum, 'unit') else "Flux"

        # Replace NaN
        finite_mask = np.isfinite(flux)
        flux_clean = np.where(finite_mask, flux, 0.0)

        # Gaussian fit
        fit_result = None
        gaussian_curve = None
        if fit_gaussian and len(x_data) > 5:
            try:
                from scipy.optimize import curve_fit

                def gaussian(x, amp, center, sigma, offset):
                    return amp * np.exp(-0.5 * ((x - center) / sigma) ** 2) + offset

                # Initial guesses: peak value, center of peak, rough width, baseline
                peak_idx = np.argmax(flux_clean)
                amp_guess = flux_clean[peak_idx] - np.median(flux_clean)
                center_guess = x_data[peak_idx]
                sigma_guess = abs(x_data[-1] - x_data[0]) / 20
                offset_guess = np.median(flux_clean)

                popt, pcov = curve_fit(
                    gaussian, x_data[finite_mask], flux[finite_mask],
                    p0=[amp_guess, center_guess, sigma_guess, offset_guess],
                    maxfev=5000,
                )

                amp, center, sigma, offset = popt
                fwhm = abs(sigma) * 2.3548  # FWHM = 2*sqrt(2*ln2)*sigma
                integrated = abs(amp) * abs(sigma) * math.sqrt(2 * math.pi)

                # Generate smooth Gaussian curve for plotting
                x_smooth = np.linspace(x_data.min(), x_data.max(), 500)
                gaussian_curve = (x_smooth, gaussian(x_smooth, *popt))

                # Convert FWHM to velocity if x-axis is frequency
                if "GHz" in x_label and center > 0:
                    fwhm_vel = (fwhm / center) * 299792.458  # km/s
                else:
                    fwhm_vel = None

                fit_result = {
                    "amplitude": round(float(amp), 6),
                    "center": round(float(center), 6),
                    "center_unit": x_label.split("(")[-1].rstrip(")"),
                    "sigma": round(float(abs(sigma)), 6),
                    "fwhm": round(float(fwhm), 4),
                    "fwhm_unit": x_label.split("(")[-1].rstrip(")"),
                    "offset": round(float(offset), 6),
                    "integrated_flux": round(float(integrated), 6),
                }
                if fwhm_vel is not None:
                    fit_result["fwhm_km_s"] = round(fwhm_vel, 2)

            except Exception as fit_err:
                logger.warning(f"[FITS] Gaussian fit failed: {fit_err}")
                fit_result = {"fit_error": str(fit_err)}

        # Plot
        fig, ax = plt.subplots(figsize=(12, 5), facecolor="#0f172a")
        ax.set_facecolor("#0f172a")

        ax.plot(x_data, flux, color="#06b6d4", linewidth=0.8, alpha=0.9,
                label="Spectrum")
        ax.fill_between(x_data, flux, alpha=0.1, color="#06b6d4")

        # Overlay Gaussian fit
        if gaussian_curve is not None:
            x_smooth, y_smooth = gaussian_curve
            ax.plot(x_smooth, y_smooth, color="#f59e0b", linewidth=2,
                    linestyle="--", label="Gaussian fit", alpha=0.9)
            # Mark center and FWHM
            if fit_result and "center" in fit_result and "fwhm" in fit_result:
                c = fit_result["center"]
                hw = fit_result["fwhm"] / 2
                half_max = fit_result["amplitude"] / 2 + fit_result["offset"]
                ax.plot([c - hw, c + hw], [half_max, half_max],
                        color="#f59e0b", linewidth=1.5, alpha=0.7)
                # Annotate
                fwhm_text = f'FWHM = {fit_result["fwhm"]:.3f} {fit_result.get("fwhm_unit", "")}'
                if "fwhm_km_s" in fit_result:
                    fwhm_text += f' ({fit_result["fwhm_km_s"]:.1f} km/s)'
                ax.text(0.02, 0.95, fwhm_text, transform=ax.transAxes,
                        color="#f59e0b", fontsize=10, va="top",
                        bbox=dict(boxstyle="round,pad=0.3",
                                  facecolor="#1e293b", edgecolor="#f59e0b",
                                  alpha=0.8))

        ax.set_xlabel(x_label, color="white", fontsize=11)
        ax.set_ylabel(flux_unit, color="white", fontsize=11)
        ax.tick_params(colors="white", labelsize=9)
        for spine in ax.spines.values():
            spine.set_color("#334155")

        pos_label = f"pixel ({x_pixel}, {y_pixel})"
        if ra_deg is not None:
            pos_label = f"RA={ra_deg:.4f}, Dec={dec_deg:.4f}"
        full_title = f"{title} -- Line Profile at {pos_label}" if title else f"Line Profile at {pos_label}"
        ax.set_title(full_title, color="white", fontsize=13, pad=10)
        ax.axhline(0, color="#475569", linewidth=0.5, linestyle="--")

        if gaussian_curve is not None:
            ax.legend(loc="upper right", fontsize=9,
                      facecolor="#1e293b", edgecolor="#334155",
                      labelcolor="white")

        img_name = f"linefit_{uuid.uuid4().hex[:10]}.png"
        img_path = os.path.join(RENDERED_DIR, img_name)
        fig.savefig(img_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)

        result = {
            "success": True,
            "image_path": f"/api/images/{img_name}",
            "caption": full_title,
            "pixel_position": {"x": x_pixel, "y": y_pixel},
            "n_channels": len(flux),
            "spectral_range": f"{x_data[0]:.4f} -- {x_data[-1]:.4f} {x_label.split('(')[-1].rstrip(')')}",
        }
        if fit_result:
            result["gaussian_fit"] = fit_result

        return result

    except Exception as e:
        logger.error(f"[FITS] fit_spectral_line failed: {e}")
        return {"success": False, "error": str(e)}

    finally:
        gc.collect()  # release memmap handles before unlink (Windows)
        if fits_path and os.path.exists(fits_path):
            try:
                os.unlink(fits_path)
            except PermissionError:
                logger.warning(f"[FITS] Could not remove temp file still in use: {fits_path}")
