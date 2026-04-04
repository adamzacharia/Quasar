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
from typing import Dict, Any, Optional, Tuple

import numpy as np
import requests as http_requests

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────
MAX_DOWNLOAD_MB = 150              # Reject files larger than this
DOWNLOAD_TIMEOUT_S = 180           # Timeout for HTTP download
RENDERED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "rendered_images")
os.makedirs(RENDERED_DIR, exist_ok=True)


def _pick_science_hdu(hdul) -> Tuple[Any, int]:
    """Find the first HDU with 2D image data (skip tables, empty primary)."""
    from astropy.io import fits as afits  # noqa: F811

    for i, hdu in enumerate(hdul):
        if hdu.data is not None and hdu.data.ndim >= 2:
            # If 3D+ cube, collapse to 2D by taking the middle slice or summing
            data = hdu.data
            while data.ndim > 2:
                mid = data.shape[0] // 2
                data = data[mid]  # take middle channel
            return data, i
    raise ValueError("No 2D image data found in FITS file")


def _download_fits(url: str, label: str = "FITS") -> str:
    """Download a FITS file from URL to a temp file. Returns path."""
    logger.info(f"[FITS] Downloading {label} from {url}")

    resp = http_requests.get(url, timeout=DOWNLOAD_TIMEOUT_S, stream=True)
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
            data_2d, hdu_idx = _pick_science_hdu(hdul)
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
        return {
            "success": True,
            "image_path": f"/api/images/{img_name}",
            "caption": caption,
        }

    except Exception as e:
        logger.error(f"[FITS] render_fits_image failed: {e}")
        return {"success": False, "error": str(e)}

    finally:
        if fits_path and os.path.exists(fits_path):
            os.unlink(fits_path)
        gc.collect()


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

        with afits.open(base_path, memmap=True) as base_hdul, \
             afits.open(contour_path, memmap=True) as cont_hdul:

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
                os.unlink(p)
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

        moment_labels = {
            0: "Integrated Intensity (Moment 0)",
            1: "Velocity Field (Moment 1)",
            2: "Velocity Dispersion (Moment 2)",
        }

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
        if order == 1:
            cmap = "RdBu_r"   # velocity field: blue = approaching, red = receding
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
        }

    except Exception as e:
        logger.error(f"[FITS] compute_moment_map failed: {e}")
        return {"success": False, "error": str(e)}

    finally:
        if fits_path and os.path.exists(fits_path):
            os.unlink(fits_path)
        gc.collect()


def extract_spectrum(
    url: str,
    ra_deg: float = None,
    dec_deg: float = None,
    x_pixel: int = None,
    y_pixel: int = None,
    title: str = "",
) -> Dict[str, Any]:
    """
    Download a FITS spectral cube and extract a 1D spectrum at a given
    position (RA/Dec or pixel coordinates).

    If no position is given, extracts the spectrum at the cube center
    (peak emission pixel of collapsed image).

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
        if ra_deg is not None and dec_deg is not None:
            # Convert RA/Dec to pixel via WCS
            from astropy.coordinates import SkyCoord
            coord = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame="icrs")
            try:
                y_pix, x_pix = cube.wcs.celestial.world_to_pixel(coord)
                x_pixel = int(round(float(x_pix)))
                y_pixel = int(round(float(y_pix)))
            except Exception:
                # Fallback to center
                y_pixel = cube.shape[1] // 2
                x_pixel = cube.shape[2] // 2
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

        # Bounds check
        ny, nx = cube.shape[1], cube.shape[2]
        x_pixel = max(0, min(x_pixel, nx - 1))
        y_pixel = max(0, min(y_pixel, ny - 1))

        # Extract 1D spectrum
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
        full_title = f"{title} — Spectrum at {pos_label}" if title else f"Spectrum at {pos_label}"
        ax.set_title(full_title, color="white", fontsize=13, pad=10)

        ax.axhline(0, color="#475569", linewidth=0.5, linestyle="--")

        img_name = f"spectrum_{uuid.uuid4().hex[:10]}.png"
        img_path = os.path.join(RENDERED_DIR, img_name)
        fig.savefig(img_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)

        return {
            "success": True,
            "image_path": f"/api/images/{img_name}",
            "caption": full_title,
            "pixel_position": {"x": x_pixel, "y": y_pixel},
            "n_channels": len(flux),
            "spectral_range": f"{x_data[0]:.4f} – {x_data[-1]:.4f} {x_label.split('(')[-1].rstrip(')')}",
        }

    except Exception as e:
        logger.error(f"[FITS] extract_spectrum failed: {e}")
        return {"success": False, "error": str(e)}

    finally:
        if fits_path and os.path.exists(fits_path):
            os.unlink(fits_path)
        gc.collect()
