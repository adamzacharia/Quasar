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
