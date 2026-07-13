"""
SkyViewClient — Sky Survey Image Retrieval for Quasar AI
Fetches cutout images from sky surveys (DSS2, 2MASS, SDSS, WISE, etc.)
via astroquery.skyview.SkyView.

CALLED BY: core/agent.py (tool execution: get_sky_image)
CALLS:     astroquery.skyview.SkyView

Registered agent tools:
    - get_sky_image(target_name, survey, radius_arcmin, pixels)
"""

import os
import warnings
from typing import Optional, Dict, Any, List
from pathlib import Path

try:
    from astroquery.skyview import SkyView
    SKYVIEW_AVAILABLE = True
except ImportError:
    SKYVIEW_AVAILABLE = False

# Canonical cached SIMBAD resolver
from integrations.simbad_resolver import _resolve_simbad_cached


class SkyViewClient:
    """
    Fetch sky survey cutout images (DSS2, 2MASS, SDSS, WISE, etc.).
    
    Returns FITS image data that can be displayed in the UI or saved to disk.
    Uses NASA SkyView virtual observatory via astroquery.
    """

    # Popular surveys grouped by wavelength
    SURVEYS = {
        # Optical
        "dss2_red": "DSS2 Red",
        "dss2_blue": "DSS2 Blue",
        "dss2_ir": "DSS2 IR",
        "dss": "DSS",
        "sdss_g": "SDSSg",
        "sdss_r": "SDSSr",
        "sdss_i": "SDSSi",
        # Near-IR
        "2mass_j": "2MASS-J",
        "2mass_h": "2MASS-H",
        "2mass_k": "2MASS-K",
        # Mid-IR
        "wise_3.4": "WISE 3.4",
        "wise_4.6": "WISE 4.6",
        "wise_12": "WISE 12",
        "wise_22": "WISE 22",
        # X-ray
        "rass": "RASS",
        # Radio
        "nvss": "NVSS",
        "first": "FIRST",
    }

    # Default survey aliases
    ALIASES = {
        "dss2": "DSS2 Red",
        "dss": "DSS",
        "sdss": "SDSSr",
        "2mass": "2MASS-J",
        "wise": "WISE 3.4",
        "optical": "DSS2 Red",
        "infrared": "2MASS-J",
        "radio": "NVSS",
        "xray": "RASS",
        "x-ray": "RASS",
    }

    def __init__(self, output_dir: str = None):
        if not SKYVIEW_AVAILABLE:
            warnings.warn("astroquery.skyview not installed. Sky image queries will fail.")
        self.output_dir = output_dir or os.path.join(
            os.path.expanduser("~"), "quasar_data", "sky_images"
        )

    def get_image(self, target: str = None, survey: str = "DSS2 Red",
                  ra: float = None, dec: float = None,
                  radius_arcmin: float = 5.0, pixels: int = 500,
                  save: bool = True) -> Dict[str, Any]:
        """
        Fetch a sky survey cutout image.
        
        Args:
            target: Astronomical target name (e.g., 'M87', 'NGC 1068')
            survey: Survey name (e.g., 'DSS2 Red', '2MASS-J', 'WISE 3.4')
            ra: RA in degrees (alternative to target)
            dec: Dec in degrees (alternative to target)
            radius_arcmin: Image radius in arcminutes (default 5')
            pixels: Image size in pixels (default 500)
            save: Whether to save the FITS file to disk
            
        Returns:
            Dict with image path, FITS HDU info, and metadata
        """
        if not SKYVIEW_AVAILABLE:
            return {"success": False, "error": "astroquery.skyview not available"}
        
        try:
            # Resolve survey alias
            survey_name = self._resolve_survey(survey)
            
            # Resolve target to coordinates if needed
            if target and ra is None:
                ra, dec = _resolve_simbad_cached(target)
                if ra is None:
                    return {"success": False, "error": f"Could not resolve target '{target}' via SIMBAD"}
            
            if ra is None or dec is None:
                return {"success": False, "error": "Provide target name or RA/Dec coordinates"}
            
            from astropy.coordinates import SkyCoord
            import astropy.units as u
            
            coord = SkyCoord(ra=ra, dec=dec, unit="deg")
            
            label = target if target else f"RA{ra:.3f}_Dec{dec:.3f}"
            print(f"[SkyView] Fetching {survey_name} image for {label} "
                  f"(radius={radius_arcmin}', {pixels}px)")
            
            # Fetch image from SkyView
            hdu_list = SkyView.get_images(
                position=coord,
                survey=[survey_name],
                radius=radius_arcmin * u.arcmin,
                pixels=pixels
            )
            
            if not hdu_list or len(hdu_list) == 0:
                return {"success": False, "error": f"No {survey_name} image available for this position"}
            
            hdu = hdu_list[0]
            result = {
                "success": True,
                "survey": survey_name,
                "target": label,
                "ra": ra,
                "dec": dec,
                "radius_arcmin": radius_arcmin,
                "pixels": pixels,
                "image_shape": list(hdu[0].data.shape) if hdu[0].data is not None else None,
            }
            
            if save:
                # Save FITS file
                os.makedirs(self.output_dir, exist_ok=True)
                safe_label = label.replace(" ", "_").replace("/", "_")
                safe_survey = survey_name.replace(" ", "_").replace("/", "_")
                filename = f"{safe_label}_{safe_survey}.fits"
                filepath = os.path.join(self.output_dir, filename)
                hdu.writeto(filepath, overwrite=True)
                result["fits_path"] = filepath
                result["fits_filename"] = filename
                print(f"[SkyView] Saved FITS: {filepath}")
                
                # Also save a PNG preview
                try:
                    png_path = self._save_preview(hdu[0].data, filepath.replace(".fits", ".png"), 
                                                   survey_name, label)
                    if png_path:
                        result["preview_path"] = png_path
                        print(f"[SkyView] Saved preview: {png_path}")
                except Exception as e:
                    print(f"[SkyView] Preview generation failed: {e}")
            
            return result
            
        except Exception as e:
            print(f"[SkyView] Image fetch error: {e}")
            import traceback
            traceback.print_exc()
            return {"success": False, "error": str(e)}

    def list_surveys(self) -> List[str]:
        """List available sky surveys."""
        if not SKYVIEW_AVAILABLE:
            return list(self.SURVEYS.values())
        try:
            return SkyView.list_surveys()
        except Exception:
            return list(self.SURVEYS.values())

    def _resolve_survey(self, survey: str) -> str:
        """Resolve survey aliases to SkyView survey names."""
        if survey is None:
            return "DSS2 Red"
        
        survey_lower = survey.lower().strip()
        
        # Check aliases first
        if survey_lower in self.ALIASES:
            return self.ALIASES[survey_lower]
        
        # Check known surveys
        if survey_lower in self.SURVEYS:
            return self.SURVEYS[survey_lower]
        
        # Return as-is (SkyView will validate)
        return survey

    def generate_finding_chart(
        self, target: str = None, ra: float = None, dec: float = None,
        survey: str = "DSS2 Red", fov_arcmin: float = 5.0,
        pixels: int = 600, title: str = None,
    ) -> Dict[str, Any]:
        """
        Generate a publication-quality finding chart with WCS axes,
        target crosshair, N/E compass arrows, and angular scale bar.

        Use DSS2 for optical or 2MASS for near-IR. The chart is rendered
        as a PNG and returned with its path for inline display.

        Args:
            target: Target name (resolved via SIMBAD)
            ra: RA in degrees (alternative to target)
            dec: Dec in degrees (alternative to target)
            survey: Sky survey (default 'DSS2 Red')
            fov_arcmin: Field of view in arcminutes (default 5')
            pixels: Image size in pixels (default 600)
            title: Custom title (default: target name + survey)

        Returns:
            Dict with image_path for inline display, plus metadata
        """
        import uuid

        if not SKYVIEW_AVAILABLE:
            return {"success": False, "error": "astroquery.skyview not available"}

        # Resolve target name
        if target and ra is None:
            ra, dec = _resolve_simbad_cached(target)
            if ra is None:
                return {"success": False, "error": f"Could not resolve target '{target}' via SIMBAD"}

        if ra is None or dec is None:
            return {"success": False, "error": "Provide target name or RA/Dec coordinates"}

        try:
            from astropy.coordinates import SkyCoord
            from astropy.wcs import WCS
            import astropy.units as u

            survey_name = self._resolve_survey(survey)
            coord = SkyCoord(ra=ra, dec=dec, unit="deg")
            label = target if target else f"RA={ra:.4f}, Dec={dec:.4f}"
            chart_title = title or f"Finding Chart: {label} ({survey_name})"

            print(f"[SkyView] Generating finding chart for {label} "
                  f"({survey_name}, {fov_arcmin}' FOV)")

            # Fetch the image
            hdu_list = SkyView.get_images(
                position=coord,
                survey=[survey_name],
                radius=fov_arcmin * u.arcmin,
                pixels=pixels,
            )

            if not hdu_list or len(hdu_list) == 0:
                return {"success": False, "error": f"No {survey_name} image at this position"}

            hdu = hdu_list[0]
            data = hdu[0].data
            header = hdu[0].header

            if data is None:
                return {"success": False, "error": "Empty image data returned"}

            # Render the finding chart
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import numpy as np
            from matplotlib.patches import FancyArrowPatch

            wcs = WCS(header)

            fig = plt.figure(figsize=(8, 8), facecolor="#0f172a")
            ax = fig.add_subplot(111, projection=wcs)

            # Image with contrast stretch
            vmin = np.nanpercentile(data, 1)
            vmax = np.nanpercentile(data, 99.5)
            ax.imshow(data, origin="lower", cmap="gray_r",
                      vmin=vmin, vmax=vmax)

            # Target crosshair at center
            px, py = wcs.world_to_pixel(coord)
            cross_size = pixels * 0.04
            ax.plot(float(px), float(py), "+", color="#ef4444",
                    markersize=18, markeredgewidth=2.5, zorder=10)
            # Circle around target
            circle = plt.Circle((float(px), float(py)), cross_size,
                                fill=False, color="#ef4444", linewidth=1.5,
                                linestyle="--", zorder=10)
            ax.add_patch(circle)

            # N/E compass arrows (top-left corner)
            arrow_origin_x = pixels * 0.12
            arrow_origin_y = pixels * 0.88
            arrow_len = pixels * 0.08

            # North arrow (up in standard orientation)
            ax.annotate("N", xy=(arrow_origin_x, arrow_origin_y + arrow_len),
                        xytext=(arrow_origin_x, arrow_origin_y),
                        arrowprops=dict(arrowstyle="->", color="white", lw=2),
                        color="white", fontsize=12, fontweight="bold",
                        ha="center", va="bottom", zorder=15)

            # East arrow (left in standard orientation)
            ax.annotate("E", xy=(arrow_origin_x - arrow_len, arrow_origin_y),
                        xytext=(arrow_origin_x, arrow_origin_y),
                        arrowprops=dict(arrowstyle="->", color="white", lw=2),
                        color="white", fontsize=12, fontweight="bold",
                        ha="right", va="center", zorder=15)

            # Scale bar (bottom-right)
            if fov_arcmin >= 2:
                bar_arcmin = 1.0
                bar_label = "1'"
            else:
                bar_arcmin = fov_arcmin / 5
                bar_label = f'{bar_arcmin * 60:.0f}"'

            bar_pixels = (bar_arcmin / fov_arcmin) * pixels * 0.5
            bar_x = pixels * 0.95 - bar_pixels
            bar_y = pixels * 0.06
            ax.plot([bar_x, bar_x + bar_pixels], [bar_y, bar_y],
                    color="white", linewidth=3, zorder=15)
            ax.text(bar_x + bar_pixels / 2, bar_y + pixels * 0.02,
                    bar_label, color="white", fontsize=11,
                    ha="center", va="bottom", fontweight="bold", zorder=15)

            # Axis labels
            ax.set_title(chart_title, color="white", fontsize=13, pad=12)
            ax.coords[0].set_axislabel("RA (J2000)", color="white", fontsize=10)
            ax.coords[1].set_axislabel("Dec (J2000)", color="white", fontsize=10)
            ax.coords[0].set_ticklabel(color="white", fontsize=8)
            ax.coords[1].set_ticklabel(color="white", fontsize=8)
            ax.coords.grid(color="white", alpha=0.15, linestyle="--")
            ax.set_facecolor("black")
            fig.patch.set_facecolor("#0f172a")

            # Save
            rendered_dir = os.path.join(
                os.path.dirname(__file__), "..", "data", "rendered_images"
            )
            os.makedirs(rendered_dir, exist_ok=True)
            img_name = f"findchart_{uuid.uuid4().hex[:10]}.png"
            img_path = os.path.join(rendered_dir, img_name)
            fig.savefig(img_path, dpi=150, bbox_inches="tight",
                        facecolor=fig.get_facecolor())
            plt.close(fig)

            return {
                "success": True,
                "image_path": f"/api/images/{img_name}",
                "caption": chart_title,
                "target": label,
                "ra_deg": ra,
                "dec_deg": dec,
                "survey": survey_name,
                "fov_arcmin": fov_arcmin,
            }

        except Exception as e:
            print(f"[SkyView] Finding chart generation failed: {e}")
            import traceback
            traceback.print_exc()
            return {"success": False, "error": str(e)}

    def _save_preview(self, data, png_path: str, survey: str, label: str) -> Optional[str]:
        """Save a PNG preview of the FITS image."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import numpy as np
            
            if data is None:
                return None
            
            fig, ax = plt.subplots(1, 1, figsize=(6, 6))
            
            # Apply log stretch for better contrast
            vmin = np.nanpercentile(data, 1)
            vmax = np.nanpercentile(data, 99)
            
            ax.imshow(data, origin="lower", cmap="gray",
                      vmin=vmin, vmax=vmax)
            ax.set_title(f"{label} — {survey}", fontsize=12)
            ax.set_xlabel("pixels")
            ax.set_ylabel("pixels")
            
            plt.tight_layout()
            plt.savefig(png_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            
            return png_path
        except ImportError:
            return None
