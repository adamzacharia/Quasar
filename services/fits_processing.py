"""
Service for processing FITS astronomical data files.

Capabilities:
  1. extract_metadata(bytes) — extract headers from local FITS bytes
  2. extract_metadata_from_url(url) — read ONLY the FITS header over HTTP
     (no full download), extract beam size, sensitivity, coordinate info
  3. generate_preview(bytes) — render 2D/3D FITS data as PNG preview

Used by: core/agent.py (tools: inspect_fits_header, FITS upload processing)
"""

import io
import base64
import logging
import numpy as np
import requests
from typing import Dict, Any, Tuple, Optional, List

logger = logging.getLogger(__name__)

# Try importing astropy and matplotlib. These imports shouldn't crash Quasar if missing,
# just gracefully disable the FITS feature if dependencies aren't installed.
try:
    from astropy.io import fits
    import matplotlib
    matplotlib.use('Agg')  # Run headlessly
    import matplotlib.pyplot as plt
    ASTROPY_AVAILABLE = True
except ImportError:
    ASTROPY_AVAILABLE = False
    logger.warning("astropy or matplotlib not found. FITS processing will be disabled.")


class FITSProcessingService:
    @staticmethod
    def extract_metadata(file_bytes: bytes) -> Dict[str, Any]:
        """Extracts key headers from a FITS file (from local bytes)."""
        if not ASTROPY_AVAILABLE:
            return {"error": "Astropy required for FITS processing"}

        try:
            with fits.open(io.BytesIO(file_bytes)) as hdul:
                header = hdul[0].header

                # Extract common relevant metadata, providing defaults
                metadata = {
                    "TELESCOP": header.get("TELESCOP", "Unknown"),
                    "INSTRUME": header.get("INSTRUME", "Unknown"),
                    "OBJECT": header.get("OBJECT", "Unknown"),
                    "DATE-OBS": header.get("DATE-OBS", "Unknown"),
                    "BMAJ (deg)": header.get("BMAJ", "Unknown"),
                    "BMIN (deg)": header.get("BMIN", "Unknown"),
                    "RESTFRQ (Hz)": header.get("RESTFRQ", "Unknown"),
                    "NAXIS": header.get("NAXIS", 0)
                }

                # Catch coordinate system Info
                for key in ["CTYPE1", "CTYPE2", "CTYPE3", "CTYPE4"]:
                    if key in header:
                        metadata[key] = header[key]

                return metadata
        except Exception as e:
            logger.error("Error extracting FITS metadata: %s", e)
            return {"error": str(e)}

    @staticmethod
    def extract_metadata_from_url(url: str) -> Dict[str, Any]:
        """
        Read ONLY the FITS header from a remote URL — no full file download.

        Uses HTTP Range requests to fetch just the first ~64 KB (enough for
        any FITS primary header), then parses it with astropy.  This lets us
        inspect beam size, sensitivity, and other metadata for dozens of files
        without downloading gigabytes of data.

        Returns:
            Dict with keys:
              - success: bool
              - url: str
              - filename: str
              - object_name: str  (OBJECT keyword)
              - telescope: str
              - date_obs: str
              - beam_major_arcsec: float | None
              - beam_minor_arcsec: float | None
              - beam_pa_deg: float | None
              - sensitivity_mjy_beam: float | None  (RMS noise)
              - rest_freq_ghz: float | None
              - image_size: str  (e.g. "512 x 512")
              - pixel_scale_arcsec: float | None
              - bunit: str  (brightness unit)
              - all_headers: dict  (full header dump)
              - error: str  (only on failure)
        """
        if not ASTROPY_AVAILABLE:
            return {"success": False, "error": "Astropy required for FITS processing", "url": url}

        filename = url.split("/")[-1].split("?")[0] if url else "unknown"
        print(f"[FITS] Reading header from {filename}...")

        try:
            # Strategy 1: HTTP Range request for just the header (fastest)
            header = FITSProcessingService._fetch_header_via_range(url)

            # Strategy 2: Use astropy's lazy-load with fsspec (fallback)
            if header is None:
                header = FITSProcessingService._fetch_header_via_lazy_open(url)

            if header is None:
                return {
                    "success": False,
                    "url": url,
                    "filename": filename,
                    "error": "Could not read FITS header from URL. The file may not be accessible.",
                }

            # ── Extract science-relevant metadata ────────────────────────
            result = FITSProcessingService._extract_science_metadata(header, url, filename)
            return result

        except Exception as e:
            logger.error("Error reading remote FITS header: %s", e)
            return {"success": False, "url": url, "filename": filename, "error": str(e)}

    # ── Internal helpers for remote header reading ──────────────────────────

    @staticmethod
    def _fetch_header_via_range(url: str, max_bytes: int = 131072):
        """
        Fetch FITS header using HTTP Range request.
        Downloads only the first 64–128 KB — enough for any FITS primary header.
        """
        try:
            max_bytes = max(2880, int(max_bytes))
            resp = requests.get(
                url,
                headers={"Range": f"bytes=0-{max_bytes - 1}"},
                timeout=30,
                stream=True,
            )
            if resp.status_code == 416:
                resp.close()
                resp = requests.get(
                    url,
                    timeout=30,
                    stream=True,
                )

            try:
                if resp.status_code not in (200, 206):
                    print(f"[FITS] Range request returned {resp.status_code}")
                    return None
                payload = bytearray()
                for chunk in resp.iter_content(chunk_size=16384):
                    if not chunk:
                        continue
                    remaining = max_bytes - len(payload)
                    if remaining <= 0:
                        break
                    payload.extend(chunk[:remaining])
                    if len(payload) >= max_bytes:
                        break
                header_bytes = bytes(payload)
            finally:
                resp.close()
            if len(header_bytes) < 2880:
                print(f"[FITS] Response too small ({len(header_bytes)} bytes)")
                return None

            return FITSProcessingService._parse_bounded_header(header_bytes)

        except Exception as e:
            print(f"[FITS] Range request failed: {e}")
            return None

    @staticmethod
    def _fetch_header_via_lazy_open(url: str):
        """
        Retry with a larger bounded header window.

        This deliberately never opens or downloads the complete remote file.
        """
        return FITSProcessingService._fetch_header_via_range(url, max_bytes=1048576)

    @staticmethod
    def _parse_bounded_header(header_bytes: bytes):
        """Parse a FITS header from a bounded byte prefix."""
        end_offset = None
        for offset in range(0, len(header_bytes) - 79, 80):
            keyword = header_bytes[offset:offset + 8].decode(
                "ascii",
                errors="ignore",
            ).strip()
            if keyword == "END":
                end_offset = offset + 80
                break
        if end_offset is None:
            print(f"[FITS] END card not found in first {len(header_bytes)} bytes")
            return None
        header_text = header_bytes[:end_offset].decode("ascii", errors="replace")
        return fits.Header.fromstring(header_text, sep="")

    @staticmethod
    def _extract_science_metadata(header, url: str, filename: str) -> Dict[str, Any]:
        """Extract beam, sensitivity, frequency, and image info from a FITS header."""

        # ── Beam size (BMAJ, BMIN in degrees → arcsec) ──────────────────
        bmaj_deg = header.get("BMAJ")
        bmin_deg = header.get("BMIN")
        bpa = header.get("BPA")

        bmaj_arcsec = round(bmaj_deg * 3600, 4) if bmaj_deg is not None else None
        bmin_arcsec = round(bmin_deg * 3600, 4) if bmin_deg is not None else None

        # ── Sensitivity / RMS noise ─────────────────────────────────────
        # ALMA images sometimes have RMS in different keywords
        rms_value = None
        for rms_key in ["RMS", "NOISE", "DATARMS", "ACTNOISE", "MAD_RMS"]:
            val = header.get(rms_key)
            if val is not None:
                try:
                    rms_value = float(val)
                    break
                except (ValueError, TypeError):
                    continue

        # Convert RMS to mJy/beam based on BUNIT
        bunit = str(header.get("BUNIT", "")).strip()
        sensitivity_mjy = None
        if rms_value is not None:
            if "Jy" in bunit:
                if "mJy" in bunit:
                    sensitivity_mjy = round(rms_value, 6)
                elif "uJy" in bunit or "microJy" in bunit:
                    sensitivity_mjy = round(rms_value / 1000.0, 6)
                else:
                    # Assume Jy/beam
                    sensitivity_mjy = round(rms_value * 1000.0, 6)
            else:
                # Unknown unit, report raw value
                sensitivity_mjy = round(rms_value * 1000.0, 6)  # assume Jy

        # ── Rest frequency ──────────────────────────────────────────────
        restfrq_hz = header.get("RESTFRQ") or header.get("RESTFREQ")
        rest_freq_ghz = None
        if restfrq_hz is not None:
            try:
                rest_freq_ghz = round(float(restfrq_hz) / 1e9, 4)
            except (ValueError, TypeError):
                pass

        # ── Image dimensions and pixel scale ────────────────────────────
        naxis1 = header.get("NAXIS1")
        naxis2 = header.get("NAXIS2")
        image_size = f"{naxis1} x {naxis2}" if naxis1 and naxis2 else None

        cdelt1 = header.get("CDELT1")
        cdelt2 = header.get("CDELT2")
        pixel_scale = None
        if cdelt2 is not None:
            try:
                pixel_scale = round(abs(float(cdelt2)) * 3600, 4)  # deg → arcsec
            except (ValueError, TypeError):
                pass

        # ── Build the full header dump (first 50 non-blank keys) ────────
        all_headers = {}
        count = 0
        for key in header:
            if key and key.strip() and count < 50:
                all_headers[key] = str(header[key])
                count += 1

        return {
            "success": True,
            "url": url,
            "filename": filename,
            "object_name": str(header.get("OBJECT", "Unknown")),
            "telescope": str(header.get("TELESCOP", header.get("TELSCOP", "Unknown"))),
            "instrument": str(header.get("INSTRUME", "Unknown")),
            "date_obs": str(header.get("DATE-OBS", "Unknown")),
            "beam_major_arcsec": bmaj_arcsec,
            "beam_minor_arcsec": bmin_arcsec,
            "beam_pa_deg": round(float(bpa), 2) if bpa is not None else None,
            "sensitivity_mjy_beam": sensitivity_mjy,
            "rms_raw_value": rms_value,
            "rms_unit": bunit if rms_value else None,
            "rest_freq_ghz": rest_freq_ghz,
            "image_size": image_size,
            "pixel_scale_arcsec": pixel_scale,
            "bunit": bunit or "Unknown",
            "all_headers": all_headers,
        }


    @staticmethod
    def generate_preview(file_bytes: bytes) -> Optional[str]:
        """
        Parses FITS array and generates a base64 encoded PNG representation.
        Handles 2D images, 3D/4D cubes (collapses spectral/stokes axes via moment-0),
        and 1D spectra.
        """
        if not ASTROPY_AVAILABLE:
            return None

        try:
            with fits.open(io.BytesIO(file_bytes)) as hdul:
                data = hdul[0].data
                
                if data is None:
                    # Sometimes data is buried in another extension
                    for ext in hdul:
                        if ext.data is not None:
                            data = ext.data
                            break
                
                if data is None:
                    return None
                
                # Squeeze the array to remove degenerate dimensions (like empty Stokes)
                data = np.squeeze(data)
                
                fig, ax = plt.subplots(figsize=(6, 6))
                
                if data.ndim == 1:
                    # 1D Spectrum
                    ax.plot(data, color='blue', linewidth=1)
                    ax.set_title("1D Spectrum Profile")
                    ax.set_ylabel("Amplitude")
                    ax.set_xlabel("Channels")
                    ax.grid(True, alpha=0.3)
                
                elif data.ndim >= 2:
                    # If 3D Cube, take a max projection or moment-0 along first axis (usually spectral)
                    if data.ndim == 3: # (Spectral, Y, X)
                        image_data = np.nanmax(data, axis=0)
                    else:
                        image_data = data
                    
                    # Compute reasonable vmin/vmax avoiding outliers (e.g. 5th - 99th percentile)
                    valid_data = image_data[~np.isnan(image_data)]
                    if len(valid_data) > 0:
                         vmin = np.percentile(valid_data, 5)
                         vmax = np.percentile(valid_data, 99)
                    else:
                         vmin, vmax = None, None
                         
                    im = ax.imshow(image_data, cmap='inferno', origin='lower', vmin=vmin, vmax=vmax)
                    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Intensity")
                    ax.set_title("2D Spatial Map (Max Projection)")
                else:
                    return None # 0D data?
                    
                # Render to PNG byte stream
                img_buf = io.BytesIO()
                plt.tight_layout()
                plt.savefig(img_buf, format='png', dpi=150, bbox_inches='tight')
                plt.close(fig)
                
                # Base64 encode
                img_buf.seek(0)
                b64_str = base64.b64encode(img_buf.read()).decode('utf-8')
                return b64_str
                
        except Exception as e:
            logger.error("Error generating FITS preview: %s", e)
            return None
