"""
Service for processing FITS astronomical data files.
Used by the chat upload API to present FITS data to GPT-4o Vision.
"""

import io
import base64
import logging
import numpy as np
from typing import Dict, Any, Tuple, Optional

logger = logging.getLogger(__name__)

# Try importing astropy and matplotlib. These imports shouldn't crash Quasar if missing, 
# just gracefully disable the FITS feature if dependencies aren't installed.
try:
    from astropy.io import fits
    import matplotlib
    matplotlib.use('Agg') # Run headlessly
    import matplotlib.pyplot as plt
    ASTROPY_AVAILABLE = True
except ImportError:
    ASTROPY_AVAILABLE = False
    logger.warning("astropy or matplotlib not found. FITS processing will be disabled.")


class FITSProcessingService:
    @staticmethod
    def extract_metadata(file_bytes: bytes) -> Dict[str, Any]:
        """Extracts key headers from a FITS file."""
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
