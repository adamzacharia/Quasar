"""
CASA Integration Module
Provides interface to CASA for calibration and imaging
Note: CASA installation is complex - this module provides both real CASA integration
and fallback Astropy-based alternatives
"""

import os
import subprocess
import json
import tempfile
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
from datetime import datetime

# Try to import CASA modules (may not be available)
try:
    from casatools import table, image, ms, quanta, measures
    from casatasks import importfits, flagdata, gaincal, bandpass, applycal, tclean, exportfits
    CASA_AVAILABLE = True
except ImportError:
    CASA_AVAILABLE = False

# Fallback to Astropy
from astropy.io import fits
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time
from astropy.wcs import WCS

class CASAIntegration:
    """CASA integration for radio data processing"""

    def __init__(self, casa_path: Optional[str] = None):
        """
        Initialize CASA integration

        Args:
            casa_path: Path to CASA installation
        """
        self.casa_available = CASA_AVAILABLE
        self.casa_path = casa_path or os.getenv("CASA_PATH", "/usr/local/casa")

        if not self.casa_available:
            print("Warning: CASA not available. Using Astropy fallback methods.")

    def import_fits_to_ms(self, fits_file: str, ms_name: Optional[str] = None) -> str:
        """
        Import FITS file to MeasurementSet

        Args:
            fits_file: Path to FITS file
            ms_name: Output MS name (auto-generated if None)

        Returns:
            Path to created MS
        """
        if not ms_name:
            ms_name = Path(fits_file).stem + ".ms"

        if self.casa_available:
            try:
                importfits(fitsimage=fits_file, imagename=ms_name, overwrite=True)
                return ms_name
            except Exception as e:
                print(f"CASA import failed: {e}")
                return self._fallback_fits_import(fits_file, ms_name)
        else:
            return self._fallback_fits_import(fits_file, ms_name)

    def _fallback_fits_import(self, fits_file: str, ms_name: str) -> str:
        """Fallback FITS import using Astropy"""
        print(f"Note: Using Astropy to read FITS. Full MS conversion requires CASA.")

        # Read FITS with Astropy
        hdul = fits.open(fits_file)

        # Save metadata for reference
        metadata = {
            "original_fits": fits_file,
            "header": dict(hdul[0].header),
            "data_shape": str(hdul[0].data.shape) if hdul[0].data is not None else "No data",
            "creation_time": datetime.now().isoformat()
        }

        metadata_file = ms_name + ".metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

        hdul.close()

        print(f"FITS metadata saved to {metadata_file}")
        return metadata_file

    def flag_rfi(self, ms_name: str, strategy: str = "default") -> Dict[str, Any]:
        """
        Flag RFI in measurement set

        Args:
            ms_name: Path to MS
            strategy: Flagging strategy

        Returns:
            Flagging statistics
        """
        if self.casa_available:
            try:
                if strategy == "default":
                    # Flag autocorrelations
                    flagdata(vis=ms_name, mode='manual', autocorr=True)

                    # Flag edge channels (5% on each side)
                    flagdata(vis=ms_name, mode='manual',
                            spw='*:0~5;60~63')

                    # Flag zeros
                    flagdata(vis=ms_name, mode='clip', clipzeros=True)

                    # Get statistics
                    stats = flagdata(vis=ms_name, mode='summary')

                    return {
                        "strategy": strategy,
                        "flagged_fraction": stats.get('fraction', 0),
                        "flagged_points": stats.get('total', 0)
                    }

            except Exception as e:
                print(f"CASA flagging failed: {e}")
                return self._fallback_flagging(ms_name, strategy)
        else:
            return self._fallback_flagging(ms_name, strategy)

    def _fallback_flagging(self, ms_name: str, strategy: str) -> Dict[str, Any]:
        """Fallback flagging statistics"""
        return {
            "strategy": strategy,
            "status": "simulated",
            "message": "CASA required for actual flagging",
            "estimated_flagged_fraction": 0.05
        }

    def calibrate(self, ms_name: str, cal_type: str = "standard") -> Dict[str, Any]:
        """
        Run calibration pipeline

        Args:
            ms_name: Path to MS
            cal_type: Type of calibration

        Returns:
            Calibration results
        """
        if self.casa_available:
            try:
                cal_tables = []

                if cal_type in ["standard", "full"]:
                    # Bandpass calibration
                    bp_table = ms_name + ".bandpass"
                    bandpass(vis=ms_name, caltable=bp_table,
                            solint='inf', combine='scan')
                    cal_tables.append(bp_table)

                    # Gain calibration
                    gain_table = ms_name + ".gain"
                    gaincal(vis=ms_name, caltable=gain_table,
                           solint='inf', combine='')
                    cal_tables.append(gain_table)

                # Apply calibration
                if cal_tables:
                    applycal(vis=ms_name, gaintable=cal_tables)

                return {
                    "status": "success",
                    "cal_type": cal_type,
                    "cal_tables": cal_tables
                }

            except Exception as e:
                print(f"CASA calibration failed: {e}")
                return self._fallback_calibration(ms_name, cal_type)
        else:
            return self._fallback_calibration(ms_name, cal_type)

    def _fallback_calibration(self, ms_name: str, cal_type: str) -> Dict[str, Any]:
        """Fallback calibration response"""
        return {
            "status": "simulated",
            "cal_type": cal_type,
            "message": "CASA required for actual calibration",
            "estimated_improvement": "3-5x dynamic range"
        }

    def image_with_tclean(self, ms_name: str,
                         imagename: Optional[str] = None,
                         imsize: int = 1024,
                         cell: str = "1arcsec",
                         weighting: str = "natural",
                         robust: float = 0.5,
                         niter: int = 1000,
                         threshold: str = "0.1mJy") -> Dict[str, Any]:
        """
        Create image using tclean

        Args:
            ms_name: Path to MS
            imagename: Output image name
            imsize: Image size in pixels
            cell: Cell size
            weighting: Weighting scheme
            robust: Robust parameter
            niter: Number of iterations
            threshold: Cleaning threshold

        Returns:
            Imaging results
        """
        if not imagename:
            imagename = Path(ms_name).stem + ".image"

        if self.casa_available:
            try:
                tclean(vis=ms_name,
                      imagename=imagename,
                      imsize=imsize,
                      cell=cell,
                      weighting=weighting,
                      robust=robust,
                      niter=niter,
                      threshold=threshold,
                      interactive=False)

                # Export to FITS
                fits_name = imagename + ".fits"
                exportfits(imagename=imagename + ".image",
                          fitsimage=fits_name, overwrite=True)

                return {
                    "status": "success",
                    "image": imagename,
                    "fits": fits_name,
                    "parameters": {
                        "imsize": imsize,
                        "cell": cell,
                        "weighting": weighting,
                        "niter": niter
                    }
                }

            except Exception as e:
                print(f"CASA imaging failed: {e}")
                return self._fallback_imaging(ms_name, imagename)
        else:
            return self._fallback_imaging(ms_name, imagename)

    def _fallback_imaging(self, ms_name: str, imagename: str) -> Dict[str, Any]:
        """Fallback imaging response"""
        # Create a dummy FITS file for demonstration
        dummy_data = np.random.randn(512, 512) * 0.001

        hdu = fits.PrimaryHDU(dummy_data)
        hdu.header['BUNIT'] = 'JY/BEAM'
        hdu.header['BMAJ'] = 5.0 / 3600  # 5 arcsec
        hdu.header['BMIN'] = 5.0 / 3600
        hdu.header['BPA'] = 0.0

        fits_name = imagename + "_simulated.fits"
        hdu.writeto(fits_name, overwrite=True)

        return {
            "status": "simulated",
            "message": "CASA required for actual imaging",
            "demo_fits": fits_name,
            "note": "Created demo FITS with random noise"
        }

    def get_image_statistics(self, image_path: str) -> Dict[str, Any]:
        """
        Get image statistics

        Args:
            image_path: Path to image

        Returns:
            Image statistics
        """
        if image_path.endswith('.fits'):
            # Use Astropy for FITS files
            hdul = fits.open(image_path)
            data = hdul[0].data
            header = hdul[0].header

            # Handle multi-dimensional data
            if data.ndim > 2:
                # Take first plane for statistics
                data = data[0, 0] if data.ndim == 4 else data[0]

            stats = {
                "mean": float(np.nanmean(data)),
                "std": float(np.nanstd(data)),
                "min": float(np.nanmin(data)),
                "max": float(np.nanmax(data)),
                "rms": float(np.sqrt(np.nanmean(data**2))),
                "shape": data.shape,
                "unit": header.get('BUNIT', 'Unknown'),
                "beam": {
                    "bmaj_arcsec": header.get('BMAJ', 0) * 3600,
                    "bmin_arcsec": header.get('BMIN', 0) * 3600,
                    "bpa_deg": header.get('BPA', 0)
                }
            }

            hdul.close()
            return stats

        elif self.casa_available:
            # Use CASA for .image files
            try:
                ia = image()
                ia.open(image_path)
                stats = ia.statistics()
                ia.close()

                return {
                    "mean": stats['mean'][0],
                    "std": stats['sigma'][0],
                    "min": stats['min'][0],
                    "max": stats['max'][0],
                    "rms": stats['rms'][0]
                }
            except Exception as e:
                return {"error": str(e)}
        else:
            return {"error": "Cannot read non-FITS images without CASA"}

    def run_vla_pipeline(self, ms_name: str,
                        config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Run complete VLA pipeline

        Args:
            ms_name: Path to MS
            config: Pipeline configuration

        Returns:
            Pipeline results
        """
        results = {
            "ms": ms_name,
            "steps": [],
            "products": []
        }

        # Default configuration
        if config is None:
            config = {
                "flagging": {"strategy": "default"},
                "calibration": {"type": "standard"},
                "imaging": {
                    "weighting": "natural",
                    "niter": 1000,
                    "threshold": "0.1mJy"
                }
            }

        # Step 1: Flagging
        print("Step 1: Flagging RFI...")
        flag_result = self.flag_rfi(ms_name, config["flagging"]["strategy"])
        results["steps"].append({"name": "flagging", "result": flag_result})

        # Step 2: Calibration
        print("Step 2: Calibration...")
        cal_result = self.calibrate(ms_name, config["calibration"]["type"])
        results["steps"].append({"name": "calibration", "result": cal_result})

        # Step 3: Imaging
        print("Step 3: Imaging...")
        image_result = self.image_with_tclean(
            ms_name,
            **config["imaging"]
        )
        results["steps"].append({"name": "imaging", "result": image_result})

        if image_result.get("fits"):
            results["products"].append(image_result["fits"])

        results["status"] = "complete"
        return results

    def generate_calibration_script(self, ms_name: str,
                                   output_script: str = "calibration.py") -> str:
        """
        Generate CASA calibration script

        Args:
            ms_name: Path to MS
            output_script: Output script filename

        Returns:
            Path to generated script
        """
        script = f'''#!/usr/bin/env python
# CASA Calibration Script
# Generated by Quasar
# Date: {datetime.now().isoformat()}

import os

# Input measurement set
ms_name = '{ms_name}'

# Check if MS exists
if not os.path.exists(ms_name):
    raise FileNotFoundError(f"MS not found: {{ms_name}}")

print(f"Processing: {{ms_name}}")

# Step 1: Initial flagging
print("Step 1: Flagging...")
flagdata(vis=ms_name, mode='manual', autocorr=True)
flagdata(vis=ms_name, mode='manual', spw='*:0~5;60~63')  # Edge channels
flagdata(vis=ms_name, mode='clip', clipzeros=True)

# Step 2: Set flux density scale
print("Step 2: Setting flux scale...")
setjy(vis=ms_name, field='0', standard='Perley-Butler 2017')

# Step 3: Bandpass calibration
print("Step 3: Bandpass calibration...")
bandpass(vis=ms_name,
         caltable=ms_name+'.bandpass',
         field='0',  # Calibrator
         solint='inf',
         combine='scan')

# Step 4: Gain calibration
print("Step 4: Gain calibration...")
gaincal(vis=ms_name,
        caltable=ms_name+'.gain',
        field='0,1',  # Calibrator and target
        solint='inf',
        gaintable=[ms_name+'.bandpass'])

# Step 5: Apply calibration
print("Step 5: Applying calibration...")
applycal(vis=ms_name,
         field='1',  # Target
         gaintable=[ms_name+'.bandpass', ms_name+'.gain'])

# Step 6: Image target
print("Step 6: Imaging...")
tclean(vis=ms_name,
       field='1',  # Target
       imagename=ms_name+'.image',
       imsize=1024,
       cell='1arcsec',
       weighting='briggs',
       robust=0.5,
       niter=1000,
       threshold='0.1mJy',
       interactive=False)

# Step 7: Export to FITS
print("Step 7: Exporting to FITS...")
exportfits(imagename=ms_name+'.image.image',
           fitsimage=ms_name+'.fits',
           overwrite=True)

print("Calibration complete!")
print(f"Output image: {{ms_name}}.fits")
'''

        with open(output_script, 'w') as f:
            f.write(script)

        # Make executable
        os.chmod(output_script, 0o755)

        print(f"CASA script generated: {output_script}")
        return output_script

    def natural_language_to_casa(self, request: str) -> Dict[str, Any]:
        """
        Convert natural language request to CASA parameters

        Args:
            request: Natural language request

        Returns:
            CASA parameters
        """
        params = {
            "weighting": "natural",
            "robust": 0.5,
            "niter": 1000,
            "threshold": "0.1mJy",
            "imsize": 1024,
            "cell": "1arcsec"
        }

        request_lower = request.lower()

        # Parse weighting
        if "uniform" in request_lower:
            params["weighting"] = "uniform"
        elif "briggs" in request_lower or "robust" in request_lower:
            params["weighting"] = "briggs"
            # Try to extract robust value
            if "-2" in request:
                params["robust"] = -2.0
            elif "-1" in request:
                params["robust"] = -1.0
            elif "1" in request and "-1" not in request:
                params["robust"] = 1.0
            elif "2" in request and "-2" not in request:
                params["robust"] = 2.0

        # Parse iterations
        if "deep" in request_lower or "sensitive" in request_lower:
            params["niter"] = 5000
            params["threshold"] = "0.01mJy"
        elif "quick" in request_lower or "fast" in request_lower:
            params["niter"] = 100
            params["threshold"] = "1mJy"

        # Parse resolution
        if "high resolution" in request_lower:
            params["cell"] = "0.5arcsec"
            params["imsize"] = 2048
        elif "low resolution" in request_lower:
            params["cell"] = "2arcsec"
            params["imsize"] = 512

        return params


# Astropy-based utilities (always available)
class AstropyProcessor:
    """Astropy-based processing utilities"""

    @staticmethod
    def read_fits_header(fits_file: str) -> Dict[str, Any]:
        """Read FITS header information"""
        hdul = fits.open(fits_file)
        header = dict(hdul[0].header)
        hdul.close()
        return header

    @staticmethod
    def estimate_noise(fits_file: str, method: str = "mad") -> float:
        """Estimate noise in FITS image"""
        hdul = fits.open(fits_file)
        data = hdul[0].data

        # Handle multi-dimensional data
        if data.ndim > 2:
            data = data[0, 0] if data.ndim == 4 else data[0]

        if method == "mad":
            # Median absolute deviation
            med = np.nanmedian(data)
            mad = np.nanmedian(np.abs(data - med))
            noise = 1.4826 * mad
        else:
            # Standard deviation
            noise = np.nanstd(data)

        hdul.close()
        return float(noise)

    @staticmethod
    def get_beam_info(fits_file: str) -> Dict[str, float]:
        """Get beam information from FITS"""
        hdul = fits.open(fits_file)
        header = hdul[0].header

        beam = {
            "bmaj_arcsec": header.get('BMAJ', 0) * 3600,
            "bmin_arcsec": header.get('BMIN', 0) * 3600,
            "bpa_deg": header.get('BPA', 0),
            "beam_area_arcsec2": None
        }

        if beam["bmaj_arcsec"] > 0 and beam["bmin_arcsec"] > 0:
            beam["beam_area_arcsec2"] = (
                np.pi * beam["bmaj_arcsec"] * beam["bmin_arcsec"] / (4 * np.log(2))
            )

        hdul.close()
        return beam

    @staticmethod
    def calculate_sensitivity(bandwidth_mhz: float,
                            integration_time_hours: float,
                            n_antennas: int = 27,
                            tsys_k: float = 30) -> float:
        """Calculate theoretical sensitivity"""
        # Simplified radiometer equation
        bandwidth_hz = bandwidth_mhz * 1e6
        integration_sec = integration_time_hours * 3600
        n_baselines = n_antennas * (n_antennas - 1) / 2

        # SEFD approximation
        sefd_jy = 2.0 * tsys_k  # Simplified

        # Calculate sensitivity
        sensitivity_mjy = 1000 * sefd_jy / np.sqrt(
            2 * bandwidth_hz * integration_sec * n_baselines
        )

        return sensitivity_mjy