# services/analysis.py
"""
Radio astronomy data analysis service (ALMA-focused)
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional

class RadioAnalysisService:
    """Analysis operations for radio astronomy data (ALMA)"""

    def __init__(self):
        self.casa_available = False
        try:
            import casatools
            self.casa_available = True
        except ImportError:
            pass

    def analyze_uv_coverage(self, ms_path: str) -> Dict[str, Any]:
        """Analyze UV coverage of a measurement set"""
        if not self.casa_available:
            return {
                "status": "analysis_pending",
                "message": f"UV analysis for {ms_path} requires CASA installation",
                "mock_results": {
                    "max_baseline_km": 16.0,
                    "min_baseline_km": 0.015,
                    "num_antennas": 50,
                    "num_baselines": 1225,
                    "uv_range": "0.015-16 km",
                    "telescope": "ALMA 12m",
                    "recommended_imaging": {
                        "cell_size_arcsec": 0.02,
                        "image_size_pixels": 4096
                    }
                }
            }

        try:
            from casatools import msmetadata
            msmd = msmetadata()
            msmd.open(ms_path)

            result = {
                "status": "success",
                "n_antennas": msmd.nantennas(),
                "n_baselines": msmd.nbaselines(),
                "n_spw": msmd.nspw(),
                "freq_range_ghz": [
                    msmd.chanfreqs(0)[0] / 1e9,
                    msmd.chanfreqs(0)[-1] / 1e9
                ],
            }

            msmd.close()
            return result

        except Exception as e:
            return {"status": "error", "message": str(e)}

    def calculate_sensitivity(self, bandwidth_mhz: float,
                            integration_time_hours: float,
                            num_antennas: int = 50,
                            system_temp_k: float = 80,
                            freq_ghz: float = 230.0) -> float:
        """
        Calculate theoretical sensitivity using the radiometer equation.
        Default parameters sized for ALMA 12m array.
        σ = SEFD / sqrt(n_pol * Δν * t * n_baselines)
        """
        bandwidth_hz = bandwidth_mhz * 1e6
        integration_sec = integration_time_hours * 3600
        n_baselines = num_antennas * (num_antennas - 1) / 2
        n_pol = 2  # Dual polarization

        # SEFD lookup for ALMA bands (approximate, Jy)
        sefd_by_band = {
            (84, 116): 3000,      # Band 3
            (125, 163): 3500,     # Band 4
            (163, 211): 4000,     # Band 5
            (211, 275): 4500,     # Band 6
            (275, 373): 6000,     # Band 7
            (385, 500): 10000,    # Band 8
            (602, 720): 20000,    # Band 9
            (787, 950): 40000,    # Band 10
        }

        sefd_jy = None
        for (low, high), sefd in sefd_by_band.items():
            if low <= freq_ghz <= high:
                sefd_jy = sefd
                break

        if sefd_jy is None:
            # Fallback: scale from system temperature
            # SEFD = 2 * k * Tsys / (η * A)  — simplified
            sefd_jy = 25 * system_temp_k  # rough approximation

        sensitivity_mjy = 1000 * sefd_jy / np.sqrt(
            n_pol * bandwidth_hz * integration_sec * n_baselines
        )

        return sensitivity_mjy

    def estimate_image_noise(self, obs_data: pd.DataFrame) -> Dict[str, float]:
        """Estimate expected image noise from observation parameters"""
        if obs_data.empty:
            return {}

        row = obs_data.iloc[0]

        # Extract parameters
        freq_min = row.get('freq_min', 211e9)
        freq_max = row.get('freq_max', 275e9)
        bandwidth_hz = freq_max - freq_min
        integration_sec = row.get('t_exptime', 3600)
        freq_ghz = (freq_min + freq_max) / 2e9

        estimated_noise = self.calculate_sensitivity(
            bandwidth_mhz=bandwidth_hz / 1e6,
            integration_time_hours=integration_sec / 3600,
            num_antennas=50,   # ALMA 12m array
            freq_ghz=freq_ghz
        )

        return {
            "theoretical_noise_mjy": estimated_noise,
            "expected_dynamic_range": 1000 / estimated_noise if estimated_noise > 0 else 0,
            "detection_limit_5sigma_mjy": 5 * estimated_noise,
            "telescope": "ALMA"
        }