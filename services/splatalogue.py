"""
SplatalogueTool — Molecular spectral line identification for Quasar AI
Queries the Splatalogue astronomical spectral line database by frequency.

Splatalogue is the authoritative database for molecular line identification,
hosted at https://splatalogue.online and queryable via a RESTful API.

Registered agent tools:
    - identify_spectral_line(frequency_ghz, tolerance_ghz)
    - search_lines_by_molecule(molecule_name)
"""

import json
from typing import Optional, List, Dict, Any

SPLATALOGUE_API = "https://splatalogue.online/c_export.php"


class SplatalogueTool:
    """Query Splatalogue for molecular spectral line identification."""

    def identify_spectral_line(
        self,
        frequency_ghz: float,
        tolerance_ghz: float = 0.01,
        top_n: int = 5,
    ) -> Dict[str, Any]:
        """
        Identify molecular spectral lines near a given frequency.

        Args:
            frequency_ghz: Rest frequency to search around (GHz)
            tolerance_ghz: Search window ± around the frequency (default: 10 MHz)
            top_n: Maximum number of candidate lines to return

        Returns:
            dict with list of candidate lines (molecule, transition, frequency, intensity)
        """
        try:
            import requests

            freq_low = frequency_ghz - tolerance_ghz
            freq_high = frequency_ghz + tolerance_ghz

            params = {
                "submit": "Search",
                "chemical_name": "",
                "freq_low": str(freq_low * 1000),   # Splatalogue uses MHz
                "freq_high": str(freq_high * 1000),
                "unit": "GHz",
                "displayLovas": "displayLovas",
                "displaySLAIM": "displaySLAIM",
                "displayJPL": "displayJPL",
                "displayCDMS": "displayCDMS",
                "displayToyaMA": "displayToyaMA",
                "displayOSU": "displayOSU",
                "displayRecomb": "displayRecomb",
                "displayLisa": "displayLisa",
                "displayRFI": "displayRFI",
                "ls1": "ls1",
                "ls5": "ls5",
                "el1": "el1",
                "outputJSON": "outputJSON",
            }

            resp = requests.get(SPLATALOGUE_API, params=params, timeout=10)
            if resp.status_code != 200:
                return self._astroquery_fallback(frequency_ghz, tolerance_ghz, top_n)

            data = resp.json()
            lines = []
            for entry in data[:top_n]:
                lines.append({
                    "molecule": entry.get("chemical_name", "Unknown"),
                    "transition": entry.get("quantum_numbers", ""),
                    "frequency_ghz": float(entry.get("orderedfreq", 0)) / 1000,
                    "offset_mhz": round((float(entry.get("orderedfreq", 0)) / 1000 - frequency_ghz) * 1000, 3),
                    "log_intensity": entry.get("sijmu2", "N/A"),
                    "source": entry.get("lovas_int", "Splatalogue"),
                })

            return {
                "query_frequency_ghz": frequency_ghz,
                "tolerance_ghz": tolerance_ghz,
                "n_matches": len(lines),
                "lines": lines,
            }

        except Exception as e:
            return self._astroquery_fallback(frequency_ghz, tolerance_ghz, top_n)

    def _astroquery_fallback(
        self, frequency_ghz: float, tolerance_ghz: float, top_n: int
    ) -> Dict[str, Any]:
        """Fallback to astroquery.splatalogue if the REST API fails."""
        try:
            from astroquery.splatalogue import Splatalogue
            import astropy.units as u

            freq_low = (frequency_ghz - tolerance_ghz) * u.GHz
            freq_high = (frequency_ghz + tolerance_ghz) * u.GHz
            table = Splatalogue.query_lines(freq_low, freq_high)

            if table is None or len(table) == 0:
                return {"query_frequency_ghz": frequency_ghz, "n_matches": 0, "lines": [], "note": "No lines found in search window."}

            lines = []
            for row in table[:top_n]:
                lines.append({
                    "molecule": str(row.get("Chemical Name", "Unknown")),
                    "transition": str(row.get("Quantum Numbers", "")),
                    "frequency_ghz": float(row.get("Freq-GHz(rest frame,redshifted)", 0)),
                    "offset_mhz": round((float(row.get("Freq-GHz(rest frame,redshifted)", 0)) - frequency_ghz) * 1000, 3),
                    "log_intensity": str(row.get("CDMS/JPL Intensity", "N/A")),
                    "source": "astroquery.splatalogue",
                })

            return {
                "query_frequency_ghz": frequency_ghz,
                "tolerance_ghz": tolerance_ghz,
                "n_matches": len(lines),
                "lines": lines,
            }

        except Exception as e2:
            # Hardcoded common lines as last resort
            common_lines = {
                "CO(1-0)": 115.271, "CO(2-1)": 230.538, "CO(3-2)": 345.796,
                "HCN(1-0)": 88.632, "HCO+(1-0)": 89.188,
                "CS(2-1)": 97.981, "CS(7-6)": 342.883,
                "SiO(v=0,2-1)": 86.847, "H2O": 22.235,
                "OH(1.6GHz)": 1.667, "CN(1-0)": 113.491,
            }
            matches = [
                {"molecule": mol, "frequency_ghz": freq_ghz,
                 "offset_mhz": round((freq_ghz - frequency_ghz) * 1000, 1), "source": "built-in catalog"}
                for mol, freq_ghz in common_lines.items()
                if abs(freq_ghz - frequency_ghz) <= tolerance_ghz
            ]
            return {
                "query_frequency_ghz": frequency_ghz,
                "n_matches": len(matches),
                "lines": matches,
                "note": f"Both Splatalogue API and astroquery failed: {e2}. Returned built-in catalog results.",
            }

    def search_lines_by_molecule(
        self,
        molecule_name: str,
        freq_min_ghz: Optional[float] = None,
        freq_max_ghz: Optional[float] = None,
        top_n: int = 10,
    ) -> Dict[str, Any]:
        """
        Search Splatalogue for all known transitions of a given molecule.

        Args:
            molecule_name: Molecule name (e.g., 'CO', 'HCN', 'CH3OH')
            freq_min_ghz: Minimum frequency filter (GHz)
            freq_max_ghz: Maximum frequency filter (GHz)
            top_n: Max results to return
        """
        try:
            from astroquery.splatalogue import Splatalogue
            import astropy.units as u

            freq_low = (freq_min_ghz or 1) * u.GHz
            freq_high = (freq_max_ghz or 1000) * u.GHz

            table = Splatalogue.query_lines(
                freq_low, freq_high, chemical_name=molecule_name
            )

            if table is None or len(table) == 0:
                return {"molecule": molecule_name, "n_matches": 0, "lines": []}

            lines = []
            for row in table[:top_n]:
                lines.append({
                    "transition": str(row.get("Quantum Numbers", "")),
                    "frequency_ghz": float(row.get("Freq-GHz(rest frame,redshifted)", 0)),
                    "log_intensity": str(row.get("CDMS/JPL Intensity", "N/A")),
                    "upper_energy_k": str(row.get("E_U (K)", "N/A")),
                })

            return {"molecule": molecule_name, "n_matches": len(lines), "lines": lines}

        except Exception as e:
            return {"error": str(e), "molecule": molecule_name}
