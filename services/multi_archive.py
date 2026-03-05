"""
MultiArchiveMatcher — Cross-archive source query for Quasar AI
Simultaneously queries multiple astronomical archives for a given source.

Archives queried:
  - Simbad (object type, coordinates, aliases)
  - NED (galaxy distances, redshifts, photometry)
  - VizieR (catalog cross-match)
  - MAST (HST, JWST, Kepler data)
  - Chandra Source Catalog (X-ray, CSC2)
  - Fermi 4FGL (gamma-ray)

Registered agent tools:
    - cross_match_source(target_name)
    - cone_search_all_archives(ra, dec, radius_arcmin)
"""

import json
from typing import Optional, Dict, Any, List
from concurrent.futures import ThreadPoolExecutor, as_completed


class MultiArchiveMatcher:
    """Query multiple astronomical archives in parallel for a given source."""

    def _query_simbad(self, target: str) -> Dict[str, Any]:
        try:
            from astroquery.simbad import Simbad
            Simbad.add_votable_fields("otype", "rv_value", "z_value", "flux(V)", "flux(B)")
            result = Simbad.query_object(target)
            if result is None:
                return {"archive": "Simbad", "found": False}
            row = result[0]
            return {
                "archive": "Simbad",
                "found": True,
                "ra_deg": float(row["RA_d"]) if "RA_d" in result.colnames else str(row.get("RA", "")),
                "dec_deg": float(row["DEC_d"]) if "DEC_d" in result.colnames else str(row.get("DEC", "")),
                "object_type": str(row.get("OTYPE", "")),
                "redshift": str(row.get("RVZ_REDSHIFT", "N/A")),
                "v_mag": str(row.get("FLUX_V", "N/A")),
            }
        except Exception as e:
            return {"archive": "Simbad", "found": False, "error": str(e)}

    def _query_ned(self, target: str) -> Dict[str, Any]:
        try:
            from astroquery.ipac.ned import Ned
            result = Ned.query_object(target)
            if result is None or len(result) == 0:
                return {"archive": "NED", "found": False}
            row = result[0]
            return {
                "archive": "NED",
                "found": True,
                "object_type": str(row.get("Type", "")),
                "redshift": str(row.get("Redshift", "N/A")),
                "ra_deg": float(row.get("RA", 0)),
                "dec_deg": float(row.get("DEC", 0)),
            }
        except Exception as e:
            return {"archive": "NED", "found": False, "error": str(e)}

    def _query_mast(self, target: str) -> Dict[str, Any]:
        try:
            from astroquery.mast import Observations
            obs = Observations.query_object(target, radius="30s")
            if obs is None or len(obs) == 0:
                return {"archive": "MAST", "found": False, "n_observations": 0}
            missions = list(set(str(o) for o in obs["obs_collection"][:20]))
            return {
                "archive": "MAST",
                "found": True,
                "n_observations": len(obs),
                "missions": missions[:10],
                "instruments": list(set(str(o) for o in obs["instrument_name"][:20]))[:8],
            }
        except Exception as e:
            return {"archive": "MAST", "found": False, "error": str(e)}

    def _query_vizier(self, target: str) -> Dict[str, Any]:
        try:
            from astroquery.vizier import Vizier
            from astropy.coordinates import SkyCoord
            import astropy.units as u

            # Resolve name to coordinates first
            from astroquery.simbad import Simbad
            coord_result = Simbad.query_object(target)
            if coord_result is None:
                return {"archive": "VizieR", "found": False, "error": "Could not resolve coordinates"}

            row = coord_result[0]
            coords = SkyCoord(row["RA"], row["DEC"], unit=(u.hourangle, u.deg), frame="icrs")
            catalogs = Vizier.query_region(coords, radius=10 * u.arcsec)

            if not catalogs:
                return {"archive": "VizieR", "found": False, "n_catalogs": 0}
            return {
                "archive": "VizieR",
                "found": True,
                "n_catalogs": len(catalogs),
                "catalog_names": list(catalogs.keys())[:10],
            }
        except Exception as e:
            return {"archive": "VizieR", "found": False, "error": str(e)}

    def _query_fermi(self, target: str) -> Dict[str, Any]:
        """Check if source is in Fermi 4FGL catalog."""
        try:
            from astroquery.heasarc import Heasarc
            heasarc = Heasarc()
            result = heasarc.query_object(target, mission="FERMI4FGL", radius="1 degree")
            if result is None or len(result) == 0:
                return {"archive": "Fermi 4FGL", "found": False}
            row = result[0]
            return {
                "archive": "Fermi 4FGL",
                "found": True,
                "source_name": str(row.get("NAME", "")),
                "flux_density": str(row.get("FLUX1000", "N/A")),
                "spectral_index": str(row.get("SPECTRAL_INDEX", "N/A")),
                "source_class": str(row.get("SOURCE_CLASS", "N/A")),
            }
        except Exception as e:
            return {"archive": "Fermi 4FGL", "found": False, "error": str(e)}

    def cross_match_source(
        self, target_name: str, archives: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Query multiple archives in parallel for a given source name.

        Args:
            target_name: Source name recognized by Simbad (e.g., "M87", "HL Tau", "NGC 1275")
            archives: List of archives to query. Defaults to all: ["simbad", "ned", "mast", "vizier", "fermi"]

        Returns:
            dict summarizing results from each archive
        """
        all_archives = {
            "simbad": self._query_simbad,
            "ned": self._query_ned,
            "mast": self._query_mast,
            "vizier": self._query_vizier,
            "fermi": self._query_fermi,
        }

        selected = {k: v for k, v in all_archives.items()
                    if archives is None or k.lower() in [a.lower() for a in archives]}

        results = {}
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {executor.submit(fn, target_name): name for name, fn in selected.items()}
            for future in as_completed(futures):
                name = futures[future]
                try:
                    results[name] = future.result(timeout=15)
                except Exception as e:
                    results[name] = {"archive": name, "found": False, "error": str(e)}

        # Summary
        found_in = [k for k, v in results.items() if v.get("found")]
        return {
            "target": target_name,
            "archives_queried": list(selected.keys()),
            "found_in": found_in,
            "n_archives_with_data": len(found_in),
            "results": results,
        }
