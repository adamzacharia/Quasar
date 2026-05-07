"""
MultiArchiveMatcher - Cross-archive source query for Quasar AI.

Simultaneously queries multiple astronomical archives for a given source.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

import pandas as pd

from integrations.eso_tap_client import ESOTAPClient
from integrations.irsa_client import IRSAClient
from integrations.mast_client import MASTClient


class MultiArchiveMatcher:
    """Query multiple astronomical archives in parallel for a given source."""

    def __init__(
        self,
        mast_client: Optional[MASTClient] = None,
        eso_client: Optional[ESOTAPClient] = None,
        irsa_client: Optional[IRSAClient] = None,
    ):
        self.mast_client = mast_client or MASTClient()
        self.eso_client = eso_client or ESOTAPClient()
        self.irsa_client = irsa_client or IRSAClient()

    def _top_values(self, df: pd.DataFrame, column: str, limit: int = 10) -> List[str]:
        if df.empty or column not in df.columns:
            return []

        values = []
        for value in df[column].dropna().astype(str):
            text = value.strip()
            if text and text not in values:
                values.append(text)
            if len(values) >= limit:
                break
        return values

    def _dataframe_result(self, archive: str, df: pd.DataFrame, **extra: Any) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "archive": archive,
            "found": not df.empty,
            "total_results": int(len(df)),
            "data": df,
        }
        if not df.empty:
            result["columns"] = list(df.columns[:20])
        result.update(extra)
        return result

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

    def deep_search_mast(
        self,
        target: str,
        mission: Optional[str] = None,
        instrument: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run a deep MAST search and keep the full observation table."""
        try:
            df = self.mast_client.search_by_target(
                target=target,
                mission=mission,
                instrument=instrument,
            )
            return self._dataframe_result(
                "MAST",
                df,
                target=target,
                mission=mission,
                instrument=instrument,
                n_observations=int(len(df)),
                missions=self._top_values(df, "telescope"),
                instruments=self._top_values(df, "instrument_name"),
                filters=self._top_values(df, "filters"),
                program_ids=self._top_values(df, "project_code"),
            )
        except Exception as e:
            return {
                "archive": "MAST",
                "found": False,
                "total_results": 0,
                "data": pd.DataFrame(),
                "error": str(e),
            }

    def deep_search_eso(
        self,
        target: str,
        instrument: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run a deep ESO archive search and keep the full observation table."""
        try:
            df = self.eso_client.search_by_target(
                target=target,
                instrument=instrument,
            )
            return self._dataframe_result(
                "ESO",
                df,
                target=target,
                instrument=instrument,
                n_observations=int(len(df)),
                instruments=self._top_values(df, "instrument_name"),
                collections=self._top_values(df, "obs_collection"),
                data_types=self._top_values(df, "dataproduct_type"),
            )
        except Exception as e:
            return {
                "archive": "ESO",
                "found": False,
                "total_results": 0,
                "data": pd.DataFrame(),
                "error": str(e),
            }

    def deep_search_irsa(
        self,
        target: str,
        catalog: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run a deep IRSA catalog search and keep the full source table."""
        try:
            resolved_catalog = catalog or "allwise"
            df = self.irsa_client.search_by_target(
                target=target,
                catalog=resolved_catalog,
            )
            return self._dataframe_result(
                "IRSA",
                df,
                target=target,
                catalog=resolved_catalog,
                n_sources=int(len(df)),
                catalogs=self._top_values(df, "instrument_name"),
            )
        except Exception as e:
            return {
                "archive": "IRSA",
                "found": False,
                "total_results": 0,
                "data": pd.DataFrame(),
                "error": str(e),
            }

    def _query_mast(self, target: str) -> Dict[str, Any]:
        try:
            result = self.deep_search_mast(target)
            if not result.get("found"):
                return {"archive": "MAST", "found": False, "n_observations": 0}
            return {
                "archive": "MAST",
                "found": True,
                "n_observations": result.get("n_observations", 0),
                "missions": result.get("missions", [])[:10],
                "instruments": result.get("instruments", [])[:8],
                "filters": result.get("filters", [])[:8],
                "program_ids": result.get("program_ids", [])[:8],
            }
        except Exception as e:
            return {"archive": "MAST", "found": False, "error": str(e)}

    def _query_eso(self, target: str) -> Dict[str, Any]:
        try:
            result = self.deep_search_eso(target)
            if not result.get("found"):
                return {"archive": "ESO", "found": False, "n_observations": 0}
            return {
                "archive": "ESO",
                "found": True,
                "n_observations": result.get("n_observations", 0),
                "instruments": result.get("instruments", [])[:8],
                "collections": result.get("collections", [])[:8],
                "data_types": result.get("data_types", [])[:8],
            }
        except Exception as e:
            return {"archive": "ESO", "found": False, "error": str(e)}

    def _query_irsa(self, target: str) -> Dict[str, Any]:
        try:
            result = self.deep_search_irsa(target)
            if not result.get("found"):
                return {"archive": "IRSA", "found": False, "n_sources": 0}
            return {
                "archive": "IRSA",
                "found": True,
                "n_sources": result.get("n_sources", 0),
                "catalogs": result.get("catalogs", [])[:8],
            }
        except Exception as e:
            return {"archive": "IRSA", "found": False, "error": str(e)}

    def _query_vizier(self, target: str) -> Dict[str, Any]:
        try:
            from astroquery.simbad import Simbad
            from astroquery.vizier import Vizier
            from astropy.coordinates import SkyCoord
            import astropy.units as u

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

        Defaults to the existing five-archive summary path. ESO and IRSA can be
        requested explicitly via the archives parameter.
        """
        all_archives = {
            "simbad": self._query_simbad,
            "ned": self._query_ned,
            "mast": self._query_mast,
            "vizier": self._query_vizier,
            "fermi": self._query_fermi,
            "eso": self._query_eso,
            "irsa": self._query_irsa,
        }
        default_archives = ["simbad", "ned", "mast", "vizier", "fermi"]
        requested = default_archives if archives is None else [a.lower() for a in archives]

        selected = {k: v for k, v in all_archives.items() if k in requested}

        results: Dict[str, Dict[str, Any]] = {}
        with ThreadPoolExecutor(max_workers=max(1, len(selected))) as executor:
            futures = {executor.submit(fn, target_name): name for name, fn in selected.items()}
            for future in as_completed(futures):
                name = futures[future]
                try:
                    results[name] = future.result(timeout=15)
                except Exception as e:
                    results[name] = {"archive": name, "found": False, "error": str(e)}

        found_in = [name for name, result in results.items() if result.get("found")]
        return {
            "target": target_name,
            "archives_queried": list(selected.keys()),
            "found_in": found_in,
            "n_archives_with_data": len(found_in),
            "results": results,
        }
