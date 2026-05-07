"""
ESOTAPClient — ESO Science Archive TAP Integration for Quasar AI
Provides TAP/ADQL query capabilities for VLT instruments (MUSE, KMOS,
X-Shooter, FORS2, HAWK-I, etc.) via the ESO programmatic interface.

CALLED BY: core/agent.py (tool execution: search_eso_archive)
CALLS:     ESO TAP endpoint (http://archive.eso.org/tap_obs) via pyvo

Reference: http://archive.eso.org/programmatic/
ESO Archive FAQ: https://archive.eso.org/cms/faq.html

Registered agent tools:
    - search_eso_archive(target_name, instrument, ra, dec, radius)
"""

import pandas as pd
import warnings
from typing import Optional, Dict, Any, List
from functools import lru_cache

# Try to import pyvo, handle if missing
try:
    import pyvo
    PYVO_AVAILABLE = True
except ImportError:
    PYVO_AVAILABLE = False

# Reuse the cached SIMBAD resolver
try:
    from integrations.alminer_client import _resolve_simbad_cached
except ImportError:
    @lru_cache(maxsize=256)
    def _resolve_simbad_cached(target_name: str):
        from astroquery.simbad import Simbad
        from astropy.coordinates import SkyCoord
        import astropy.units as u
        result = Simbad.query_object(target_name)
        if result is None or len(result) == 0:
            return (None, None)
        coord = SkyCoord(result['RA'][0], result['DEC'][0],
                         unit=(u.hourangle, u.deg))
        return (coord.ra.deg, coord.dec.deg)


class ESOTAPClient:
    """
    Query ESO Science Archive via TAP (ADQL queries).
    
    Provides access to VLT instrument data:
    - MUSE (integral field spectrograph, 0.48-0.93μm)
    - KMOS (multi-object spectrograph, 0.8-2.5μm)
    - X-Shooter (echelle spectrograph, 0.3-2.5μm)
    - FORS2 (imager/spectrograph, 0.33-1.1μm)
    - HAWK-I (wide-field imager, 0.85-2.5μm)
    - UVES (echelle spectrograph, 0.3-1.1μm)
    - SPHERE (high-contrast imager)
    - GRAVITY (interferometric beam combiner)
    - And more VLT/VLTI instruments
    
    Uses the ESO TAP service at http://archive.eso.org/tap_obs
    which exposes the ivoa.ObsCore table following VO standards.
    """

    TAP_URL = "http://archive.eso.org/tap_obs"
    
    # Common ESO/VLT instruments
    INSTRUMENTS = [
        "MUSE", "KMOS", "XSHOOTER", "FORS2", "HAWK-I", "HAWKI",
        "UVES", "SPHERE", "GRAVITY", "CRIRES", "VISIR", "ESPRESSO",
        "FLAMES", "VIRCAM", "OMEGACAM", "MATISSE", "PIONIER"
    ]

    def __init__(self):
        if not PYVO_AVAILABLE:
            warnings.warn("pyvo not installed. ESO TAP queries will fail.")
        self._tap_service = None

    def _get_tap_service(self):
        """Get or create a cached TAP service instance."""
        if self._tap_service is None:
            if not PYVO_AVAILABLE:
                raise ImportError("pyvo is required for ESO TAP queries")
            self._tap_service = pyvo.dal.TAPService(self.TAP_URL)
        return self._tap_service

    def search_by_target(self, target: str, instrument: str = None,
                         collection: str = None, radius_arcmin: float = 1.0,
                         max_results: int = 500) -> pd.DataFrame:
        """
        Search ESO archive by target name.
        
        Resolves the target name via SIMBAD, then performs a cone search
        against the ESO TAP ObsCore table.
        
        Args:
            target: Astronomical target name (e.g., 'NGC 1068', 'Eta Carinae')
            instrument: ESO instrument filter (e.g., 'MUSE', 'KMOS', 'XSHOOTER')
            collection: Data collection filter
            radius_arcmin: Search radius in arcminutes (default 1')
            max_results: Maximum number of results
            
        Returns:
            DataFrame of matching observations
        """
        print(f"[ESO] Searching for '{target}'"
              f"{f' [{instrument}]' if instrument else ''}")
        
        # Resolve target name
        ra, dec = _resolve_simbad_cached(target)
        if ra is None:
            print(f"[ESO] Could not resolve target '{target}' via SIMBAD")
            return pd.DataFrame()
        
        print(f"[ESO] Resolved to RA={ra:.4f}, Dec={dec:.4f}")
        return self.search_by_position(
            ra, dec, radius_arcmin=radius_arcmin,
            instrument=instrument, collection=collection,
            max_results=max_results
        )

    def search_by_position(self, ra: float, dec: float, radius_arcmin: float = 1.0,
                           instrument: str = None, collection: str = None,
                           max_results: int = 500) -> pd.DataFrame:
        """
        Positional cone search against ESO TAP ObsCore.
        
        Args:
            ra: Right Ascension in degrees (ICRS)
            dec: Declination in degrees (ICRS)
            radius_arcmin: Search radius in arcminutes
            instrument: ESO instrument filter
            collection: Data collection filter
            max_results: Maximum results
        """
        try:
            service = self._get_tap_service()
            
            # Convert radius to degrees for ADQL
            radius_deg = radius_arcmin / 60.0
            
            # Build ADQL query
            where_clauses = [
                f"CONTAINS(POINT('ICRS', s_ra, s_dec), "
                f"CIRCLE('ICRS', {ra:.6f}, {dec:.6f}, {radius_deg:.6f})) = 1"
            ]
            
            if instrument:
                # Normalize instrument name
                instr = instrument.upper().replace("-", "").replace(" ", "")
                # Handle common aliases
                instr_map = {
                    "XSHOOTER": "XSHOOTER",
                    "X-SHOOTER": "XSHOOTER",
                    "HAWKI": "HAWKI",
                    "HAWK-I": "HAWKI",
                }
                instr = instr_map.get(instr, instr)
                where_clauses.append(f"instrument_name LIKE '%{instr}%'")
            
            if collection:
                where_clauses.append(f"obs_collection = '{collection}'")
            
            where = " AND ".join(where_clauses)
            
            adql = f"""
            SELECT TOP {max_results} *
            FROM ivoa.ObsCore
            WHERE {where}
            ORDER BY t_min DESC
            """
            
            print(f"[ESO] Executing TAP query (radius={radius_arcmin}')")
            result = service.search(adql)
            df = result.to_table().to_pandas()
            
            if df.empty:
                print(f"[ESO] No results found")
                return pd.DataFrame()
            
            print(f"[ESO] Found {len(df)} observations")
            return self._standardize_columns(df)
            
        except ImportError:
            print("[ESO] pyvo not available")
            return pd.DataFrame()
        except Exception as e:
            print(f"[ESO] Search error: {e}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()

    def execute_adql(self, query: str) -> pd.DataFrame:
        """
        Execute a raw ADQL query against the ESO TAP service.
        
        Args:
            query: ADQL query string
            
        Returns:
            DataFrame of results
        """
        try:
            service = self._get_tap_service()
            print(f"[ESO] Executing custom ADQL query")
            result = service.search(query)
            df = result.to_table().to_pandas()
            print(f"[ESO] Query returned {len(df)} rows")
            return self._standardize_columns(df)
        except Exception as e:
            print(f"[ESO] ADQL query error: {e}")
            return pd.DataFrame()

    def list_instruments(self) -> List[str]:
        """
        List available ESO instruments by querying distinct values.
        Falls back to static list if query fails.
        """
        try:
            service = self._get_tap_service()
            result = service.search(
                "SELECT DISTINCT instrument_name FROM ivoa.ObsCore "
                "ORDER BY instrument_name"
            )
            instruments = [str(row["instrument_name"]) for row in result]
            return instruments
        except Exception:
            return self.INSTRUMENTS

    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize ESO TAP output columns to match Quasar's format.
        The ObsCore table already uses standard IVOA column names,
        so mostly we just add convenience columns.
        """
        if df.empty:
            return df
        
        # ObsCore standard columns are already well-named:
        # s_ra, s_dec, target_name, instrument_name, obs_collection,
        # t_exptime, s_resolution, dataproduct_type, access_url, etc.
        
        # Add telescope label
        if "obs_collection" in df.columns:
            df["telescope"] = df["obs_collection"]
        else:
            df["telescope"] = "ESO"
        
        # Map obs_id to project_code if not present
        if "project_code" not in df.columns and "obs_id" in df.columns:
            df["project_code"] = df["obs_id"]
        
        # Ensure instrument_name is clean
        if "instrument_name" in df.columns:
            df["instrument_name"] = df["instrument_name"].astype(str).str.strip()
        
        return df
