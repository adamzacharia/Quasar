"""
IRSAClient — IRSA (Infrared Science Archive) Integration for Quasar AI
Provides catalog-query capabilities for infrared surveys such as WISE,
2MASS, Spitzer, and other IRSA-hosted catalogs.

CALLED BY: core/agent.py (tool execution: search_irsa)
CALLS:     astroquery.ipac.irsa

Registered agent tools:
    - search_irsa(target_name, catalog, radius_arcsec)
"""

import pandas as pd
import warnings
from typing import Optional, Dict, Any, List

# Try to import astroquery IRSA, handle if missing
try:
    from astroquery.ipac.irsa import Irsa
    IRSA_AVAILABLE = True
except ImportError:
    try:
        # Older astroquery versions
        from astroquery.irsa import Irsa
        IRSA_AVAILABLE = True
    except ImportError:
        IRSA_AVAILABLE = False

# Canonical cached SIMBAD resolver
from integrations.simbad_resolver import _resolve_simbad_cached


class IRSAClient:
    """
    Query IRSA catalog services for infrared survey sources.
    
    Supported catalogs include:
    - AllWISE Source Catalog (wise_allwise_p3as_psd)
    - 2MASS All-Sky Point Source Catalog (fp_psc)
    - 2MASS All-Sky Extended Source Catalog (fp_xsc)
    - Spitzer Enhanced Imaging Products (SEIP)
    - And many more IRSA-hosted catalogs
    """

    # Common IRSA catalogs
    POPULAR_CATALOGS = {
        "allwise": "allwise_p3as_psd",
        "wise": "allwise_p3as_psd",
        "2mass_psc": "fp_psc",
        "2mass": "fp_psc",
        "2mass_xsc": "fp_xsc",
        "seip": "slphotdr4",
    }

    def __init__(self):
        if not IRSA_AVAILABLE:
            warnings.warn("astroquery.ipac.irsa not installed. IRSA queries will fail.")

    def search_by_target(self, target: str, catalog: str = "allwise_p3as_psd",
                         radius_arcsec: float = 30.0,
                         max_results: int = 500) -> pd.DataFrame:
        """
        Search IRSA catalog by target name.
        
        Args:
            target: Astronomical target name (e.g., 'M31', 'NGC 253')
            catalog: IRSA catalog name (default: AllWISE)
            radius_arcsec: Search radius in arcseconds
            max_results: Maximum results
            
        Returns:
            DataFrame of matching sources
        """
        if not IRSA_AVAILABLE:
            print("[IRSA] ERROR: astroquery IRSA not available")
            return pd.DataFrame()
        
        try:
            # Resolve catalog alias
            catalog = self._resolve_catalog(catalog)
            
            print(f"[IRSA] Searching '{target}' in catalog '{catalog}' "
                  f"(radius={radius_arcsec}\")")
            
            # Resolve target name to coordinates
            ra, dec = _resolve_simbad_cached(target)
            if ra is None:
                print(f"[IRSA] Could not resolve target '{target}' via SIMBAD")
                return pd.DataFrame()
            
            return self.search_by_position(
                ra, dec, radius_arcsec=radius_arcsec,
                catalog=catalog, max_results=max_results
            )
            
        except Exception as e:
            print(f"[IRSA] Target search error: {e}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()

    def search_by_position(self, ra: float, dec: float, radius_arcsec: float = 30.0,
                           catalog: str = "allwise_p3as_psd",
                           max_results: int = 500) -> pd.DataFrame:
        """
        Cone search by RA/Dec in an IRSA catalog.
        
        Args:
            ra: Right Ascension in degrees
            dec: Declination in degrees
            radius_arcsec: Search radius in arcseconds
            catalog: IRSA catalog name
            max_results: Maximum results
        """
        if not IRSA_AVAILABLE:
            return pd.DataFrame()
        
        try:
            from astropy.coordinates import SkyCoord
            import astropy.units as u
            
            # Resolve catalog alias
            catalog = self._resolve_catalog(catalog)
            
            coord = SkyCoord(ra=ra, dec=dec, unit="deg")
            radius = radius_arcsec * u.arcsec
            
            print(f"[IRSA] Cone search: RA={ra:.4f}, Dec={dec:.4f}, "
                  f"radius={radius_arcsec}\", catalog={catalog}")
            
            result = Irsa.query_region(
                coord, catalog=catalog, radius=radius
            )
            
            if result is None or len(result) == 0:
                print("[IRSA] No results found")
                return pd.DataFrame()
            
            df = result.to_pandas()
            
            if len(df) > max_results:
                df = df.head(max_results)
            
            print(f"[IRSA] Found {len(df)} sources")
            return self._standardize_columns(df, catalog)
            
        except Exception as e:
            print(f"[IRSA] Position search error: {e}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()

    def _resolve_catalog(self, catalog: str) -> str:
        """Resolve catalog aliases to IRSA catalog names."""
        if catalog is None:
            return "allwise_p3as_psd"
        catalog_lower = catalog.lower().strip()
        return self.POPULAR_CATALOGS.get(catalog_lower, catalog)

    def _standardize_columns(self, df: pd.DataFrame, catalog: str) -> pd.DataFrame:
        """Standardize IRSA output columns for Quasar data cards."""
        if df.empty:
            return df
        
        # Try to map RA/Dec columns (different catalogs use different names)
        ra_cols = ["ra", "RA", "RAJ2000", "ra_01"]
        dec_cols = ["dec", "DEC", "DEJ2000", "dec_01"]
        
        for col in ra_cols:
            if col in df.columns:
                df["s_ra"] = df[col]
                break
        
        for col in dec_cols:
            if col in df.columns:
                df["s_dec"] = df[col]
                break
        
        # Add standard labels
        df["telescope"] = "IRSA"
        df["instrument_name"] = catalog.upper()
        
        # Try to build target_name from designation columns
        name_cols = ["designation", "source_id", "cntr", "objid"]
        for col in name_cols:
            if col in df.columns:
                df["target_name"] = df[col].astype(str)
                break
        
        return df
