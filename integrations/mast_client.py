"""
MASTClient — Deep MAST Archive Integration for Quasar AI
Provides full query capabilities for JWST, HST, TESS, Kepler data
via astroquery.mast.

CALLED BY: core/agent.py (tool execution: search_mast, search_mast_by_criteria)
CALLS:     astroquery.mast.Observations

Registered agent tools:
    - search_mast(target_name, mission, instrument, radius)
    - search_mast_by_criteria(mission, instrument, proposal_id, filters, ...)
    - get_mast_products()  (file-level product listing)
"""

import pandas as pd
import warnings
from typing import Optional, Dict, Any, List, Tuple

# Try to import astroquery.mast, handle if missing
try:
    from astroquery.mast import Observations
    MAST_AVAILABLE = True
except ImportError:
    MAST_AVAILABLE = False

# Canonical cached SIMBAD resolver
from integrations.simbad_resolver import _resolve_simbad_cached


class MASTClient:
    """
    Deep query client for MAST archive (JWST, HST, TESS, Kepler).
    
    Wraps astroquery.mast.Observations to provide:
    - Target name search with mission/instrument filters
    - Positional cone search
    - Rich criteria-based search (program ID, date range, filter, instrument)
    - File-level product listing for download
    """

    # Standard MAST missions
    SUPPORTED_MISSIONS = ["JWST", "HST", "TESS", "Kepler", "K2",
                          "GALEX", "IUE", "FUSE", "Swift"]
    
    # JWST instruments
    JWST_INSTRUMENTS = ["NIRCAM", "NIRSPEC", "MIRI", "NIRISS", "FGS"]
    
    # HST instruments  
    HST_INSTRUMENTS = ["ACS", "WFC3", "COS", "STIS", "NICMOS", "WFPC2"]

    def __init__(self):
        if not MAST_AVAILABLE:
            warnings.warn("astroquery.mast not installed. MAST queries will fail.")

    def search_by_target(self, target: str, mission: str = None,
                         instrument: str = None, radius: str = "30s",
                         max_results: int = 500) -> pd.DataFrame:
        """
        Search MAST by target name with optional mission/instrument filters.
        
        Args:
            target: Astronomical target name (e.g., 'M87', 'Carina Nebula')
            mission: Filter by mission (e.g., 'JWST', 'HST')
            instrument: Filter by instrument (e.g., 'NIRCAM', 'ACS')
            radius: Search radius (e.g., '30s' for 30 arcsec)
            max_results: Maximum number of results
            
        Returns:
            DataFrame of matching observations
        """
        if not MAST_AVAILABLE:
            print("[MAST] ERROR: astroquery.mast not available")
            return pd.DataFrame()
        
        try:
            print(f"[MAST] Searching for '{target}'"
                  f"{f' [{mission}]' if mission else ''}"
                  f"{f' [{instrument}]' if instrument else ''}")
            
            # Build criteria dict
            criteria = {}
            if mission:
                criteria["obs_collection"] = mission.upper()
            if instrument:
                criteria["instrument_name"] = instrument.upper()
            
            if criteria:
                # Use query_criteria with target coordinates
                ra, dec = _resolve_simbad_cached(target)
                if ra is None:
                    print(f"[MAST] Could not resolve target '{target}' via SIMBAD")
                    return pd.DataFrame()
                
                from astropy.coordinates import SkyCoord
                import astropy.units as u
                coord = SkyCoord(ra=ra, dec=dec, unit="deg")
                
                # Parse radius
                radius_val = self._parse_radius(radius)
                
                obs = Observations.query_criteria(
                    coordinates=coord,
                    radius=radius_val,
                    **criteria
                )
            else:
                # Simple target query
                obs = Observations.query_object(target, radius=radius)
            
            if obs is None or len(obs) == 0:
                print(f"[MAST] No results found for '{target}'")
                return pd.DataFrame()
            
            df = obs.to_pandas()
            
            # Limit results
            if len(df) > max_results:
                df = df.head(max_results)
            
            print(f"[MAST] Found {len(df)} observations")
            return self._standardize_columns(df)
            
        except Exception as e:
            print(f"[MAST] Search error: {e}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()

    def search_by_position(self, ra: float, dec: float, radius_arcmin: float = 1.0,
                           mission: str = None, instrument: str = None,
                           max_results: int = 500) -> pd.DataFrame:
        """
        Cone search by RA/Dec with optional mission/instrument filters.
        
        Args:
            ra: Right Ascension in degrees (ICRS)
            dec: Declination in degrees (ICRS)
            radius_arcmin: Search radius in arcminutes
            mission: Filter by mission
            instrument: Filter by instrument
            max_results: Maximum results
        """
        if not MAST_AVAILABLE:
            return pd.DataFrame()
        
        try:
            from astropy.coordinates import SkyCoord
            import astropy.units as u
            
            coord = SkyCoord(ra=ra, dec=dec, unit="deg")
            radius = radius_arcmin * u.arcmin
            
            criteria = {}
            if mission:
                criteria["obs_collection"] = mission.upper()
            if instrument:
                criteria["instrument_name"] = instrument.upper()
            
            print(f"[MAST] Cone search: RA={ra:.4f}, Dec={dec:.4f}, "
                  f"radius={radius_arcmin}'")
            
            obs = Observations.query_criteria(
                coordinates=coord,
                radius=radius,
                **criteria
            )
            
            if obs is None or len(obs) == 0:
                return pd.DataFrame()
            
            df = obs.to_pandas()
            if len(df) > max_results:
                df = df.head(max_results)
            
            print(f"[MAST] Found {len(df)} observations")
            return self._standardize_columns(df)
            
        except Exception as e:
            print(f"[MAST] Position search error: {e}")
            return pd.DataFrame()

    def search_by_criteria(self, mission: str = None, instrument: str = None,
                           proposal_id: str = None, filters: str = None,
                           target_name: str = None, dataproduct_type: str = None,
                           date_range: Tuple[str, str] = None,
                           max_results: int = 500) -> pd.DataFrame:
        """
        Advanced criteria-based search with rich filtering.
        
        Args:
            mission: Mission name (JWST, HST, etc.)
            instrument: Instrument name
            proposal_id: Specific proposal/program ID
            filters: Filter name (e.g., 'F200W', 'F444W')
            target_name: Target name (resolves to coordinates internally)
            dataproduct_type: 'image', 'spectrum', 'cube', etc.
            date_range: Tuple of (start_date, end_date) as ISO strings
            max_results: Maximum results
        """
        if not MAST_AVAILABLE:
            return pd.DataFrame()
        
        try:
            criteria = {}
            
            if mission:
                criteria["obs_collection"] = mission.upper()
            if instrument:
                criteria["instrument_name"] = instrument.upper()
            if proposal_id:
                criteria["proposal_id"] = proposal_id
            if filters:
                criteria["filters"] = filters
            if dataproduct_type:
                criteria["dataproduct_type"] = dataproduct_type
            if target_name:
                criteria["objectname"] = target_name
                criteria["radius"] = "3s"  # Default radius for name search
            if date_range and len(date_range) == 2:
                criteria["t_min"] = [date_range[0], date_range[1]]
            
            if not criteria:
                print("[MAST] No search criteria provided")
                return pd.DataFrame()
            
            criteria_str = ", ".join(f"{k}={v}" for k, v in criteria.items())
            print(f"[MAST] Criteria search: {criteria_str}")
            
            obs = Observations.query_criteria(**criteria)
            
            if obs is None or len(obs) == 0:
                print("[MAST] No results found")
                return pd.DataFrame()
            
            df = obs.to_pandas()
            if len(df) > max_results:
                df = df.head(max_results)
            
            print(f"[MAST] Found {len(df)} observations")
            return self._standardize_columns(df)
            
        except Exception as e:
            print(f"[MAST] Criteria search error: {e}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()

    def get_product_list(self, observations: pd.DataFrame,
                         productType: str = None,
                         extension: str = None) -> pd.DataFrame:
        """
        Get file-level product list for observations.
        
        Args:
            observations: DataFrame from a previous MAST search
            productType: Filter by type ('SCIENCE', 'CALIBRATION', 'PREVIEW')
            extension: Filter by file extension ('fits', 'jpg', etc.)
            
        Returns:
            DataFrame with file-level product info (filenames, sizes, URLs)
        """
        if not MAST_AVAILABLE or observations.empty:
            return pd.DataFrame()
        
        try:
            from astropy.table import Table
            
            # Convert back to astropy table for MAST API
            obs_table = Table.from_pandas(observations)
            
            products = Observations.get_product_list(obs_table)
            
            if products is None or len(products) == 0:
                return pd.DataFrame()
            
            df = products.to_pandas()
            
            # Apply filters
            if productType:
                df = df[df["productType"].str.upper() == productType.upper()]
            if extension:
                df = df[df["productFilename"].str.endswith(f".{extension}")]
            
            print(f"[MAST] Found {len(df)} data products")
            return df
            
        except Exception as e:
            print(f"[MAST] Product list error: {e}")
            return pd.DataFrame()

    def download_products(self, products: pd.DataFrame = None,
                          observations: pd.DataFrame = None,
                          download_dir: str = None,
                          productType: str = "SCIENCE",
                          extension: str = "fits",
                          max_files: int = 10) -> Dict[str, Any]:
        """
        Download FITS files and data products from MAST.
        
        Args:
            products: DataFrame from get_product_list (preferred)
            observations: DataFrame from search (will get products first)
            download_dir: Directory to save files (default ~/quasar_data/mast/)
            productType: Filter by type: 'SCIENCE', 'CALIBRATION', 'PREVIEW'
            extension: Filter by extension: 'fits', 'jpg', etc.
            max_files: Maximum number of files to download (safety limit)
            
        Returns:
            Dict with download paths, file count, and total size
        """
        if not MAST_AVAILABLE:
            return {"success": False, "error": "astroquery.mast not available"}
        
        import os
        download_dir = download_dir or os.path.join(
            os.path.expanduser("~"), "quasar_data", "mast"
        )
        os.makedirs(download_dir, exist_ok=True)
        
        try:
            from astropy.table import Table
            
            # Get products if only observations provided
            if products is None or products.empty:
                if observations is None or observations.empty:
                    return {"success": False, "error": "No data to download. Run search_mast and get_mast_products first."}
                products = self.get_product_list(observations, 
                                                 productType=productType,
                                                 extension=extension)
            
            if products.empty:
                return {"success": False, "error": "No matching data products found."}
            
            # Apply safety limit
            if len(products) > max_files:
                print(f"[MAST] Limiting download to {max_files} files (of {len(products)} available)")
                products = products.head(max_files)
            
            # Convert to astropy table for MAST API
            prod_table = Table.from_pandas(products)
            
            print(f"[MAST] Downloading {len(products)} files to {download_dir}")
            
            manifest = Observations.download_products(
                prod_table,
                download_dir=download_dir
            )
            
            if manifest is None:
                return {"success": False, "error": "Download returned no results"}
            
            manifest_df = manifest.to_pandas()
            downloaded = manifest_df[manifest_df["Status"] == "COMPLETE"] if "Status" in manifest_df.columns else manifest_df
            
            paths = list(downloaded["Local Path"].values) if "Local Path" in downloaded.columns else []
            
            print(f"[MAST] Downloaded {len(downloaded)} files")
            
            return {
                "success": True,
                "downloaded_files": len(downloaded),
                "total_attempted": len(products),
                "download_dir": download_dir,
                "file_paths": paths[:20],  # Limit paths in response
                "note": f"Downloaded {len(downloaded)} files to {download_dir}"
            }
            
        except Exception as e:
            print(f"[MAST] Download error: {e}")
            import traceback
            traceback.print_exc()
            return {"success": False, "error": f"Download failed: {str(e)}"}

    def _parse_radius(self, radius_str: str):
        """Parse radius string like '30s', '1m', '0.5d' into astropy quantity."""
        import astropy.units as u
        
        if isinstance(radius_str, (int, float)):
            return radius_str * u.deg
        
        radius_str = str(radius_str).strip().lower()
        if radius_str.endswith("s"):
            return float(radius_str[:-1]) * u.arcsec
        elif radius_str.endswith("m"):
            return float(radius_str[:-1]) * u.arcmin
        elif radius_str.endswith("d"):
            return float(radius_str[:-1]) * u.deg
        else:
            try:
                return float(radius_str) * u.arcsec
            except ValueError:
                return 30 * u.arcsec

    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize MAST output columns to match Quasar's expected format.
        Aligns with the same column conventions as ALMA data cards.
        """
        if df.empty:
            return df
        
        # Rename key columns for consistency
        rename_map = {
            "s_ra": "s_ra",
            "s_dec": "s_dec",
            "target_name": "target_name",
            "obs_collection": "telescope",
            "instrument_name": "instrument_name",
            "filters": "filters",
            "proposal_id": "project_code",
            "t_exptime": "t_exptime",
            "dataproduct_type": "dataproduct_type",
            "obs_id": "obs_id",
            "calib_level": "calib_level",
            "t_min": "t_min",
            "t_max": "t_max",
        }
        
        df = df.rename(columns={k: v for k, v in rename_map.items() 
                                if k in df.columns and k != v})
        
        # Ensure telescope column exists
        if "telescope" not in df.columns and "obs_collection" in df.columns:
            df["telescope"] = df["obs_collection"]
        elif "telescope" not in df.columns:
            df["telescope"] = "MAST"
        
        return df
