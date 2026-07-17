"""
ALminer Client Integration
Wraps alminer functionality for Quasar
"""

import pandas as pd
from typing import Optional, Dict, Any, List, Union
import warnings
import os

# Try to import alminer, handle if missing
try:
    import alminer
    ALMINER_AVAILABLE = True
except ImportError:
    ALMINER_AVAILABLE = False

# Try to import matplotlib for plotting
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    
# ── Cached SIMBAD resolution (canonical impl: integrations/simbad_resolver.py) ──
from integrations.simbad_resolver import _resolve_simbad_cached


# Canonical ALMA result columns — guaranteed present on BOTH the TAP and
# ALminer paths by _standardize_columns so downstream code sees a deterministic
# schema regardless of which source answered. (C14)
_CANONICAL_ALMA_COLUMNS = [
    "s_ra", "s_dec", "t_exptime", "Band", "resolution", "sensitivity",
    "bandwidth", "freq_min", "freq_max", "freq_min_ghz", "freq_max_ghz",
    "telescope", "instrument_name", "access_url",
]


class ALminerClient:
    """Client for interacting with ALMA archive via ALminer"""

    def __init__(self):
        """Initialize ALminer client"""
        if not ALMINER_AVAILABLE:
            warnings.warn("alminer not installed. ALMA searches will fail.")
        
        # Ensure we have a downloads directory
        self.download_dir = "./downloads"
        if not os.path.exists(self.download_dir):
            os.makedirs(self.download_dir, exist_ok=True)
        
        # Cached TAP service — reuse TCP connection across queries
        self._tap_service = None

    def _get_tap_service(self):
        """Get or create a cached pyvo TAPService instance."""
        if self._tap_service is None:
            import pyvo
            from integrations.tap import _TimeoutHTTPSession
            session = _TimeoutHTTPSession(timeout=30.0)
            try:
                self._tap_service = pyvo.dal.TAPService(
                    'https://almascience.nrao.edu/tap', session=session)
            except TypeError:
                # Older pyvo without session support
                self._tap_service = pyvo.dal.TAPService('https://almascience.nrao.edu/tap')
        return self._tap_service
            
    def search_by_target(self, target_name: str, public: bool = True) -> pd.DataFrame:
        """
        Search ALMA archive by target name.
        Strategy: 
        1. Resolve name to RA/Dec via SIMBAD (cached)
        2. Race TAP and ALminer in parallel - return whichever succeeds first
        """
        print(f"[ALMA] Starting search for '{target_name}'")
        
        # Step 1: Resolve target name using cached SIMBAD lookup
        try:
            ra_deg, dec_deg = _resolve_simbad_cached(target_name)
            if ra_deg is None:
                print(f"[ALMA] SIMBAD could not resolve '{target_name}'")
                return pd.DataFrame()
            print(f"[ALMA] Resolved to RA={ra_deg:.4f}, Dec={dec_deg:.4f}")
        except Exception as e:
            print(f"[ALMA] SIMBAD resolution failed: {e}")
            return pd.DataFrame()
        
        # Step 2: Race TAP and ALminer in parallel
        return self._parallel_search(ra_deg, dec_deg, radius=0.05, target_name=target_name)
    
    def _parallel_search(self, ra: float, dec: float, radius: float = 0.05, target_name: str = "") -> pd.DataFrame:
        """
        Primary-plus-fallback cone search: try TAP, then ALminer, SEQUENTIALLY.
        Each source is bounded by a per-source timeout on a daemon thread, so
        there is no two-thread race and no stranded in-flight search (C14).
        Returns one canonical column schema via _standardize_columns.
        """
        import threading

        def _run_bounded(fn, timeout):
            box = {}

            def _target():
                try:
                    box["df"] = fn()
                except Exception as exc:  # ImportError, network, parse — all "no result"
                    box["err"] = exc

            t = threading.Thread(target=_target, daemon=True)
            t.start()
            t.join(timeout)
            return box.get("df")

        def tap_search():
            print("[ALMA] TAP search starting...")
            service = self._get_tap_service()
            query = (
                "SELECT * FROM ivoa.obscore "
                f"WHERE CONTAINS(POINT('ICRS', s_ra, s_dec), "
                f"CIRCLE('ICRS', {ra}, {dec}, {radius})) = 1"
            )
            res = service.search(query)
            return res.to_table().to_pandas()

        def alminer_search():
            import alminer
            print("[ALMA] ALminer search starting...")
            return alminer.conesearch(ra, dec, search_radius=radius, print_targets=False)

        for name, fn in (("TAP", tap_search), ("ALminer", alminer_search)):
            df = _run_bounded(fn, 60)
            if df is not None and not df.empty:
                print(f"[ALMA] {name} found {len(df)} results")
                return self._standardize_columns(df)
        print("[ALMA] Both TAP and ALminer returned no results (or timed out)")
        return pd.DataFrame()

    def search_by_position(self, ra: float, dec: float, radius: float = 0.016, public: bool = True) -> pd.DataFrame:
        """
        Search ALMA archive by position (cone search)
        radius is in degrees (default ~1 arcmin)
        """
        if not ALMINER_AVAILABLE:
            print("[ALminer] ERROR: alminer not available")
            return pd.DataFrame()

        try:
            print(f"[ALminer] Starting conesearch: RA={ra}, Dec={dec}, radius={radius}°")
            print("[ALminer] Connecting to ALMA archive (this may take 30-60 seconds)...")
            
            # Use threading with timeout to prevent infinite hang
            import threading
            result_holder = [None]
            error_holder = [None]
            
            def do_search():
                try:
                    result_holder[0] = alminer.conesearch(ra, dec, search_radius=radius, public=public, print_targets=False)
                except Exception as e:
                    error_holder[0] = e
            
            search_thread = threading.Thread(target=do_search, daemon=True)
            search_thread.start()
            search_thread.join(timeout=120)  # 2 minute timeout
            
            if search_thread.is_alive():
                print("[ALminer] ERROR: Search timed out after 2 minutes!")
                return pd.DataFrame()
            
            if error_holder[0]:
                raise error_holder[0]
            
            df = result_holder[0]
            if df is None:
                print("[ALminer] Search returned None")
                return pd.DataFrame()
                
            print(f"[ALminer] Search complete! Found {len(df)} results")
            return self._standardize_columns(df)
        except Exception as e:
            print(f"[ALminer] Position search error: {e}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()

    def search_by_keywords(self, keywords: Dict[str, Any], public: bool = True) -> pd.DataFrame:
        """
        Search by ALMA keywords (PI name, proposal ID, etc.)
        Example keywords: {'pi_name': 'Smith', 'proposal_id': '2017.1.000'}
        """
        if not ALMINER_AVAILABLE:
            return pd.DataFrame()
            
        try:
            # Construct dictionary for alminer.key_search
            # It accepts arguments like scientist='Name', project_code='ID'
            # We map generic keywords to what alminer expects
            # Commonly used: project_code, source_name_alma, scientific_category, pi_name
            
            # Pass the keywords directly if they match alminer arguments
            df = alminer.key_search(public=public, print_targets=False, **keywords)
            return self._standardize_columns(df)
        except Exception as e:
            print(f"ALminer keyword search error: {e}")
            return pd.DataFrame()
    
    def search_by_sql(self, query: str) -> pd.DataFrame:
        """
        Execute a custom TAP query using ALminer
        """
        if not ALMINER_AVAILABLE:
            return pd.DataFrame()
            
        try:
            df = alminer.run_tap_query(query, print_targets=False)
            return self._standardize_columns(df)
        except Exception as e:
            print(f"ALminer SQL/TAP search error: {e}")
            return pd.DataFrame()

    def search_by_frequency(self, min_freq_ghz: float, max_freq_ghz: float, public: bool = True) -> pd.DataFrame:
        """
        Search by frequency range
        Using key_search frequency parameters if supported, or falling back.
        ALminer doesn't have a direct frequency range search function easily wrapped like target(),
        but we can use key_search with frequency parameters or TAP.
        
        Using TAP is more robust for range queries.
        """
        if not ALMINER_AVAILABLE:
            return pd.DataFrame()
            
        try:
            service = self._get_tap_service()
            
            query = f'''
            SELECT *
            FROM ivoa.obscore
            WHERE frequency >= {min_freq_ghz}
              AND frequency <= {max_freq_ghz}
            '''
            res = service.search(query)
            df = res.to_table().to_pandas()
            
            if not df.empty:
                print(f"[ALMA] Frequency search found {len(df)} results")
                return self._standardize_columns(df)
            else:
                print("[ALMA] Frequency search returned no results")
                return pd.DataFrame()
                
        except ImportError:
            print("[ALMA] pyvo not available for frequency search")
            return pd.DataFrame()
        except Exception as e:
            print(f"[ALMA] Frequency search error: {e}")
            return pd.DataFrame()

    def get_run_summary(self, df: pd.DataFrame) -> str:
        """Get a text summary of the results"""
        if not ALMINER_AVAILABLE or df.empty:
            return "No data available to summarize."
        
        # Provide a simple custom summary or leverage alminer summary calls if they return text
        # alminer.summary() usually prints to stdout.
        # We can construct manual summary
        summary = f"Found {len(df)} observations.\n"
        if 'project_code' in df.columns:
            projects = df['project_code'].nunique()
            summary += f"Unique Projects: {projects}\n"
        if 'target_name' in df.columns:
            targets = df['target_name'].nunique()
            summary += f"Unique Targets: {targets}\n"
        
        return summary

    def plot_sky_distribution(self, df: pd.DataFrame, filename: str = "alma_sky_plot.png") -> bytes:
        """
        Generate sky distribution plot
        Returns image as bytes (Fix 5 - prevents MediaFileStorageError)
        """
        if not ALMINER_AVAILABLE or not MATPLOTLIB_AVAILABLE or df.empty:
            return b""

        try:
            import tempfile
            # Save to temp file, read bytes, cleanup
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                tmp_path = tmp.name
            
            alminer.plot_sky(df, savefig=tmp_path)
            
            with open(tmp_path, 'rb') as f:
                image_bytes = f.read()
            
            # Cleanup temp file
            try:
                os.remove(tmp_path)
            except OSError:
                pass
            
            return image_bytes
        except Exception as e:
            print(f"Error plotting sky distribution: {e}")
            return b""

    def plot_freq_coverage(self, df: pd.DataFrame, filename: str = "alma_freq_plot.png") -> bytes:
        """
        Generate frequency coverage plot
        Returns image as bytes (Fix 5 - prevents MediaFileStorageError)
        """
        if not ALMINER_AVAILABLE or not MATPLOTLIB_AVAILABLE or df.empty:
            return b""

        try:
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                tmp_path = tmp.name
            
            alminer.plot_bands(df, savefig=tmp_path)
            
            with open(tmp_path, 'rb') as f:
                image_bytes = f.read()
            
            try:
                os.remove(tmp_path)
            except OSError:
                pass
            
            return image_bytes
        except Exception as e:
            print(f"Error plotting frequency coverage: {e}")
            return b""

    def plot_overview(self, df: pd.DataFrame, filename: str = "alma_overview_plot.png") -> bytes:
        """
        Generate overview plot (Integration time vs Sensitivity usually)
        Returns image as bytes (Fix 5 - prevents MediaFileStorageError)
        """
        if not ALMINER_AVAILABLE or not MATPLOTLIB_AVAILABLE or df.empty:
            return b""

        try:
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                tmp_path = tmp.name
            
            alminer.plot_overview(df, savefig=tmp_path)
            
            with open(tmp_path, 'rb') as f:
                image_bytes = f.read()
            
            try:
                os.remove(tmp_path)
            except OSError:
                pass
            
            return image_bytes
        except Exception as e:
            print(f"Error plotting overview: {e}")
            return b""

    def download_data(self, df: pd.DataFrame, dry_run: bool = False) -> str:
        """
        Download data for the observations in the dataframe
        """
        if not ALMINER_AVAILABLE or df.empty:
            return "ALminer not available or empty dataframe."

        try:
            # alminer.download_data(df, fitsonly=..., dryrun=...)
            # We'll default to just FITS to save space/time, and dry_run for safety
            
            alminer.download_data(df, download_dir=self.download_dir, dryrun=dry_run, fitsonly=True)
            
            if dry_run:
                return f"Dry run complete. Would download to {self.download_dir}"
            else:
                return f"Download initiated to {self.download_dir}"
        except Exception as e:
            return f"Download failed: {e}"

    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize ALminer output columns to match Quasar's expected format
        """
        if df.empty:
            return df

        # Rename columns to match UI expectations
        rename_map = {
            'ra': 's_ra',
            'dec': 's_dec',
            'integration_time': 't_exptime',
            'band_number': 'Band',
            's_resolution': 'resolution',
            'sensitivity_10kms': 'sensitivity',
            'bandwidth': 'bandwidth',
            'min_freq_ghz': 'freq_min',
            'max_freq_ghz': 'freq_max'
        }
        
        # Apply renaming for columns that exist
        df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
        
        # Ensure standard columns exist
        df['telescope'] = 'ALMA'
        df['instrument_name'] = 'ALMA'
        
        # Add frequency aliases for CLI display compatibility
        if 'freq_min' in df.columns and 'freq_min_ghz' not in df.columns:
            df['freq_min_ghz'] = df['freq_min']
        if 'freq_max' in df.columns and 'freq_max_ghz' not in df.columns:
            df['freq_max_ghz'] = df['freq_max']
        
        # Construct useful Archive URL
        if 'member_ous_uid' in df.columns:
            df['access_url'] = df['member_ous_uid'].apply(
                lambda x: f"https://almascience.nrao.edu/aq/?member_ous_id={x}" if pd.notna(x) else ""
            )
        elif 'obs_publisher_did' in df.columns:
             pass

        # C14: guarantee a stable canonical column set on BOTH source paths.
        for _col in _CANONICAL_ALMA_COLUMNS:
            if _col not in df.columns:
                df[_col] = pd.NA

        return df

    def get_line_coverage(self, df: pd.DataFrame, line_freq: float, z: float = 0.0, line_name: str = "Line") -> pd.DataFrame:
        """
        Check which observations cover a specific line frequency (GHz)
        """
        if not ALMINER_AVAILABLE or df.empty:
            return pd.DataFrame()
            
        try:
            # alminer.line_coverage returns a filtered dataframe
            result = alminer.line_coverage(df, line_freq=line_freq, z=z, line_name=line_name, print_targets=False)
            return self._standardize_columns(result)
        except Exception as e:
            print(f"Line coverage check failed: {e}")
            return pd.DataFrame()

    def get_co_lines(self, df: pd.DataFrame, z: float = 0.0) -> pd.DataFrame:
        """
        Check for CO, 13CO, and C18O lines in the observations
        """
        if not ALMINER_AVAILABLE or df.empty:
            return pd.DataFrame()
            
        try:
            # alminer.CO_lines returns a DataFrame of matching observations
            result = alminer.CO_lines(df, z=z, print_targets=False)
            return self._standardize_columns(result)
        except Exception as e:
            print(f"CO lines check failed: {e}")
            return pd.DataFrame()

    def search_by_catalog(self, catalog_data: Dict[str, List[Any]]) -> pd.DataFrame:
        """
        Search by a catalog of objects.
        catalog_data expected format: {"Name": [...], "RAJ2000": [...], "DEJ2000": [...]}
        """
        if not ALMINER_AVAILABLE:
            return pd.DataFrame()

        try:
            # Convert dictionary to DataFrame for alminer
            cat_df = pd.DataFrame(catalog_data)
            result = alminer.catalog(cat_df, print_targets=False)
            return self._standardize_columns(result)
        except Exception as e:
            print(f"Catalog search failed: {e}")
            return pd.DataFrame()
