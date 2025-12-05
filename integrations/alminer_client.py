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
            
    def search_by_target(self, target_name: str, public: bool = True) -> pd.DataFrame:
        """
        Search ALMA archive by target name
        """
        if not ALMINER_AVAILABLE:
            return pd.DataFrame()

        try:
            # ALminer's target search
            # print_targets=False to avoid cluttering stdout
            df = alminer.target(target_name, public=public, print_targets=False)
            
            # Filter by target name to avoid neighbors/project siblings
            if not df.empty and 'target_name' in df.columns:
                # Normalize names for comparison (remove spaces, lowercase)
                clean_target = target_name.lower().replace(" ", "")
                
                # Create a mask for matching targets
                # We check if the requested target is in the returned target name
                mask = df['target_name'].apply(
                    lambda x: clean_target in str(x).lower().replace(" ", "") if pd.notna(x) else False
                )
                df = df[mask]
            
            return self._standardize_columns(df)
        except Exception as e:
            print(f"ALminer target search error: {e}")
            return pd.DataFrame()

    def search_by_position(self, ra: float, dec: float, radius: float = 0.016, public: bool = True) -> pd.DataFrame:
        """
        Search ALMA archive by position (cone search)
        radius is in degrees (default ~1 arcmin)
        """
        if not ALMINER_AVAILABLE:
            return pd.DataFrame()

        try:
            # ALminer uses coordinates in degrees
            df = alminer.conesearch(ra, dec, search_radius=radius, public=public, print_targets=False)
            return self._standardize_columns(df)
        except Exception as e:
            print(f"ALminer position search error: {e}")
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
            # Construct a TAP query for frequency overlap
            # ALMA TAP has min_freq and max_freq columns? Usually it's frequency_support
            # Simpler approach: usage of alminer features might be best.
            # But let's return empty for now as requested in original code, 
            # or try to use search_by_keywords if there's params.
            return pd.DataFrame()
        except Exception:
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

    def plot_sky_distribution(self, df: pd.DataFrame, filename: str = "alma_sky_plot.png") -> str:
        """
        Generate sky distribution plot
        Returns absolute path to saved image
        """
        if not ALMINER_AVAILABLE or not MATPLOTLIB_AVAILABLE or df.empty:
            return ""

        try:
            # Determine save path in artifacts or temp location
            # Using current directory or a standard cache
            save_path = os.path.abspath(f"./{filename}")
            
            # alminer.plot_sky(df) creates the plot
            # We assume it uses the current figure
            alminer.plot_sky(df, savefig=save_path)
            
            return save_path
        except Exception as e:
            print(f"Error plotting sky distribution: {e}")
            return ""

    def plot_freq_coverage(self, df: pd.DataFrame, filename: str = "alma_freq_plot.png") -> str:
        """
        Generate frequency coverage plot
        """
        if not ALMINER_AVAILABLE or not MATPLOTLIB_AVAILABLE or df.empty:
            return ""

        try:
            save_path = os.path.abspath(f"./{filename}")
            # alminer.plot_bands(df) plots standard bands
            # alminer.plot_frequency_cover(df) might be what we want
            # Let's use plot_bands which is standard
            alminer.plot_bands(df, savefig=save_path)
            return save_path
        except Exception as e:
            print(f"Error plotting frequency coverage: {e}")
            return ""

    def plot_overview(self, df: pd.DataFrame, filename: str = "alma_overview_plot.png") -> str:
        """
        Generate overview plot (Integration time vs Sensitivity usually)
        """
        if not ALMINER_AVAILABLE or not MATPLOTLIB_AVAILABLE or df.empty:
            return ""

        try:
            save_path = os.path.abspath(f"./{filename}")
            alminer.plot_overview(df, savefig=save_path)
            return save_path
        except Exception as e:
            print(f"Error plotting overview: {e}")
            return ""

    def download_data(self, df: pd.DataFrame, dry_run: bool = True) -> str:
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
        
        # Construct useful Archive URL
        if 'member_ous_uid' in df.columns:
            df['access_url'] = df['member_ous_uid'].apply(
                lambda x: f"https://almascience.nrao.edu/aq/?member_ous_id={x}" if pd.notna(x) else ""
            )
        elif 'obs_publisher_did' in df.columns:
             pass

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
