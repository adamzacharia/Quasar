"""
ALminer Visualization Service for Quasar
Uses alminer's native plotting functions for better visualizations
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, List, Tuple
import tempfile
import os
import io

# Try to import alminer
try:
    import alminer
    ALMINER_AVAILABLE = True
except ImportError:
    ALMINER_AVAILABLE = False

# Try to import matplotlib
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

# Common spectral lines (rest frequency in GHz)
COMMON_LINES = {
    "CO(1-0)": 115.271,
    "CO(2-1)": 230.538,
    "CO(3-2)": 345.796,
    "CO(4-3)": 461.041,
    "13CO(1-0)": 110.201,
    "13CO(2-1)": 220.399,
    "HCN(1-0)": 88.632,
    "HCN(3-2)": 265.886,
    "HCO+(1-0)": 89.189,
    "CS(2-1)": 97.981,
    "[CI] 492": 492.161,
    "[CI] 809": 809.342,
    "H2O 22GHz": 22.235,
}


class ALminerVisualizationService:
    """
    Visualization service using alminer's native plotting functions
    
    Available plots:
    - plot_overview: Summary of frequencies, resolution, LAS, velocity resolution
    - plot_line_overview: Highlight observations at a specific frequency
    - plot_bands: Detailed frequency plots per band with CO marking
    - plot_observations: Exact frequency ranges per observation
    - plot_sky: Sky distribution of targets
    """
    
    def __init__(self):
        self.common_lines = COMMON_LINES
        self._ensure_reports_dir()
    
    def _ensure_reports_dir(self):
        """Ensure the reports directory exists"""
        if not os.path.exists("reports"):
            os.makedirs("reports", exist_ok=True)
    
    def _normalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize DataFrame column names from TAP format to alminer expected format.
        
        TAP returns:           alminer expects:
        s_ra, s_dec       ->   RAJ2000, DEJ2000
        freq_min          ->   min_freq_GHz  
        freq_max          ->   max_freq_GHz
        (calculated)      ->   central_freq_GHz
        Band              ->   band
        s_resolution      ->   ang_res_arcsec
        t_exptime         ->   int_time
        """
        if df.empty:
            return df
        
        df = df.copy()
        
        # Column mapping from TAP to alminer
        column_map = {
            's_ra': 'RAJ2000',
            's_dec': 'DEJ2000',
            'freq_min': 'min_freq_GHz',
            'freq_max': 'max_freq_GHz',
            'Band': 'band',
            's_resolution': 'ang_res_arcsec',
            't_exptime': 'int_time',
            'project_code': 'project_code',
            'target_name': 'target_name',
        }
        
        # Rename columns that exist
        rename_dict = {k: v for k, v in column_map.items() if k in df.columns and v not in df.columns}
        if rename_dict:
            df = df.rename(columns=rename_dict)
        
        # Calculate central frequency if we have min/max
        if 'min_freq_GHz' in df.columns and 'max_freq_GHz' in df.columns:
            if 'central_freq_GHz' not in df.columns:
                df['central_freq_GHz'] = (df['min_freq_GHz'] + df['max_freq_GHz']) / 2
        
        # If freq_min/freq_max exist but not min_freq_GHz, copy them
        if 'freq_min' in df.columns and 'min_freq_GHz' not in df.columns:
            df['min_freq_GHz'] = df['freq_min']
        if 'freq_max' in df.columns and 'max_freq_GHz' not in df.columns:
            df['max_freq_GHz'] = df['freq_max']
        
        # Calculate central freq from original columns if still missing
        if 'central_freq_GHz' not in df.columns:
            if 'freq_min' in df.columns and 'freq_max' in df.columns:
                df['central_freq_GHz'] = (df['freq_min'] + df['freq_max']) / 2
        
        # Ensure RAJ2000/DEJ2000 exist
        if 'RAJ2000' not in df.columns and 's_ra' in df.columns:
            df['RAJ2000'] = df['s_ra']
        if 'DEJ2000' not in df.columns and 's_dec' in df.columns:
            df['DEJ2000'] = df['s_dec']
        
        # Ensure band column exists
        if 'band' not in df.columns and 'Band' in df.columns:
            df['band'] = df['Band']
        
        return df
    
    def _plot_to_bytes(self, plot_func, *args, **kwargs) -> bytes:
        """
        Execute an alminer plot function and return as bytes for Streamlit display
        
        Args:
            plot_func: The alminer plotting function
            *args, **kwargs: Arguments to pass to the function
            
        Returns:
            PNG image bytes
        """
        if not ALMINER_AVAILABLE or not MATPLOTLIB_AVAILABLE:
            return b""
        
        try:
            # Use a temporary file
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                tmp_path = tmp.name
            
            # Execute the plot function with savefig
            kwargs['savefig'] = tmp_path.replace('.png', '')  # alminer adds .pdf, we want .png
            kwargs['showfig'] = False
            
            plot_func(*args, **kwargs)
            
            # alminer saves as PDF by default, but we can capture the current figure
            plt.savefig(tmp_path, format='png', dpi=150, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            plt.close('all')
            
            # Read the bytes
            with open(tmp_path, 'rb') as f:
                image_bytes = f.read()
            
            # Cleanup
            try:
                os.remove(tmp_path)
            except OSError:
                pass
            
            return image_bytes
            
        except Exception as e:
            print(f"Plot generation failed: {e}")
            plt.close('all')
            return b""
    
    # =========================================================================
    # 1. OVERVIEW PLOT
    # =========================================================================
    
    def plot_overview(self, df: pd.DataFrame) -> bytes:
        """
        Generate alminer overview plot showing:
        - Observed frequencies distribution
        - Angular resolution distribution
        - Largest Angular Scale (LAS) distribution
        - Frequency and velocity resolution
        
        Args:
            df: DataFrame from alminer query
            
        Returns:
            PNG image bytes
        """
        if not ALMINER_AVAILABLE or df.empty:
            return b""
        
        try:
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                tmp_path = tmp.name
            
            # Normalize columns for alminer
            normalized_df = self._normalize_columns(df)
            
            # Call alminer's plot_overview
            alminer.plot_overview(normalized_df, showfig=False)
            
            # Save current figure
            plt.savefig(tmp_path, format='png', dpi=150, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            plt.close('all')
            
            with open(tmp_path, 'rb') as f:
                image_bytes = f.read()
            
            os.remove(tmp_path)
            return image_bytes
            
        except Exception as e:
            print(f"plot_overview failed: {e}")
            plt.close('all')
            return b""
    
    # =========================================================================
    # 2. LINE OVERVIEW PLOT
    # =========================================================================
    
    def plot_line_overview(
        self, 
        df: pd.DataFrame, 
        line_freq: float,
        z: float = 0.0
    ) -> bytes:
        """
        Generate overview plot highlighting observations at a specific frequency
        
        Args:
            df: DataFrame from alminer query
            line_freq: Line frequency in GHz (rest frame)
            z: Redshift to apply
            
        Returns:
            PNG image bytes
        """
        if not ALMINER_AVAILABLE or df.empty:
            return b""
        
        try:
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                tmp_path = tmp.name
            
            normalized_df = self._normalize_columns(df)
            alminer.plot_line_overview(normalized_df, line_freq=line_freq, z=z, showfig=False)
            
            plt.savefig(tmp_path, format='png', dpi=150, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            plt.close('all')
            
            with open(tmp_path, 'rb') as f:
                image_bytes = f.read()
            
            os.remove(tmp_path)
            return image_bytes
            
        except Exception as e:
            print(f"plot_line_overview failed: {e}")
            plt.close('all')
            return b""
    
    # =========================================================================
    # 3. BANDS PLOT
    # =========================================================================
    
    def plot_bands(
        self, 
        df: pd.DataFrame,
        mark_CO: bool = True,
        z: float = 0.0,
        mark_freq: Optional[List[float]] = None
    ) -> bytes:
        """
        Generate detailed plot of observed frequencies in each ALMA band
        
        Args:
            df: DataFrame from alminer query
            mark_CO: Mark CO, 13CO, C18O lines
            z: Redshift for line marking
            mark_freq: Additional frequencies to mark
            
        Returns:
            PNG image bytes
        """
        if not ALMINER_AVAILABLE or df.empty:
            return b""
        
        try:
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                tmp_path = tmp.name
            
            normalized_df = self._normalize_columns(df)
            if mark_freq:
                alminer.plot_bands(normalized_df, mark_freq=mark_freq, z=z, showfig=False)
            else:
                alminer.plot_bands(normalized_df, mark_CO=mark_CO, z=z, showfig=False)
            
            plt.savefig(tmp_path, format='png', dpi=150, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            plt.close('all')
            
            with open(tmp_path, 'rb') as f:
                image_bytes = f.read()
            
            os.remove(tmp_path)
            return image_bytes
            
        except Exception as e:
            print(f"plot_bands failed: {e}")
            plt.close('all')
            return b""
    
    # =========================================================================
    # 4. OBSERVATIONS PLOT
    # =========================================================================
    
    def plot_observations(
        self, 
        df: pd.DataFrame,
        mark_CO: bool = True,
        z: float = 0.0,
        mark_freq: Optional[List[float]] = None
    ) -> bytes:
        """
        Generate plot showing exact frequency ranges for each observation
        
        Args:
            df: DataFrame from alminer query
            mark_CO: Mark CO lines
            z: Redshift for line marking
            mark_freq: Additional frequencies to mark
            
        Returns:
            PNG image bytes
        """
        if not ALMINER_AVAILABLE or df.empty:
            return b""
        
        try:
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                tmp_path = tmp.name
            
            normalized_df = self._normalize_columns(df)
            if mark_freq:
                alminer.plot_observations(normalized_df, mark_freq=mark_freq, z=z, showfig=False)
            else:
                alminer.plot_observations(normalized_df, mark_CO=mark_CO, z=z, showfig=False)
            
            plt.savefig(tmp_path, format='png', dpi=150, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            plt.close('all')
            
            with open(tmp_path, 'rb') as f:
                image_bytes = f.read()
            
            os.remove(tmp_path)
            return image_bytes
            
        except Exception as e:
            print(f"plot_observations failed: {e}")
            plt.close('all')
            return b""
    
    # =========================================================================
    # 5. SKY DISTRIBUTION PLOT
    # =========================================================================
    
    def plot_sky(self, df: pd.DataFrame) -> bytes:
        """
        Generate sky distribution plot of targets
        
        Args:
            df: DataFrame from alminer query
            
        Returns:
            PNG image bytes
        """
        if not ALMINER_AVAILABLE or df.empty:
            return b""
        
        try:
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                tmp_path = tmp.name
            
            normalized_df = self._normalize_columns(df)
            alminer.plot_sky(normalized_df, showfig=False)
            
            plt.savefig(tmp_path, format='png', dpi=150, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            plt.close('all')
            
            with open(tmp_path, 'rb') as f:
                image_bytes = f.read()
            
            os.remove(tmp_path)
            return image_bytes
            
        except Exception as e:
            print(f"plot_sky failed: {e}")
            plt.close('all')
            return b""
    
    # =========================================================================
    # LINE COVERAGE UTILITIES
    # =========================================================================
    
    def check_line_coverage(
        self,
        df: pd.DataFrame,
        line_freq: float,
        z: float = 0.0,
        line_name: str = "Line"
    ) -> pd.DataFrame:
        """
        Check which observations cover a specific spectral line
        
        Args:
            df: DataFrame from alminer query
            line_freq: Rest frequency in GHz
            z: Source redshift
            line_name: Name of the line for display
            
        Returns:
            Filtered DataFrame with observations covering the line
        """
        if not ALMINER_AVAILABLE or df.empty:
            return pd.DataFrame()
        
        try:
            normalized_df = self._normalize_columns(df)
            return alminer.line_coverage(normalized_df, line_freq=line_freq, z=z, 
                                        line_name=line_name, print_targets=False)
        except Exception as e:
            print(f"line_coverage check failed: {e}")
            return pd.DataFrame()
    
    def check_CO_coverage(self, df: pd.DataFrame, z: float = 0.0) -> pd.DataFrame:
        """
        Check for CO, 13CO, and C18O line coverage
        
        Args:
            df: DataFrame from alminer query
            z: Source redshift
            
        Returns:
            DataFrame with CO line coverage info
        """
        if not ALMINER_AVAILABLE or df.empty:
            return pd.DataFrame()
        
        try:
            normalized_df = self._normalize_columns(df)
            return alminer.CO_lines(normalized_df, z=z, print_targets=False)
        except Exception as e:
            print(f"CO_lines check failed: {e}")
            return pd.DataFrame()
    
    # =========================================================================
    # SUMMARY STATISTICS
    # =========================================================================
    
    def get_summary_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Get summary statistics for the observations
        
        Args:
            df: DataFrame from alminer query
            
        Returns:
            Dictionary with summary statistics
        """
        if df.empty:
            return {}
        
        stats = {
            "n_observations": len(df),
            "n_projects": df['project_code'].nunique() if 'project_code' in df.columns else 0,
            "n_targets": df['target_name'].nunique() if 'target_name' in df.columns else 0,
        }
        
        # Band distribution
        if 'band' in df.columns:
            stats["bands"] = df['band'].value_counts().to_dict()
        
        # Frequency range
        if 'min_freq_ghz' in df.columns and 'max_freq_ghz' in df.columns:
            stats["freq_range_ghz"] = (
                float(df['min_freq_ghz'].min()),
                float(df['max_freq_ghz'].max())
            )
        
        # Resolution range
        if 's_resolution' in df.columns:
            valid_res = df['s_resolution'].dropna()
            if not valid_res.empty:
                stats["resolution_range_arcsec"] = (
                    float(valid_res.min()),
                    float(valid_res.max())
                )
        
        return stats
    
    # =========================================================================
    # REPORT GENERATION
    # =========================================================================
    
    def save_table(self, df: pd.DataFrame, filename: str = "observations") -> str:
        """
        Export query results as a CSV table
        
        Args:
            df: DataFrame from alminer query
            filename: Output filename (without extension)
            
        Returns:
            Path to saved file or empty string on failure
        """
        if df.empty:
            return ""
        
        try:
            # Ensure tables directory exists
            os.makedirs("tables", exist_ok=True)
            
            if ALMINER_AVAILABLE:
                alminer.save_table(df, filename=filename)
                filepath = f"tables/{filename}.csv"
            else:
                # Fallback to pandas
                filepath = f"tables/{filename}.csv"
                df.to_csv(filepath, index=False)
            
            return filepath
            
        except Exception as e:
            print(f"save_table failed: {e}")
            return ""
    
    def save_source_reports(
        self, 
        df: pd.DataFrame, 
        mark_CO: bool = True,
        z: float = 0.0
    ) -> List[str]:
        """
        Generate and save overview plots for each target source
        
        Args:
            df: DataFrame from alminer query
            mark_CO: Mark CO lines on plots
            z: Redshift for line marking
            
        Returns:
            List of saved report filenames
        """
        if not ALMINER_AVAILABLE or df.empty:
            return []
        
        try:
            # Ensure reports directory exists
            os.makedirs("reports", exist_ok=True)
            
            alminer.save_source_reports(df, mark_CO=mark_CO, z=z)
            
            # Return list of generated files
            if 'target_name' in df.columns:
                targets = df['target_name'].unique()
                return [f"reports/{t}.pdf" for t in targets if os.path.exists(f"reports/{t}.pdf")]
            return []
            
        except Exception as e:
            print(f"save_source_reports failed: {e}")
            return []
    
    # =========================================================================
    # DOWNLOAD FUNCTIONALITY
    # =========================================================================
    
    def download_data(
        self,
        df: pd.DataFrame,
        location: str = "./data",
        fitsonly: bool = True,
        dryrun: bool = True,
        archive_mirror: str = "ESO",
        filename_must_include: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Download data from ALMA archive
        
        Args:
            df: DataFrame from alminer query
            location: Download directory
            fitsonly: Download only FITS products (not raw data)
            dryrun: If True, only estimate disk space needed
            archive_mirror: ESO, NRAO, or NAOJ
            filename_must_include: List of strings that must be in filename
            
        Returns:
            Dictionary with download info (disk space, file count, etc.)
        """
        if not ALMINER_AVAILABLE or df.empty:
            return {"error": "ALminer not available or empty DataFrame"}
        
        try:
            import io
            import sys
            from contextlib import redirect_stdout
            
            # Capture output from alminer
            captured = io.StringIO()
            
            with redirect_stdout(captured):
                alminer.download_data(
                    df,
                    location=location,
                    fitsonly=fitsonly,
                    dryrun=dryrun,
                    archive_mirror=archive_mirror,
                    filename_must_include=filename_must_include or [],
                    print_urls=False
                )
            
            output = captured.getvalue()
            
            # Parse the output
            result = {
                "dryrun": dryrun,
                "location": location,
                "fitsonly": fitsonly,
                "raw_output": output
            }
            
            # Extract disk space info
            for line in output.split('\n'):
                if 'Needed disk space' in line:
                    result["disk_space"] = line.split('=')[-1].strip()
                elif 'Number of files' in line:
                    result["file_count"] = line.split('=')[-1].strip()
                elif 'Member OUSs' in line and 'download' in line:
                    result["mous_count"] = line.split('=')[-1].strip()
            
            return result
            
        except Exception as e:
            print(f"download_data failed: {e}")
            return {"error": str(e)}
    
    def get_download_urls(
        self,
        df: pd.DataFrame,
        fitsonly: bool = True,
        filename_must_include: Optional[List[str]] = None
    ) -> List[str]:
        """
        Get list of URLs for downloading data
        
        Args:
            df: DataFrame from alminer query
            fitsonly: Only FITS products
            filename_must_include: Filter filenames
            
        Returns:
            List of download URLs
        """
        if not ALMINER_AVAILABLE or df.empty:
            return []
        
        try:
            import io
            import sys
            from contextlib import redirect_stdout
            
            captured = io.StringIO()
            
            with redirect_stdout(captured):
                alminer.download_data(
                    df,
                    fitsonly=fitsonly,
                    dryrun=True,
                    filename_must_include=filename_must_include or [],
                    print_urls=True
                )
            
            output = captured.getvalue()
            
            # Extract URLs from output
            urls = []
            for line in output.split('\n'):
                if line.startswith('http'):
                    urls.append(line.strip())
            
            return urls
            
        except Exception as e:
            print(f"get_download_urls failed: {e}")
            return []
    
    def get_summary(self, df: pd.DataFrame) -> str:
        """
        Get alminer summary as string
        
        Args:
            df: DataFrame from alminer query
            
        Returns:
            Summary string
        """
        if not ALMINER_AVAILABLE or df.empty:
            return "No data available"
        
        try:
            import io
            from contextlib import redirect_stdout
            
            captured = io.StringIO()
            with redirect_stdout(captured):
                alminer.summary(df)
            
            return captured.getvalue()
            
        except Exception as e:
            return f"Summary failed: {e}"


# Singleton instance
_alminer_viz_service = None

def get_alminer_visualization_service() -> ALminerVisualizationService:
    """Get or create singleton visualization service"""
    global _alminer_viz_service
    if _alminer_viz_service is None:
        _alminer_viz_service = ALminerVisualizationService()
    return _alminer_viz_service
