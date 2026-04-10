"""
Data Processing Service for Quasar
Handles real data fetching and processing
"""

import pandas as pd
from typing import Dict, List, Any, Optional
from datetime import datetime
import numpy as np

class DataProcessor:
    """Process and enrich NRAO archive data"""
    
    def __init__(self, tap_client, ads_service=None):
        self.tap_client = tap_client
        self.ads_service = ads_service
    
    def fetch_source_data(self, source_name: str) -> Dict[str, Any]:
        """
        Fetch comprehensive data for a source
        Returns dict with observations, papers, and summary
        """
        result = {
            'source': source_name,
            'observations': None,
            'papers': [],
            'summary': None,
            'statistics': {}
        }
        
        # Fetch observations
        try:
            observations = self.tap_client.search_by_source_name(source_name, max_results=50)
            
            if not observations.empty:
                result['observations'] = observations
                
                # Calculate statistics
                result['statistics'] = {
                    'total_observations': len(observations),
                    'facilities': observations['facility_name'].unique().tolist() if 'facility_name' in observations.columns else [],
                    'date_range': {
                        'start': observations['obs_date'].min() if 'obs_date' in observations.columns else None,
                        'end': observations['obs_date'].max() if 'obs_date' in observations.columns else None
                    },
                    'frequency_range': {
                        'min': observations['freq_min_ghz'].min() if 'freq_min_ghz' in observations.columns else None,
                        'max': observations['freq_max_ghz'].max() if 'freq_max_ghz' in observations.columns else None
                    },
                    'total_size_gb': observations['size_gb'].sum() if 'size_gb' in observations.columns else 0,
                    'total_integration_hours': observations['duration_hours'].sum() if 'duration_hours' in observations.columns else 0
                }
        except Exception as e:
            print(f"Error fetching observations: {e}")
        
        # Fetch papers if ADS service available
        if self.ads_service:
            try:
                papers = self.ads_service.search_by_target(source_name, max_results=10)
                result['papers'] = papers
            except Exception as e:
                print(f"Error fetching papers: {e}")
        
        # Generate summary
        result['summary'] = self.generate_summary(result)
        
        return result
    
    def generate_summary(self, data: Dict[str, Any]) -> str:
        """Generate AI-style summary of the data"""
        source = data['source']
        stats = data['statistics']
        
        if data['observations'] is not None and len(data['observations']) > 0:
            # Format facilities list
            facilities_str = ', '.join(stats.get('facilities', [])) if stats.get('facilities') else 'various facilities'
            
            # Format date range
            date_start = stats['date_range'].get('start', 'Unknown')
            date_end = stats['date_range'].get('end', 'Unknown')
            if pd.notna(date_start) and pd.notna(date_end):
                date_range_str = f"{date_start} to {date_end}"
            else:
                date_range_str = "various dates"
            
            # Format frequency range
            freq_min = stats['frequency_range'].get('min')
            freq_max = stats['frequency_range'].get('max')
            if freq_min is not None and freq_max is not None:
                freq_range_str = f"{freq_min:.1f} - {freq_max:.1f} GHz"
            else:
                freq_range_str = "multiple frequency bands"
            
            summary = f"""
### {source} — Archive Summary

Found **{stats['total_observations']} observations** in the NRAO archives.

**Key Statistics:**
- **Facilities:** {facilities_str}
- **Date Range:** {date_range_str}
- **Frequency Coverage:** {freq_range_str}
- **Total Data Volume:** {stats['total_size_gb']:.1f} GB
- **Total Integration Time:** {stats['total_integration_hours']:.1f} hours

"""
            
            if len(data['papers']) > 0:
                summary += f"**{len(data['papers'])} related scientific papers** found in NASA ADS.\n"
            
            summary += "\n*You can ask me to generate analysis code or find more specific information about these observations.*"
            
        else:
            summary = f"""
### No Results Found

No observations found for **{source}** in the NRAO archives.

**Suggestions:**
- Check the spelling of the source name
- Try alternative designations (e.g., '3C 273' vs '3C273')
- Common sources to try:
  - Quasars: 3C 273, 3C 48, PKS 2155-304
  - Galaxies: M31, M87, NGC 1275, Cygnus A
  - Pulsars: Crab Pulsar, PSR J0437-4715
"""
        
        return summary.strip()
    
    def get_archive_urls(self, observations: pd.DataFrame) -> List[str]:
        """Extract direct archive URLs from observations"""
        urls = []
        
        if 'obs_publisher_did' in observations.columns:
            for obs_id in observations['obs_publisher_did'][:10]:  # Limit to 10 URLs
                if pd.notna(obs_id):
                    obs_id_str = str(obs_id)
                    # Build NRAO archive URL based on facility
                    if 'VLA' in obs_id_str.upper():
                        url = f"https://data.nrao.edu/portal/vla_project/{obs_id_str}"
                    elif 'VLBA' in obs_id_str.upper():
                        url = f"https://data.nrao.edu/portal/vlba_project/{obs_id_str}"
                    elif 'ALMA' in obs_id_str.upper():
                        url = f"https://almascience.nrao.edu/aq/?project_code={obs_id_str}"
                    else:
                        url = f"https://data.nrao.edu/portal/search?obs_id={obs_id_str}"
                    urls.append(url)
        
        return urls
    
    def format_observations_for_display(self, observations: pd.DataFrame) -> pd.DataFrame:
        """Format observations dataframe for UI display"""
        if observations.empty:
            return observations
        
        # Select and rename columns for display
        display_cols = {
            'obs_date': 'Date',
            'target_name': 'Target',
            'facility_name': 'Facility',
            'freq_min_ghz': 'Min Freq (GHz)',
            'freq_max_ghz': 'Max Freq (GHz)',
            'duration_hours': 'Duration (hrs)',
            'size_gb': 'Size (GB)',
            'configuration': 'Config',
            'obs_publisher_did': 'Observation ID'
        }
        
        # Filter to available columns
        available_cols = [col for col in display_cols.keys() if col in observations.columns]
        
        if not available_cols:
            return observations
        
        # Create display dataframe
        display_df = observations[available_cols].copy()
        
        # Rename columns
        display_df.columns = [display_cols[col] for col in available_cols]
        
        # Format numeric columns
        if 'Min Freq (GHz)' in display_df.columns:
            display_df['Min Freq (GHz)'] = display_df['Min Freq (GHz)'].round(2)
        if 'Max Freq (GHz)' in display_df.columns:
            display_df['Max Freq (GHz)'] = display_df['Max Freq (GHz)'].round(2)
        if 'Duration (hrs)' in display_df.columns:
            display_df['Duration (hrs)'] = display_df['Duration (hrs)'].round(2)
        if 'Size (GB)' in display_df.columns:
            display_df['Size (GB)'] = display_df['Size (GB)'].round(2)
        
        return display_df
