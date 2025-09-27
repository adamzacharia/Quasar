# services/search.py
"""
Search service for NRAO archives
"""

from typing import Optional, Dict, Any, List, Tuple
import pandas as pd
from datetime import datetime

class SearchService:
    """High-level search operations for NRAO data"""

    def __init__(self, tap_client):
        self.tap_client = tap_client

    def cone_search(self, ra: float, dec: float, radius: float,
                   facility: Optional[str] = None,
                   max_results: int = 100) -> pd.DataFrame:
        """Perform cone search"""
        return self.tap_client.cone_search(ra, dec, radius, facility, max_results)

    def search_by_target(self, target_name: str,
                        facility: Optional[str] = None,
                        date_range: Optional[str] = None,
                        max_results: int = 100) -> pd.DataFrame:
        """Search by target name"""
        # Parse date range if provided as string
        dates = None
        if date_range:
            parts = date_range.split(',')
            if len(parts) == 2:
                dates = (
                    datetime.strptime(parts[0].strip(), '%Y-%m-%d'),
                    datetime.strptime(parts[1].strip(), '%Y-%m-%d')
                )

        return self.tap_client.search_by_target(target_name, facility, dates, max_results)

    def search_by_frequency(self, min_freq_ghz: float, max_freq_ghz: float,
                           facility: Optional[str] = None,
                           max_results: int = 100) -> pd.DataFrame:
        """Search by frequency range"""
        return self.tap_client.search_by_frequency(
            min_freq_ghz, max_freq_ghz, facility, max_results
        )

    def get_observation_details(self, obs_id: str) -> Dict[str, Any]:
        """Get detailed observation information"""
        return self.tap_client.get_observation_details(obs_id)

    def advanced_search(self, query: str) -> pd.DataFrame:
        """Execute advanced ADQL query"""
        return self.tap_client.execute_query(query)

    def find_calibrators(self, ra: float, dec: float,
                        max_separation: float = 10.0) -> pd.DataFrame:
        """Find potential calibrator sources near target"""
        # Search for bright sources near target
        results = self.cone_search(ra, dec, max_separation, max_results=50)

        if results.empty:
            return results

        # Filter for potential calibrators (simplified logic)
        # In reality, this would check flux density, compactness, etc.
        calibrators = results[
            results['target_name'].str.contains(
                '3C|PKS|B1950|J2000|NVSS|FIRST',
                case=False,
                na=False
            )
        ]

        return calibrators

    def search_source(self, source_name: str, max_results: int = 100) -> pd.DataFrame:
        """
        Smart search that handles any source name properly
        """
        # Try the enhanced search
        results = self.tap_client.search_by_source_name(source_name, max_results)
        
        if results.empty:
            # Try alternative search methods
            # Remove common prefixes/suffixes
            clean_name = source_name.strip()
            for prefix in ['3C', 'NGC', 'IC', 'M', 'PKS', 'B1950', 'J2000']:
                if clean_name.startswith(prefix):
                    # Try with and without space
                    results = self.tap_client.search_by_target(clean_name, max_results=max_results)
                    if not results.empty:
                        break
        
        return results
