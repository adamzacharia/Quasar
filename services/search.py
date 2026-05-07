# services/search.py
"""
Search Service — Facade over ALminerClient for all ALMA archive searches.

CALLED BY: core/agent.py (tool execution: search_by_target, search_by_position, etc.)
CALLS:     integrations/alminer_client.py (ALminerClient)

DATA FLOW:
    agent._search_by_target(name) → SearchService.search_by_target(name)
    → ALminerClient.search_by_target(name) → alminer.target(name)
    → Returns pd.DataFrame of matching observations

Also provides: plot generation, data download, line coverage checks,
and catalog search — all delegated to ALminerClient.
"""

from typing import Optional, Dict, Any, List, Tuple
import pandas as pd
from datetime import datetime
from integrations.alminer_client import ALminerClient

class SearchService:
    """High-level search operations for NRAO data"""

    def __init__(self):
        self.alminer_client = ALminerClient()

    def cone_search(self, ra: float, dec: float, radius: float,
                   facility: Optional[str] = None,
                   max_results: int = 100) -> pd.DataFrame:
        """Perform cone search using ALminer"""
        try:
            return self.alminer_client.search_by_position(ra, dec, radius)
        except Exception as e:
            print(f"ALminer search failed: {e}")
            return pd.DataFrame()

    def search_by_target(self, target_name: str,
                        facility: Optional[str] = None,
                        date_range: Optional[str] = None,
                        max_results: int = 100) -> pd.DataFrame:
        """Search by target name using ALminer"""
        try:
            return self.alminer_client.search_by_target(target_name)
        except Exception as e:
            print(f"ALminer search failed: {e}")
            return pd.DataFrame()

    def search_by_frequency(self, min_freq_ghz: float, max_freq_ghz: float,
                           facility: Optional[str] = None,
                           max_results: int = 100) -> pd.DataFrame:
        """Search by frequency range"""
        return self.alminer_client.search_by_frequency(min_freq_ghz, max_freq_ghz)

    def get_observation_details(self, obs_id: str) -> Dict[str, Any]:
        """Get detailed observation information"""
        return {}

    def advanced_search(self, query: str) -> pd.DataFrame:
        """Execute advanced ADQL/TAP query"""
        try:
            return self.alminer_client.search_by_sql(query)
        except Exception as e:
            print(f"Advanced search failed: {e}")
            return pd.DataFrame()

    def search_alma_with_keywords(self, keywords: Dict[str, Any]) -> pd.DataFrame:
        """Search ALMA using specific keywords (project_code, pi_name, etc.)"""
        try:
            return self.alminer_client.search_by_keywords(keywords)
        except Exception as e:
            print(f"Keyword search failed: {e}")
            return pd.DataFrame()

    def plot_alma_results(self, df: pd.DataFrame, plot_type: str = "sky") -> bytes:
        """
        Generate plots for ALMA results
        plot_type: 'sky', 'frequency', 'overview'
        Returns image as bytes (Fix 5)
        """
        if plot_type == "sky":
            return self.alminer_client.plot_sky_distribution(df)
        elif plot_type == "frequency":
            return self.alminer_client.plot_freq_coverage(df)
        elif plot_type == "overview":
            return self.alminer_client.plot_overview(df)
        return b""

    def download_alma_data(self, df: pd.DataFrame, dry_run: bool = True) -> str:
        """Download data"""
        return self.alminer_client.download_data(df, dry_run=dry_run)

    def find_calibrators(self, ra: float, dec: float,
                        max_separation: float = 10.0) -> pd.DataFrame:
        """Find potential calibrator sources near target"""
        return pd.DataFrame()

    def search_source(self, source_name: str, max_results: int = 100) -> pd.DataFrame:
        """
        Smart search that handles any source name properly
        """
        try:
            return self.alminer_client.search_by_target(source_name)
        except Exception:
            return pd.DataFrame()

    def check_line_coverage(self, line_freq_ghz: float, z: float = 0.0, line_name: str = "Line") -> pd.DataFrame:
        """
        Check if specific frequency line is covered in loaded observations.
        Note: This usually requires a prior search to have results, 
        but alminer.line_coverage takes a DataFrame.
        This tool might need to be chainable or take a 'last_search_context'.
        For now, let's assume the agent passes the dataframe (not possible via JSON).
        Wait, standard tools take simple types. 
        REALITY CHECK: line_coverage filters EXISTING results. 
        So this tool only makes sense if we have a stateful DataFrame.
        The Agent has `self.last_run_result`. 
        We can't pass a DataFrame to an OpenAI tool call.
        
        Alternatives:
        1. Tool doesn't take DF, instead it uses 'latest_search_results' from agent state implicitly? No, simpler to just run a fresh search + filter?
        2. Or user runs this AFTER a search. The Agent code executes it.
        
        Let's Define the tool takes (line_freq, z, line_name). 
        The Agent implementation (`core/agent.py`) will inject `self.last_search_results` when calling this method.
        So this method signature should accept DF? 
        The TOOL definition (for LLM) will hide the DF argument.
        The AGENT (execution) will provide it.
        """
        # This generic signature expects DF to be passed by caller (Agent)
        # We'll handle the injection in agent.py
        pass # The logic is in alminer_client. We just expose the method on the service.
        
    def check_line_coverage_on_last(self, df: pd.DataFrame, line_freq_ghz: float, z: float = 0.0, line_name: str = "Line") -> pd.DataFrame:
        return self.alminer_client.get_line_coverage(df, line_freq_ghz, z, line_name)

    def check_co_lines_on_last(self, df: pd.DataFrame, z: float = 0.0) -> pd.DataFrame:
        return self.alminer_client.get_co_lines(df, z)
        
    def search_catalog(self, catalog_list: List[Dict[str, Any]]) -> pd.DataFrame:
        return self.alminer_client.search_by_catalog(catalog_list)
