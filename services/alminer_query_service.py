"""
ALminer Query Service for Quasar
Provides advanced query capabilities using alminer's native functions:
- keysearch: Query by keywords (proposal_abstract, scientific_category, etc.)
- catalog: Query by coordinate catalog  
- conesearch: Query by RA/Dec position
- target: Query by source name (existing)
"""

import pandas as pd
from typing import Optional, Dict, Any, List, Union
import io
from contextlib import redirect_stdout

# Try to import alminer
try:
    import alminer
    ALMINER_AVAILABLE = True
except ImportError:
    ALMINER_AVAILABLE = False

# Valid ALMA science categories
SCIENCE_CATEGORIES = [
    "Active galaxies",
    "Cosmology", 
    "Disks and planet formation",
    "Galaxy evolution",
    "ISM and star formation",
    "Local Universe",
    "Solar system",
    "Stars and stellar evolution",
    "Sun"
]

# Common science keywords (subset - ALMA has many more)
COMMON_SCIENCE_KEYWORDS = [
    "Disks around low-mass stars",
    "Disks around high-mass stars",
    "Debris disks",
    "Exo-planets",
    "Solar system - Trans-Neptunian Objects",
    "Solar system - Comets",
    "Outflows, jets, feedback",
    "High-z Universe",
    "Gravitational lenses",
    "Galaxy chemistry",
    "Galaxy structure & evolution",
    "Starburst galaxies",
    "Active Galactic Nuclei",
    "Luminous and Ultra-Luminous Infra-Red Galaxies",
    "High-mass star formation",
    "Low-mass star formation",
    "Astrochemistry",
    "Inter-Stellar Medium",
    "Photon-Dominated Regions",
    "Pre-stellar cores",
    "Circumstellar shells",
    "Asymptotic Giant Branch stars",
    "Post-AGB stars",
    "Evolved stars - Shaping/physical structure",
    "Main sequence stars",
    "Magnetic fields"
]

# Valid keysearch keywords
VALID_KEYWORDS = [
    "target_name",
    "proposal_id", 
    "proposal_abstract",
    "scientific_category",
    "science_keyword",
    "pi_name",
    "pol_states",
    "em_resolution",
    "member_ous_uid"
]


class ALminerQueryService:
    """
    Advanced query service using alminer's native query functions
    """
    
    def __init__(self, tap_service: str = "ESO"):
        """
        Initialize query service
        
        Args:
            tap_service: ESO, NRAO, or NAOJ
        """
        self.tap_service = tap_service
        self.science_categories = SCIENCE_CATEGORIES
        self.science_keywords = COMMON_SCIENCE_KEYWORDS
        self.valid_keywords = VALID_KEYWORDS
    
    def _capture_output(self, func, *args, **kwargs) -> tuple:
        """Execute function and capture both result and printed output"""
        captured = io.StringIO()
        with redirect_stdout(captured):
            result = func(*args, **kwargs)
        return result, captured.getvalue()
    
    # =========================================================================
    # KEYSEARCH - Query by keywords
    # =========================================================================
    
    def keysearch(
        self,
        search_dict: Dict[str, List[str]],
        public: Optional[bool] = True,
        published: Optional[bool] = None,
        print_targets: bool = False
    ) -> pd.DataFrame:
        """
        Query ALMA archive by keywords
        
        Args:
            search_dict: Dictionary of keywords and values
                Examples:
                - {"proposal_abstract": ["star formation outflow"]}
                - {"scientific_category": ["Galaxy evolution"]}
                - {"science_keyword": ["High-mass star formation"]}
                - {"target_name": ["Sgr A*", "M87"]}
                - {"proposal_id": ["2023.1.00001.S"]}
                - {"pi_name": ["Smith"]}
            public: True for public only, False for proprietary, None for both
            published: True for published only, False for unpublished, None for both
            print_targets: Print target names
            
        Returns:
            DataFrame with query results
            
        Notes:
            - Multiple keywords are combined with AND
            - Multiple values for same keyword use OR
            - Quoted phrases are searched as exact phrases
        """
        if not ALMINER_AVAILABLE:
            return pd.DataFrame()
        
        try:
            result, output = self._capture_output(
                alminer.keysearch,
                search_dict,
                tap_service=self.tap_service,
                public=public,
                published=published,
                print_targets=print_targets,
                print_query=False
            )
            return result if result is not None else pd.DataFrame()
            
        except Exception as e:
            print(f"keysearch failed: {e}")
            return pd.DataFrame()
    
    def search_by_abstract(
        self,
        keywords: List[str],
        public: Optional[bool] = True
    ) -> pd.DataFrame:
        """
        Search proposals by abstract keywords
        
        Args:
            keywords: List of keywords (combined with OR) or single phrase
            public: Public data filter
            
        Returns:
            DataFrame with results
        """
        return self.keysearch({"proposal_abstract": keywords}, public=public)
    
    def search_by_category(
        self,
        category: str,
        public: Optional[bool] = True
    ) -> pd.DataFrame:
        """
        Search by ALMA scientific category
        
        Args:
            category: One of the valid categories (see SCIENCE_CATEGORIES)
            public: Public data filter
            
        Returns:
            DataFrame with results
        """
        if category not in SCIENCE_CATEGORIES:
            print(f"Warning: '{category}' not in known categories: {SCIENCE_CATEGORIES}")
        return self.keysearch({"scientific_category": [category]}, public=public)
    
    def search_by_science_keyword(
        self,
        keyword: str,
        public: Optional[bool] = True
    ) -> pd.DataFrame:
        """
        Search by ALMA science keyword
        
        Args:
            keyword: Science keyword (e.g., "High-mass star formation")
            public: Public data filter
            
        Returns:
            DataFrame with results
        """
        return self.keysearch({"science_keyword": [f'"{keyword}"']}, public=public)
    
    def search_by_pi(
        self,
        pi_name: str,
        public: Optional[bool] = True
    ) -> pd.DataFrame:
        """
        Search by Principal Investigator name
        
        Args:
            pi_name: PI name (partial match)
            public: Public data filter
            
        Returns:
            DataFrame with results
        """
        return self.keysearch({"pi_name": [pi_name]}, public=public)
    
    def search_by_proposal_id(
        self,
        proposal_ids: List[str],
        public: Optional[bool] = True
    ) -> pd.DataFrame:
        """
        Search by ALMA proposal ID(s)
        
        Args:
            proposal_ids: List of proposal IDs (e.g., ["2023.1.00001.S"])
            public: Public data filter
            
        Returns:
            DataFrame with results
        """
        return self.keysearch({"proposal_id": proposal_ids}, public=public)
    
    def search_polarization(
        self,
        pol_states: List[str] = ["XY", "YX"],
        public: Optional[bool] = True
    ) -> pd.DataFrame:
        """
        Search for full polarization data
        
        Args:
            pol_states: Polarization states to search for
            public: Public data filter
            
        Returns:
            DataFrame with results
        """
        return self.keysearch({"pol_states": pol_states}, public=public)
    
    # =========================================================================
    # CONESEARCH - Query by position
    # =========================================================================
    
    def conesearch(
        self,
        ra: float,
        dec: float,
        search_radius: float = 1.0,
        point: bool = False,
        public: Optional[bool] = True,
        published: Optional[bool] = None
    ) -> pd.DataFrame:
        """
        Query ALMA archive by sky position
        
        Args:
            ra: Right ascension in degrees (ICRS)
            dec: Declination in degrees (ICRS)
            search_radius: Search radius in arcminutes
            point: If True, search if position is in any observation
                   If False, search for overlap with cone
            public: Public data filter
            published: Published data filter
            
        Returns:
            DataFrame with results
        """
        if not ALMINER_AVAILABLE:
            return pd.DataFrame()
        
        try:
            result, output = self._capture_output(
                alminer.conesearch,
                ra=ra,
                dec=dec,
                search_radius=search_radius,
                point=point,
                tap_service=self.tap_service,
                public=public,
                published=published,
                print_targets=False,
                print_query=False
            )
            return result if result is not None else pd.DataFrame()
            
        except Exception as e:
            print(f"conesearch failed: {e}")
            return pd.DataFrame()
    
    # =========================================================================
    # CATALOG - Query by coordinate catalog
    # =========================================================================
    
    def catalog_search(
        self,
        catalog_df: pd.DataFrame,
        search_radius: float = 1.0,
        point: bool = False,
        public: Optional[bool] = True,
        published: Optional[bool] = None
    ) -> pd.DataFrame:
        """
        Query ALMA archive for a catalog of sources
        
        Args:
            catalog_df: DataFrame with columns:
                - Name: source name
                - RAJ2000: RA in degrees  
                - DEJ2000: Dec in degrees
            search_radius: Search radius in arcminutes
            point: If True, search if position is in any observation
            public: Public data filter
            published: Published data filter
            
        Returns:
            DataFrame with combined results
        """
        if not ALMINER_AVAILABLE:
            return pd.DataFrame()
        
        # Validate catalog format
        required_cols = ['Name', 'RAJ2000', 'DEJ2000']
        if not all(col in catalog_df.columns for col in required_cols):
            print(f"Catalog must have columns: {required_cols}")
            return pd.DataFrame()
        
        try:
            result, output = self._capture_output(
                alminer.catalog,
                catalog_df,
                search_radius=search_radius,
                point=point,
                tap_service=self.tap_service,
                public=public,
                published=published,
                print_targets=False,
                print_query=False
            )
            return result if result is not None else pd.DataFrame()
            
        except Exception as e:
            print(f"catalog search failed: {e}")
            return pd.DataFrame()
    
    # =========================================================================
    # TARGET - Query by name (enhanced)
    # =========================================================================
    
    def target_search(
        self,
        sources: Union[str, List[str]],
        search_radius: float = 1.0,
        point: bool = True,
        public: Optional[bool] = True,
        published: Optional[bool] = None
    ) -> pd.DataFrame:
        """
        Query ALMA archive by target name(s)
        
        Args:
            sources: Target name(s) - resolved via SIMBAD/NED/VizieR
            search_radius: Search radius in arcminutes
            point: If True, search if position is in any observation
            public: Public data filter
            published: Published data filter
            
        Returns:
            DataFrame with results
        """
        if not ALMINER_AVAILABLE:
            return pd.DataFrame()
        
        if isinstance(sources, str):
            sources = [sources]
        
        try:
            result, output = self._capture_output(
                alminer.target,
                sources,
                search_radius=search_radius,
                point=point,
                tap_service=self.tap_service,
                public=public,
                published=published,
                print_targets=False,
                print_query=False
            )
            return result if result is not None else pd.DataFrame()
            
        except Exception as e:
            print(f"target search failed: {e}")
            return pd.DataFrame()
    
    # =========================================================================
    # COMBINED SEARCHES
    # =========================================================================
    
    def advanced_search(
        self,
        target_name: Optional[str] = None,
        ra: Optional[float] = None,
        dec: Optional[float] = None,
        search_radius: float = 1.0,
        scientific_category: Optional[str] = None,
        science_keyword: Optional[str] = None,
        proposal_abstract: Optional[str] = None,
        pi_name: Optional[str] = None,
        proposal_id: Optional[str] = None,
        public: Optional[bool] = True,
        published: Optional[bool] = None
    ) -> pd.DataFrame:
        """
        Combined advanced search with multiple parameters
        
        Args:
            target_name: Source name (uses SIMBAD resolution)
            ra, dec: Position in degrees (alternative to target_name)
            search_radius: Radius in arcminutes
            scientific_category: ALMA science category filter
            science_keyword: ALMA science keyword filter
            proposal_abstract: Keywords in proposal abstract
            pi_name: PI name filter
            proposal_id: Proposal ID filter
            public: Public data filter
            published: Published data filter
            
        Returns:
            DataFrame with results (intersection of all filters)
        """
        # Start with position search
        if target_name:
            df = self.target_search(target_name, search_radius=search_radius, 
                                   public=public, published=published)
        elif ra is not None and dec is not None:
            df = self.conesearch(ra, dec, search_radius=search_radius,
                                public=public, published=published)
        else:
            # No position constraint - start with keyword search
            search_dict = {}
            if scientific_category:
                search_dict["scientific_category"] = [scientific_category]
            if science_keyword:
                search_dict["science_keyword"] = [f'"{science_keyword}"']
            if proposal_abstract:
                search_dict["proposal_abstract"] = [proposal_abstract]
            if pi_name:
                search_dict["pi_name"] = [pi_name]
            if proposal_id:
                search_dict["proposal_id"] = [proposal_id]
            
            if search_dict:
                df = self.keysearch(search_dict, public=public, published=published)
            else:
                return pd.DataFrame()
        
        if df.empty:
            return df
        
        # Apply additional filters if we started with position search
        if target_name or (ra is not None and dec is not None):
            if scientific_category and 'scientific_category' in df.columns:
                df = df[df['scientific_category'].str.contains(scientific_category, case=False, na=False)]
            if proposal_id and 'proposal_id' in df.columns:
                df = df[df['proposal_id'].str.contains(proposal_id, na=False)]
            if pi_name and 'obs_creator_name' in df.columns:
                df = df[df['obs_creator_name'].str.contains(pi_name, case=False, na=False)]
        
        return df


# Singleton instance
_alminer_query_service = None

def get_alminer_query_service(tap_service: str = "ESO") -> ALminerQueryService:
    """Get or create singleton query service"""
    global _alminer_query_service
    if _alminer_query_service is None:
        _alminer_query_service = ALminerQueryService(tap_service)
    return _alminer_query_service
