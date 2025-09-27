"""
NRAO TAP (Table Access Protocol) Service Integration
Provides access to NRAO's Virtual Observatory for querying archive metadata
"""

import os
from typing import Optional, Dict, Any, List
import pandas as pd
from datetime import datetime
import pyvo as vo
from astropy import units as u
from astropy.coordinates import SkyCoord
import warnings
# Try to import the warning, but don't fail if it doesn't exist
try:
    from pyvo.utils.exceptions import W06
    warnings.filterwarnings('ignore', category=W06)
except (ImportError, AttributeError):
    # Older pyvo version or warning doesn't exist
    pass

class NRAOTapClient:
    """Client for interacting with NRAO's TAP service"""

    # NRAO TAP service endpoints
    TAP_URLS = {
        "production": "https://data-query.nrao.edu/tap",
        "test": "https://data-query.nrao.edu/tap"
    }

    # Common ADQL query templates
    QUERIES = {
        "cone_search": """
            SELECT TOP {max_results}
                s_ra, s_dec, target_name, obs_publisher_did,
                facility_name, instrument_name,
                t_min, t_max, t_exptime,
                freq_min, freq_max,
                pol_states,
                access_estsize,
                configuration,
                num_antennas
            FROM tap_schema.obscore
            WHERE 1=CONTAINS(POINT('ICRS', s_ra, s_dec),
                           CIRCLE('ICRS', {ra}, {dec}, {radius}))
            {filters}
            ORDER BY t_min DESC
        """,

        "target_search": """
            SELECT TOP {max_results}
                s_ra, s_dec, target_name, obs_publisher_did,
                facility_name, instrument_name,
                t_min, t_max, t_exptime,
                freq_min, freq_max,
                pol_states,
                access_estsize,
                configuration
            FROM tap_schema.obscore
            WHERE target_name LIKE '%{target}%'
            {filters}
            ORDER BY t_min DESC
        """,

        "frequency_search": """
            SELECT TOP {max_results}
                s_ra, s_dec, target_name, obs_publisher_did,
                facility_name, instrument_name,
                t_min, t_max, t_exptime,
                freq_min, freq_max,
                pol_states,
                access_estsize,
                configuration
            FROM tap_schema.obscore
            WHERE freq_min <= {max_freq} AND freq_max >= {min_freq}
            {filters}
            ORDER BY t_min DESC
        """,

        "observation_details": """
            SELECT *
            FROM tap_schema.obscore
            WHERE obs_publisher_did = '{obs_id}'
        """,

        "project_search": """
            SELECT TOP {max_results}
                s_ra, s_dec, target_name, obs_publisher_did,
                facility_name, instrument_name,
                t_min, t_max, t_exptime,
                freq_min, freq_max,
            FROM tap_schema.obscore
            ORDER BY t_min DESC
        """
    }

    def __init__(self, service_url: Optional[str] = None, timeout: int = 60):
        """
        Initialize TAP client

        Args:
            service_url: TAP service URL (defaults to production)
            timeout: Query timeout in seconds
        """
        self.service_url = service_url or self.TAP_URLS["production"]
        self.timeout = timeout
        self.tap_service = None
        self._connect()

    def _connect(self):
        """Connect to TAP service"""
        try:
            self.tap_service = vo.dal.TAPService(self.service_url)
        except Exception as e:
            raise ConnectionError(f"Failed to connect to TAP service: {str(e)}")

    def test_connection(self) -> Dict[str, Any]:
        """Test TAP service connection and get service info"""
        try:
            # Try a simple query to test connection
            query = "SELECT TOP 1 * FROM tap_schema.obscore"
            result = self.tap_service.search(query, maxrec=1)

            # Get service capabilities
            tables = self.get_available_tables()

            return {
                "status": "connected",
                "service_url": self.service_url,
                "version": "1.1",  # TAP version
                "table_count": len(tables),
                "tables": tables[:5]  # First 5 tables
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }

    def get_available_tables(self) -> List[str]:
        """Get list of available tables in the TAP service"""
        try:
            # Query the TAP schema for available tables
            query = """
                SELECT table_name
                FROM TAP_SCHEMA.tables
                WHERE schema_name = 'ivoa'
            """
            result = self.tap_service.search(query)
            return [row['table_name'] for row in result]
        except:
            # Fallback to known tables
            return ["obscore", "obs_radio", "calib"]

    def execute_query(self, query: str, max_records: int = 10000) -> pd.DataFrame:
        """
        Execute raw ADQL query

        Args:
            query: ADQL query string
            max_records: Maximum records to return

        Returns:
            DataFrame with query results
        """
        try:
            result = self.tap_service.search(query, maxrec=max_records)

            # Convert to pandas DataFrame
            if len(result) > 0:
                df = result.to_table().to_pandas()
                return self._process_dataframe(df)
            else:
                return pd.DataFrame()

        except Exception as e:
            raise RuntimeError(f"Query execution failed: {str(e)}")

    def cone_search(self, ra: float, dec: float, radius: float = 0.5,
                   facility: Optional[str] = None,
                   max_results: int = 100) -> pd.DataFrame:
        """
        Perform cone search around coordinates

        Args:
            ra: Right ascension in degrees
            dec: Declination in degrees
            radius: Search radius in degrees
            facility: Optional facility filter (VLA, VLBA, ALMA, GBT)
            max_results: Maximum results to return

        Returns:
            DataFrame with search results
        """
        filters = self._build_facility_filter(facility)

        query = self.QUERIES["cone_search"].format(
            ra=ra, dec=dec, radius=radius,
            max_results=max_results,
            filters=filters
        )

        return self.execute_query(query, max_results)

    def search_by_target(self, target_name: str,
                        facility: Optional[str] = None,
                        date_range: Optional[tuple] = None,
                        max_results: int = 100) -> pd.DataFrame:
        """
        Search by target name

        Args:
            target_name: Name of astronomical target
            facility: Optional facility filter
            date_range: Optional (start_date, end_date) tuple
            max_results: Maximum results

        Returns:
            DataFrame with search results
        """
        filters = self._build_facility_filter(facility)

        if date_range:
            filters += self._build_date_filter(date_range)

        query = self.QUERIES["target_search"].format(
            target=target_name,
            max_results=max_results,
            filters=filters
        )

        return self.execute_query(query, max_results)

    def search_by_frequency(self, min_freq_ghz: float, max_freq_ghz: float,
                           facility: Optional[str] = None,
                           max_results: int = 100) -> pd.DataFrame:
        """
        Search by frequency range

        Args:
            min_freq_ghz: Minimum frequency in GHz
            max_freq_ghz: Maximum frequency in GHz
            facility: Optional facility filter
            max_results: Maximum results

        Returns:
            DataFrame with search results
        """
        # Convert GHz to Hz for query
        min_freq_hz = min_freq_ghz * 1e9
        max_freq_hz = max_freq_ghz * 1e9

        filters = self._build_facility_filter(facility)

        query = self.QUERIES["frequency_search"].format(
            min_freq=min_freq_hz,
            max_freq=max_freq_hz,
            max_results=max_results,
            filters=filters
        )

        return self.execute_query(query, max_results)

    def get_observation_details(self, obs_id: str) -> Dict[str, Any]:
        """
        Get detailed information for specific observation

        Args:
            obs_id: Observation/execution block ID

        Returns:
            Dictionary with observation details
        """
        query = self.QUERIES["observation_details"].format(obs_id=obs_id)
        df = self.execute_query(query, max_results=1)

        if df.empty:
            raise ValueError(f"No observation found with ID: {obs_id}")

        # Convert first row to dictionary
        obs = df.iloc[0].to_dict()

        # Add computed fields
        obs['freq_ghz'] = {
            'min': obs.get('freq_min', 0) / 1e9,
            'max': obs.get('freq_max', 0) / 1e9,
            'center': (obs.get('freq_min', 0) + obs.get('freq_max', 0)) / 2e9
        }

        obs['time_range'] = {
            'start': self._mjd_to_datetime(obs.get('t_min')),
            'end': self._mjd_to_datetime(obs.get('t_max')),
            'duration_hours': obs.get('t_exptime', 0) / 3600
        }

        obs['size_gb'] = obs.get('access_estsize', 0) / 1e9

        return obs

    def search_by_project(self, project_code: str,
                         max_results: int = 100) -> pd.DataFrame:
        """
        Search by project code

        Args:
            project_code: Project/proposal code (e.g., 'VLA/23A-001')
            max_results: Maximum results

        Returns:
            DataFrame with search results
        """
        query = self.QUERIES["project_search"].format(
            project_code=project_code,
            max_results=max_results
        )

        return self.execute_query(query, max_results)

    def resolve_target_coordinates(self, target_name: str) -> tuple:
        """
        Resolve target name to coordinates using Simbad

        Args:
            target_name: Astronomical object name

        Returns:
            (ra, dec) tuple in degrees
        """
        try:
            coord = SkyCoord.from_name(target_name)
            return coord.ra.degree, coord.dec.degree
        except Exception as e:
            raise ValueError(f"Could not resolve target '{target_name}': {str(e)}")

    def _build_facility_filter(self, facility: Optional[str]) -> str:
        """Build facility filter clause"""
        if not facility:
            return ""

        facility_upper = facility.upper()
        if facility_upper in ["VLA", "VLBA", "ALMA", "GBT"]:
            return f" AND facility_name = '{facility_upper}'"
        else:
            return ""

    def _build_date_filter(self, date_range: tuple) -> str:
        """Build date range filter clause"""
        if not date_range or len(date_range) != 2:
            return ""

        start_mjd = self._datetime_to_mjd(date_range[0])
        end_mjd = self._datetime_to_mjd(date_range[1])

        return f" AND t_min >= {start_mjd} AND t_max <= {end_mjd}"

    def _process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process and enhance dataframe with computed columns"""
        if df.empty:
            return df

        # Add frequency in GHz
        if 'freq_min' in df.columns:
            df['freq_min_ghz'] = df['freq_min'] / 1e9
            df['freq_max_ghz'] = df['freq_max'] / 1e9

        # Convert MJD to datetime
        if 't_min' in df.columns:
            df['obs_date'] = pd.to_datetime(df['t_min'] - 40587, unit='D', origin='1970-01-01')

        # Add size in GB
        if 'access_estsize' in df.columns:
            df['size_gb'] = df['access_estsize'] / 1e9

        # Add duration in hours
        if 't_exptime' in df.columns:
            df['duration_hours'] = df['t_exptime'] / 3600

        return df

    @staticmethod
    def _mjd_to_datetime(mjd: float) -> datetime:
        """Convert Modified Julian Date to datetime"""
        if pd.isna(mjd):
            return None
        # MJD epoch is 1858-11-17
        return pd.to_datetime(mjd - 40587, unit='D', origin='1970-01-01')

    @staticmethod
    def _datetime_to_mjd(dt: datetime) -> float:
        """Convert datetime to Modified Julian Date"""
        # Convert to pandas timestamp then to MJD
        ts = pd.Timestamp(dt)
        return (ts - pd.Timestamp('1970-01-01')).total_seconds() / 86400 + 40587

    def search_by_source_name(self, source_name: str, max_results: int = 100) -> pd.DataFrame:
        """
        Enhanced search for specific sources like '3C 273'
        Handles both coordinate-based and name-based searches

        Args:
            source_name: Name of the astronomical source (e.g., '3C 273', 'M31', 'Cygnus A')
            max_results: Maximum number of results to return

        Returns:
            DataFrame with observation results
        """
        # First try to resolve coordinates using SIMBAD
        try:
            ra, dec = self.resolve_target_coordinates(source_name)
            # Do a cone search with 0.1 degree radius
            results = self.cone_search(ra, dec, radius=0.1, max_results=max_results)

            # If we got results, add the searched source name
            if not results.empty:
                results['searched_source'] = source_name
                return results
        except:
            pass

        # Fallback to name-based search
        # Handle various name formats (3C 273, 3C273, 3C_273)
        name_variants = [
            source_name,
            source_name.replace(' ', ''),
            source_name.replace(' ', '_'),
            source_name.replace('_', ' ')
        ]

        # Build query conditions for all name variants
        queries = []
        for variant in name_variants:
            queries.append(f"target_name LIKE '%{variant}%'")

        # Construct ADQL query
        query = f"""
            SELECT TOP {max_results}
                s_ra, s_dec, target_name, obs_publisher_did,
                facility_name, instrument_name,
                t_min, t_max, t_exptime,
                freq_min, freq_max,
                pol_states,
                access_estsize,
                access_url,
                proposal_id
            FROM tap_schema.obscore
            WHERE ({' OR '.join(queries)})
            ORDER BY t_min DESC
        """

        return self.execute_query(query, max_results)