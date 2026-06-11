"""
NRAO TAP (Table Access Protocol) Service Integration - FIXED VERSION 2
Corrects the table name issue for VLA/VLBA queries
"""

import os
from typing import Optional, Dict, Any, List
import pandas as pd
import requests
from datetime import datetime
import pyvo as vo
from astropy import units as u
from astropy.coordinates import SkyCoord
from astroquery.alma import Alma
from astroquery.simbad import Simbad
import warnings
warnings.filterwarnings('ignore')


class _TimeoutHTTPSession(requests.Session):
    """requests.Session that enforces a default timeout on every request.

    pyvo issues requests without a timeout, which can hang indefinitely on
    slow or filtered networks. Injecting this session guarantees every TAP
    HTTP call fails fast instead of blocking the agent.
    """

    def __init__(self, timeout: float = 30.0):
        super().__init__()
        self._default_timeout = timeout

    def request(self, method, url, **kwargs):
        kwargs.setdefault("timeout", self._default_timeout)
        return super().request(method, url, **kwargs)


def _sanitize_adql(value: str) -> str:
    """Escape user input for safe ADQL interpolation."""
    # Remove/escape characters that could break ADQL string literals
    return value.replace("'", "''").replace("\\", "\\\\").replace(";", "").replace("--", "")


class NRAOTapClient:
    """Client for NRAO's TAP service covering VLA, VLBA, EVLA, and GBT.

    Connection is lazy: the first search triggers the TAP endpoint
    connection and ObsCore table discovery, keeping construction free of
    network calls (safe to instantiate at agent startup).
    """

    # NRAO TAP service endpoints
    TAP_URLS = {
        "nrao": "https://data-query.nrao.edu/tap",  # VLA/VLBA/GBT data
        "alma": "https://almascience.nrao.edu/tap"   # ALMA data
    }

    # Instrument name variants per facility in the NRAO ObsCore table
    FACILITY_INSTRUMENTS = {
        "VLA": ["VLA", "EVLA", "JVLA"],
        "VLBA": ["VLBA"],
        "GBT": ["GBT"],
    }
    DEFAULT_INSTRUMENTS = ["VLA", "VLBA", "EVLA", "JVLA"]

    def __init__(self, service_url: Optional[str] = None, timeout: int = 60, lazy: bool = True):
        """
        Initialize TAP client. With lazy=True (default) no network calls
        are made until the first search.
        """
        self.nrao_url = service_url or self.TAP_URLS["nrao"]
        self.alma_url = self.TAP_URLS["alma"]
        self.timeout = timeout
        self.obscore_table: Optional[str] = None
        self.nrao_tap = None
        self._connected = False
        self._alma = None
        if not lazy:
            self._ensure_connected()

    @property
    def alma(self):
        """Lazily constructed astroquery ALMA interface."""
        if self._alma is None:
            self._alma = Alma()
        return self._alma

    @classmethod
    def instruments_for_facility(cls, facility: Optional[str]) -> List[str]:
        """Map a facility name (VLA/VLBA/GBT) to ObsCore instrument_name values."""
        if not facility:
            return list(cls.DEFAULT_INSTRUMENTS)
        return list(cls.FACILITY_INSTRUMENTS.get(facility.strip().upper(), cls.DEFAULT_INSTRUMENTS))

    def _ensure_connected(self) -> bool:
        """Connect to the NRAO TAP service and discover the ObsCore table name."""
        if self._connected and self.nrao_tap is not None:
            return True

        try:
            # Fast reachability probe so unreachable networks fail in seconds,
            # not minutes (pyvo's own requests carry no timeout).
            probe = requests.get(f"{self.nrao_url}/capabilities", timeout=10)
            probe.raise_for_status()

            session = _TimeoutHTTPSession(timeout=self.timeout)
            try:
                self.nrao_tap = vo.dal.TAPService(self.nrao_url, session=session)
            except TypeError:
                # Older pyvo without session support
                self.nrao_tap = vo.dal.TAPService(self.nrao_url)
            print(f"[NRAO TAP] Connected: {self.nrao_url}")

            # Try different table names to find the correct one
            for table_name in ["obscore", "ivoa.obscore", "ObsCore", "tap_schema.obscore"]:
                try:
                    test_query = f"SELECT TOP 1 obs_publisher_did FROM {table_name}"
                    self.nrao_tap.search(test_query)
                    self.obscore_table = table_name
                    print(f"[NRAO TAP] Using table: {table_name}")
                    break
                except Exception:
                    continue

            if not self.obscore_table:
                print("[NRAO TAP] Could not determine ObsCore table name, using 'obscore' as default")
                self.obscore_table = "obscore"

            self._connected = True
        except Exception as e:
            print(f"[NRAO TAP] Connection failed: {e}")
            self.nrao_tap = None
            self.obscore_table = self.obscore_table or "obscore"
            self._connected = False

        return self._connected

    def search_by_source_name(self, source_name: str, max_results: int = 100) -> pd.DataFrame:
        """
        Search for a source across all NRAO facilities
        """
        print(f"\n🔍 Searching NRAO archives for: {source_name}")
        print("="*60)
        
        all_results = []
        
        # 1. Search VLA/VLBA using TAP
        vla_vlba_df = self.search_vla_vlba(source_name, max_results)
        if not vla_vlba_df.empty:
            all_results.append(vla_vlba_df)
            
        # 2. Search ALMA
        alma_df = self.search_alma(source_name)
        if not alma_df.empty:
            all_results.append(alma_df)
            
        # Combine results
        if all_results:
            combined = pd.concat(all_results, ignore_index=True, sort=False)
            print(f"\n✅ Total observations found: {len(combined)}")
            return combined
        else:
            print(f"\n❌ No observations found for {source_name}")
            return pd.DataFrame()

    def _postprocess_obscore_df(self, df: pd.DataFrame,
                                instruments: Optional[List[str]] = None) -> pd.DataFrame:
        """Standardize an NRAO ObsCore result DataFrame (freq, dates, size, URLs)."""
        if df.empty:
            return df

        # Convert wavelength to frequency if em_min/max present
        if 'em_min' in df.columns and 'em_max' in df.columns:
            c = 299792458.0  # speed of light in m/s
            df['freq_max_ghz'] = (c / df['em_min']) / 1e9  # min wavelength = max frequency
            df['freq_min_ghz'] = (c / df['em_max']) / 1e9  # max wavelength = min frequency

        # Convert MJD to datetime
        if 't_min' in df.columns:
            try:
                df['obs_date'] = pd.to_datetime(df['t_min'] - 40587, unit='D', origin='1970-01-01')
            except Exception:
                df['obs_date'] = df['t_min']

        # Add size in GB
        if 'access_estsize' in df.columns:
            df['size_gb'] = df['access_estsize'] / 1e9

        # Add archive URLs
        if 'obs_publisher_did' in df.columns:
            df['archive_url'] = df['obs_publisher_did'].apply(
                lambda x: f"https://data.nrao.edu/portal/#/search/{x}" if pd.notna(x) else ""
            )

        # Filter to requested instruments if instrument_name exists
        if instruments and 'instrument_name' in df.columns:
            df = df[df['instrument_name'].isin(instruments)]

        return df

    def search_by_position(self, ra: float, dec: float, radius: float = 0.5,
                           instruments: Optional[List[str]] = None,
                           max_results: int = 100) -> pd.DataFrame:
        """
        Cone search of the NRAO archive (VLA/VLBA/GBT) by sky position.

        Args:
            ra: Right ascension in degrees (ICRS)
            dec: Declination in degrees (ICRS)
            radius: Search radius in degrees
            instruments: ObsCore instrument_name values to keep
                         (default: VLA/VLBA/EVLA/JVLA)
            max_results: Maximum rows to return
        """
        if not self._ensure_connected() or not self.nrao_tap:
            print("[NRAO TAP] Service not available")
            return pd.DataFrame()

        instruments = instruments or list(self.DEFAULT_INSTRUMENTS)
        inst_list = ", ".join(f"'{_sanitize_adql(i)}'" for i in instruments)

        query = f"""
        SELECT TOP {int(max_results)}
            obs_publisher_did,
            target_name,
            s_ra, s_dec,
            t_min, t_max,
            t_exptime,
            em_min, em_max,
            instrument_name,
            facility_name,
            access_estsize
        FROM {self.obscore_table}
        WHERE 1=CONTAINS(POINT('ICRS', s_ra, s_dec),
                       CIRCLE('ICRS', {float(ra)}, {float(dec)}, {float(radius)}))
        AND instrument_name IN ({inst_list})
        ORDER BY t_min DESC
        """

        try:
            try:
                results = self.nrao_tap.search(query)
            except Exception:
                # Fallback without frequency columns
                query = f"""
                SELECT TOP {int(max_results)}
                    obs_publisher_did,
                    target_name,
                    s_ra, s_dec,
                    t_min, t_max,
                    t_exptime,
                    instrument_name,
                    facility_name,
                    access_estsize
                FROM {self.obscore_table}
                WHERE 1=CONTAINS(POINT('ICRS', s_ra, s_dec),
                               CIRCLE('ICRS', {float(ra)}, {float(dec)}, {float(radius)}))
                AND instrument_name IN ({inst_list})
                ORDER BY t_min DESC
                """
                results = self.nrao_tap.search(query)

            if results and len(results) > 0:
                df = results.to_table().to_pandas()
                df = self._postprocess_obscore_df(df, instruments=None)
                print(f"[NRAO TAP] Position search found {len(df)} observations")
                return df

            print("[NRAO TAP] No observations found at position")
            return pd.DataFrame()
        except Exception as e:
            print(f"[NRAO TAP] Position search error: {e}")
            return pd.DataFrame()

    def search_by_frequency_range(self, min_freq_ghz: float, max_freq_ghz: float,
                                  instruments: Optional[List[str]] = None,
                                  max_results: int = 100) -> pd.DataFrame:
        """
        Search the NRAO archive for observations whose spectral coverage
        overlaps [min_freq_ghz, max_freq_ghz].

        ObsCore stores wavelength bounds (em_min/em_max in meters), so the
        overlap condition in wavelength space is:
            em_min <= c/min_freq  AND  em_max >= c/max_freq
        """
        if not self._ensure_connected() or not self.nrao_tap:
            print("[NRAO TAP] Service not available")
            return pd.DataFrame()

        if min_freq_ghz <= 0 or max_freq_ghz <= 0 or max_freq_ghz < min_freq_ghz:
            print(f"[NRAO TAP] Invalid frequency range: {min_freq_ghz}-{max_freq_ghz} GHz")
            return pd.DataFrame()

        c = 299792458.0
        lam_max = c / (min_freq_ghz * 1e9)  # longest wavelength of the requested range
        lam_min = c / (max_freq_ghz * 1e9)  # shortest wavelength of the requested range

        instruments = instruments or list(self.DEFAULT_INSTRUMENTS)
        inst_list = ", ".join(f"'{_sanitize_adql(i)}'" for i in instruments)

        query = f"""
        SELECT TOP {int(max_results)}
            obs_publisher_did,
            target_name,
            s_ra, s_dec,
            t_min, t_max,
            t_exptime,
            em_min, em_max,
            instrument_name,
            facility_name,
            access_estsize
        FROM {self.obscore_table}
        WHERE em_min <= {lam_max}
        AND em_max >= {lam_min}
        AND instrument_name IN ({inst_list})
        ORDER BY t_min DESC
        """

        try:
            results = self.nrao_tap.search(query)
            if results and len(results) > 0:
                df = results.to_table().to_pandas()
                df = self._postprocess_obscore_df(df, instruments=None)
                print(f"[NRAO TAP] Frequency search found {len(df)} observations")
                return df
            print("[NRAO TAP] No observations found in frequency range")
            return pd.DataFrame()
        except Exception as e:
            print(f"[NRAO TAP] Frequency search error: {e}")
            return pd.DataFrame()

    def search_vla_vlba(self, source_name: str, max_results: int = 100,
                        instruments: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Search VLA/VLBA/GBT observations using a TAP query.

        Args:
            source_name: Target name (resolved to coordinates when possible)
            max_results: Maximum rows to return
            instruments: ObsCore instrument_name values to keep
                         (default: VLA/VLBA/EVLA/JVLA)
        """
        if not self._ensure_connected() or not self.nrao_tap:
            print("[NRAO TAP] Service not available")
            return pd.DataFrame()

        instruments = instruments or list(self.DEFAULT_INSTRUMENTS)
        inst_filter = ", ".join(f"'{_sanitize_adql(i)}'" for i in instruments)

        try:
            # First try to resolve to coordinates
            try:
                coord = SkyCoord.from_name(source_name)
                ra, dec = coord.ra.degree, coord.dec.degree
                print(f"📍 Resolved {source_name} to RA={ra:.4f}, Dec={dec:.4f}")
                
                # Use the discovered table name
                query = f"""
                SELECT TOP {max_results}
                    obs_publisher_did,
                    target_name,
                    s_ra, s_dec,
                    t_min, t_max,
                    t_exptime,
                    em_min, em_max,
                    instrument_name,
                    facility_name,
                    pol_states,
                    access_estsize
                FROM {self.obscore_table}
                WHERE 1=CONTAINS(POINT('ICRS', s_ra, s_dec),
                               CIRCLE('ICRS', {ra}, {dec}, 0.1))
                AND instrument_name IN ({inst_filter})
                ORDER BY t_min DESC
                """
                
                # Try with frequency columns if em_min/max doesn't work
                try:
                    results = self.nrao_tap.search(query)
                except:
                    # Fallback query without frequency columns
                    query = f"""
                    SELECT TOP {max_results}
                        obs_publisher_did,
                        target_name,
                        s_ra, s_dec,
                        t_min, t_max,
                        t_exptime,
                        instrument_name,
                        facility_name,
                        access_estsize
                    FROM {self.obscore_table}
                    WHERE 1=CONTAINS(POINT('ICRS', s_ra, s_dec),
                                   CIRCLE('ICRS', {ra}, {dec}, 0.1))
                    ORDER BY t_min DESC
                    """
                    results = self.nrao_tap.search(query)
                    
            except Exception as coord_error:
                # Fallback to name search
                print(f"🔍 Coordinate resolution failed, searching by name pattern: {source_name}")
                
                # Handle different name formats
                safe_name = _sanitize_adql(source_name)
                name_patterns = [
                    safe_name,
                    safe_name.replace(' ', ''),
                    safe_name.replace(' ', '_'),
                    safe_name.replace('_', ' '),
                    safe_name.upper(),
                    safe_name.replace(' ', '%')
                ]
                
                conditions = " OR ".join([f"target_name LIKE '%{p}%'" for p in name_patterns])
                
                query = f"""
                SELECT TOP {max_results}
                    obs_publisher_did,
                    target_name,
                    s_ra, s_dec,
                    t_min, t_max,
                    t_exptime,
                    instrument_name,
                    facility_name,
                    access_estsize
                FROM {self.obscore_table}
                WHERE ({conditions})
                ORDER BY t_min DESC
                """
                
                try:
                    results = self.nrao_tap.search(query)
                except Exception as e:
                    # Try simplest possible query
                    query = f"""
                    SELECT TOP {max_results}
                        *
                    FROM {self.obscore_table}
                    WHERE target_name LIKE '%{_sanitize_adql(source_name)}%'
                    """
                    results = self.nrao_tap.search(query)
            
            # Process results
            if results and len(results) > 0:
                df = results.to_table().to_pandas()

                if not df.empty:
                    df = self._postprocess_obscore_df(df, instruments=instruments)

                    if not df.empty:
                        print(f"[NRAO TAP] Found {len(df)} observations ({'/'.join(instruments)})")
                        if 'instrument_name' in df.columns:
                            for inst in df['instrument_name'].unique():
                                count = len(df[df['instrument_name'] == inst])
                                print(f"   - {inst}: {count} observations")
                    else:
                        print("[NRAO TAP] No observations found after instrument filtering")

                    return df

            print("[NRAO TAP] No observations found")
            return pd.DataFrame()

        except Exception as e:
            print(f"[NRAO TAP] Search error: {e}")
            # Return empty dataframe on error
            return pd.DataFrame()

    def search_alma(self, source_name: str, radius=5*u.arcmin) -> pd.DataFrame:
        """
        Search ALMA observations using astroquery.alma
        """
        try:
            # Try direct object query first
            print(f"🔍 Searching ALMA for {source_name}...")
            
            results = None
            try:
                results = self.alma.query_object(source_name, public=True, science=True)
            except:
                # If that fails, try coordinate search
                try:
                    coord = SkyCoord.from_name(source_name)
                    results = self.alma.query_region(coord, radius=radius, public=True, science=True)
                except:
                    pass
            
            if results and len(results) > 0:
                df = results.to_pandas()
                
                # Standardize column names to match VLA/VLBA format
                column_mapping = {
                    'Project code': 'project_code',
                    'Source name': 'target_name',
                    'RA': 's_ra',
                    'Dec': 's_dec',
                    'Band': 'band',
                    'Integration': 't_exptime',
                    'Release date': 'release_date',
                    'Frequency support': 'frequency_support',
                    'Member ous id': 'member_ous_uid'
                }
                
                df = df.rename(columns=column_mapping)
                
                # Add facility info
                df['instrument_name'] = 'ALMA'
                df['facility_name'] = 'ALMA'
                
                # Use Project code as obs_publisher_did if not present
                if 'obs_publisher_did' not in df.columns and 'project_code' in df.columns:
                    df['obs_publisher_did'] = df['project_code']
                
                # Add archive URLs
                if 'member_ous_uid' in df.columns:
                    df['archive_url'] = df['member_ous_uid'].apply(
                        lambda x: f"https://almascience.nrao.edu/aq/?member_ous_id={x}" if pd.notna(x) else ""
                    )
                
                print(f"✅ Found {len(df)} ALMA observations")
                
                # Show band summary if available
                if 'band' in df.columns:
                    bands = df['band'].value_counts()
                    for band, count in bands.items():
                        print(f"   - Band {band}: {count} observations")
                        
                return df
            else:
                print("❌ No ALMA observations found")
                return pd.DataFrame()
                
        except Exception as e:
            print(f"❌ Error searching ALMA: {e}")
            return pd.DataFrame()

    def get_archive_url(self, obs_id: str, facility: str = "VLA") -> str:
        """
        Generate direct archive URL for an observation
        """
        if facility.upper() == "ALMA":
            return f"https://almascience.nrao.edu/aq/?member_ous_id={obs_id}"
        else:
            return f"https://data.nrao.edu/portal/#/search/{obs_id}"

    def get_observation_details(self, obs_id: str) -> Dict[str, Any]:
        """
        Get detailed information about a specific observation.
        """
        safe_id = _sanitize_adql(obs_id)
        if not self.nrao_tap or not self.obscore_table:
            return {"error": "TAP service not connected"}

        try:
            query = f"""
            SELECT *
            FROM {self.obscore_table}
            WHERE obs_publisher_did = '{safe_id}'
               OR obs_publisher_did LIKE '%{safe_id}%'
            """
            results = self.nrao_tap.search(query)
            if results and len(results) > 0:
                df = results.to_table().to_pandas()
                if not df.empty:
                    return df.iloc[0].to_dict()
            return {"message": f"No observation found with ID: {obs_id}"}
        except Exception as e:
            return {"error": str(e)}

    def test_connection(self) -> Dict[str, Any]:
        """
        Test connections to all services
        """
        self._ensure_connected()

        results = {
            "status": "disconnected",
            "nrao_tap": "disconnected",
            "alma": "disconnected",
            "obscore_table": self.obscore_table
        }
        
        # Test NRAO TAP
        if self.nrao_tap and self.obscore_table:
            try:
                test_query = f"SELECT TOP 1 obs_publisher_did FROM {self.obscore_table}"
                self.nrao_tap.search(test_query)
                results["nrao_tap"] = "connected"
            except Exception:
                pass
        
        # Test ALMA
        try:
            self.alma.help()
            results["alma"] = "connected"
        except Exception:
            pass

        # Overall status
        if results["nrao_tap"] == "connected" or results["alma"] == "connected":
            results["status"] = "connected"
            
        return results

    def format_results_summary(self, df: pd.DataFrame) -> str:
        """
        Format a nice summary of search results
        """
        if df.empty:
            return "No observations found"
        
        summary = []
        summary.append(f"\n{'='*70}")
        summary.append(f"SEARCH RESULTS SUMMARY")
        summary.append(f"{'='*70}")
        summary.append(f"Total observations: {len(df)}")
        
        if 'instrument_name' in df.columns:
            summary.append("\nBy Instrument:")
            for inst, count in df['instrument_name'].value_counts().items():
                summary.append(f"  • {inst}: {count} observations")
        
        if 'freq_min_ghz' in df.columns and 'freq_max_ghz' in df.columns:
            freq_cols = df[['freq_min_ghz', 'freq_max_ghz']].dropna()
            if not freq_cols.empty:
                freq_min = freq_cols['freq_min_ghz'].min()
                freq_max = freq_cols['freq_max_ghz'].max()
                summary.append(f"\nFrequency range: {freq_min:.2f} - {freq_max:.2f} GHz")
        
        if 'obs_date' in df.columns:
            try:
                dates = pd.to_datetime(df['obs_date'], errors='coerce').dropna()
                if not dates.empty:
                    summary.append(f"Date range: {dates.min().date()} to {dates.max().date()}")
            except:
                pass
        
        if 'size_gb' in df.columns:
            total_size = df['size_gb'].sum()
            summary.append(f"Total data size: {total_size:.2f} GB")
        
        summary.append(f"{'='*70}\n")
        
        return "\n".join(summary)


# Test function
def test_nrao_access():
    """
    Test the fixed NRAO TAP client - Version 2
    """
    print("\n" + "="*70)
    print("TESTING FIXED NRAO TAP CLIENT - VERSION 2")
    print("="*70)
    
    # Initialize client
    client = NRAOTapClient()
    
    # Test connections
    print("\n1. Testing connections...")
    status = client.test_connection()
    for service, state in status.items():
        if service == "obscore_table":
            print(f"   📋 ObsCore table: {state}")
        else:
            symbol = "✅" if state == "connected" else "❌"
            print(f"   {symbol} {service}: {state}")
    
    # Search for 3C 273
    print("\n2. Searching for '3C 273'...")
    df = client.search_by_source_name("3C 273", max_results=20)
    
    if not df.empty:
        # Print summary
        print(client.format_results_summary(df))
        
        # Show first few observations
        print("Sample observations:")
        cols_to_show = ['target_name', 'instrument_name', 'obs_date', 'freq_min_ghz', 'freq_max_ghz']
        available_cols = [c for c in cols_to_show if c in df.columns]
        if available_cols:
            print(df[available_cols].head())
        
        # Show archive URLs
        if 'archive_url' in df.columns:
            print("\nDirect archive links (first 3):")
            for url in df['archive_url'].head(3).unique():
                if pd.notna(url) and url:
                    print(f"  📎 {url}")
    
    return df


if __name__ == "__main__":
    # Run the test
    results = test_nrao_access()
    
    print("\n" + "="*70)
    print("NEXT STEPS:")
    print("="*70)
    print("""
1. This fixed version should now work with VLA/VLBA data
2. Copy this to replace your tap.py:
   cp integrations/tap_fixed.py integrations/tap.py

3. Your app will now correctly fetch both VLA/VLBA AND ALMA data!
    """)
