#!/usr/bin/env python3
"""
Alternative: Use ALMA TAP which actually works
or direct HTTP requests to data.nrao.edu
"""

import requests
import pandas as pd
from typing import Dict, List, Optional
import json

class DirectNRAOAccess:
    """
    Alternative method using direct HTTP requests
    Since TAP is broken and astroquery needs installation
    """
    
    def __init__(self):
        self.base_url = "https://data.nrao.edu"
        self.session = requests.Session()
        
    def search_archive_web(self, source_name: str) -> Dict:
        """
        Search using the web interface backend
        This mimics what the web interface does
        """
        search_url = f"{self.base_url}/portal/search"
        
        # Parameters that the web interface uses
        params = {
            'source': source_name,
            'telescope': 'vla',  # or 'vlba'
            'format': 'json'  # Request JSON if available
        }
        
        try:
            response = self.session.get(search_url, params=params, timeout=10)
            
            if response.status_code == 200:
                # The response might be HTML, we'd need to parse it
                # This is a fallback method
                return {
                    'status': 'success',
                    'message': f'Search submitted for {source_name}',
                    'url': response.url,
                    'note': 'Visit the URL to see results'
                }
            
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    def search_vlass_cutouts(self, ra: float, dec: float) -> str:
        """
        For VLASS data specifically, we can get cutouts directly
        """
        base_url = "https://www.cadc-ccda.hia-iha.nrc-cnrc.gc.ca/vlass"
        
        params = {
            'ra': ra,
            'dec': dec,
            'radius': 0.05,  # 3 arcmin
            'epoch': '1',
            'type': 'quicklook'
        }
        
        response = requests.get(base_url, params=params)
        return response.url


class WorkingALMATAP:
    """
    ALMA TAP actually works! Use this for ALMA data
    """
    
    def __init__(self):
        self.tap_url = "https://almascience.nrao.edu/tap"
        
    def search_alma_direct(self, source_name: str) -> pd.DataFrame:
        """
        Search ALMA using direct TAP query (no PyVO needed)
        """
        query = f"""
        SELECT TOP 20 
            target_name, 
            s_ra, 
            s_dec,
            t_min,
            frequency,
            velocity_resolution,
            proposal_id,
            schedblock_name,
            member_ous_uid
        FROM ivoa.obscore
        WHERE target_name LIKE '%{source_name}%'
        ORDER BY t_min DESC
        """
        
        # Format for TAP
        params = {
            'REQUEST': 'doQuery',
            'LANG': 'ADQL',
            'FORMAT': 'json',
            'QUERY': query
        }
        
        try:
            response = requests.get(
                f"{self.tap_url}/sync",
                params=params,
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                
                # Parse the JSON response
                if 'data' in data:
                    rows = data['data']
                    columns = [col['name'] for col in data['metadata']]
                    
                    df = pd.DataFrame(rows, columns=columns)
                    return df
                    
        except Exception as e:
            print(f"ALMA TAP error: {e}")
            
        return pd.DataFrame()


def test_alternatives():
    """Test alternative methods that don't require astroquery"""
    
    print("=" * 60)
    print("TESTING ALTERNATIVE METHODS")
    print("=" * 60)
    
    print("\n1. Testing ALMA TAP (this actually works!)...")
    alma = WorkingALMATAP()
    df = alma.search_alma_direct("3C 273")
    
    if not df.empty:
        print(f"✅ Found {len(df)} ALMA observations!")
        print("\nALMA data preview:")
        print(df[['target_name', 's_ra', 's_dec', 'proposal_id']].head())
    else:
        print("No ALMA observations found")
    
    print("\n2. Testing direct web search...")
    direct = DirectNRAOAccess()
    result = direct.search_archive_web("3C 273")
    print(f"Status: {result['status']}")
    print(f"URL: {result.get('url', 'N/A')}")
    print(f"Note: {result.get('note', '')}")
    
    print("\n" + "=" * 60)
    print("SUMMARY OF WORKING METHODS:")
    print("=" * 60)
    print("""
    Without astroquery, you can:
    
    1. ✅ Use ALMA TAP (works great for ALMA data)
       - Endpoint: https://almascience.nrao.edu/tap
       - Table: ivoa.obscore
       - Full ObsCore implementation
    
    2. ⚠️ Use web interface at data.nrao.edu
       - Manual search and download
       - Can be automated with selenium
    
    3. ❌ VLA/VLBA TAP doesn't properly exist
       - No ivoa.obscore table
       - That's why your searches fail
    
    RECOMMENDATION: Install astroquery!
    pip install astroquery
    
    It's the official supported method for VLA/VLBA data.
    """)


if __name__ == "__main__":
    test_alternatives()
