#!/usr/bin/env python3
"""
SOLUTION FOR YOUR QUASAR PROJECT
Replace broken TAP with working astroquery.nrao
"""

# First, install astroquery if not already installed:
# pip install astroquery

from astroquery.nrao import Nrao
from astropy.coordinates import SkyCoord
from astropy import units as u
import pandas as pd

def search_nrao_working(source_name: str):
    """
    The ACTUAL working method to search NRAO archives
    """
    print(f"\nSearching NRAO for: {source_name}")
    print("=" * 50)
    
    try:
        # Method 1: Search by source name directly
        results = Nrao.query_object(
            object_name=source_name,
            telescope='jansky_vla'  # or 'vlba'
        )
        
        if results and len(results) > 0:
            print(f"✅ Found {len(results)} observations by name!")
            df = results.to_pandas()
            
            # Show what we found
            if 'Source' in df.columns:
                print(f"\nSources found: {df['Source'].unique()}")
            if 'Configuration' in df.columns:
                print(f"Configurations: {df['Configuration'].unique()}")
            if 'Obs Band' in df.columns:
                print(f"Bands observed: {df['Obs Band'].unique()}")
            
            return df
            
    except Exception as e:
        print(f"Name search failed: {e}")
    
    try:
        # Method 2: Resolve to coordinates and search
        coord = SkyCoord.from_name(source_name)
        print(f"Resolved to: RA={coord.ra.degree:.4f}, Dec={coord.dec.degree:.4f}")
        
        results = Nrao.query_region(
            coordinates=coord,
            radius=0.1 * u.deg,
            telescope='jansky_vla'
        )
        
        if results and len(results) > 0:
            print(f"✅ Found {len(results)} observations by position!")
            return results.to_pandas()
            
    except Exception as e:
        print(f"Coordinate search failed: {e}")
    
    print("❌ No observations found")
    return pd.DataFrame()


# TEST IT
print("""
================================================================================
SOLUTION: Use astroquery.nrao instead of broken TAP
================================================================================

The research reveals that:
1. VLA/VLBA don't have a working TAP service with ivoa.obscore
2. Only ALMA has proper TAP at almascience.nrao.edu/tap
3. The official method is astroquery.nrao for VLA/VLBA

Let's test the working method:
""")

# Test with 3C 273
df = search_nrao_working("3C 273")

if not df.empty:
    print(f"\n📊 Data Preview:")
    print(df[['Source', 'Obs Date', 'Configuration', 'Obs Band', 'Exposure']].head())

print("""
================================================================================
HOW TO FIX YOUR QUASAR PROJECT:
================================================================================

1. INSTALL: pip install astroquery

2. REPLACE your tap.py search with:

from astroquery.nrao import Nrao

def search_by_source_name(source_name):
    results = Nrao.query_object(
        object_name=source_name,
        telescope='jansky_vla'
    )
    if results:
        return results.to_pandas()
    return pd.DataFrame()

3. For DOWNLOADS:
   - astroquery gives you metadata only
   - Go to https://data.nrao.edu
   - Request the data (you'll get wget commands by email)
   - Use those wget commands to download

4. For ALMA data (TAP works!):
   
import pyvo
service = pyvo.dal.TAPService("https://almascience.nrao.edu/tap")
query = "SELECT * FROM ivoa.obscore WHERE target_name LIKE '%3C 273%'"
results = service.search(query)

================================================================================
""")
