#!/usr/bin/env python3
"""
IMMEDIATE FIX - Force correct table and run search
This bypasses the auto-discovery and uses ivoa.obscore directly
"""

import pandas as pd
import pyvo as vo
from astropy.coordinates import SkyCoord

print("=" * 60)
print("QUICK FIX - Using ivoa.obscore directly")
print("=" * 60)

# Connect to TAP service
tap_url = "https://data-query.nrao.edu/tap"
service = vo.dal.TAPService(tap_url)

print(f"Connected to: {tap_url}")

# FORCE the correct table
TABLE = "ivoa.obscore"
print(f"Using table: {TABLE}")

# Test with simple query first
print("\n1. Testing table access...")
test_query = f"SELECT TOP 1 * FROM {TABLE}"

try:
    result = service.run_sync(test_query)
    print("✅ Table is accessible!")
except Exception as e:
    print(f"❌ Error accessing table: {e}")
    print("\nTrying alternative tables...")
    
    # Try alternatives
    alternatives = ["ivoa.ObsCore", "ObsCore", "obscore"]
    for alt_table in alternatives:
        try:
            test_query = f"SELECT TOP 1 * FROM {alt_table}"
            result = service.run_sync(test_query)
            print(f"✅ Found working table: {alt_table}")
            TABLE = alt_table
            break
        except:
            continue

# Now search for 3C 273
print(f"\n2. Searching for 3C 273 in {TABLE}...")

# Method 1: Try by name
try:
    query = f"""
    SELECT TOP 100 
        s_ra, s_dec, target_name, 
        obs_publisher_did,
        t_min, t_max,
        freq_min, freq_max,
        facility_name,
        access_url
    FROM {TABLE}
    WHERE target_name LIKE '%3C 273%'
       OR target_name LIKE '%3C273%'
    ORDER BY t_min DESC
    """
    
    result = service.run_sync(query)
    df = result.to_table().to_pandas()
    
    if not df.empty:
        print(f"✅ Found {len(df)} observations by name!")
        
        # Check for download URLs
        if 'access_url' in df.columns:
            urls = df['access_url'].dropna()
            if not urls.empty:
                print(f"✅ Found {len(urls)} download URLs!")
                print("\nExample URLs:")
                for url in urls.head(3):
                    print(f"  - {url}")
        
        # Show facilities
        if 'facility_name' in df.columns:
            print(f"\nFacilities: {df['facility_name'].unique()}")
            
    else:
        print("No results from name search, trying coordinates...")
        
        # Method 2: Try by coordinates
        coord = SkyCoord.from_name("3C 273")
        ra, dec = coord.ra.degree, coord.dec.degree
        print(f"Resolved to RA={ra:.4f}, Dec={dec:.4f}")
        
        query = f"""
        SELECT TOP 100
            s_ra, s_dec, target_name,
            obs_publisher_did,
            t_min, t_max,
            freq_min, freq_max,
            facility_name,
            access_url
        FROM {TABLE}
        WHERE s_ra BETWEEN {ra - 0.1} AND {ra + 0.1}
          AND s_dec BETWEEN {dec - 0.1} AND {dec + 0.1}
        ORDER BY t_min DESC
        """
        
        result = service.run_sync(query)
        df = result.to_table().to_pandas()
        
        if not df.empty:
            print(f"✅ Found {len(df)} observations by coordinates!")
            if 'access_url' in df.columns:
                urls = df['access_url'].dropna()
                print(f"   Download URLs: {len(urls)}")

except Exception as e:
    print(f"❌ Search failed: {e}")
    print("\nThis might mean:")
    print("1. The TAP service is temporarily down")
    print("2. The table structure has changed")
    print("3. Network issues")

print("\n" + "=" * 60)
print("SOLUTION")
print("=" * 60)
print("""
To fix your project permanently:

1. Set environment variable before running:
   export OBSCORE_TABLE="ivoa.obscore"
   
2. Or add to your .env file:
   OBSCORE_TABLE=ivoa.obscore
   
3. Or modify tap.py to force this table:
   self.obscore_table = "ivoa.obscore"  # Force correct table

The issue is that auto-discovery is picking the wrong table
(tap_schema.tap_schema.obscore) which is slow/broken.
""")
