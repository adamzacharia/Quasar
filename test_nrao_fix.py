#!/usr/bin/env python3
"""
Quick test to verify NRAO integration is working correctly
Run this after updating tap.py to see if it's fetching real data
"""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def test_nrao_integration():
    """Test that we're getting real NRAO data"""
    
    print("=" * 60)
    print("TESTING NRAO INTEGRATION")
    print("=" * 60)
    
    try:
        # Import the fixed client
        from integrations.tap_fixed import NRAOTapClient
        
        # Initialize
        print("\n1. Initializing TAP client...")
        client = NRAOTapClient()
        
        print(f"   ✅ Connected to: {client.service_url}")
        print(f"   ✅ Using table: {client.obscore_table}")
        print(f"   ✅ Available columns: {len(client._table_columns)}")
        
        # Key check: Do we have access_url column?
        has_urls = 'access_url' in {c.lower() for c in client._table_columns}
        if has_urls:
            print(f"   ✅ Has access_url column (download links available!)")
        else:
            print(f"   ⚠️  No access_url column (no direct download links)")
        
        # Test search
        print("\n2. Testing search for '3C 273'...")
        df = client.search_by_source_name("3C 273")
        
        if df.empty:
            print("   ❌ No results found")
            print("   (This might mean the service is down or the source name needs adjustment)")
        else:
            print(f"   ✅ Found {len(df)} observations!")
            
            # Show key information
            if 'facility_name' in df.columns:
                facilities = df['facility_name'].value_counts()
                print(f"\n   Facilities:")
                for fac, count in facilities.items():
                    print(f"     - {fac}: {count} observations")
            
            if 'freq_min_ghz' in df.columns and 'freq_max_ghz' in df.columns:
                freq_min = df['freq_min_ghz'].min()
                freq_max = df['freq_max_ghz'].max()
                print(f"\n   Frequency range: {freq_min:.2f} - {freq_max:.2f} GHz")
            
            if 'obs_date' in df.columns:
                dates = pd.to_datetime(df['obs_date'])
                print(f"   Date range: {dates.min()} to {dates.max()}")
            
            # Most important: Download URLs
            if 'access_url' in df.columns:
                urls = df['access_url'].dropna()
                if not urls.empty:
                    print(f"\n   ✅✅✅ FOUND {len(urls)} DOWNLOAD URLs!")
                    print(f"   Example URLs:")
                    for url in urls.head(3):
                        print(f"     - {url[:80]}...")
                else:
                    print(f"\n   ⚠️  access_url column exists but is empty")
            else:
                print(f"\n   ⚠️  No access_url column in results")
        
        # Test coordinate resolution
        print("\n3. Testing coordinate resolution...")
        try:
            ra, dec = client.resolve_target_coordinates("M31")
            print(f"   ✅ Resolved M31 to RA={ra:.4f}, Dec={dec:.4f}")
        except Exception as e:
            print(f"   ⚠️  Could not resolve coordinates: {e}")
        
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        
        if not df.empty and has_urls:
            print("✅✅✅ SUCCESS! Your NRAO integration is working correctly!")
            print("     - Correct table detected")
            print("     - Real observations found")
            print("     - Download URLs available")
        elif not df.empty:
            print("⚠️  PARTIAL SUCCESS: Found data but no download URLs")
            print("    The table might not have access_url column")
        else:
            print("❌ No data returned - check if TAP service is up")
        
    except ImportError:
        print("❌ Could not import tap_fixed.py")
        print("   Make sure the file exists in integrations/")
    except Exception as e:
        print(f"❌ Error during test: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("NEXT STEPS")
    print("=" * 60)
    print("""
1. If successful, replace your tap.py with tap_fixed.py:
   cp integrations/tap_fixed.py integrations/tap.py

2. Update UI to use real data instead of mock data

3. Add OpenAlex for paper searches (no API key needed!)

4. Your app will now show:
   - Real NRAO observations
   - Direct download links
   - Actual observation metadata
   - Free access to papers via OpenAlex
    """)

if __name__ == "__main__":
    # Need pandas for the test
    try:
        import pandas as pd
    except ImportError:
        print("Please install pandas: pip install pandas")
        sys.exit(1)
    
    test_nrao_integration()
