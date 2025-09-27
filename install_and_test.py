#!/usr/bin/env python3
"""
Install and test astroquery for NRAO data access
"""

import subprocess
import sys

print("=" * 60)
print("INSTALLING ASTROQUERY FOR NRAO DATA ACCESS")
print("=" * 60)

# Install astroquery
print("\nInstalling astroquery...")
try:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "astroquery", "--upgrade"])
    print("✅ astroquery installed successfully!")
except Exception as e:
    print(f"❌ Failed to install: {e}")
    print("\nTry manually: pip install astroquery")
    sys.exit(1)

print("\nNow testing astroquery.nrao...")
print("-" * 40)

try:
    from astroquery.nrao import Nrao
    from astropy.coordinates import SkyCoord
    from astropy import units as u
    
    print("✅ Import successful!")
    
    # Test with a real query
    print("\nTesting search for 3C 273...")
    
    # Method 1: Direct object search
    results = Nrao.query_object(
        object_name="3C 273",
        telescope='jansky_vla'
    )
    
    if results:
        print(f"✅ Found {len(results)} VLA observations!")
        
        # Convert to pandas to see the data
        import pandas as pd
        df = results.to_pandas()
        
        print("\nColumns available:")
        print(list(df.columns))
        
        print("\nFirst 3 observations:")
        if 'Source' in df.columns and 'Obs Date' in df.columns:
            print(df[['Source', 'Obs Date', 'Configuration', 'Obs Band']].head(3))
        
        print("\n✅ SUCCESS! astroquery.nrao is working!")
        print("\nYou can now use this instead of broken TAP!")
    else:
        print("No observations found (but the query worked)")
        
except ImportError as e:
    print(f"❌ Still can't import astroquery: {e}")
    print("\nTry:")
    print("1. Exit Python and restart")
    print("2. pip install --force-reinstall astroquery")
    
except Exception as e:
    print(f"Error during test: {e}")

print("\n" + "=" * 60)
print("NEXT STEPS:")
print("=" * 60)
print("""
1. Replace your tap.py with tap_astroquery.py:
   cp integrations/tap_astroquery.py integrations/tap.py

2. Update your imports in any file using tap:
   from integrations.tap import NRAOTapClient

3. Your searches will now work using astroquery instead of TAP!

Note: This gives you metadata. For actual data files:
- Go to https://data.nrao.edu
- Search for your project
- Request download (you'll get wget commands by email)
""")
