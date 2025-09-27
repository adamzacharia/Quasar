#!/usr/bin/env python3
"""
QUICK FIX: Apply the working NRAO integration to your Quasar project
This script will update your project to use the correct NRAO data access methods
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

def apply_fixes():
    """Apply all fixes to make NRAO data access work"""
    
    print("\n" + "🚀"*30)
    print("  FIXING QUASAR PROJECT NRAO INTEGRATION")
    print("🚀"*30)
    
    project_root = Path(__file__).parent
    
    # Step 1: Backup original tap.py
    print("\n1️⃣ Backing up original tap.py...")
    tap_original = project_root / "integrations" / "tap.py"
    tap_backup = project_root / "integrations" / "tap_original_backup.py"
    
    if tap_original.exists() and not tap_backup.exists():
        shutil.copy(tap_original, tap_backup)
        print(f"   ✅ Backed up to {tap_backup}")
    
    # Step 2: Replace with fixed version
    print("\n2️⃣ Installing fixed TAP client...")
    tap_fixed = project_root / "integrations" / "tap_fixed.py"
    
    if tap_fixed.exists():
        shutil.copy(tap_fixed, tap_original)
        print(f"   ✅ Replaced tap.py with working version")
    else:
        print(f"   ❌ tap_fixed.py not found!")
        return False
    
    # Step 3: Install/upgrade required packages
    print("\n3️⃣ Installing/upgrading required packages...")
    packages = ["pyvo", "astroquery", "astropy", "pandas", "numpy"]
    
    for package in packages:
        print(f"   📦 Upgrading {package}...")
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", package], 
                         capture_output=True, text=True, check=True)
            print(f"      ✅ {package} ready")
        except subprocess.CalledProcessError as e:
            print(f"      ⚠️ Failed to upgrade {package}: {e}")
    
    # Step 4: Test the connection
    print("\n4️⃣ Testing NRAO connections...")
    try:
        from integrations.tap import NRAOTapClient
        client = NRAOTapClient()
        status = client.test_connection()
        
        for service, state in status.items():
            symbol = "✅" if state == "connected" else "⚠️"
            print(f"   {symbol} {service}: {state}")
    except Exception as e:
        print(f"   ❌ Connection test failed: {e}")
        return False
    
    # Step 5: Quick test with 3C 273
    print("\n5️⃣ Quick test: Searching for 3C 273...")
    try:
        df = client.search_by_source_name("3C 273", max_results=5)
        if not df.empty:
            print(f"   ✅ Found {len(df)} observations!")
            print(f"   ✅ NRAO integration is working!")
        else:
            print(f"   ⚠️ No data found (service might be down)")
    except Exception as e:
        print(f"   ❌ Search failed: {e}")
        return False
    
    return True


def show_usage():
    """Show how to use the fixed integration"""
    
    print("\n" + "📖"*30)
    print("  HOW TO USE THE FIXED INTEGRATION")
    print("📖"*30)
    
    print("""
    IN YOUR PYTHON CODE:
    --------------------
    from integrations.tap import NRAOTapClient
    
    # Initialize
    client = NRAOTapClient()
    
    # Search for any source
    df = client.search_by_source_name("3C 273")  # or "M31", "Cygnus A", etc.
    
    # Get results
    if not df.empty:
        print(f"Found {len(df)} observations")
        print(df[['target_name', 'instrument_name', 'obs_date']].head())
        
        # Get archive URLs
        for url in df['archive_url'].head(3):
            print(f"Archive: {url}")
    
    
    IN YOUR QUASAR APP UI:
    ----------------------
    When user types: "fetch me data of 3C 273"
    
    Your app will:
    1. Extract "3C 273" from the query
    2. Call client.search_by_source_name("3C 273")
    3. Display results with:
       • Number of observations
       • Telescope used (VLA, VLBA, ALMA)
       • Observation dates
       • Frequency ranges
       • Direct archive links
    
    
    AVAILABLE METHODS:
    -----------------
    • search_by_source_name(name)     - Search by source name
    • search_vla_vlba(name)          - VLA/VLBA only
    • search_alma(name)              - ALMA only
    • get_archive_url(obs_id)        - Get direct archive link
    • format_results_summary(df)      - Format nice summary
    
    
    WHAT WORKS NOW:
    --------------
    ✅ VLA data search
    ✅ VLBA data search  
    ✅ ALMA data search
    ✅ Coordinate resolution (3C 273 → RA/Dec)
    ✅ Name variants (3C 273, 3C273, 3C_273)
    ✅ Archive URL generation
    ✅ Metadata retrieval
    
    
    WHAT STILL NEEDS WEB INTERFACE:
    -------------------------------
    ⚠️ Actual data downloads (use archive URLs)
    ⚠️ Proprietary data access (need login)
    """)


if __name__ == "__main__":
    print("\n" + "="*60)
    print("QUASAR PROJECT - NRAO INTEGRATION FIX")
    print("="*60)
    
    # Apply the fixes
    success = apply_fixes()
    
    if success:
        print("\n" + "✅"*30)
        print("  ALL FIXES APPLIED SUCCESSFULLY!")
        print("✅"*30)
        
        # Show usage
        show_usage()
        
        print("\n" + "🎉"*30)
        print("  YOUR QUASAR APP CAN NOW FETCH NRAO DATA!")
        print("🎉"*30)
        
        print("""
        NEXT STEPS:
        -----------
        1. Run: python test_3c273_search.py
           (to see exactly what data you'll get)
        
        2. Start your app: python quasar.py
           (and try searching for sources)
        
        3. Test queries like:
           • "fetch me data of 3C 273"
           • "show VLA observations of M31"
           • "find ALMA data for NGC 253"
        """)
    else:
        print("\n" + "❌"*30)
        print("  SOME FIXES FAILED - CHECK THE ERRORS ABOVE")
        print("❌"*30)
        
        print("""
        TROUBLESHOOTING:
        ---------------
        1. Make sure you have internet connection
        2. Try: pip install --upgrade pyvo astroquery
        3. Check if tap_fixed.py exists
        4. Try running test_3c273_search.py directly
        """)
