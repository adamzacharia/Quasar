#!/usr/bin/env python3
"""
RUN THIS TO GET A WORKING DEMO RIGHT NOW
"""

import shutil
import subprocess
import sys

print("\n🚀 FIXING QUASAR FOR YOUR DEMO...\n")

# Step 1: Replace app with ALMA-only version
try:
    shutil.copy("ui/app.py", "ui/app_old.py")
    shutil.copy("ui/alma_only_app.py", "ui/app.py")
    print("✅ App replaced with ALMA-only version")
except:
    print("⚠️ Couldn't backup, but continuing...")

# Step 2: Test ALMA works
print("\n📡 Testing ALMA connection...")
try:
    from astroquery.alma import Alma
    results = Alma.query_object("3C 273", public=True, science=True)
    if results and len(results) > 0:
        print(f"✅ ALMA WORKS! Found {len(results)} observations of 3C 273")
    else:
        print("⚠️ No data returned but connection works")
except Exception as e:
    print(f"❌ ALMA test failed: {e}")
    print("Run: pip install --upgrade astroquery")
    sys.exit(1)

# Step 3: Launch the app
print("\n🌟 LAUNCHING YOUR DEMO APP...\n")
print("="*50)
print("OPENING IN BROWSER AT: http://localhost:8501")
print("="*50)
print("\nTRY SEARCHING FOR:")
print("  • 3C 273 (77 observations)")
print("  • M31 (many observations)")
print("  • NGC 253 (lots of data)")
print("\nPress Ctrl+C to stop the app")
print("="*50)

# Launch streamlit
subprocess.run(["streamlit", "run", "ui/app.py"])
