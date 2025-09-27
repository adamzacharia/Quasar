#!/usr/bin/env python3
"""
QUICK FIX FOR YOUR DEMO - ALMA ONLY VERSION
"""

import shutil

print("\n" + "="*60)
print("SETTING UP ALMA-ONLY QUASAR FOR YOUR DEMO")
print("="*60)

# Backup original
shutil.copy("ui/app.py", "ui/app_backup.py")
print("✅ Backed up original app.py")

# Replace with ALMA-only version
shutil.copy("ui/alma_only_app.py", "ui/app.py")
print("✅ Replaced with ALMA-only version")

print("\n" + "="*60)
print("READY FOR YOUR DEMO!")
print("="*60)

print("""
TO RUN:
-------
streamlit run ui/app.py

WHAT IT DOES:
------------
✅ Searches ALMA (which we KNOW works)
✅ Shows real observation data
✅ Provides archive links
✅ Shows bands, integration times, projects
✅ Downloads as CSV
✅ NO BROKEN VLA/VLBA stuff

TRY THESE SOURCES:
-----------------
• 3C 273
• M31
• NGC 253
• Cen A
• Sgr A*

Your demo will work! 🎉
""")
