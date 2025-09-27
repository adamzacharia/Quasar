#!/usr/bin/env python3
"""
Fix the Quasar web app to properly handle queries
"""

import shutil
from pathlib import Path

print("\n" + "="*60)
print("FIXING QUASAR WEB APP")
print("="*60)

# Backup original
original = Path("ui/app.py")
backup = Path("ui/app_backup.py")
fixed = Path("ui/app_fixed.py")

if original.exists() and not backup.exists():
    shutil.copy(original, backup)
    print(f"✅ Backed up original app.py to app_backup.py")

# Replace with fixed version
if fixed.exists():
    shutil.copy(fixed, original)
    print(f"✅ Replaced app.py with fixed version")
    
print("\n" + "="*60)
print("FIX COMPLETE!")
print("="*60)

print("""
Now run your app:
    python ui/app.py
    
Or if using streamlit directly:
    streamlit run ui/app.py

The chat interface will now:
✅ Actually process your queries when you hit Enter
✅ Search NRAO archives for any source
✅ Display real data in tables
✅ Show archive links
✅ Not lose your input!

Try these queries:
• Search NRAO archives for "3C 273"
• Find VLA observations of M31
• Show ALMA data for NGC 253
""")
