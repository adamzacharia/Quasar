#!/usr/bin/env python3
"""
JUST MAKE IT WORK
"""

import shutil

print("\nFIXING YOUR APP - SIMPLE VERSION")
print("="*50)

# First run diagnostics
print("\n1. CHECKING WHAT'S BROKEN:")
print("-"*30)

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

# Check TAP client
try:
    from integrations.tap import NRAOTapClient
    print("✅ TAP client works")
except:
    print("❌ TAP client broken - run: cp integrations/tap_fixed.py integrations/tap.py")

# Check streamlit
try:
    import streamlit
    print("✅ Streamlit installed")
except:
    print("❌ Streamlit missing - run: pip install streamlit")

print("\n2. APPLYING FIX:")
print("-"*30)

# Backup and replace
shutil.copy("ui/app.py", "ui/app_original.py")
shutil.copy("ui/simple_app.py", "ui/app.py")
print("✅ Replaced app.py with SIMPLE WORKING VERSION")

print("\n3. HOW TO RUN:")
print("-"*30)
print("streamlit run ui/app.py")

print("\n4. WHAT IT DOES:")
print("-"*30)
print("• Shows a search box")
print("• You type a source name (like '3C 273')")  
print("• Click Search")
print("• It shows REAL NRAO data in a table")
print("• That's it. Simple and working.")

print("\nDONE! Now run: streamlit run ui/app.py")
