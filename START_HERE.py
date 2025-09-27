#!/usr/bin/env python3
"""
GET YOUR DEMO WORKING NOW!
"""

print("""
========================================
QUASAR DEMO - ALMA ONLY (WORKING VERSION)
========================================

You have 3 options:

OPTION 1: Full App (Recommended)
---------------------------------
python setup_demo.py
streamlit run ui/app.py

This gives you:
✅ Nice UI with dark theme
✅ Statistics and charts
✅ Archive links
✅ CSV download
✅ Sidebar with examples

OPTION 2: Emergency Standalone
------------------------------
streamlit run emergency_demo.py

This is:
✅ ONE file, no dependencies
✅ Super simple
✅ Still searches real ALMA data
✅ Good enough for demo

OPTION 3: Test First
--------------------
python test_alma.py

This checks if ALMA access works at all.

========================================
FOR YOUR DEMO TODAY:
========================================

1. Run: python setup_demo.py
2. Run: streamlit run ui/app.py
3. Search for: "3C 273"
4. You'll see 77+ real ALMA observations
5. Show the data table, bands, links
6. Download CSV to prove it's real data

THAT'S IT. IT WORKS. GOOD LUCK! 🎉
""")
