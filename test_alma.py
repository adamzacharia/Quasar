#!/usr/bin/env python3
"""
TEST ALMA - Make sure it works before your demo!
"""

print("\nTesting ALMA access for your demo...")
print("="*40)

try:
    from astroquery.alma import Alma
    print("✅ ALMA module imported")
    
    # Test search
    print("\nSearching for 3C 273...")
    results = Alma.query_object("3C 273", public=True, science=True)
    
    if results and len(results) > 0:
        print(f"✅ SUCCESS! Found {len(results)} observations")
        print("\nYOUR DEMO WILL WORK! 🎉")
    else:
        print("⚠️ No results, but connection works")
        
except Exception as e:
    print(f"❌ Error: {e}")
    print("\nFix: pip install --upgrade astroquery")

print("\nIf you see SUCCESS above, run:")
print("python setup_demo.py")
print("streamlit run ui/app.py")
