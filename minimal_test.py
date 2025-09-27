#!/usr/bin/env python3
"""
ABSOLUTE MINIMAL TEST - Does NRAO search work AT ALL?
No fancy UI, no complex imports, just raw functionality.
"""

print("\n" + "="*60)
print("TESTING IF WE CAN FETCH NRAO DATA AT ALL")
print("="*60)

# Most basic imports
import pyvo
import pandas as pd

print("\n1. Connecting to NRAO TAP service...")
try:
    tap = pyvo.dal.TAPService("https://data-query.nrao.edu/tap")
    print("✅ Connected!")
except Exception as e:
    print(f"❌ Can't connect: {e}")
    exit(1)

print("\n2. Searching for 3C 273...")
try:
    # Simplest possible query
    query = """
    SELECT TOP 10 
        target_name, instrument_name, t_min
    FROM obscore
    WHERE target_name LIKE '%3C%273%'
    """
    
    results = tap.search(query)
    print(f"✅ Query executed!")
    
    if len(results) > 0:
        print(f"✅ Found {len(results)} observations!")
        
        # Convert to pandas and show
        df = results.to_table().to_pandas()
        print("\nResults:")
        print(df)
    else:
        print("⚠️ No results (but query worked)")
        
except Exception as e:
    print(f"❌ Query failed: {e}")
    print("\nTrying different table name...")
    
    try:
        # Try without schema
        query2 = """
        SELECT TOP 10 *
        FROM obscore
        """
        results = tap.search(query2)
        print("✅ Basic query works - table is 'obscore'")
    except:
        print("❌ Can't query obscore table")

print("\n" + "="*60)
print("TEST COMPLETE")
print("="*60)
