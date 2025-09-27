#!/usr/bin/env python3
"""
Quick fix to force the correct table
"""

import os

# Set environment variable to force the correct table
os.environ['OBSCORE_TABLE'] = 'ivoa.obscore'

print("Forcing correct table: ivoa.obscore")
print("=" * 60)

# Now run the test with the correct table
from integrations.tap_fixed import NRAOTapClient

print("Testing with ivoa.obscore...")
client = NRAOTapClient()

print(f"Using table: {client.obscore_table}")
print(f"Columns available: {len(client._table_columns)}")

# Quick test
print("\nSearching for 3C 273...")
df = client.search_by_source_name("3C 273")

if not df.empty:
    print(f"✅ Found {len(df)} observations!")
    if 'access_url' in df.columns:
        urls = df['access_url'].dropna()
        if not urls.empty:
            print(f"✅ Found {len(urls)} download URLs")
else:
    print("No results found")
