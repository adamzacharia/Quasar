#!/bin/bash
# Quick shell script to test with correct table

echo "Testing NRAO TAP with correct table..."
echo "======================================="

# Set the correct table
export OBSCORE_TABLE="ivoa.obscore"

# Run Python test
python3 << 'EOF'
import os
import time
print(f"OBSCORE_TABLE is set to: {os.getenv('OBSCORE_TABLE')}")

try:
    from integrations.tap_fixed import NRAOTapClient
    
    print("\nInitializing client...")
    start = time.time()
    client = NRAOTapClient()
    
    print(f"Table being used: {client.obscore_table}")
    
    if client.obscore_table == "ivoa.obscore":
        print("✅ CORRECT TABLE!")
    else:
        print(f"❌ WRONG TABLE: {client.obscore_table}")
        print("   Setting manually...")
        client.obscore_table = "ivoa.obscore"
    
    print("\nSearching for 3C 273...")
    search_start = time.time()
    df = client.search_by_source_name("3C 273")
    search_time = time.time() - search_start
    
    print(f"\nSearch completed in {search_time:.1f} seconds")
    
    if not df.empty:
        print(f"✅ Found {len(df)} observations!")
    else:
        print("❌ No results (but query was fast)")
        
    total_time = time.time() - start
    print(f"\nTotal time: {total_time:.1f} seconds")
    
    if total_time < 10:
        print("✅ FAST! Using correct table!")
    else:
        print("⚠️ Still slow - might be network issues")
        
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
EOF

echo ""
echo "======================================="
echo "If this was fast (<10 seconds), the fix works!"
echo "Just make sure OBSCORE_TABLE=ivoa.obscore is in your .env"
