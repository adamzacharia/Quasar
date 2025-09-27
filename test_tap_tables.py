#!/usr/bin/env python3
"""
Discover the correct table names in NRAO TAP service
"""

import pyvo

def discover_nrao_tables():
    """Find out what tables are actually available in NRAO TAP"""
    
    print("\n" + "="*70)
    print("DISCOVERING NRAO TAP SERVICE TABLES")
    print("="*70)
    
    try:
        # Connect to NRAO TAP
        tap_service = pyvo.dal.TAPService("https://data-query.nrao.edu/tap")
        print(f"✅ Connected to: https://data-query.nrao.edu/tap")
        
        # Try to get table list
        print("\n🔍 Querying available tables...")
        
        # Method 1: Try querying TAP_SCHEMA.tables
        try:
            query = "SELECT table_name, description FROM TAP_SCHEMA.tables"
            result = tap_service.search(query)
            
            if result:
                print(f"\n✅ Found {len(result)} tables:")
                for row in result:
                    print(f"   • {row['table_name']}")
                    if row.get('description'):
                        print(f"     {row['description'][:100]}")
        except Exception as e:
            print(f"❌ Could not query TAP_SCHEMA: {e}")
        
        # Method 2: Try common table names
        print("\n🔍 Testing common table names...")
        test_tables = [
            "ivoa.ObsCore",
            "ivoa.obscore", 
            "ObsCore",
            "obscore",
            "tap_schema.obscore",
            "TAP_SCHEMA.ObsCore",
            "public.obscore",
            "nrao.obscore"
        ]
        
        for table in test_tables:
            try:
                query = f"SELECT TOP 1 * FROM {table}"
                result = tap_service.search(query)
                print(f"✅ WORKS: {table}")
                
                # Get column names
                if result:
                    cols = result.to_table().colnames
                    print(f"   Columns: {', '.join(cols[:5])}...")
                    return table  # Return the working table name
                    
            except Exception as e:
                error_msg = str(e)
                if "not found" in error_msg.lower():
                    print(f"❌ NOT FOUND: {table}")
                else:
                    print(f"❌ ERROR with {table}: {error_msg[:50]}...")
        
        # Method 3: Try a different approach - query without table
        print("\n🔍 Trying alternative query methods...")
        try:
            # Some TAP services have default tables
            query = "SELECT TOP 1 * FROM obscore"
            result = tap_service.search(query)
            print("✅ Default table 'obscore' works!")
            return "obscore"
        except:
            pass
            
    except Exception as e:
        print(f"\n❌ Connection error: {e}")
    
    return None

if __name__ == "__main__":
    working_table = discover_nrao_tables()
    
    if working_table:
        print(f"\n" + "="*70)
        print(f"✅ SUCCESS! Use this table name: {working_table}")
        print("="*70)
        print(f"\nUpdate your queries to use: FROM {working_table}")
    else:
        print("\n⚠️ Could not find working table name")
        print("The service might be down or using a different structure")
