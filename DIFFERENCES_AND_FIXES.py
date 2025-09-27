#!/usr/bin/env python3
"""
QUASAR PROJECT - KEY DIFFERENCES & HOW TO FIX

This explains what was wrong and how the new code fixes it.
"""

print("""
================================================================================
WHY YOUR QUASAR PROJECT WASN'T FETCHING REAL NRAO DATA
================================================================================

THE MAIN PROBLEMS:
-----------------
1. ❌ WRONG TABLE: Your code was using 'tap_schema.obscore' 
   ✅ FIX: Should use 'ivoa.obscore' (where the real data is)

2. ❌ NO COLUMN CHECKING: Querying columns that don't exist causes errors
   ✅ FIX: Check which columns exist before building queries

3. ❌ NO DOWNLOAD URLS: Manually constructing URLs that don't work
   ✅ FIX: Use 'access_url' column which has real download links

4. ❌ SINGLE QUERY APPROACH: If one query fails, everything fails
   ✅ FIX: Multiple fallback strategies

5. ❌ NASA ADS REQUIRES API KEY: Limited access to papers
   ✅ FIX: Use OpenAlex (FREE, no API key needed!)

================================================================================
HOW TO UPDATE YOUR PROJECT
================================================================================

STEP 1: Replace tap.py
----------------------
""")

print("""
# Option A: Use the fixed version directly
cp integrations/tap_fixed.py integrations/tap.py

# Option B: Or update your existing tap.py with these key changes:

1. Add the auto-table discovery method from tap_fixed.py
2. Add the column checking before queries  
3. Include 'access_url' in WANTED_COLUMNS
4. Add fallback query strategies
""")

print("""
================================================================================
STEP 2: Test the Fix
================================================================================
""")

# Create test code
test_code = '''
from integrations.tap_fixed import NRAOTapClient

# Initialize the fixed client
client = NRAOTapClient()

# This will show you which table it's using
print(f"Using table: {client.obscore_table}")
print(f"Available columns: {len(client._table_columns)}")

# Test search for 3C 273
print("\\nSearching for 3C 273...")
df = client.search_by_source_name("3C 273")

if not df.empty:
    print(f"✅ Found {len(df)} observations!")
    
    # Check if we have download URLs
    if 'access_url' in df.columns:
        urls = df['access_url'].dropna()
        if not urls.empty:
            print(f"✅ Found {len(urls)} direct download URLs!")
            print(f"   Example URL: {urls.iloc[0]}")
    
    # Show what facilities observed this source
    if 'facility_name' in df.columns:
        facilities = df['facility_name'].value_counts()
        print(f"\\nFacilities that observed 3C 273:")
        for facility, count in facilities.items():
            print(f"  - {facility}: {count} observations")
else:
    print("❌ No data found")
'''

print("Run this test code:")
print("-" * 40)
print(test_code)
print("-" * 40)

print("""
================================================================================
STEP 3: Add OpenAlex for Papers (No API Key Required!)
================================================================================
""")

openalex_code = '''
import requests

class OpenAlexClient:
    """Free paper search - no API key needed!"""
    
    def __init__(self, email="adamandspace@gmail.com"):
        self.base_url = "https://api.openalex.org"
        self.email = email
    
    def search_papers(self, source_name, max_results=10):
        """Search for papers about an astronomical source"""
        
        # Build query
        query = f'{source_name} AND (radio OR VLA OR ALMA OR interferometry)'
        
        params = {
            'search': query,
            'filter': 'has_doi:true',
            'per_page': max_results,
            'mailto': self.email
        }
        
        response = requests.get(
            f"{self.base_url}/works",
            params=params,
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            papers = data.get('results', [])
            
            for paper in papers[:5]:
                print(f"📄 {paper.get('title', 'Unknown')}")
                print(f"   Citations: {paper.get('cited_by_count', 0)}")
                print(f"   Year: {paper.get('publication_year', 'Unknown')}")
                
                # Check for open access
                oa = paper.get('open_access', {})
                if oa.get('is_oa'):
                    print(f"   FREE PDF: {oa.get('oa_url')}")
                print()
            
            return papers
        return []

# Test it
client = OpenAlexClient()
papers = client.search_papers("3C 273")
print(f"Found {len(papers)} papers about 3C 273")
'''

print("OpenAlex integration (add to services/openalex_service.py):")
print("-" * 40)
print(openalex_code)
print("-" * 40)

print("""
================================================================================
KEY INSIGHTS FROM YOUR WORKING CODE
================================================================================

1. TABLE DISCOVERY IS CRITICAL
   Your working code smartly finds the right table by:
   - Checking multiple tables
   - Scoring them by available columns
   - Preferring 'ivoa.obscore' over 'tap_schema.obscore'

2. COLUMN VALIDATION PREVENTS ERRORS
   - Check which columns exist BEFORE querying
   - Only SELECT columns that are actually there
   - Have fallbacks if expected columns are missing

3. ACCESS_URL IS THE KEY TO DOWNLOADS
   - This column contains actual NRAO archive URLs
   - No need to construct URLs manually
   - Direct links to download the data

4. MULTIPLE QUERY STRATEGIES
   - Try ADQL geometry functions first
   - Fall back to simple RA/Dec box if not supported
   - Try different name formats for source searches

5. OPENALEX > NASA ADS
   - Completely free, no API key
   - Rich metadata including citations
   - Open access PDF links when available

================================================================================
SUMMARY
================================================================================

Your original project was querying the WRONG table (tap_schema.obscore) which
either doesn't exist or has incomplete data. The REAL data is in ivoa.obscore.

The fixed code:
✅ Auto-discovers the correct table
✅ Validates columns before querying
✅ Gets real download URLs from access_url
✅ Has multiple fallback strategies
✅ Uses OpenAlex for papers (FREE!)

Now your searches for "3C 273" will return REAL NRAO observations with
direct download links to the actual data!

================================================================================
""")
