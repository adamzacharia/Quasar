
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Mock cgi to prevent pyvo crash on Python 3.13
import types
mock_cgi = types.ModuleType("cgi")
mock_cgi.parse_header = lambda x: (x, {}) # minimalistic mock
sys.modules["cgi"] = mock_cgi

from integrations.ads_client import ADSService

def test_ads_direct():
    print("--- Testing ADSService Direct Usage ---")
    ads_key = os.getenv("NASA_ADS_API_KEY")
    if not ads_key:
        print("WARNING: No NASA_ADS_API_KEY found. Expecting mocks.")
    
    service = ADSService(api_key=ads_key)
    
    print("1. Testing search_natural_language...")
    try:
        # Simple query
        res = service.search_natural_language("Find papers on Sz65 spectral analysis", max_results=3)
        print(f"Result Query: {res.get('query')}")
        print(f"Papers found: {len(res.get('papers', []))}")
        if res.get('papers'):
             print(f"Top paper: {res['papers'][0]['title']}")
    except Exception as e:
        print(f"ERROR in search_natural_language: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_ads_direct()

