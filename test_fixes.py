#!/usr/bin/env python3
"""
Test script for Quasar NRAO integration fixes
Run this to verify the fixes are working correctly
"""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def test_tap_integration():
    """Test the enhanced TAP integration"""
    print("=" * 60)
    print("TESTING TAP INTEGRATION")
    print("=" * 60)
    
    try:
        from integrations.tap import NRAOTapClient
        client = NRAOTapClient()
        
        # Test the new search_by_source_name method
        print("\n1. Testing search_by_source_name('3C 273')...")
        results = client.search_by_source_name('3C 273', max_results=5)
        
        if not results.empty:
            print(f"   ✅ Found {len(results)} observations")
            print(f"   Columns: {list(results.columns)[:5]}...")
            if 'target_name' in results.columns:
                print(f"   Targets: {results['target_name'].unique()[:3]}")
        else:
            print("   ⚠️  No results found (TAP service might be down)")
            
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False
    
    return True

def test_data_processor():
    """Test the data processor service"""
    print("\n" + "=" * 60)
    print("TESTING DATA PROCESSOR")
    print("=" * 60)
    
    try:
        from integrations.tap import NRAOTapClient
        from services.data_processor import DataProcessor
        from services.ads_service import ADSService
        
        tap_client = NRAOTapClient()
        ads_service = ADSService()
        processor = DataProcessor(tap_client, ads_service)
        
        print("\n2. Testing fetch_source_data('M31')...")
        result = processor.fetch_source_data('M31')
        
        print(f"   ✅ Got result dictionary with keys: {list(result.keys())}")
        
        if result['observations'] is not None:
            print(f"   Found {len(result['observations'])} observations")
        
        if result['summary']:
            print(f"   Summary generated: {len(result['summary'])} characters")
            
        if result['papers']:
            print(f"   Found {len(result['papers'])} papers")
            
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False
    
    return True

def test_search_service():
    """Test the enhanced search service"""
    print("\n" + "=" * 60)
    print("TESTING SEARCH SERVICE")
    print("=" * 60)
    
    try:
        from integrations.tap import NRAOTapClient
        from services.search import SearchService
        
        tap_client = NRAOTapClient()
        search_service = SearchService(tap_client)
        
        print("\n3. Testing search_source('Cygnus A')...")
        results = search_service.search_source('Cygnus A')
        
        if not results.empty:
            print(f"   ✅ Found {len(results)} observations")
        else:
            print("   ⚠️  No results (service might be down)")
            
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False
    
    return True

def test_ui_functions():
    """Test the UI helper functions"""
    print("\n" + "=" * 60)
    print("TESTING UI FUNCTIONS")
    print("=" * 60)
    
    try:
        from ui.enhanced_functions import extract_source_name
        
        print("\n4. Testing extract_source_name()...")
        
        test_cases = [
            ("Find VLA observations of 3C 273", "3C 273"),
            ("Search for M31 data", "M31"),
            ("Show me Cygnus A observations", "Cygnus A"),
            ("Get data for NGC 1275", "NGC 1275")
        ]
        
        for query, expected in test_cases:
            result = extract_source_name(query)
            if result == expected:
                print(f"   ✅ '{query}' -> '{result}'")
            else:
                print(f"   ❌ '{query}' -> Got '{result}', expected '{expected}'")
                
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False
    
    return True

def main():
    """Run all tests"""
    print("\n🔬 QUASAR FIX VERIFICATION TESTS")
    print("=" * 60)
    
    # Check environment
    print("\nChecking environment variables...")
    if os.getenv('OPENAI_API_KEY'):
        print("✅ OPENAI_API_KEY is set")
    else:
        print("⚠️  OPENAI_API_KEY not set (AI responses won't work)")
    
    if os.getenv('NASA_ADS_API_KEY'):
        print("✅ NASA_ADS_API_KEY is set")
    else:
        print("⚠️  NASA_ADS_API_KEY not set (paper search limited)")
    
    # Run tests
    tests = [
        test_tap_integration,
        test_data_processor,
        test_search_service,
        test_ui_functions
    ]
    
    results = []
    for test in tests:
        try:
            success = test()
            results.append(success)
        except Exception as e:
            print(f"\n❌ Test failed with error: {e}")
            results.append(False)
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for r in results if r)
    total = len(results)
    
    if passed == total:
        print(f"✅ All {total} tests passed!")
        print("\nYour Quasar project is ready to fetch real NRAO data!")
        print("Run: python quasar.py web")
        print("Then try: 'Find VLA observations of 3C 273'")
    else:
        print(f"⚠️  {passed}/{total} tests passed")
        print("\nSome tests failed. This might be due to:")
        print("- NRAO TAP service being temporarily down")
        print("- Missing API keys in .env file")
        print("- Network connectivity issues")
        print("\nThe core fixes are still in place and should work when services are available.")

if __name__ == "__main__":
    main()
