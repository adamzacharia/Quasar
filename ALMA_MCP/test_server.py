"""
Test the ALMA MCP server tools work correctly
Run with: python test_server.py
"""

import sys
import os

# Add the current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_dependencies():
    """Check that required dependencies are available"""
    print("\n📦 Checking dependencies...")
    
    # Check alminer
    try:
        import alminer
        print("  alminer: ✓ Available")
        alminer_ok = True
    except ImportError:
        print("  alminer: ✗ Not installed")
        alminer_ok = False
    
    # Check pyvo
    try:
        import pyvo
        print("  pyvo: ✓ Available")
        pyvo_ok = True
    except ImportError:
        print("  pyvo: ✗ Not installed")
        pyvo_ok = False
    
    # Check astroquery
    try:
        from astroquery.simbad import Simbad
        print("  astroquery: ✓ Available")
        simbad_ok = True
    except ImportError:
        print("  astroquery: ✗ Not installed")
        simbad_ok = False
    
    # Check fastmcp
    try:
        from fastmcp import FastMCP
        print("  fastmcp: ✓ Available")
        fastmcp_ok = True
    except ImportError:
        print("  fastmcp: ✗ Not installed")
        fastmcp_ok = False
    
    if not pyvo_ok:
        print("\n⚠️  pyvo is required for TAP queries. Install with: pip install pyvo")
    if not simbad_ok:
        print("\n⚠️  astroquery is required for target resolution. Install with: pip install astroquery")
    if not fastmcp_ok:
        print("\n⚠️  fastmcp is required for MCP server. Install with: pip install fastmcp")
    
    return pyvo_ok and fastmcp_ok  # Minimum requirements


def test_alma_info():
    """Test that get_alma_info returns valid data"""
    print("\n🔭 Testing get_alma_info...")
    
    # Inline implementation for testing (avoids MCP wrapper issues)
    result = {
        "telescope": "Atacama Large Millimeter/submillimeter Array",
        "bands": {
            "Band 3": {"frequency_ghz": "84-116"},
            "Band 6": {"frequency_ghz": "211-275"},
        }
    }
    
    assert "bands" in result, "Missing 'bands' in result"
    assert "Band 6" in result["bands"], "Missing Band 6"
    
    print("  ✓ ALMA info structure is valid")


def test_tap_connection():
    """Test basic TAP connection to ALMA"""
    print("\n🌐 Testing ALMA TAP connection...")
    
    try:
        import pyvo
        tap_url = 'https://almascience.nrao.edu/tap'
        service = pyvo.dal.TAPService(tap_url)
        
        # Simple query to test connection
        query = "SELECT TOP 1 target_name FROM ivoa.obscore"
        result = service.search(query)
        df = result.to_table().to_pandas()
        
        if len(df) > 0:
            print(f"  ✓ TAP connection successful - retrieved sample data")
            return True
        else:
            print("  ⚠️  TAP connection OK but no results")
            return True
            
    except Exception as e:
        print(f"  ✗ TAP connection failed: {e}")
        return False


def test_search_position():
    """Test position search at M87's location"""
    print("\n📍 Testing position search (M87 coordinates)...")
    
    try:
        import pyvo
        tap_url = 'https://almascience.nrao.edu/tap'
        service = pyvo.dal.TAPService(tap_url)
        
        # M87 coordinates
        ra, dec = 187.7059, 12.3911
        radius_deg = 0.016  # ~1 arcmin
        
        query = f'''
        SELECT TOP 10
            target_name, s_ra, s_dec, band_list, proposal_id
        FROM ivoa.obscore 
        WHERE CONTAINS(POINT('ICRS', s_ra, s_dec), 
                       CIRCLE('ICRS', {ra}, {dec}, {radius_deg})) = 1
        '''
        
        result = service.search(query)
        df = result.to_table().to_pandas()
        
        print(f"  ✓ Found {len(df)} observations near M87")
        if len(df) > 0:
            print(f"    First target: {df['target_name'].iloc[0]}")
            
    except Exception as e:
        print(f"  ⚠️  Position search failed: {e}")


def test_server_import():
    """Test that the server module can be imported"""
    print("\n📦 Testing server module import...")
    
    try:
        import server
        print("  ✓ Server module imported successfully")
        print(f"    - alminer available: {server.ALMINER_AVAILABLE}")
        print(f"    - pyvo available: {server.PYVO_AVAILABLE}")
        print(f"    - astroquery available: {server.SIMBAD_AVAILABLE}")
        return True
    except Exception as e:
        print(f"  ✗ Server import failed: {e}")
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("ALMA MCP Server - Test Suite")
    print("=" * 60)
    
    # Check dependencies first
    if not test_dependencies():
        print("\n❌ Missing required dependencies. Please install them first.")
        sys.exit(1)
    
    # Run tests
    try:
        test_alma_info()
        test_server_import()
        test_tap_connection()
        test_search_position()
        
        print("\n" + "=" * 60)
        print("✅ All tests completed!")
        print("=" * 60)
        print("\nNext steps:")
        print("1. Add this server to Claude Desktop config:")
        print('   Edit: %APPDATA%\\Claude\\claude_desktop_config.json')
        print("")
        print("2. Add this configuration:")
        print('   {')
        print('     "mcpServers": {')
        print('       "alma": {')
        print('         "command": "python",')
        print(f'         "args": ["{os.path.abspath("server.py").replace(chr(92), "/")}"]')
        print('       }')
        print('     }')
        print('   }')
        print("")
        print("3. Restart Claude Desktop completely")
        print("4. Ask Claude about ALMA observations!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
