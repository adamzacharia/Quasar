"""Test script for new ALMA MCP tools - verify registration"""
import sys
sys.path.insert(0, '.')

from ALMA_MCP.server import mcp

print("=" * 60)
print("Verifying ALMA MCP Server Tools")
print("=" * 60)

# Get all registered tools
tools = mcp._tool_manager._tools if hasattr(mcp, '_tool_manager') else {}

print(f"\nTotal tools registered: {len(tools)}")
print("\nRegistered tools:")
for i, tool_name in enumerate(sorted(tools.keys()), 1):
    print(f"  {i:2d}. {tool_name}")

# Verify new tools are present
expected_new_tools = [
    'search_alma_by_source_name',
    'search_alma_by_bibliography',
    'search_alma_by_member_ous',
    'search_alma_by_data_type',
    'search_alma_by_science_keyword',
    'search_alma_by_abstract',
    'search_alma_by_sensitivity',
    'query_alma_multiple_sources',
]

print("\n" + "-" * 40)
print("Checking new tools:")
all_present = True
for tool in expected_new_tools:
    if tool in tools:
        print(f"  ✅ {tool}")
    else:
        print(f"  ❌ {tool} - MISSING")
        all_present = False

print("\n" + "=" * 60)
if all_present:
    print("SUCCESS: All new tools registered correctly!")
else:
    print("WARNING: Some tools are missing!")
print("=" * 60)
