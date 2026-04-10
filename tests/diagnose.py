#!/usr/bin/env python3
"""
SIMPLE TEST - What's actually broken?
"""

print("Testing what works and what doesn't...")

# Test 1: Can we import the TAP client?
try:
    from integrations.tap import NRAOTapClient
    print("[OK] TAP client imports OK")
except Exception as e:
    print(f"[FAIL] TAP client broken: {e}")

# Test 2: Can we import Streamlit?
try:
    import streamlit as st
    print("[OK] Streamlit imports OK")
except Exception as e:
    print(f"[FAIL] Streamlit broken: {e}")

# Test 3: Does core.agent exist?
try:
    from core.agent import QuasarAgent
    print("[OK] Core agent exists")
except Exception as e:
    print(f"[FAIL] Core agent missing: {e}")

# Test 4: Can we actually search for data?
try:
    client = NRAOTapClient()
    print("[OK] TAP client initializes")
    
    # Try a simple search
    df = client.search_by_source_name("3C 273", max_results=5)
    if not df.empty:
        print(f"[OK] Can fetch data! Found {len(df)} observations")
    else:
        print("[!] No data returned")
except Exception as e:
    print(f"[FAIL] Can't fetch data: {e}")

print("\nNow you know what's broken!")
