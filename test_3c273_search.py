#!/usr/bin/env python3
"""
Test what happens when you search for '3C 273' with the fixed NRAO integration
This shows exactly what data you'll get and how to use it in your Quasar app
"""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Import the fixed TAP client
from integrations.tap_fixed import NRAOTapClient

def demonstrate_3c273_search():
    """
    Show exactly what happens when searching for 3C 273
    """
    print("\n" + "🌟"*35)
    print("  QUASAR APP: FETCHING DATA FOR 3C 273")
    print("🌟"*35)
    
    # Initialize the client
    print("\n1️⃣ Initializing NRAO connections...")
    client = NRAOTapClient()
    
    # Search for 3C 273
    print("\n2️⃣ User query: 'fetch me data of 3C 273'")
    print("   Processing...")
    
    df = client.search_by_source_name("3C 273", max_results=50)
    
    if df.empty:
        print("\n❌ No data found for 3C 273")
        return None
    
    print("\n3️⃣ DATA RETRIEVED SUCCESSFULLY!")
    print("="*70)
    
    # Show what we found
    print(f"📊 Total observations found: {len(df)}")
    
    # Group by facility
    if 'instrument_name' in df.columns:
        print("\n📡 By Telescope:")
        for inst in df['instrument_name'].unique():
            inst_df = df[df['instrument_name'] == inst]
            print(f"   • {inst}: {len(inst_df)} observations")
            
            # Show date range for each instrument
            if 'obs_date' in inst_df.columns:
                dates = inst_df['obs_date']
                if not dates.empty:
                    print(f"     Date range: {dates.min().date()} to {dates.max().date()}")
    
    # Frequency coverage
    if 'freq_min_ghz' in df.columns and 'freq_max_ghz' in df.columns:
        print(f"\n📻 Frequency Coverage:")
        print(f"   • Minimum: {df['freq_min_ghz'].min():.2f} GHz")
        print(f"   • Maximum: {df['freq_max_ghz'].max():.2f} GHz")
        
        # Show common bands
        bands = []
        if any((df['freq_min_ghz'] < 2) & (df['freq_max_ghz'] > 1)):
            bands.append("L-band (1-2 GHz)")
        if any((df['freq_min_ghz'] < 6) & (df['freq_max_ghz'] > 4)):
            bands.append("C-band (4-8 GHz)")
        if any((df['freq_min_ghz'] < 12) & (df['freq_max_ghz'] > 8)):
            bands.append("X-band (8-12 GHz)")
        if any((df['freq_min_ghz'] < 18) & (df['freq_max_ghz'] > 12)):
            bands.append("Ku-band (12-18 GHz)")
        if any((df['freq_min_ghz'] < 26) & (df['freq_max_ghz'] > 18)):
            bands.append("K-band (18-26 GHz)")
        if any((df['freq_min_ghz'] < 40) & (df['freq_max_ghz'] > 26)):
            bands.append("Ka-band (26-40 GHz)")
        
        if bands:
            print(f"   • Bands observed: {', '.join(bands)}")
    
    # Total data size
    if 'size_gb' in df.columns:
        total_size = df['size_gb'].sum()
        print(f"\n💾 Total Data Volume: {total_size:.2f} GB")
    
    # Show sample observations
    print("\n4️⃣ SAMPLE OBSERVATIONS (Most Recent):")
    print("="*70)
    
    # Sort by date if available
    if 'obs_date' in df.columns:
        df_sorted = df.sort_values('obs_date', ascending=False)
    else:
        df_sorted = df
    
    # Display top 5 observations
    for idx, row in df_sorted.head(5).iterrows():
        print(f"\n📌 Observation {idx+1}:")
        
        if 'obs_publisher_did' in row:
            print(f"   ID: {row['obs_publisher_did']}")
        
        if 'instrument_name' in row:
            print(f"   Telescope: {row['instrument_name']}")
            
        if 'obs_date' in row:
            print(f"   Date: {row['obs_date'].date() if hasattr(row['obs_date'], 'date') else row['obs_date']}")
            
        if 'freq_min_ghz' in row and 'freq_max_ghz' in row:
            print(f"   Frequency: {row['freq_min_ghz']:.2f} - {row['freq_max_ghz']:.2f} GHz")
            
        if 'configuration' in row and row['configuration']:
            print(f"   Configuration: {row['configuration']}")
            
        if 't_exptime' in row:
            exp_hours = row['t_exptime'] / 3600 if row['t_exptime'] else 0
            print(f"   Exposure: {exp_hours:.2f} hours")
            
        if 'size_gb' in row:
            print(f"   Data Size: {row['size_gb']:.2f} GB")
            
        if 'archive_url' in row and row['archive_url']:
            print(f"   📎 Archive Link: {row['archive_url']}")
    
    print("\n5️⃣ HOW TO ACCESS THE DATA:")
    print("="*70)
    print("""
    Option 1: Direct Archive Access
    --------------------------------
    • Click on any archive link above
    • Login with your NRAO account (if you have proprietary access)
    • Download the data using wget commands provided
    
    Option 2: Through Your Quasar App UI
    -------------------------------------
    • The app now displays all this information
    • Click 'Download' button next to any observation
    • Links open in the NRAO archive interface
    
    Option 3: Programmatic Access (Advanced)
    -----------------------------------------
    • Use the obs_publisher_did to construct download requests
    • Note: Direct download API not yet available
    • Must use web interface for actual data retrieval
    """)
    
    return df


def show_integration_instructions():
    """
    Show how to integrate this into the Quasar UI
    """
    print("\n" + "🔧"*35)
    print("  HOW TO INTEGRATE INTO YOUR QUASAR APP")
    print("🔧"*35)
    
    print("""
    1. REPLACE YOUR TAP.PY:
    -----------------------
    cp integrations/tap_fixed.py integrations/tap.py
    
    2. UPDATE YOUR UI CODE:
    -----------------------
    In your UI handler (ui/app.py or ui/enhanced_functions.py), when user asks
    for data about a source, call:
    
    ```python
    from integrations.tap_fixed import NRAOTapClient
    
    def handle_data_search(query):
        # Extract source name (e.g., "3C 273" from "fetch me data of 3C 273")
        source_name = extract_source_name(query)
        
        # Get the data
        client = NRAOTapClient()
        df = client.search_by_source_name(source_name)
        
        # Format for display
        summary = client.format_results_summary(df)
        
        # Return formatted results to UI
        return {
            'summary': summary,
            'data': df.to_dict('records'),
            'total_count': len(df),
            'archive_links': df['archive_url'].tolist() if 'archive_url' in df else []
        }
    ```
    
    3. WHAT THE USER SEES:
    ----------------------
    When they type "fetch me data of 3C 273", they'll see:
    
    • Summary of all observations
    • List of observations with details
    • Direct links to NRAO archive
    • Frequency and date information
    • Data sizes and configurations
    
    4. REQUIRED PACKAGES:
    --------------------
    pip install --upgrade pyvo astroquery astropy pandas
    
    5. ERROR HANDLING:
    -----------------
    The fixed client handles:
    • Service outages gracefully
    • Name resolution failures
    • Empty results
    • Different name formats (3C 273, 3C273, 3C_273)
    """)


if __name__ == "__main__":
    print("\n" + "="*70)
    print("QUASAR PROJECT - NRAO DATA ACCESS TEST")
    print("Testing what happens when user asks: 'fetch me data of 3C 273'")
    print("="*70)
    
    # Run the demonstration
    results = demonstrate_3c273_search()
    
    # Show integration instructions
    show_integration_instructions()
    
    print("\n" + "✅"*35)
    print("  TEST COMPLETE!")
    print("✅"*35)
    
    if results is not None and not results.empty:
        print(f"\n✨ Success! Found {len(results)} observations of 3C 273")
        print("✨ Your Quasar app can now fetch real NRAO data!")
    else:
        print("\n⚠️ No data found - check your internet connection")
        print("⚠️ Or the NRAO TAP service might be temporarily down")
