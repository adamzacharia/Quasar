#!/usr/bin/env python3
"""
SIMPLEST POSSIBLE WORKING QUASAR APP
Just searches NRAO and shows results. That's it.
"""

import streamlit as st
import pandas as pd
import sys
from pathlib import Path

# Add parent to path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent.parent))

# Try to import the TAP client
try:
    from integrations.tap import NRAOTapClient
except:
    try:
        from integrations.tap_fixed import NRAOTapClient
    except:
        st.error("Can't import TAP client! Run: python apply_nrao_fix.py")
        st.stop()

# Page setup
st.set_page_config(page_title="Quasar", page_icon="🔭", layout="wide")

# Simple CSS - just dark background
st.markdown("""
<style>
.stApp {
    background: linear-gradient(to bottom, #0a0e27 0%, #1a1e3a 100%);
}
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("# 🌌 QUASAR - Radio Astronomy Data Search")
st.markdown("### Search NRAO Archives (VLA, VLBA, ALMA)")

# Initialize session state
if 'tap_client' not in st.session_state:
    with st.spinner("Connecting to NRAO..."):
        try:
            st.session_state.tap_client = NRAOTapClient()
            st.success("✅ Connected to NRAO archives")
        except Exception as e:
            st.error(f"Can't connect to NRAO: {e}")
            st.stop()

# MAIN SEARCH BOX
st.markdown("---")
source_name = st.text_input(
    "Enter source name (e.g., 3C 273, M31, Cygnus A):",
    placeholder="3C 273",
    key="search_input"
)

# Search button
if st.button("🔍 Search NRAO Archives", type="primary"):
    if source_name:
        with st.spinner(f"Searching for {source_name}..."):
            try:
                # ACTUALLY SEARCH FOR DATA
                df = st.session_state.tap_client.search_by_source_name(
                    source_name, 
                    max_results=100
                )
                
                if df is not None and not df.empty:
                    # SUCCESS! Show the data
                    st.success(f"✅ Found {len(df)} observations of {source_name}")
                    
                    # Summary stats
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        if 'instrument_name' in df.columns:
                            instruments = df['instrument_name'].value_counts()
                            st.metric("Telescopes", len(instruments))
                            for inst, count in instruments.items():
                                st.write(f"• {inst}: {count}")
                    
                    with col2:
                        if 'freq_min_ghz' in df.columns and 'freq_max_ghz' in df.columns:
                            st.metric("Frequency Range", 
                                    f"{df['freq_min_ghz'].min():.1f}-{df['freq_max_ghz'].max():.1f} GHz")
                    
                    with col3:
                        if 'size_gb' in df.columns:
                            st.metric("Total Size", f"{df['size_gb'].sum():.1f} GB")
                    
                    # Show the actual data table
                    st.markdown("### 📊 Observation Data")
                    
                    # Select columns that exist
                    show_cols = []
                    for col in ['obs_publisher_did', 'target_name', 'instrument_name', 
                               'obs_date', 'freq_min_ghz', 'freq_max_ghz', 't_exptime', 
                               'size_gb', 'archive_url']:
                        if col in df.columns:
                            show_cols.append(col)
                    
                    if show_cols:
                        st.dataframe(df[show_cols], use_container_width=True)
                    else:
                        st.dataframe(df, use_container_width=True)
                    
                    # Archive links
                    if 'archive_url' in df.columns:
                        st.markdown("### 🔗 Direct Archive Links")
                        urls = df['archive_url'].dropna().unique()[:5]
                        for url in urls:
                            if url:
                                st.markdown(f"📎 [{url}]({url})")
                    
                    # Store data for download
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download as CSV",
                        data=csv,
                        file_name=f"{source_name}_observations.csv",
                        mime="text/csv"
                    )
                    
                else:
                    st.warning(f"No observations found for {source_name}")
                    
            except Exception as e:
                st.error(f"Error: {e}")
                st.info("Try a well-known source like '3C 273', 'M31', or 'Cygnus A'")
    else:
        st.warning("Please enter a source name")

# Quick examples in sidebar
with st.sidebar:
    st.markdown("### 💡 Quick Examples")
    
    example_sources = ["3C 273", "M31", "Cygnus A", "NGC 253", "Sagittarius A*"]
    
    for source in example_sources:
        if st.button(source, key=f"ex_{source}"):
            st.session_state.search_input = source
            st.rerun()
    
    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.markdown("""
    This searches:
    - **VLA** (Very Large Array)
    - **VLBA** (Very Long Baseline Array)  
    - **ALMA** (Atacama Large Millimeter Array)
    
    Data comes from NRAO's TAP service.
    """)

# Footer
st.markdown("---")
st.markdown("*Quasar - Radio Astronomy Data Discovery*")
