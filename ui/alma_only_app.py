#!/usr/bin/env python3
"""
SIMPLE ALMA-ONLY QUASAR APP - FOR YOUR DEMO TODAY
Only uses astroquery.alma which WE KNOW WORKS
"""

import streamlit as st
import pandas as pd
from astroquery.alma import Alma
from astropy.coordinates import SkyCoord
import astropy.units as u

# Page setup
st.set_page_config(page_title="Quasar - ALMA Data Search", page_icon="🔭", layout="wide")

# Dark theme
st.markdown("""
<style>
.stApp {
    background: linear-gradient(to bottom, #0a0e27 0%, #1a1e3a 100%);
}
h1, h2, h3 {
    color: white !important;
}
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("# 🌌 QUASAR - Radio Astronomy Intelligence System")
st.markdown("### ALMA Data Search Portal")

# Initialize ALMA
if 'alma' not in st.session_state:
    st.session_state.alma = Alma()

# SEARCH SECTION
st.markdown("---")
col1, col2 = st.columns([3, 1])

with col1:
    source_name = st.text_input(
        "🔍 Enter astronomical source name:",
        placeholder="e.g., 3C 273, M31, NGC 253",
        key="source"
    )

with col2:
    st.markdown("<br>", unsafe_allow_html=True)
    search_button = st.button("Search ALMA", type="primary", use_container_width=True)

# SEARCH LOGIC
if search_button and source_name:
    with st.spinner(f"🔭 Searching ALMA archives for **{source_name}**..."):
        try:
            # Search ALMA
            results = Alma.query_object(source_name, public=True, science=True)
            
            if results and len(results) > 0:
                # Convert to pandas
                df = results.to_pandas()
                
                # SUCCESS!
                st.success(f"✅ Found **{len(df)} ALMA observations** of {source_name}!")
                
                # Quick Stats
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if 'Band' in df.columns:
                        bands = df['Band'].value_counts()
                        st.metric("📡 Bands Observed", len(bands))
                        with st.expander("Band Details"):
                            for band, count in bands.items():
                                st.write(f"Band {band}: {count} obs")
                
                with col2:
                    if 'Integration' in df.columns:
                        total_time = df['Integration'].sum()
                        st.metric("⏱️ Total Integration", f"{total_time:.1f}s")
                
                with col3:
                    if 'Project code' in df.columns:
                        projects = df['Project code'].nunique()
                        st.metric("📂 Projects", projects)
                
                # Data Table
                st.markdown("### 📊 Observation Data")
                
                # Select important columns if they exist
                show_columns = []
                important_cols = ['Project code', 'Source name', 'RA', 'Dec', 'Band', 
                                'Integration', 'Release date', 'Frequency support', 
                                'Member ous id']
                
                for col in important_cols:
                    if col in df.columns:
                        show_columns.append(col)
                
                if show_columns:
                    display_df = df[show_columns]
                else:
                    display_df = df
                
                # Show data
                st.dataframe(display_df, use_container_width=True, height=400)
                
                # Archive Links
                st.markdown("### 🔗 Access Data")
                if 'Member ous id' in df.columns:
                    st.info("Click links below to access data in ALMA Science Archive:")
                    
                    # Show first 5 unique links
                    unique_ids = df['Member ous id'].dropna().unique()[:5]
                    for uid in unique_ids:
                        if pd.notna(uid):
                            url = f"https://almascience.nrao.edu/aq/?member_ous_id={uid}"
                            st.markdown(f"📎 [Observation: {uid}]({url})")
                
                # Download CSV
                csv = df.to_csv(index=False)
                st.download_button(
                    label="💾 Download Results as CSV",
                    data=csv,
                    file_name=f"ALMA_{source_name.replace(' ', '_')}_observations.csv",
                    mime="text/csv"
                )
                
            else:
                st.warning(f"No ALMA observations found for **{source_name}**")
                st.info("Try searching for: 3C 273, M31, NGC 253, or Cen A")
                
        except Exception as e:
            st.error(f"Search failed: {str(e)}")
            st.info("Make sure the source name is correct. Examples: '3C 273', 'M31', 'NGC 253'")

# EXAMPLES IN SIDEBAR
with st.sidebar:
    st.markdown("### 🌟 Popular Sources")
    st.markdown("Click to search:")
    
    examples = ["3C 273", "M31", "NGC 253", "Cen A", "Sgr A*", "M87", "NGC 1068"]
    
    for source in examples:
        if st.button(f"🔭 {source}", key=f"ex_{source}"):
            st.session_state.source = source
            st.rerun()
    
    st.markdown("---")
    st.markdown("### ℹ️ About ALMA")
    st.markdown("""
    **ALMA** (Atacama Large Millimeter/submillimeter Array) is the world's largest ground-based telescope, located in Chile.
    
    This app searches the ALMA Science Archive for public observations.
    """)
    
    st.markdown("---")
    st.markdown("### 📊 What You Get")
    st.markdown("""
    - Observation dates
    - Frequency bands
    - Integration times
    - Project codes
    - Direct archive links
    - Downloadable CSV
    """)

# WELCOME MESSAGE (if no search yet)
if not source_name:
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; padding: 2rem;'>
        <h3 style='color: #00d4ff;'>Welcome to Quasar! 🌟</h3>
        <p style='color: #a0a0a0;'>
            Search the ALMA archives for any astronomical source.<br>
            Try searching for <b>3C 273</b> or click an example in the sidebar.
        </p>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.caption("Quasar - Radio Astronomy Data Discovery | ALMA Archive Search")
