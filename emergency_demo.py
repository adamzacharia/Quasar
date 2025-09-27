#!/usr/bin/env python3
"""
EMERGENCY STANDALONE DEMO - Copy this ONE file anywhere and run it!
No dependencies on your project structure.
"""

import streamlit as st
from astroquery.alma import Alma

st.set_page_config(page_title="ALMA Search Demo", page_icon="🔭", layout="wide")

st.title("🔭 ALMA Telescope Data Search")

# Search box
source = st.text_input("Enter source name (e.g., 3C 273):", "3C 273")

if st.button("Search ALMA"):
    with st.spinner(f"Searching for {source}..."):
        try:
            results = Alma.query_object(source, public=True, science=True)
            if results and len(results) > 0:
                df = results.to_pandas()
                st.success(f"Found {len(df)} observations!")
                st.dataframe(df)
                
                # Download button
                csv = df.to_csv(index=False)
                st.download_button("Download CSV", csv, f"{source}_alma.csv", "text/csv")
            else:
                st.warning("No observations found")
        except Exception as e:
            st.error(f"Error: {e}")

# Examples
st.sidebar.markdown("### Try these sources:")
for src in ["3C 273", "M31", "NGC 253", "Cen A"]:
    if st.sidebar.button(src):
        st.session_state.source = src
        st.rerun()
