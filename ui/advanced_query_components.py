"""
Advanced Query UI Component for Quasar
Provides UI for advanced ALminer query types: keysearch, conesearch, catalog
"""

import streamlit as st
import pandas as pd
from typing import Optional, List

# Import query service
try:
    from services.alminer_query_service import (
        get_alminer_query_service,
        SCIENCE_CATEGORIES,
        COMMON_SCIENCE_KEYWORDS
    )
    QUERY_SERVICE_AVAILABLE = True
except ImportError:
    QUERY_SERVICE_AVAILABLE = False
    SCIENCE_CATEGORIES = []
    COMMON_SCIENCE_KEYWORDS = []


def render_advanced_query_panel(key_prefix: str = "adv_query") -> Optional[pd.DataFrame]:
    """
    Render advanced query panel in sidebar or main area
    
    Args:
        key_prefix: Widget key prefix
        
    Returns:
        DataFrame with results if query was executed, None otherwise
    """
    if not QUERY_SERVICE_AVAILABLE:
        st.warning("Advanced query service not available")
        return None
    
    query_service = get_alminer_query_service()
    
    st.markdown("### 🔬 Advanced Query")
    
    # Query type selection
    query_type = st.selectbox(
        "Query Type",
        [
            "🎯 Target Name",
            "📍 Position (RA/Dec)",
            "🔖 Science Category",
            "🏷️ Science Keyword",
            "📝 Proposal Abstract",
            "👤 PI Name",
            "📋 Proposal ID",
            "🔧 Custom keysearch"
        ],
        key=f"{key_prefix}_type"
    )
    
    result_df = None
    
    # Common options
    col1, col2 = st.columns(2)
    with col1:
        public_opt = st.selectbox(
            "Data Type",
            ["Public Only", "Proprietary Only", "All Data"],
            key=f"{key_prefix}_public"
        )
    with col2:
        published_opt = st.selectbox(
            "Published",
            ["All", "Published Only", "Unpublished Only"],
            key=f"{key_prefix}_published"
        )
    
    # Convert to API values
    public = True if public_opt == "Public Only" else (False if public_opt == "Proprietary Only" else None)
    published = True if published_opt == "Published Only" else (False if published_opt == "Unpublished Only" else None)
    
    # =========================================================================
    # TARGET NAME SEARCH
    # =========================================================================
    if query_type == "🎯 Target Name":
        target_name = st.text_input(
            "Target Name(s)",
            placeholder="e.g., M87, Sgr A*, Orion KL",
            help="Comma-separated for multiple targets. Uses SIMBAD/NED/VizieR resolution.",
            key=f"{key_prefix}_target"
        )
        
        search_radius = st.slider(
            "Search Radius (arcmin)",
            min_value=0.1,
            max_value=10.0,
            value=1.0,
            step=0.1,
            key=f"{key_prefix}_radius"
        )
        
        point_search = st.checkbox(
            "Point search (ignore radius)",
            value=True,
            key=f"{key_prefix}_point"
        )
        
        if st.button("🔍 Search", key=f"{key_prefix}_target_btn"):
            if target_name:
                targets = [t.strip() for t in target_name.split(",")]
                with st.spinner(f"Searching for {', '.join(targets)}..."):
                    result_df = query_service.target_search(
                        targets,
                        search_radius=search_radius,
                        point=point_search,
                        public=public,
                        published=published
                    )
    
    # =========================================================================
    # POSITION SEARCH
    # =========================================================================
    elif query_type == "📍 Position (RA/Dec)":
        col1, col2 = st.columns(2)
        with col1:
            ra = st.number_input(
                "RA (degrees)",
                min_value=0.0,
                max_value=360.0,
                value=266.42,  # Sgr A*
                step=0.01,
                key=f"{key_prefix}_ra"
            )
        with col2:
            dec = st.number_input(
                "Dec (degrees)",
                min_value=-90.0,
                max_value=90.0,
                value=-29.01,  # Sgr A*
                step=0.01,
                key=f"{key_prefix}_dec"
            )
        
        search_radius = st.slider(
            "Search Radius (arcmin)",
            min_value=0.1,
            max_value=60.0,
            value=1.0,
            step=0.1,
            key=f"{key_prefix}_cone_radius"
        )
        
        if st.button("🔍 Search", key=f"{key_prefix}_cone_btn"):
            with st.spinner(f"Searching around RA={ra:.3f}, Dec={dec:.3f}..."):
                result_df = query_service.conesearch(
                    ra=ra,
                    dec=dec,
                    search_radius=search_radius,
                    point=False,
                    public=public,
                    published=published
                )
    
    # =========================================================================
    # SCIENCE CATEGORY SEARCH
    # =========================================================================
    elif query_type == "🔖 Science Category":
        category = st.selectbox(
            "ALMA Science Category",
            SCIENCE_CATEGORIES if SCIENCE_CATEGORIES else ["Galaxy evolution"],
            key=f"{key_prefix}_category"
        )
        
        if st.button("🔍 Search", key=f"{key_prefix}_cat_btn"):
            with st.spinner(f"Searching category: {category}..."):
                result_df = query_service.search_by_category(category, public=public)
    
    # =========================================================================
    # SCIENCE KEYWORD SEARCH
    # =========================================================================
    elif query_type == "🏷️ Science Keyword":
        keyword = st.selectbox(
            "ALMA Science Keyword",
            COMMON_SCIENCE_KEYWORDS if COMMON_SCIENCE_KEYWORDS else ["High-mass star formation"],
            key=f"{key_prefix}_keyword"
        )
        
        if st.button("🔍 Search", key=f"{key_prefix}_kw_btn"):
            with st.spinner(f"Searching keyword: {keyword}..."):
                result_df = query_service.search_by_science_keyword(keyword, public=public)
    
    # =========================================================================
    # PROPOSAL ABSTRACT SEARCH
    # =========================================================================
    elif query_type == "📝 Proposal Abstract":
        abstract_query = st.text_area(
            "Search Terms",
            placeholder='e.g., "high-mass star formation" outflow disk\n\nUse quotes for exact phrases.',
            height=100,
            key=f"{key_prefix}_abstract"
        )
        
        st.caption("Words are combined with AND. Use quotes for phrases. Multiple values use OR.")
        
        if st.button("🔍 Search", key=f"{key_prefix}_abs_btn"):
            if abstract_query:
                with st.spinner("Searching proposal abstracts..."):
                    result_df = query_service.search_by_abstract([abstract_query], public=public)
    
    # =========================================================================
    # PI NAME SEARCH
    # =========================================================================
    elif query_type == "👤 PI Name":
        pi_name = st.text_input(
            "PI Name",
            placeholder="e.g., Smith",
            key=f"{key_prefix}_pi"
        )
        
        if st.button("🔍 Search", key=f"{key_prefix}_pi_btn"):
            if pi_name:
                with st.spinner(f"Searching PI: {pi_name}..."):
                    result_df = query_service.search_by_pi(pi_name, public=public)
    
    # =========================================================================
    # PROPOSAL ID SEARCH
    # =========================================================================
    elif query_type == "📋 Proposal ID":
        proposal_ids = st.text_input(
            "Proposal ID(s)",
            placeholder="e.g., 2023.1.00001.S, 2022.1.00002.S",
            help="Comma-separated for multiple IDs",
            key=f"{key_prefix}_proposal"
        )
        
        if st.button("🔍 Search", key=f"{key_prefix}_prop_btn"):
            if proposal_ids:
                ids = [p.strip() for p in proposal_ids.split(",")]
                with st.spinner(f"Searching proposals: {', '.join(ids)}..."):
                    result_df = query_service.search_by_proposal_id(ids, public=public)
    
    # =========================================================================
    # CUSTOM KEYSEARCH
    # =========================================================================
    elif query_type == "🔧 Custom keysearch":
        st.caption("Build a custom query with multiple keywords")
        
        # Keyword 1
        col1, col2 = st.columns([1, 2])
        with col1:
            key1 = st.selectbox(
                "Keyword 1",
                ["target_name", "proposal_abstract", "scientific_category", 
                 "science_keyword", "pi_name", "proposal_id", "pol_states"],
                key=f"{key_prefix}_key1"
            )
        with col2:
            val1 = st.text_input(
                "Value(s) 1",
                placeholder="Comma-separated values",
                key=f"{key_prefix}_val1"
            )
        
        # Keyword 2 (optional)
        col1, col2 = st.columns([1, 2])
        with col1:
            key2 = st.selectbox(
                "Keyword 2 (optional)",
                ["(none)", "target_name", "proposal_abstract", "scientific_category",
                 "science_keyword", "pi_name", "proposal_id", "pol_states"],
                key=f"{key_prefix}_key2"
            )
        with col2:
            val2 = st.text_input(
                "Value(s) 2",
                placeholder="Comma-separated values",
                key=f"{key_prefix}_val2"
            )
        
        if st.button("🔍 Search", key=f"{key_prefix}_custom_btn"):
            search_dict = {}
            if val1:
                search_dict[key1] = [v.strip() for v in val1.split(",")]
            if key2 != "(none)" and val2:
                search_dict[key2] = [v.strip() for v in val2.split(",")]
            
            if search_dict:
                with st.spinner("Running custom keysearch..."):
                    result_df = query_service.keysearch(search_dict, public=public, published=published)
    
    # =========================================================================
    # DISPLAY RESULTS
    # =========================================================================
    if result_df is not None:
        if not result_df.empty:
            st.success(f"✅ Found {len(result_df)} observations")
            
            # Store in session state for visualization
            st.session_state['advanced_query_results'] = result_df
            st.session_state['advanced_query_source'] = query_type
            
            # Show preview
            with st.expander("📊 Preview Results", expanded=True):
                display_cols = ['target_name', 'project_code', 'band', 's_ra', 's_dec']
                available_cols = [c for c in display_cols if c in result_df.columns]
                if available_cols:
                    st.dataframe(result_df[available_cols].head(20), use_container_width=True)
                else:
                    st.dataframe(result_df.head(20), use_container_width=True)
        else:
            st.warning("No observations found matching your criteria")
    
    return result_df


def render_query_sidebar() -> Optional[pd.DataFrame]:
    """
    Render compact query controls in sidebar
    
    Returns:
        DataFrame if query executed
    """
    if not QUERY_SERVICE_AVAILABLE:
        return None
    
    with st.sidebar:
        st.markdown("---")
        st.markdown("### 🔬 Advanced Search")
        
        query_type = st.radio(
            "Search by",
            ["Target", "Position", "Category", "Abstract"],
            key="sidebar_query_type",
            horizontal=True
        )
        
        query_service = get_alminer_query_service()
        result_df = None
        
        if query_type == "Target":
            target = st.text_input("Target name", key="sb_target")
            if st.button("Search", key="sb_target_btn") and target:
                with st.spinner("Searching..."):
                    result_df = query_service.target_search(target)
        
        elif query_type == "Position":
            ra = st.number_input("RA (°)", value=0.0, key="sb_ra")
            dec = st.number_input("Dec (°)", value=0.0, key="sb_dec")
            if st.button("Search", key="sb_pos_btn"):
                with st.spinner("Searching..."):
                    result_df = query_service.conesearch(ra, dec)
        
        elif query_type == "Category":
            cat = st.selectbox("Category", SCIENCE_CATEGORIES, key="sb_cat")
            if st.button("Search", key="sb_cat_btn"):
                with st.spinner("Searching..."):
                    result_df = query_service.search_by_category(cat)
        
        elif query_type == "Abstract":
            abstract = st.text_input("Keywords", key="sb_abstract")
            if st.button("Search", key="sb_abs_btn") and abstract:
                with st.spinner("Searching..."):
                    result_df = query_service.search_by_abstract([abstract])
        
        if result_df is not None and not result_df.empty:
            st.success(f"Found {len(result_df)} obs")
            st.session_state['advanced_query_results'] = result_df
        
        return result_df
