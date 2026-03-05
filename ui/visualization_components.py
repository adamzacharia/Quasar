"""
ALminer Visualization UI Components for Quasar
Streamlit components using alminer's native plotting functions
"""

import streamlit as st
import pandas as pd
from typing import Optional, Dict, Any, List

# Import visualization service
try:
    from services.visualization_service import get_alminer_visualization_service, COMMON_LINES
    VIZ_SERVICE_AVAILABLE = True
except ImportError:
    VIZ_SERVICE_AVAILABLE = False
    COMMON_LINES = {}


def render_visualization_panel(df: pd.DataFrame, key_prefix: str = "viz") -> None:
    """
    Render the ALminer visualization panel with tabs for different plot types
    
    Args:
        df: DataFrame with observation data (from alminer query)
        key_prefix: Prefix for widget keys to avoid conflicts
    """
    if df.empty:
        st.info("📊 No data to visualize. Run a search first!")
        return
    
    if not VIZ_SERVICE_AVAILABLE:
        st.error("Visualization service not available. Please check imports.")
        return
    
    viz = get_alminer_visualization_service()
    
    st.markdown("### 📈 ALminer Visualizations")
    
    # Create tabs for different visualization types
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Overview",
        "📡 Bands",
        "🔭 Observations",
        "🌌 Sky Map",
        "🔬 Line Check",
        "📥 Export/Download"
    ])
    
    # =========================================================================
    # TAB 1: Overview Plot
    # =========================================================================
    with tab1:
        st.markdown("#### Observation Overview")
        st.caption("Frequency distribution, angular resolution, LAS, and velocity resolution")
        
        if st.button("🔄 Generate Overview Plot", key=f"{key_prefix}_overview_btn"):
            with st.spinner("Generating overview plot..."):
                img_bytes = viz.plot_overview(df)
                if img_bytes:
                    st.image(img_bytes, caption="ALminer Overview Plot", use_container_width=True)
                else:
                    st.warning("Could not generate overview plot.")
        
        # Show summary stats
        stats = viz.get_summary_stats(df)
        if stats:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Observations", stats.get("n_observations", 0))
            with col2:
                st.metric("Projects", stats.get("n_projects", 0))
            with col3:
                st.metric("Targets", stats.get("n_targets", 0))
    
    # =========================================================================
    # TAB 2: Bands Plot
    # =========================================================================
    with tab2:
        st.markdown("#### Frequency Coverage by Band")
        st.caption("Detailed view of observed frequencies in each ALMA band")
        
        col1, col2 = st.columns(2)
        
        with col1:
            mark_co = st.checkbox("Mark CO lines", value=True, key=f"{key_prefix}_bands_co")
        
        with col2:
            redshift = st.number_input(
                "Redshift (z)",
                min_value=0.0,
                max_value=10.0,
                value=0.0,
                step=0.01,
                key=f"{key_prefix}_bands_z"
            )
        
        # Custom frequency marking
        custom_freq_str = st.text_input(
            "Mark additional frequencies (comma-separated GHz)",
            placeholder="e.g., 115.27, 230.54, 345.80",
            key=f"{key_prefix}_bands_custom"
        )
        
        custom_freqs = None
        if custom_freq_str:
            try:
                custom_freqs = [float(f.strip()) for f in custom_freq_str.split(",") if f.strip()]
            except:
                st.warning("Invalid frequency format")
        
        if st.button("🔄 Generate Band Plot", key=f"{key_prefix}_bands_btn"):
            with st.spinner("Generating band plot..."):
                img_bytes = viz.plot_bands(df, mark_CO=mark_co, z=redshift, mark_freq=custom_freqs)
                if img_bytes:
                    st.image(img_bytes, caption="Frequency Coverage by Band", use_container_width=True)
                else:
                    st.warning("Could not generate band plot.")
    
    # =========================================================================
    # TAB 3: Observations Plot
    # =========================================================================
    with tab3:
        st.markdown("#### Individual Observation Frequencies")
        st.caption("Exact frequency ranges for each observation")
        
        col1, col2 = st.columns(2)
        
        with col1:
            mark_co_obs = st.checkbox("Mark CO lines", value=True, key=f"{key_prefix}_obs_co")
        
        with col2:
            redshift_obs = st.number_input(
                "Redshift (z)",
                min_value=0.0,
                max_value=10.0,
                value=0.0,
                step=0.01,
                key=f"{key_prefix}_obs_z"
            )
        
        if st.button("🔄 Generate Observation Plot", key=f"{key_prefix}_obs_btn"):
            with st.spinner("Generating observation plot..."):
                img_bytes = viz.plot_observations(df, mark_CO=mark_co_obs, z=redshift_obs)
                if img_bytes:
                    st.image(img_bytes, caption="Observation Frequency Ranges", use_container_width=True)
                else:
                    st.warning("Could not generate observation plot.")
    
    # =========================================================================
    # TAB 4: Sky Distribution
    # =========================================================================
    with tab4:
        st.markdown("#### Sky Distribution")
        st.caption("Distribution of targets on the sky")
        
        if st.button("🔄 Generate Sky Plot", key=f"{key_prefix}_sky_btn"):
            with st.spinner("Generating sky plot..."):
                img_bytes = viz.plot_sky(df)
                if img_bytes:
                    st.image(img_bytes, caption="Sky Distribution", use_container_width=True)
                else:
                    st.warning("Could not generate sky plot.")
    
    # =========================================================================
    # TAB 5: Line Coverage Check
    # =========================================================================
    with tab5:
        st.markdown("#### Spectral Line Coverage")
        st.caption("Check which observations cover specific spectral lines")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Line selection
            line_options = list(COMMON_LINES.keys()) if COMMON_LINES else ["CO(1-0)", "CO(2-1)", "CO(3-2)"]
            selected_line = st.selectbox(
                "Select Line",
                line_options,
                key=f"{key_prefix}_line_select"
            )
            
            # Get frequency for selected line
            line_freq = COMMON_LINES.get(selected_line, 115.27) if COMMON_LINES else 115.27
            st.caption(f"Rest frequency: {line_freq:.3f} GHz")
        
        with col2:
            line_z = st.number_input(
                "Redshift (z)",
                min_value=0.0,
                max_value=10.0,
                value=0.0,
                step=0.01,
                key=f"{key_prefix}_line_z"
            )
            
            obs_freq = line_freq / (1 + line_z)
            st.caption(f"Observed frequency: {obs_freq:.3f} GHz")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔍 Check Line Coverage", key=f"{key_prefix}_line_check_btn"):
                with st.spinner("Checking line coverage..."):
                    coverage_df = viz.check_line_coverage(df, line_freq, z=line_z, line_name=selected_line)
                    
                    if not coverage_df.empty:
                        st.success(f"✅ {len(coverage_df)} observations cover {selected_line}")
                        st.dataframe(coverage_df[['target_name', 'project_code', 'band']].head(20) 
                                    if all(c in coverage_df.columns for c in ['target_name', 'project_code', 'band'])
                                    else coverage_df.head(20),
                                    use_container_width=True)
                    else:
                        st.warning(f"No observations cover {selected_line} at z={line_z}")
        
        with col2:
            if st.button("🔍 Check CO Coverage", key=f"{key_prefix}_co_check_btn"):
                with st.spinner("Checking CO line coverage..."):
                    co_df = viz.check_CO_coverage(df, z=line_z)
                    
                    if not co_df.empty:
                        st.success(f"✅ {len(co_df)} observations have CO line coverage")
                    else:
                        st.warning("No CO line coverage found")
        
        # Line overview plot
        st.markdown("---")
        if st.button("📊 Generate Line Overview Plot", key=f"{key_prefix}_line_plot_btn"):
            with st.spinner("Generating line overview plot..."):
                img_bytes = viz.plot_line_overview(df, line_freq=line_freq, z=line_z)
                if img_bytes:
                    st.image(img_bytes, caption=f"Observations covering {selected_line}", use_container_width=True)
                else:
                    st.warning("Could not generate line overview plot.")
    
    # =========================================================================
    # TAB 6: Export/Download
    # =========================================================================
    with tab6:
        st.markdown("#### Export & Download")
        st.caption("Save tables, generate reports, and download data from ALMA archive")
        
        # Section 1: Save Table
        st.markdown("##### 📄 Export Results Table")
        col1, col2 = st.columns([2, 1])
        
        with col1:
            table_filename = st.text_input(
                "Filename (without extension)",
                value="alma_observations",
                key=f"{key_prefix}_table_filename"
            )
        
        with col2:
            if st.button("💾 Save CSV", key=f"{key_prefix}_save_table_btn"):
                with st.spinner("Saving table..."):
                    filepath = viz.save_table(df, filename=table_filename)
                    if filepath:
                        st.success(f"✅ Saved to `{filepath}`")
                    else:
                        st.error("Failed to save table")
        
        st.markdown("---")
        
        # Section 2: Source Reports
        st.markdown("##### 📊 Generate Source Reports")
        st.caption("Create overview plots for each target (saved as PDFs in 'reports' folder)")
        
        col1, col2 = st.columns(2)
        
        with col1:
            report_mark_co = st.checkbox("Mark CO lines", value=True, key=f"{key_prefix}_report_co")
        
        with col2:
            report_z = st.number_input(
                "Redshift",
                min_value=0.0,
                max_value=10.0,
                value=0.0,
                step=0.01,
                key=f"{key_prefix}_report_z"
            )
        
        if st.button("📊 Generate All Source Reports", key=f"{key_prefix}_gen_reports_btn"):
            with st.spinner("Generating reports for each source..."):
                reports = viz.save_source_reports(df, mark_CO=report_mark_co, z=report_z)
                if reports:
                    st.success(f"✅ Generated {len(reports)} reports in 'reports' folder")
                else:
                    st.warning("No reports generated")
        
        st.markdown("---")
        
        # Section 3: Download Data
        st.markdown("##### 📥 Download ALMA Data")
        st.warning("⚠️ Downloading raw ALMA data can require significant disk space!")
        
        col1, col2 = st.columns(2)
        
        with col1:
            download_location = st.text_input(
                "Download location",
                value="./data",
                key=f"{key_prefix}_download_loc"
            )
            
            fits_only = st.checkbox(
                "FITS only (smaller size)",
                value=True,
                key=f"{key_prefix}_fits_only"
            )
        
        with col2:
            archive_mirror = st.selectbox(
                "Archive mirror",
                ["ESO", "NRAO", "NAOJ"],
                key=f"{key_prefix}_archive_mirror"
            )
            
            filename_filter = st.text_input(
                "Filename must include (comma-sep)",
                placeholder="e.g., _sci, .pbcor, cont",
                key=f"{key_prefix}_filename_filter"
            )
        
        filter_list = None
        if filename_filter:
            filter_list = [f.strip() for f in filename_filter.split(",") if f.strip()]
        
        if st.button("🔍 Check Download Size (Dry Run)", key=f"{key_prefix}_dryrun_btn"):
            with st.spinner("Checking required disk space..."):
                result = viz.download_data(
                    df,
                    location=download_location,
                    fitsonly=fits_only,
                    dryrun=True,
                    archive_mirror=archive_mirror,
                    filename_must_include=filter_list
                )
                
                if "error" in result:
                    st.error(f"Error: {result['error']}")
                else:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Disk Space Required", result.get("disk_space", "Unknown"))
                    with col2:
                        st.metric("Files to Download", result.get("file_count", "Unknown"))
                    with col3:
                        st.metric("Member OUSs", result.get("mous_count", "Unknown"))
                    
                    with st.expander("Raw output"):
                        st.text(result.get("raw_output", ""))
        
        # Summary section
        st.markdown("---")
        st.markdown("##### 📋 ALminer Summary")
        if st.button("Show Summary", key=f"{key_prefix}_summary_btn"):
            summary = viz.get_summary(df)
            st.text(summary)


def render_quick_visualization_buttons(df: pd.DataFrame, container=None) -> None:
    """
    Render quick visualization buttons inline with search results
    
    Args:
        df: DataFrame with observation data
        container: Streamlit container to render in (default: st)
    """
    if df.empty or not VIZ_SERVICE_AVAILABLE:
        return
    
    ctx = container if container else st
    viz = get_alminer_visualization_service()
    
    ctx.markdown("##### 📊 Quick Plots")
    
    col1, col2, col3 = ctx.columns(3)
    
    with col1:
        if ctx.button("📈 Overview", key="quick_overview"):
            with st.spinner("Generating..."):
                img_bytes = viz.plot_overview(df)
                if img_bytes:
                    st.session_state['quick_viz_img'] = img_bytes
                    st.session_state['quick_viz_caption'] = "Overview"
    
    with col2:
        if ctx.button("📡 Bands", key="quick_bands"):
            with st.spinner("Generating..."):
                img_bytes = viz.plot_bands(df, mark_CO=True)
                if img_bytes:
                    st.session_state['quick_viz_img'] = img_bytes
                    st.session_state['quick_viz_caption'] = "Frequency Bands"
    
    with col3:
        if ctx.button("🌌 Sky", key="quick_sky"):
            with st.spinner("Generating..."):
                img_bytes = viz.plot_sky(df)
                if img_bytes:
                    st.session_state['quick_viz_img'] = img_bytes
                    st.session_state['quick_viz_caption'] = "Sky Distribution"
    
    # Display quick viz if available
    if 'quick_viz_img' in st.session_state and st.session_state['quick_viz_img']:
        st.image(st.session_state['quick_viz_img'], 
                caption=st.session_state.get('quick_viz_caption', ''),
                use_container_width=True)
