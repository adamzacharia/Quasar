"""
UI Components for Quasar Web Interface
Reusable Streamlit components and widgets
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
from typing import Optional, Dict, Any, List
import json

def create_metric_card(title: str, value: Any, delta: Optional[str] = None,
                       color: str = "blue") -> None:
    """Create a styled metric card"""
    col = st.container()
    with col:
        st.markdown(
            f"""
            <div style="
                background: linear-gradient(135deg, #{color}40 0%, #{color}20 100%);
                border-radius: 10px;
                padding: 1rem;
                border-left: 4px solid #{color};
            ">
                <p style="margin: 0; font-size: 0.9rem; color: #666;">{title}</p>
                <p style="margin: 0; font-size: 1.8rem; font-weight: bold;">{value}</p>
                {f'<p style="margin: 0; font-size: 0.8rem; color: #888;">{delta}</p>' if delta else ''}
            </div>
            """,
            unsafe_allow_html=True
        )

def create_search_form() -> Dict[str, Any]:
    """Create an advanced search form"""
    with st.form("advanced_search"):
        st.subheader("🔍 Advanced Search")

        col1, col2, col3 = st.columns(3)

        with col1:
            search_type = st.selectbox(
                "Search Type",
                ["Cone Search", "Target Name", "Frequency Range", "Project Code", "Custom ADQL"]
            )

            facility = st.multiselect(
                "Facilities",
                ["VLA", "VLBA", "ALMA", "GBT"],
                default=[]
            )

        with col2:
            date_from = st.date_input("Start Date", value=None)
            date_to = st.date_input("End Date", value=datetime.now())

        with col3:
            max_results = st.number_input(
                "Max Results",
                min_value=10,
                max_value=10000,
                value=100
            )

            sort_by = st.selectbox(
                "Sort By",
                ["Date (Newest)", "Date (Oldest)", "Size", "Duration"]
            )

        # Dynamic fields based on search type
        if search_type == "Cone Search":
            col1, col2, col3 = st.columns(3)
            with col1:
                ra = st.number_input("RA (degrees)", value=0.0, format="%.6f")
            with col2:
                dec = st.number_input("Dec (degrees)", value=0.0, format="%.6f")
            with col3:
                radius = st.number_input("Radius (degrees)", value=0.5, min_value=0.01, max_value=10.0)
            params = {"ra": ra, "dec": dec, "radius": radius}

        elif search_type == "Target Name":
            target = st.text_input("Target Name", placeholder="e.g., M31, Cygnus A, NGC 1234")
            resolve = st.checkbox("Resolve coordinates with Simbad")
            params = {"target": target, "resolve": resolve}

        elif search_type == "Frequency Range":
            col1, col2 = st.columns(2)
            with col1:
                min_freq = st.number_input("Min Frequency (GHz)", value=1.0, min_value=0.01)
            with col2:
                max_freq = st.number_input("Max Frequency (GHz)", value=2.0, min_value=0.01)
            params = {"min_freq": min_freq, "max_freq": max_freq}

        elif search_type == "Project Code":
            project = st.text_input("Project Code", placeholder="e.g., VLA/23A-001")
            params = {"project": project}

        else:  # Custom ADQL
            adql_query = st.text_area(
                "ADQL Query",
                placeholder="SELECT TOP 100 * FROM ivoa.obscore WHERE ...",
                height=100
            )
            params = {"query": adql_query}

        submitted = st.form_submit_button("Search", type="primary", use_container_width=True)

        if submitted:
            return {
                "type": search_type,
                "facilities": facility,
                "date_range": (date_from, date_to),
                "max_results": max_results,
                "sort_by": sort_by,
                "params": params
            }

    return None

def display_results_table(df: pd.DataFrame, key: str = "results") -> Optional[str]:
    """Display results in an interactive table with selection"""
    if df.empty:
        st.warning("No results to display")
        return None

    # Column selection
    available_cols = df.columns.tolist()
    default_cols = [
        'target_name', 'facility_name', 'obs_date',
        'freq_min_ghz', 'freq_max_ghz', 'duration_hours', 'size_gb'
    ]
    display_cols = [col for col in default_cols if col in available_cols]

    selected_cols = st.multiselect(
        "Display Columns",
        available_cols,
        default=display_cols,
        key=f"{key}_cols"
    )

    if not selected_cols:
        selected_cols = display_cols

    # Add selection column
    df_display = df[selected_cols].copy()

    # Display with selection
    selected_row = st.dataframe(
        df_display,
        use_container_width=True,
        height=400,
        hide_index=True,
        selection_mode="single-row",
        key=f"{key}_table"
    )

    # Return selected observation ID if any
    if selected_row and 'obs_publisher_did' in df.columns:
        selected_idx = selected_row.selection.rows[0] if selected_row.selection.rows else None
        if selected_idx is not None:
            return df.iloc[selected_idx]['obs_publisher_did']

    return None

def create_sky_plot(df: pd.DataFrame) -> go.Figure:
    """Create an interactive sky coverage plot"""
    if 's_ra' not in df.columns or 's_dec' not in df.columns:
        return None

    fig = go.Figure()

    # Add scatter plot for each facility
    if 'facility_name' in df.columns:
        for facility in df['facility_name'].unique():
            facility_data = df[df['facility_name'] == facility]

            # Prepare hover text
            hover_text = []
            for _, row in facility_data.iterrows():
                text = f"<b>{row.get('target_name', 'Unknown')}</b><br>"
                text += f"RA: {row['s_ra']:.3f}°<br>"
                text += f"Dec: {row['s_dec']:.3f}°<br>"
                if 'freq_min_ghz' in row:
                    text += f"Freq: {row['freq_min_ghz']:.1f}-{row.get('freq_max_ghz', 0):.1f} GHz<br>"
                if 'obs_date' in row:
                    text += f"Date: {row['obs_date']}"
                hover_text.append(text)

            fig.add_trace(go.Scatter(
                x=facility_data['s_ra'],
                y=facility_data['s_dec'],
                mode='markers',
                name=facility,
                marker=dict(
                    size=8,
                    opacity=0.7,
                    line=dict(width=1, color='white')
                ),
                hovertext=hover_text,
                hoverinfo='text'
            ))
    else:
        # Single trace if no facility info
        fig.add_trace(go.Scatter(
            x=df['s_ra'],
            y=df['s_dec'],
            mode='markers',
            marker=dict(size=8, color='blue', opacity=0.6),
            text=df.get('target_name', 'Unknown'),
            hovertemplate='<b>%{text}</b><br>RA: %{x:.3f}°<br>Dec: %{y:.3f}°'
        ))

    fig.update_layout(
        title="Sky Coverage (Equatorial Coordinates)",
        xaxis_title="Right Ascension (degrees)",
        yaxis_title="Declination (degrees)",
        height=500,
        hovermode='closest',
        showlegend=True,
        template="plotly_dark"
    )

    # Reverse x-axis (RA increases right to left in sky convention)
    fig.update_xaxes(autorange="reversed")

    return fig

def create_frequency_timeline(df: pd.DataFrame) -> go.Figure:
    """Create frequency vs time plot"""
    if 'obs_date' not in df.columns or 'freq_min_ghz' not in df.columns:
        return None

    # Prepare data
    df_plot = df.dropna(subset=['obs_date', 'freq_min_ghz'])

    if df_plot.empty:
        return None

    fig = go.Figure()

    # Add markers for observations
    if 'facility_name' in df_plot.columns:
        for facility in df_plot['facility_name'].unique():
            facility_data = df_plot[df_plot['facility_name'] == facility]

            fig.add_trace(go.Scatter(
                x=facility_data['obs_date'],
                y=facility_data['freq_min_ghz'],
                mode='markers',
                name=facility,
                marker=dict(
                    size=facility_data.get('duration_hours', 5) * 2,  # Size by duration
                    opacity=0.6,
                    line=dict(width=1, color='white')
                ),
                text=facility_data.get('target_name', 'Unknown'),
                hovertemplate='<b>%{text}</b><br>Date: %{x}<br>Freq: %{y:.2f} GHz'
            ))
    else:
        fig.add_trace(go.Scatter(
            x=df_plot['obs_date'],
            y=df_plot['freq_min_ghz'],
            mode='markers',
            marker=dict(size=8, color='blue', opacity=0.6),
            text=df_plot.get('target_name', 'Unknown'),
            hovertemplate='<b>%{text}</b><br>Date: %{x}<br>Freq: %{y:.2f} GHz'
        ))

    fig.update_layout(
        title="Observation Timeline",
        xaxis_title="Observation Date",
        yaxis_title="Frequency (GHz)",
        height=400,
        hovermode='closest',
        showlegend=True
    )

    return fig

def create_observation_card(obs_details: Dict[str, Any]) -> None:
    """Create a detailed observation information card"""
    st.markdown("### 📡 Observation Details")

    # Basic info in columns
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Target**")
        st.write(obs_details.get('target_name', 'Unknown'))

        st.markdown("**Facility**")
        st.write(obs_details.get('facility_name', 'Unknown'))

        st.markdown("**Configuration**")
        st.write(obs_details.get('configuration', 'N/A'))

    with col2:
        st.markdown("**Frequency Range**")
        if 'freq_ghz' in obs_details:
            freq = obs_details['freq_ghz']
            st.write(f"{freq['min']:.2f} - {freq['max']:.2f} GHz")
        else:
            st.write("N/A")

        st.markdown("**Polarization**")
        st.write(obs_details.get('pol_states', 'N/A'))

        st.markdown("**Antennas**")
        st.write(obs_details.get('num_antennas', 'N/A'))

    with col3:
        st.markdown("**Time Range**")
        if 'time_range' in obs_details:
            time_info = obs_details['time_range']
            st.write(f"{time_info['start']} to {time_info['end']}")
            st.write(f"Duration: {time_info['duration_hours']:.2f} hours")
        else:
            st.write("N/A")

        st.markdown("**Data Size**")
        st.write(f"{obs_details.get('size_gb', 0):.2f} GB")

    # Additional details in expander
    with st.expander("More Details"):
        st.json(obs_details)

def create_download_button(obs_id: str, size_gb: float = 0) -> None:
    """Create a download button with warning for large files"""
    if size_gb > 10:
        st.warning(f"⚠️ Large file ({size_gb:.1f} GB). Download may take significant time.")

    col1, col2 = st.columns([3, 1])
    with col1:
        st.text_input("Download Directory", value="./data", key=f"download_dir_{obs_id}")
    with col2:
        if st.button("📥 Download", key=f"download_btn_{obs_id}"):
            st.info(f"Download functionality requires implementation of DataLink client")
            st.code(f"# Command to download:\ncurl -O https://data.nrao.edu/download/{obs_id}")

def create_pipeline_options() -> Dict[str, Any]:
    """Create pipeline configuration options"""
    with st.expander("🔧 Pipeline Configuration", expanded=False):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Flagging Options**")
            flag_autocorr = st.checkbox("Flag autocorrelations", value=True)
            flag_edge = st.checkbox("Flag edge channels", value=True)
            edge_percent = st.slider("Edge %", 0, 20, 5) if flag_edge else 0

            st.markdown("**Calibration**")
            cal_type = st.selectbox(
                "Calibration Type",
                ["Standard", "Bandpass only", "Phase only", "Full"]
            )

        with col2:
            st.markdown("**Imaging Parameters**")
            weighting = st.selectbox("Weighting", ["natural", "uniform", "briggs"])
            robust = st.slider("Robust", -2.0, 2.0, 0.5) if weighting == "briggs" else 0.5

            cell_size = st.number_input("Cell size (arcsec)", value=1.0, min_value=0.01)
            image_size = st.number_input("Image size (pixels)", value=1024, min_value=64, step=64)

        return {
            "flagging": {
                "autocorr": flag_autocorr,
                "edge": flag_edge,
                "edge_percent": edge_percent
            },
            "calibration": {
                "type": cal_type
            },
            "imaging": {
                "weighting": weighting,
                "robust": robust,
                "cell": f"{cell_size}arcsec",
                "imsize": image_size
            }
        }

def create_status_indicator(status: str, message: str = "") -> None:
    """Create a status indicator with color coding"""
    colors = {
        "success": "green",
        "running": "blue",
        "warning": "orange",
        "error": "red",
        "idle": "gray"
    }

    color = colors.get(status, "gray")
    icon = {
        "success": "✅",
        "running": "🔄",
        "warning": "⚠️",
        "error": "❌",
        "idle": "⭕"
    }.get(status, "⭕")

    st.markdown(
        f"""
        <div style="
            padding: 0.5rem;
            border-radius: 5px;
            background-color: {color}22;
            border-left: 4px solid {color};
        ">
            <span style="font-size: 1.2rem;">{icon}</span>
            <span style="margin-left: 0.5rem;">{message}</span>
        </div>
        """,
        unsafe_allow_html=True
    )

def render_help_section():
    """Render help and documentation section"""
    with st.expander("📚 Help & Documentation"):
        st.markdown("""
        ### Quick Start Guide

        **1. Search for Data:**
        - Use natural language in the chat interface
        - Or use the direct search panel for specific queries

        **2. Common Search Examples:**
        - `"Find VLA observations of M31"`
        - `"Search for pulsars at 1.4 GHz"`
        - `"Show me all data from project VLA/23A-001"`

        **3. ADQL Query Examples:**
        ```sql
        -- Find bright sources
        SELECT TOP 100 * FROM ivoa.obscore
        WHERE facility_name='VLA' AND flux > 1.0

        -- Search by date range
        SELECT * FROM ivoa.obscore
        WHERE t_min > 59000 AND t_max < 60000
        ```

        **4. Keyboard Shortcuts:**
        - `Ctrl + K`: Clear chat
        - `Ctrl + Enter`: Submit query

        **5. Need Help?**
        - Check the [NRAO Archive Documentation](https://science.nrao.edu/facilities/vla/archive/index)
        - View [TAP Service Documentation](https://www.ivoa.net/documents/TAP/)
        """)

def create_export_options(data: pd.DataFrame, filename_prefix: str = "quasar_export"):
    """Create data export options"""
    col1, col2, col3 = st.columns(3)

    with col1:
        # CSV export
        csv = data.to_csv(index=False)
        st.download_button(
            label="📄 Download CSV",
            data=csv,
            file_name=f"{filename_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

    with col2:
        # JSON export
        json_str = data.to_json(orient='records', indent=2)
        st.download_button(
            label="📋 Download JSON",
            data=json_str,
            file_name=f"{filename_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )

    with col3:
        # VOTable export (placeholder)
        st.button("🌟 Export as VOTable", disabled=True, help="Coming soon")

def create_filter_sidebar(df: pd.DataFrame) -> pd.DataFrame:
    """Create filtering options in sidebar"""
    st.sidebar.markdown("### 🔽 Filters")

    filtered_df = df.copy()

    # Facility filter
    if 'facility_name' in df.columns:
        facilities = st.sidebar.multiselect(
            "Facilities",
            df['facility_name'].unique(),
            default=df['facility_name'].unique()
        )
        filtered_df = filtered_df[filtered_df['facility_name'].isin(facilities)]

    # Frequency filter
    if 'freq_min_ghz' in df.columns and 'freq_max_ghz' in df.columns:
        freq_min = float(df['freq_min_ghz'].min())
        freq_max = float(df['freq_max_ghz'].max())

        freq_range = st.sidebar.slider(
            "Frequency Range (GHz)",
            freq_min, freq_max,
            (freq_min, freq_max)
        )
        filtered_df = filtered_df[
            (filtered_df['freq_min_ghz'] >= freq_range[0]) &
            (filtered_df['freq_max_ghz'] <= freq_range[1])
        ]

    # Size filter
    if 'size_gb' in df.columns:
        max_size = st.sidebar.number_input(
            "Max Size (GB)",
            min_value=0.0,
            value=float(df['size_gb'].max()),
            step=1.0
        )
        filtered_df = filtered_df[filtered_df['size_gb'] <= max_size]

    # Date filter
    if 'obs_date' in df.columns:
        df['obs_date'] = pd.to_datetime(df['obs_date'])
        date_min = df['obs_date'].min()
        date_max = df['obs_date'].max()

        date_range = st.sidebar.date_input(
            "Date Range",
            value=(date_min, date_max),
            min_value=date_min,
            max_value=date_max
        )

        if len(date_range) == 2:
            filtered_df = filtered_df[
                (filtered_df['obs_date'] >= pd.Timestamp(date_range[0])) &
                (filtered_df['obs_date'] <= pd.Timestamp(date_range[1]))
            ]

    # Show filter stats
    st.sidebar.markdown(f"**Filtered: {len(filtered_df)}/{len(df)} observations**")

    return filtered_df