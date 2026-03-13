"""
Quasar Web Interface - Complete with ALL Features
"""

import streamlit as st
import pandas as pd
import os
import sys
from pathlib import Path
import json
import re
from dotenv import load_dotenv
from openai import OpenAI
import requests
from typing import Dict, List, Optional
import types

# -----------------------------------------------------------------------------
# CRITICAL FIX FOR PYTHON 3.13 (Missing cgi module required by pyvo/astropy)
# -----------------------------------------------------------------------------
if sys.version_info >= (3, 13):
    if "cgi" not in sys.modules:
        mock_cgi = types.ModuleType("cgi")
        mock_cgi.parse_header = lambda x: (x, {})  # Minimal mock for pyvo
        sys.modules["cgi"] = mock_cgi
# -----------------------------------------------------------------------------

# Add project root to path for imports - FORCE ABSOLUTE PATH
root_path = Path(__file__).resolve().parent.parent
if str(root_path) not in sys.path:
    sys.path.append(str(root_path))

from core.prompts import ENTITY_EXTRACTION_PROMPT
from services.rag_service import RAGService
from services.conversation_service import ConversationService

# Import visualization components
try:
    from ui.visualization_components import render_visualization_panel, render_quick_visualization_buttons
    VIZ_COMPONENTS_AVAILABLE = True
except ImportError:
    VIZ_COMPONENTS_AVAILABLE = False

# Import advanced query components
try:
    from ui.advanced_query_components import render_advanced_query_panel
    ADV_QUERY_AVAILABLE = True
except ImportError:
    ADV_QUERY_AVAILABLE = False

from services.auth import AuthService

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Quasar - Radio Astronomy Assistant",
    page_icon="🔭",
    layout="wide",
    initial_sidebar_state="expanded"  # Show document upload sidebar
)


# ... (rest of the file) ...

# [OMITTED: Sidebar code removed]

# NASA ADS Client - Fixed with simple queries that work


# Hide Streamlit branding
hide_st_style = """
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* ============================================================
       PLAYGROUND-STYLE THEME - Solid Dark Teal
       ============================================================ */
    
    /* Main background - solid dark navy */
    .stApp {
        background: #0a1628 !important;
    }
    
    /* Remove any pseudo-elements */
    .stApp::before, .stApp::after {
        display: none !important;
    }
    
    /* SIDEBAR STYLING - 1.5x wider, lighter navy */
    [data-testid="stSidebar"] {
        display: block !important;
        min-width: 350px !important;
        width: 350px !important;
        background: #112340 !important;
        border-right: 1px solid rgba(255, 255, 255, 0.08) !important;
    }
    
    /* Hide collapse button */
    [data-testid="stSidebar"] [data-testid="stSidebarCollapseButton"] {
        display: none !important;
    }
    
    /* Sidebar content */
    [data-testid="stSidebar"] > div {
        background: transparent !important;
        padding: 0.5rem 1rem !important;
    }
    
    /* Sidebar text */
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #f472b6 !important;
    }
    
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] label {
        color: #94a3b8 !important;
    }
    
    [data-testid="stSidebar"] hr {
        border-color: rgba(255, 255, 255, 0.1) !important;
    }
    
    /* New Chat button - pink accent */
    [data-testid="stSidebar"] button[kind="primary"] {
        background: #f472b6 !important;
        color: #0a1628 !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
    }
    
    /* Regular buttons */
    [data-testid="stSidebar"] button {
        background: transparent !important;
        color: #94a3b8 !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        border-radius: 8px !important;
    }
    
    /* MAIN AREA */
    .main > div {
        background: transparent !important;
    }
    
    /* Text colors */
    .stMarkdown {
        color: #e2e8f0 !important;
    }
    
    h1, h2, h3, h4, h5, h6 {
        color: #ffffff !important;
    }
    
    p {
        color: #ffffff !important;
    }
    
    /* Model selector dropdown */
    [data-testid="stSelectbox"] {
        max-width: 250px !important;
    }
    
    [data-testid="stSelectbox"] > div > div {
        background: #1a3a5c !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        border-radius: 8px !important;
        color: white !important;
    }
    
    /* Tool calling card */
    .tool-card {
        background: rgba(255, 255, 255, 0.02) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 12px !important;
        padding: 1.5rem !important;
        max-width: 400px !important;
        margin: 1rem auto !important;
    }
    
    /* Chat input - floating modern style */
    [data-testid="stBottom"] {
        background: transparent !important;
        padding: 1rem 2rem !important;
    }
    
    [data-testid="stBottom"] > div {
        background: transparent !important;
    }
    
    [data-testid="stChatInput"] {
        background: rgba(26, 58, 92, 0.8) !important;
        border: 1px solid rgba(255, 255, 255, 0.15) !important;
        border-radius: 28px !important;
        backdrop-filter: blur(10px) !important;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3) !important;
    }
    
    [data-testid="stChatInput"]:focus-within {
        border-color: #f472b6 !important;
        box-shadow: 0 4px 25px rgba(244, 114, 182, 0.2) !important;
    }
    
    [data-testid="stChatInput"] textarea {
        background: transparent !important;
        color: white !important;
    }
    
    [data-testid="stChatInput"] textarea::placeholder {
        color: #64748b !important;
    }
    
    /* Chat messages */
    .stChatMessage {
        background: rgba(255, 255, 255, 0.03) !important;
        border-radius: 12px !important;
    }
    
    /* Links */
    a {
        color: #f472b6 !important;
        text-decoration: none !important;
    }
    
    a:hover {
        text-decoration: underline !important;
    }
    
    /* Expander styling */
    .stExpander {
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 8px !important;
        background: rgba(255, 255, 255, 0.02) !important;
    }
    
    /* File uploader styling - make visible */
    [data-testid="stFileUploader"] {
        background: rgba(255, 255, 255, 0.03) !important;
        border: 1px dashed rgba(255, 255, 255, 0.2) !important;
        border-radius: 8px !important;
        padding: 0.75rem !important;
    }
    
    [data-testid="stFileUploader"] label {
        color: #94a3b8 !important;
    }
    
    [data-testid="stFileUploader"] button {
        background: #f472b6 !important;
        color: #0a1628 !important;
    }
    
    /* External links styling */
    .external-link {
        color: #f472b6 !important;
        font-size: 1.1rem !important;
        padding: 0.25rem 0 !important;
        display: block !important;
    }
    
    /* Logo text */
    .logo-text {
        color: #f472b6 !important;
        font-size: 1.5rem !important;
        font-weight: 600 !important;
        margin-top: 0.5rem !important;
    }
    
    /* Footer text */
    .footer-text {
        color: #64748b !important;
        font-size: 0.75rem !important;
        text-align: center !important;
    }
</style>
"""

st.markdown(hide_st_style, unsafe_allow_html=True)

def render_sidebar_logo():
    """Render logo and QUASAR text in sidebar using SVG"""
    # SVG Logo (Quasar stylized Q)
    logo_svg = """
    <svg viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
        <defs>
            <linearGradient id="grad1" x1="0%" y1="0%" x2="100%" y2="0%">
                <stop offset="0%" style="stop-color:#f472b6;stop-opacity:1" />
                <stop offset="100%" style="stop-color:#c084fc;stop-opacity:1" />
            </linearGradient>
            <filter id="glow">
                <feGaussianBlur stdDeviation="2.5" result="coloredBlur"/>
                <feMerge>
                    <feMergeNode in="coloredBlur"/>
                    <feMergeNode in="SourceGraphic"/>
                </feMerge>
            </filter>
        </defs>
        <circle cx="50" cy="50" r="35" stroke="url(#grad1)" stroke-width="6" fill="transparent" filter="url(#glow)"/>
        <line x1="68" y1="68" x2="88" y2="88" stroke="url(#grad1)" stroke-width="6" stroke-linecap="round" filter="url(#glow)"/>
        <circle cx="50" cy="50" r="15" fill="#f472b6" opacity="0.8">
            <animate attributeName="opacity" values="0.8;0.4;0.8" dur="3s" repeatCount="indefinite" />
        </circle>
    </svg>
    """
    
    st.markdown(f"""
    <div style='text-align: center; padding: 0.5rem 0 1rem 0;'>
        <div style="width: 80px; height: 80px; margin: 0 auto;">
            {logo_svg}
        </div>
        <div style='color: #f472b6; font-size: 1.8rem; font-weight: 700; margin-top: 0.5rem; letter-spacing: 3px; text-shadow: 0 0 10px rgba(244, 114, 182, 0.3);'>QUASAR</div>
    </div>
    """, unsafe_allow_html=True)


def render_model_selector_top():
    """Render model selector at top left of main area (like Playground)"""
    available_models = [
        # ── Anthropic Claude (Latest) ──────────────────────────────
        "claude-sonnet-4-6",       # Latest Sonnet — best coding, default free/pro
        "claude-opus-4-6",         # Latest Opus — smartest, 1M context (beta)
        "claude-sonnet-4-5",       # Sonnet 4.5 — coding/computer use
        "claude-opus-4-5",         # Opus 4.5 — best coding & agents
        "claude-haiku-4-5",        # Haiku 4.5 — fastest & cheapest
        "claude-opus-4-1",         # Opus 4.1 — agentic tasks
        "claude-sonnet-4",         # Sonnet 4
        "claude-opus-4",           # Opus 4
        "claude-3-7-sonnet-20250219",  # Claude 3.7 Sonnet — hybrid reasoning
        "claude-3-5-sonnet-20241022",  # Claude 3.5 Sonnet (Oct 2024)
        "claude-3-5-haiku-20241022",   # Claude 3.5 Haiku
        # ── OpenAI GPT (Latest) ───────────────────────────────────
        "gpt-5.4",                 # GPT-5.4 — latest flagship (Mar 2026)
        "gpt-5.4-2026-03-05",      # GPT-5.4 snapshot
        "gpt-5",                   # GPT-5
        "gpt-5-mini",              # GPT-5 Mini — lower latency
        "gpt-5-nano",              # GPT-5 Nano — cheapest
        "gpt-4o",                  # GPT-4o — omni flagship
        "gpt-4o-mini",             # GPT-4o Mini — fast & cheap
        "gpt-4.1",                 # GPT-4.1 — coding & long-context (1M)
        "gpt-4.1-mini",            # GPT-4.1 Mini — balanced
        "gpt-4.1-nano",            # GPT-4.1 Nano — fastest/cheapest
        "o3",                      # o3 — advanced reasoning
        "o4-mini",                 # o4-mini — reasoning, lower cost
        # ── Google Gemini (Latest) ────────────────────────────────
        "gemini-3.1-pro",          # Gemini 3.1 Pro — best reasoning
        "gemini-3.1-flash",        # Gemini 3.1 Flash
        "gemini-3.1-flash-lite",   # Gemini 3.1 Flash Lite
        "gemini-3-flash",          # Gemini 3 Flash (default app)
        "gemini-3-pro",            # Gemini 3 Pro
        "gemini-2.5-pro",          # Gemini 2.5 Pro
        "gemini-2.5-flash",        # Gemini 2.5 Flash
        "gemini-2.5-flash-lite",   # Gemini 2.5 Flash Lite
        "gemini-2.0-flash",        # Gemini 2.0 Flash
    ]
    
    selected_model = st.selectbox(
        "Model:",
        available_models,
        index=0,
        key="model_selector",
        label_visibility="collapsed"
    )
    
    # Update agent if exists
    if 'agent' in st.session_state:
        if st.session_state.agent.config.model != selected_model:
            st.session_state.agent.config.model = selected_model
            st.session_state.agent.set_model(selected_model)
    
    return selected_model

def render_main_content_center():
    """Render center content with logo and welcome message"""
    import base64
    from pathlib import Path
    
    # Load logo
    logo_path = Path(__file__).parent / "assets" / "quasar_logo.png"
    logo_b64 = ""
    if logo_path.exists():
        with open(logo_path, "rb") as f:
            logo_b64 = base64.b64encode(f.read()).decode()
    
    # Get selected model
    selected_model = st.session_state.get('model_selector', 'gpt-4o')
    
    # Center content with logo
    st.markdown("<div style='height: 60px;'></div>", unsafe_allow_html=True)
    
    if logo_b64:
        st.markdown(f"""
        <div style='text-align: center;'>
            <img src="data:image/png;base64,{logo_b64}" style="width: 100px; height: 100px;" />
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("<div style='text-align: center; font-size: 5rem;'>🔭</div>", unsafe_allow_html=True)
    
    # Welcome text with examples (no model link)
    st.markdown("""
    <div style='text-align: center; max-width: 500px; margin: 1.5rem auto;'>
        <p style='color: #94a3b8; font-size: 0.95rem; line-height: 1.6;'>
            Your AI assistant for radio astronomy research.<br><br>
            Try: <span style='color: #f472b6;'>@archive</span> Find ALMA data for Sz65<br>
            Or: <span style='color: #f472b6;'>@paper</span> Papers on protoplanetary disks
        </p>
    </div>
    """, unsafe_allow_html=True)

def render_footer():
    """Render footer text"""
    st.markdown("""
    <div style='text-align: center; color: #64748b; font-size: 0.75rem; padding: 1rem 0;'>
        Always fact-check your results. Quasar is designed for radio astronomy research.
    </div>
    """, unsafe_allow_html=True)


def format_message_with_tags(content):
    """Format message content to replace command prefixes with visual tags"""
    if not isinstance(content, str):
        return content
        
    # Case insensitive replacement for rendering
    # We use regex to match the start of the string or word boundaries
    
    # @Archive -> Red Badge
    content = re.sub(r'(?i)(@archive)', r'<span class="cmd-tag archive-tag">📡 Archive</span>', content)
    
    # @search -> Blue Badge
    content = re.sub(r'(?i)(@search)', r'<span class="cmd-tag search-tag">🧠 Search</span>', content)
    
    # @paper -> Green Badge (if used)
    content = re.sub(r'(?i)(@paper)', r'<span class="cmd-tag paper-tag">📚 Paper</span>', content)
    
    return content

from core.agent import QuasarAgent, AgentConfig








def auto_save_conversation():
    """Auto-save current conversation to database"""
    try:
        if 'conv_service' in st.session_state and 'current_conv_id' in st.session_state:
            if st.session_state.messages:
                st.session_state.conv_service.save_full_conversation(
                    st.session_state.current_conv_id,
                    st.session_state.messages
                )
                # Update title from first user message (only if "New Chat")
                first_msg = next((m for m in st.session_state.messages if m.get("role") == "user"), None)
                if first_msg:
                    convs = st.session_state.conv_service.get_user_conversations(
                        st.session_state.get('user_id', ''), limit=1
                    )
                    if convs and convs[0]["id"] == st.session_state.current_conv_id:
                        if convs[0]["title"] == "New Chat":
                            title = st.session_state.conv_service.generate_title_from_message(
                                first_msg.get("content", "")
                            )
                            st.session_state.conv_service.update_conversation_title(
                                st.session_state.current_conv_id, title
                            )
    except Exception as e:
        pass  # Silent fail for background save


def display_papers(papers_df, source_name):

    """Enhanced display with arXiv/published differentiation"""

    if papers_df is None or papers_df.empty:
        st.warning("No papers found in NASA ADS")
        return

    st.markdown(f"### 📚 Research Papers on {source_name}")

    # Calculate statistics
    arxiv_papers = papers_df[papers_df.get('is_arxiv', False) == True] if 'is_arxiv' in papers_df.columns else pd.DataFrame()
    published_papers = papers_df[papers_df.get('is_arxiv', False) == False] if 'is_arxiv' in papers_df.columns else papers_df
    radio_papers = papers_df[papers_df.get('is_radio_paper', False) == True] if 'is_radio_paper' in papers_df.columns else papers_df

    # Display statistics
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("📊 Total Papers", len(papers_df))
    with col2:
        st.metric("📚 Published", len(published_papers))
    with col3:
        st.metric("📄 arXiv", len(arxiv_papers))
    with col4:
        st.metric("📡 Radio Papers", len(radio_papers))
    with col5:
        total_citations = papers_df['citation_count'].sum() if 'citation_count' in papers_df.columns else 0
        st.metric("📈 Total Citations", f"{total_citations:,}")

    # Filter controls
    st.markdown("### 🔍 Filter Papers")
    col1, col2, col3 = st.columns(3)

    with col1:
        show_only_radio = st.checkbox("📡 Only Radio Papers", value=True)
    with col2:
        show_arxiv = st.checkbox("📄 Show arXiv", value=True)
    with col3:
        show_published = st.checkbox("📚 Show Published", value=True)

    # Apply filters
    display_df = papers_df.copy()

    if show_only_radio and 'is_radio_paper' in display_df.columns:
        display_df = display_df[display_df['is_radio_paper'] == True]

    if not show_arxiv and 'is_arxiv' in display_df.columns:
        display_df = display_df[display_df['is_arxiv'] == False]

    if not show_published and 'is_arxiv' in display_df.columns:
        display_df = display_df[display_df['is_arxiv'] == True]

    # Display papers
    st.markdown("### 📄 Paper Details")

    if display_df.empty:
        st.info("No papers match the selected filters")
    else:
        for idx, row in display_df.head(30).iterrows():
            # Prepare display elements
            title_display = row['title'][:100] + "..." if len(row['title']) > 100 else row['title']
            cites = row.get('citation_count', 0)

            # Create badges
            badges = []
            if row.get('is_arxiv', False):
                badges.append('<span class="arxiv-badge">arXiv</span>')
            else:
                badges.append('<span class="published-badge">Published</span>')

            if row.get('is_radio_paper', False):
                badges.append('📡')

            relevance = row.get('relevance_score', 0)
            if relevance >= 10:
                badges.append('⭐⭐⭐')
            elif relevance >= 5:
                badges.append('⭐⭐')
            elif relevance >= 1:
                badges.append('⭐')

            badges_html = ' '.join(badges)

            with st.expander(f"[{row['year']}] {title_display} ({cites} citations)"):
                st.markdown(f"{badges_html}", unsafe_allow_html=True)

                col1, col2 = st.columns([3, 1])

                with col1:
                    st.markdown(f"**Title:** {row['title']}")
                    st.markdown(f"**Authors:** {row['authors']}")
                    st.markdown(f"**Journal:** {row.get('journal', 'N/A')}")

                    if row.get('abstract'):
                        st.markdown("**Abstract:**")
                        st.text(row['abstract'])

                with col2:
                    st.metric("Citations", cites)
                    st.metric("Relevance", f"{row.get('relevance_score', 0):.0f}")

                    # Links section
                    st.markdown("**Links:**")

                    if row.get('bibcode'):
                        st.markdown(f"🔗 [NASA ADS](https://ui.adsabs.harvard.edu/abs/{row['bibcode']})")

                    if row.get('arxiv_id'):
                        arxiv_id = row['arxiv_id']
                        st.markdown(f"📄 [arXiv](https://arxiv.org/abs/{arxiv_id})")
                        if row.get('pdf_url'):
                            st.markdown(f"📥 [PDF](https://arxiv.org/pdf/{arxiv_id}.pdf)")

                    if row.get('doi'):
                        st.markdown(f"📖 [DOI](https://doi.org/{row['doi']})")

    # Download button
    csv = display_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Filtered Papers",
        data=csv,
        file_name=f"{source_name.replace(' ', '_')}_papers.csv",
        mime="text/csv",
        key=f"papers_{source_name}_{id(papers_df)}_{len(st.session_state.messages)}"
    )

    # Search tips
    with st.expander("💡 Search Tips"):
        st.markdown("""
        **To find more radio astronomy papers:**
        - The search automatically includes radio telescopes (VLA, ALMA, VLBI, etc.)
        - Papers are scored by relevance to radio astronomy
        - arXiv preprints are included and marked with badges
        - Try searching for: `object:"source_name" AND (VLA OR ALMA)`

        **Understanding the indicators:**
        - 📡 = Confirmed radio astronomy paper
        - ⭐⭐⭐ = Source name in title (highest relevance)
        - ⭐⭐ = Radio telescope mentioned
        - ⭐ = Radio terms in abstract
        """)

def display_results_with_summary(df, source_name, agent):
    """Display observation data table with summary below"""

    if df is None or df.empty:
        st.warning("No observations found")
        return

    # Statistics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Observations", len(df))

    with col2:
        if 'telescope' in df.columns:
            st.metric("Telescopes", df['telescope'].nunique())

    with col3:
        if 'access_url' in df.columns:
            has_urls = (~df['access_url'].isna()).sum()
            st.metric("With URLs", has_urls)

    with col4:
        if 'Band' in df.columns:
            alma_count = len(df[df['telescope'] == 'ALMA'])
            st.metric("ALMA Obs", alma_count)

    # Data table
    st.markdown("### 📊 Observation Data")

    display_cols = []
    # Columns to display in order
    target_cols = [
        'obs_publisher_did', 'target_name', 'telescope', 'Band', 
        'resolution', 'sensitivity', 'bandwidth',
        't_min', 'freq_min', 'freq_max', 'access_url'
    ]
    
    for col in target_cols:
        if col in df.columns:
            display_cols.append(col)

    # Configure columns for clickable links
    column_config = {
        "access_url": st.column_config.LinkColumn(
            "Access URL",
            help="Click to access the data",
            validate=".*", # Allow any URL format to avoid regex issues
            max_chars=100,
        )
    }

    if display_cols:
        st.dataframe(
            df[display_cols], 
            use_container_width=True, 
            height=400,
            column_config=column_config
        )
    else:
        st.dataframe(
            df, 
            use_container_width=True, 
            height=400,
            column_config=column_config
        )

    # Summary below table
    st.markdown("### 🎯 Analysis Summary")
    with st.container():
        summary = agent.generate_summary(df, source_name)
        st.info(summary)

    # =========================================================================
    # ADVANCED VISUALIZATIONS PANEL
    # =========================================================================
    if VIZ_COMPONENTS_AVAILABLE:
        st.markdown("---")
        with st.expander("📈 **Advanced Visualizations** (click to expand)", expanded=False):
            render_visualization_panel(df, key_prefix=f"viz_{source_name.replace(' ', '_')}")
    
    # Download button
    csv = df.to_csv(index=False)
    st.download_button(
        label="📥 Download Data CSV",
        data=csv,
        file_name=f"{source_name.replace(' ', '_')}_observations.csv",
        mime="text/csv",
        key=f"data_{source_name}_{id(df)}_{len(st.session_state.messages)}"
    )

def render_chat_interface():
    """Main chat interface with streaming support"""

    # Display message history
    for msg_idx, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            if message["role"] == "user":
                formatted_content = format_message_with_tags(message["content"])
                st.markdown(formatted_content, unsafe_allow_html=True)
            else:
                if message.get("type") == "papers" and "data" in message:
                    display_papers(message["data"], message.get("source", "Unknown"))
                elif message.get("type") == "data" and "data" in message:
                    display_results_with_summary(
                        message["data"],
                        message.get("source", "Unknown"),
                        st.session_state.agent
                    )
                elif message.get("type") == "image" and "image" in message:
                    st.image(message["image"], caption=message.get("caption", "Plot"))
                else:
                    st.markdown(message.get("content", ""))

    # Chat input
    if prompt := st.chat_input("Ask me anything about radio astronomy..."):
        # Fix 6: Clear any previous error state
        if 'error_message' in st.session_state:
            st.session_state.error_message = None
        if 'last_error' in st.session_state:
            st.session_state.last_error = None
        
        # SYNC: Ensure agent memory matches UI before processing
        if hasattr(st.session_state, 'agent') and hasattr(st.session_state.agent, 'memory'):
            if hasattr(st.session_state.agent.memory, 'sync_from_ui'):
                st.session_state.agent.memory.sync_from_ui(st.session_state.messages)
            
        st.session_state.messages.append({"role": "user", "content": prompt})


        with st.chat_message("user"):
            st.markdown(format_message_with_tags(prompt), unsafe_allow_html=True)

        with st.chat_message("assistant"):
            # Check if we are waiting for clarification
            # BUT only if the new message looks like a clarification (short, entity-like)
            # If it's a full question, treat it as a new intent
            is_question = "?" in prompt or len(prompt.split()) > 5
            
            if st.session_state.get('awaiting_clarification') and not is_question:
                st.session_state.awaiting_clarification = False
                
                # Extract source name from the clarification response
                # Use the same robust extraction as the main query to handle "the source is Sz65"
                try:
                    extraction_prompt = ENTITY_EXTRACTION_PROMPT.format(query=prompt)
                    response = st.session_state.agent.client.responses.create(
                        model="gpt-4o",
                        input=extraction_prompt,
                        instructions="Extract entities from the user's message. Respond with JSON only.",
                        text={"format": {"type": "json_object"}},
                        temperature=0.1
                    )
                    # Extract text from Responses API
                    output_text = response.output_text if hasattr(response, 'output_text') else str(response)
                    result = json.loads(output_text)
                    extracted_name = result.get("source_name")
                    
                    # Use extracted name if found, otherwise fallback to raw input (cleaned)
                    if extracted_name:
                        source_name = extracted_name
                    else:
                        # Simple cleanup fallback
                        source_name = prompt.replace("the source is", "").replace("source is", "").strip()
                except:
                    source_name = prompt
                
                with st.spinner(f"🔍 Searching for {source_name}..."):
                    # Directly search with the provided source name
                    if any(word in prompt.lower() for word in ['paper', 'publication', 'article']):
                         result = st.session_state.agent.search_papers(source_name)
                         result_type = "papers"
                    else:
                         result = st.session_state.agent.search_service.search_source(source_name)
                         result_type = "data"

                if result is not None and not result.empty:
                    if result_type == "papers":
                        display_papers(result, source_name)
                        st.session_state.messages.append({
                            "role": "assistant",
                            "data": result,
                            "source": source_name,
                            "type": "papers"
                        })
                    else:
                        display_results_with_summary(result, source_name, st.session_state.agent)
                        st.session_state.messages.append({
                            "role": "assistant",
                            "data": result,
                            "source": source_name,
                            "type": "data"
                        })
                else:
                    response = f"No results found for {source_name}"
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                return

            # Quick intent detection for immediate feedback
            with st.spinner("🤔 Thinking..."):
                # Force everything to GENERAL to use the Smart Agent (Native Tools)
                # The Agent is now smart enough to handle "Search for X", "Plot Y", etc. natively.
                intent = "general"
                
                # Check for Papers specifically if we want specialized paper UI
                if "paper" in prompt.lower():
                    # We can keep paper handling separate OR move it to a tool too.
                    # For now, let's trust the agent or keep it simple.
                    # Let's try handling even papers via the agent if possible, 
                    # but current agent.py handles papers via process_query still?
                    # Let's keep strict intent detection JUST for papers if needed,
                    # but for Data/Search, use general.
                    pass

                # Override: If it looks like a search, it is GENERAL (Agent handles tool calls)
                if "search" in prompt.lower() or "find" in prompt.lower() or "get" in prompt.lower():
                    intent = "general"

            if intent == "general":
                # === THINKING BUBBLE FEATURE ===
                # Create a placeholder for the thinking bubble that we can clear
                thinking_placeholder = st.empty()
                
                with thinking_placeholder.container():
                    with st.expander("🧠 **Thinking...**", expanded=True):
                        # Show planning steps
                        if prompt.lower().startswith("@archive"):
                            st.markdown("📡 **Command detected:** `@archive`")
                            st.markdown("🔍 Planning to search ALMA archive...")
                        elif prompt.lower().startswith("@search"):
                            st.markdown("🧠 **Command detected:** `@search`")
                            st.markdown("📚 Consulting ALMA Manual (RAG)...")
                        elif prompt.lower().startswith("@paper"):
                            st.markdown("📚 **Command detected:** `@paper`")
                            st.markdown("🔎 Will search NASA ADS for papers...")
                        else:
                            st.markdown("💭 Analyzing query intent...")
                
                # Stream general responses - ALWAYS use Responses API
                message_placeholder = st.empty()
                
                # Use Responses API method (default and only path)
                full_response = st.session_state.agent.stream_response_api(
                    prompt, 
                    message_placeholder,
                    user_id=st.session_state.get('user_id', 'user')
                )
                
                # Clear the thinking bubble once response is complete
                thinking_placeholder.empty()

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": full_response,
                    "type": "general"
                })
                
                # Check for side-effects (Tool results)
                # Use getattr to prevent AttributeError if agent is stale
                last_run_result = getattr(st.session_state.agent, 'last_run_result', None)
                
                # Handle tool results
                if last_run_result:

                    result = last_run_result
                    
                    if result.get("type") == "image":
                        # Fix 5: Handle both bytes (new) and path (legacy) image data
                        image_data = result.get("image_bytes") or result.get("path")
                        if image_data:
                            st.image(image_data, caption=result.get("caption", "Generated Plot"))
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": "",
                                "image": image_data,  # Store bytes or path
                                "caption": result.get("caption"),
                                "type": "image"
                            })
                    
                    elif result.get("type") == "papers":
                        # Render the papers table
                        display_papers(result["data"], result.get("source", "Search"))
                        st.session_state.messages.append({
                            "role": "assistant",
                            "data": result["data"],
                            "source": result.get("source", "Search"),
                            "type": "papers"
                        })
                    
                    elif result.get("type") == "data":
                        # Render the data table
                        display_results_with_summary(result["data"], result.get("source", "Search"), st.session_state.agent)
                        st.session_state.messages.append({
                            "role": "assistant",
                            "data": result["data"],
                            "source": result.get("source", "Search"),
                            "type": "data"
                        })
                    
                    elif result.get("type") == "analysis":
                        st.json(result["data"])
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": str(result["data"]),
                            "type": "analysis"
                        })

                    # Clear result after handling
                    st.session_state.agent.last_run_result = None

            elif intent == "papers":
                # Search for papers
                with st.spinner("📚 Searching NASA ADS for radio astronomy papers..."):
                    result, source_name, result_type = st.session_state.agent.process_query(
                        prompt,
                        user_id=st.session_state.get('user_id', 'user')
                    )

                if result is not None and not result.empty:
                    display_papers(result, source_name)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "data": result,
                        "source": source_name,
                        "type": "papers"
                    })
                else:
                    if result_type == "clarification":
                        st.session_state.awaiting_clarification = True
                        response = source_name
                    else:
                        response = f"No papers found for {source_name}" if source_name else "Could not identify what to search for"
                    
                    st.markdown(response)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response,
                        "type": "general"
                    })

            elif intent == "data":
                # Search for observation data
                with st.spinner("🔍 Searching ALMA and VLA/VLBA archives..."):
                    result, source_name, result_type = st.session_state.agent.process_query(
                        prompt,
                        user_id=st.session_state.get('user_id', 'user')
                    )

                if result is not None and not result.empty:
                    display_results_with_summary(result, source_name, st.session_state.agent)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "data": result,
                        "source": source_name,
                        "type": "data"
                    })
                else:
                    if result_type == "clarification":
                        st.session_state.awaiting_clarification = True
                        response = source_name 
                    else:
                        response = f"No observations found for {source_name}" if source_name else "Could not identify what to search for"
                    
                    st.markdown(response)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response,
                        "type": "general"
                    })
        
        # Auto-save conversation after each interaction
        auto_save_conversation()

def render_login_page():

    """Render the login/registration page"""
    st.markdown("""
    <div style='text-align: center; padding: 2rem;'>
        <h1 style='color: #00d4ff;'>🌌 QUASAR</h1>
        <p style='color: #a0a0a0;'>Radio Astronomy Intelligence System</p>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["Login", "Register"])

    if 'auth_service' not in st.session_state:
        st.session_state.auth_service = AuthService()

    with tab1:
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submit = st.form_submit_button("Login", use_container_width=True)
            
            if submit:
                success, user_id, msg = st.session_state.auth_service.login_user(username, password)
                if success:
                    st.session_state.user_id = user_id
                    st.session_state.username = username
                    st.success(msg)
                    st.rerun()
                else:
                    st.error(msg)

    with tab2:
        with st.form("register_form"):
            new_user = st.text_input("New Username")
            new_pass = st.text_input("New Password", type="password")
            confirm_pass = st.text_input("Confirm Password", type="password")
            submit = st.form_submit_button("Register", use_container_width=True)
            
            if submit:
                if new_pass != confirm_pass:
                    st.error("Passwords do not match")
                else:
                    success, msg = st.session_state.auth_service.register_user(new_user, new_pass)
                    if success:
                        st.success(msg + " - Please login")
                    else:
                        st.error(msg)

def main():
    """Main application"""
    
    # Check authentication
    if 'user_id' not in st.session_state:
        render_login_page()
        return

    # Initialize session state for uploads
    if 'upload_status' not in st.session_state:
        st.session_state.upload_status = None
    
    # Get user's RAG service (with personal collection)
    user_id = st.session_state.user_id
    
    # Sidebar FIRST (always show) - Playground style
    with st.sidebar:
        # Logo and QUASAR text at top
        render_sidebar_logo()
        
        st.divider()
        
        # New Chat button
        if st.button("+ New chat", use_container_width=True, type="primary"):
            # Save current conversation first
            if st.session_state.messages:
                conv_service = st.session_state.get('conv_service')
                if conv_service:
                    conv_service.save_full_conversation(
                        st.session_state.current_conv_id, 
                        st.session_state.messages
                    )
            
            # Create new conversation
            if 'conv_service' in st.session_state:
                new_id = st.session_state.conv_service.create_conversation(user_id)
                st.session_state.current_conv_id = new_id
            st.session_state.messages = []
            if hasattr(st.session_state, 'agent') and hasattr(st.session_state.agent, 'memory'):
                st.session_state.agent.memory.clear()
            st.rerun()
        
        st.divider()
        
        # Thread History section
        with st.expander("Thread history", expanded=False):
            # Initialize conversation service
            if 'conv_service' not in st.session_state:
                st.session_state.conv_service = ConversationService()
            conv_service = st.session_state.conv_service
            
            # Initialize current conversation if needed
            if 'current_conv_id' not in st.session_state:
                st.session_state.current_conv_id = conv_service.get_or_create_current(user_id)
            
            # List recent conversations
            conversations = conv_service.get_user_conversations(user_id, limit=10)
            
            if conversations:
                for conv in conversations:
                    is_current = conv["id"] == st.session_state.get("current_conv_id")
                    label = f"{'▶ ' if is_current else ''}{conv['title'][:30]}"
                    
                    if st.button(label, key=f"conv_{conv['id']}", use_container_width=True, 
                                disabled=is_current):
                        # Save current before switching
                        if st.session_state.messages:
                            conv_service.save_full_conversation(
                                st.session_state.current_conv_id,
                                st.session_state.messages
                            )
                        
                        # Load selected conversation
                        st.session_state.current_conv_id = conv["id"]
                        st.session_state.messages = conv_service.get_conversation_messages(conv["id"])
                        st.rerun()
            else:
                st.caption("No conversations yet")
        
        st.divider()
        
        # =====================================================================
        # ADD CUSTOM TOOLS
        # =====================================================================
        with st.expander("Add Tools", expanded=False):
            st.markdown("Add your own function tools:")
            
            tool_name = st.text_input("Tool name", placeholder="my_custom_tool", key="tool_name_input")
            tool_desc = st.text_area("Description", placeholder="What does this tool do?", 
                                     height=60, key="tool_desc_input")
            tool_code = st.text_area("Python code", height=120, key="tool_code_input",
                                     placeholder='''def my_custom_tool(param1: str) -> dict:
    """Your tool logic here"""
    return {"result": "success"}''')
            
            if st.button("+ Add Tool", use_container_width=True):
                if tool_name and tool_code:
                    # Save tool definition to file
                    tools_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "user_tools", user_id)
                    os.makedirs(tools_dir, exist_ok=True)
                    tool_file = os.path.join(tools_dir, f"{tool_name}.py")
                    with open(tool_file, "w") as f:
                        f.write(f'"""{tool_desc}"""\n\n{tool_code}')
                    st.success(f"✓ Tool '{tool_name}' saved!")
                    st.info("Restart app to load new tools")
                else:
                    st.warning("Please fill in name and code")
        
        st.divider()

        # =====================================================================
        # DOCUMENT UPLOAD (Personal RAG)
        # =====================================================================
        st.markdown("**My Documents**")
        
        # Initialize RAG service
        rag_service = None
        try:
            if 'rag_service' not in st.session_state or st.session_state.get('rag_user_id') != user_id:
                st.session_state.rag_service = RAGService(user_id=user_id)
                st.session_state.rag_user_id = user_id
            rag_service = st.session_state.rag_service
        except:
            try:
                st.session_state.rag_service = RAGService()
                rag_service = st.session_state.rag_service
            except:
                pass
        
        if rag_service:
            uploaded_file = st.file_uploader("Add to your knowledge base", type=["pdf", "txt"], 
                                             label_visibility="collapsed", key="doc_upload")
            
            if uploaded_file is not None:
                if st.button("Upload", use_container_width=True):
                    upload_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "user_uploads", user_id)
                    os.makedirs(upload_dir, exist_ok=True)
                    file_path = os.path.join(upload_dir, uploaded_file.name)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    try:
                        result = rag_service.ingest_document(file_path, personal=True)
                        if result.get("success"):
                            st.success(f"✓ Added {uploaded_file.name}")
                        else:
                            st.error("Upload failed")
                    except Exception as e:
                        st.error(f"Error: {e}")
        
        st.divider()
        
        # =====================================================================
        # LINKS AND LOG OUT
        # =====================================================================
        st.markdown("""
        <div style='font-size: 0.9rem; margin-bottom: 0.5rem;'>
            <a href='/documentation' target='_self' style='color: #f472b6; text-decoration: none;'>Documentation</a>
        </div>
        """, unsafe_allow_html=True)
        
        # Log out button
        if st.button("Log out", use_container_width=True):
            del st.session_state.user_id
            del st.session_state.username
            st.rerun()

            
    # Model selector at top left (Playground style)
    # Model selector at top left (Playground style)
    # Side-by-side layout: "Model:" label + Dropdown
    top_cols = st.columns([2, 5])
    with top_cols[0]:
        sub_c1, sub_c2 = st.columns([1, 3])
        with sub_c1:
             st.markdown("<div style='padding-top: 10px; font-weight: 600; white-space: nowrap;'>Model:</div>", unsafe_allow_html=True)
        with sub_c2:
             selected_model = render_model_selector_top()
    
    # Enable Responses API by default (no toggle needed)
    st.session_state['use_responses_api'] = True

    # Initialize session state
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # API key setup
    AGENT_VERSION = "3.9"  # Bump to force re-init (Chat History)


    
    # Check if agent exists AND has the new attributes
    agent_is_stale = False
    if 'agent' in st.session_state:
        if not hasattr(st.session_state.agent, 'last_run_result'):
            agent_is_stale = True

    if 'agent' not in st.session_state or st.session_state.get('agent_version') != AGENT_VERSION or agent_is_stale:
        st.session_state.agent_version = AGENT_VERSION
        openai_key = os.getenv('OPENAI_API_KEY')
        ads_key = os.getenv('NASA_ADS_API_KEY')

        if not openai_key:
            st.error("Set OPENAI_API_KEY in .env file")
            return

        if not ads_key:
            st.warning("NASA ADS API key not found. Paper search won't work.")
            st.info("Get a free API key at: https://ui.adsabs.harvard.edu/user/settings/token")

        # Create config with keys
        config = AgentConfig(api_key=openai_key)
        # Manually set ADS key if provided, to avoid init issues if class definition is stale
        if ads_key and hasattr(config, 'ads_api_key'):
            config.ads_api_key = ads_key
        
        try:
            # Try new signature with rag_service
            st.session_state.agent = QuasarAgent(config=config, rag_service=rag_service)
        except TypeError:
            # Fallback for stale cache (old signature)
            st.session_state.agent = QuasarAgent(config=config)
            
        st.session_state.agent_version = AGENT_VERSION

    # Welcome message (Playground style - logo in center, tool card)
    if not st.session_state.messages:
        # Render centered content with logo and tool card
        render_main_content_center()

    render_chat_interface()


if __name__ == "__main__":
    main()