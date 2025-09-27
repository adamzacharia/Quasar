"""
Quasar Web Interface - FIXED VERSION with Working NRAO Integration
Streamlit-based UI for radio astronomy data discovery and analysis
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
import sys
from pathlib import Path
import time
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Import the fixed TAP client
try:
    from integrations.tap import NRAOTapClient
except:
    from integrations.tap_fixed import NRAOTapClient

# Page configuration
st.set_page_config(
    page_title="Quasar - Radio Astronomy Assistant",
    page_icon="🔭",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Hide Streamlit branding
hide_st_style = """
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
"""
st.markdown(hide_st_style, unsafe_allow_html=True)

# Working CSS with stars (keeping your beautiful design)
st.markdown("""
<style>
    /* Dark gradient background */
    .stApp {
        background: linear-gradient(to bottom, #0a0e27 0%, #1a1e3a 100%);
    }

    /* Add stars as pseudo-element */
    .stApp::after {
        content: '';
        position: fixed;
        width: 100%;
        height: 100%;
        top: 0;
        left: 0;
        background-image:
            radial-gradient(2px 2px at 20% 30%, white, transparent),
            radial-gradient(2px 2px at 60% 70%, white, transparent),
            radial-gradient(1px 1px at 50% 50%, white, transparent),
            radial-gradient(3px 3px at 80% 10%, white, transparent),
            radial-gradient(2px 2px at 90% 60%, white, transparent),
            radial-gradient(1px 1px at 30% 80%, white, transparent),
            radial-gradient(2px 2px at 70% 40%, white, transparent);
        background-size: 200% 200%;
        animation: twinkle 20s ease-in-out infinite;
        pointer-events: none;
        z-index: 1;
    }

    @keyframes twinkle {
        0%, 100% { opacity: 0.5; }
        50% { opacity: 1; }
    }

    /* Ensure content is visible above stars */
    .main > div {
        position: relative;
        z-index: 2;
    }

    /* Header styling */
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        background: linear-gradient(90deg, #00d4ff 0%, #7a5fff 50%, #ff006e 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem;
        position: relative;
        z-index: 10;
    }

    /* Make text visible */
    .stMarkdown {
        color: #e0e0e0;
    }

    h1, h2, h3, h4, h5, h6 {
        color: #ffffff !important;
    }

    /* Chat message styling */
    .stChatMessage {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border-radius: 10px;
    }

    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'tap_client' not in st.session_state:
    st.session_state.tap_client = None
if 'current_data' not in st.session_state:
    st.session_state.current_data = None
if 'initialized' not in st.session_state:
    st.session_state.initialized = False

def initialize_tap_client():
    """Initialize TAP client"""
    try:
        return NRAOTapClient()
    except Exception as e:
        st.error(f"Failed to connect to NRAO TAP service: {str(e)}")
        return None

def render_header():
    """Render application header with animation"""
    header_html = """
    <div style='position: relative; z-index: 20; padding: 0.5rem 0;'>
        <div class="main-header">🌌 QUASAR 🔭</div>
        <center><p style='color: #00d4ff; font-size: 1rem; margin-top: -0.5rem;'>Radio Astronomy Intelligence System</p></center>
    </div>
    """
    st.markdown(header_html, unsafe_allow_html=True)

def extract_source_name(query):
    """Extract source name from natural language query"""
    # Common patterns
    query_lower = query.lower()
    
    # Remove common words
    remove_words = ['search', 'find', 'show', 'get', 'fetch', 'data', 'for', 'of', 'the', 
                   'observations', 'from', 'nrao', 'archives', 'source', 'me', 'recent', 
                   'most', 'vla', 'alma', 'vlba', 'summarize', 'about']
    
    # If query contains quotes, extract quoted text
    if '"' in query:
        import re
        quoted = re.findall(r'"([^"]*)"', query)
        if quoted:
            return quoted[0]
    
    # Otherwise, remove common words and get what's left
    words = query.split()
    source_words = []
    for word in words:
        if word.lower() not in remove_words:
            source_words.append(word)
    
    if source_words:
        return ' '.join(source_words).strip()
    
    return None

def search_nrao_data(source_name):
    """Search NRAO archives for a source"""
    if not st.session_state.tap_client:
        st.session_state.tap_client = initialize_tap_client()
    
    if not st.session_state.tap_client:
        return None
    
    try:
        # Search for the source
        with st.spinner(f"🔍 Searching NRAO archives for {source_name}..."):
            df = st.session_state.tap_client.search_by_source_name(source_name, max_results=50)
            
            if not df.empty:
                st.session_state.current_data = df
                return df
            else:
                st.warning(f"No observations found for {source_name}")
                return None
                
    except Exception as e:
        st.error(f"Error searching NRAO: {str(e)}")
        return None

def display_data_results(df):
    """Display search results in a nice format"""
    if df is None or df.empty:
        return
    
    st.success(f"✅ Found {len(df)} observations!")
    
    # Show summary statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Observations", len(df))
    
    with col2:
        if 'instrument_name' in df.columns:
            instruments = df['instrument_name'].value_counts()
            st.metric("Telescopes", len(instruments))
    
    with col3:
        if 'freq_min_ghz' in df.columns and 'freq_max_ghz' in df.columns:
            freq_range = f"{df['freq_min_ghz'].min():.1f}-{df['freq_max_ghz'].max():.1f} GHz"
            st.metric("Frequency Range", freq_range)
    
    with col4:
        if 'size_gb' in df.columns:
            total_size = df['size_gb'].sum()
            st.metric("Total Data Size", f"{total_size:.1f} GB")
    
    # Show breakdown by telescope
    if 'instrument_name' in df.columns:
        st.markdown("### 📡 Observations by Telescope")
        for inst in df['instrument_name'].unique():
            inst_df = df[df['instrument_name'] == inst]
            st.markdown(f"**{inst}**: {len(inst_df)} observations")
    
    # Display the data table
    st.markdown("### 📊 Observation Details")
    
    # Select columns to display
    display_columns = []
    possible_cols = ['obs_publisher_did', 'target_name', 'instrument_name', 'obs_date', 
                     'freq_min_ghz', 'freq_max_ghz', 't_exptime', 'size_gb', 'archive_url']
    
    for col in possible_cols:
        if col in df.columns:
            display_columns.append(col)
    
    if display_columns:
        display_df = df[display_columns].head(20)
        st.dataframe(display_df, use_container_width=True, height=400)
    
    # Show archive links
    if 'archive_url' in df.columns:
        st.markdown("### 🔗 Archive Links")
        st.info("Click any link below to access the data in NRAO archive:")
        
        # Show first 5 unique links
        unique_urls = df['archive_url'].dropna().unique()[:5]
        for url in unique_urls:
            if url:
                st.markdown(f"📎 [{url}]({url})")

def process_query(query):
    """Process user query - FIXED VERSION"""
    # Add user message to chat
    st.session_state.messages.append({"role": "user", "content": query})
    
    # Parse the query
    query_lower = query.lower()
    
    # Check if it's a data search query
    search_keywords = ['search', 'find', 'show', 'get', 'fetch', 'observations', 'data', 
                      'vla', 'alma', 'vlba', 'nrao', 'archives']
    
    is_search_query = any(keyword in query_lower for keyword in search_keywords)
    
    if is_search_query:
        # Extract source name
        source_name = extract_source_name(query)
        
        if source_name:
            # Add assistant response
            response = f"Searching NRAO archives for **{source_name}**..."
            st.session_state.messages.append({"role": "assistant", "content": response})
            
            # Search for data
            df = search_nrao_data(source_name)
            
            if df is not None and not df.empty:
                # Create summary response
                summary = f"""
I found **{len(df)} observations** of {source_name} in the NRAO archives.

**Summary:**
"""
                if 'instrument_name' in df.columns:
                    for inst in df['instrument_name'].unique():
                        count = len(df[df['instrument_name'] == inst])
                        summary += f"\n• {inst}: {count} observations"
                
                if 'freq_min_ghz' in df.columns and 'freq_max_ghz' in df.columns:
                    summary += f"\n• Frequency range: {df['freq_min_ghz'].min():.1f} - {df['freq_max_ghz'].max():.1f} GHz"
                
                if 'obs_date' in df.columns:
                    try:
                        dates = pd.to_datetime(df['obs_date'], errors='coerce').dropna()
                        if not dates.empty:
                            summary += f"\n• Date range: {dates.min().date()} to {dates.max().date()}"
                    except:
                        pass
                
                st.session_state.messages.append({"role": "assistant", "content": summary})
            else:
                response = f"No observations found for {source_name} in the NRAO archives."
                st.session_state.messages.append({"role": "assistant", "content": response})
        else:
            response = "I couldn't identify a source name in your query. Please specify a source like '3C 273', 'M31', or 'Cygnus A'."
            st.session_state.messages.append({"role": "assistant", "content": response})
    else:
        # General response
        response = """
I'm ready to help you explore radio astronomy data! I can:

• Search NRAO archives (VLA, VLBA, ALMA) for any astronomical source
• Provide observation details, frequencies, and dates
• Generate direct links to the data archives

Try asking me something like:
- "Search the NRAO archives for 3C 273"
- "Find VLA observations of M31"
- "Show me ALMA data for NGC 253"
"""
        st.session_state.messages.append({"role": "assistant", "content": response})

def main():
    """Main application - FIXED"""
    # Initialize TAP client on first run
    if not st.session_state.initialized:
        st.session_state.tap_client = initialize_tap_client()
        st.session_state.initialized = True
    
    # Render header
    render_header()
    
    # Sidebar with examples
    with st.sidebar:
        st.markdown("### 💡 Example Queries")
        examples = [
            'Search NRAO archives for "3C 273"',
            "Find VLA observations of M31",
            "Show ALMA data for NGC 253",
            "Get Cygnus A observations",
            "Search for Sagittarius A*"
        ]
        
        st.markdown("Click any example to try it:")
        for example in examples:
            if st.button(example, key=f"ex_{example}"):
                # Process the example query
                process_query(example)
                st.rerun()
    
    # Welcome message if no messages
    if not st.session_state.messages:
        st.markdown("""
        <div style='text-align: center; padding: 2rem;'>
            <h3 style='color: #00d4ff;'>Welcome to Quasar! 🌟</h3>
            <p style='color: #a0a0a0; font-size: 1rem;'>
                I'm your AI assistant for exploring radio astronomy data.<br>
                Ask me about observations, targets, or any radio astronomy questions!
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # If this is an assistant message about found data, display it
            if message["role"] == "assistant" and "found" in message["content"].lower() and st.session_state.current_data is not None:
                display_data_results(st.session_state.current_data)
    
    # Chat input - FIXED to actually process the query
    if prompt := st.chat_input("Ask about radio astronomy observations..."):
        # Clear current data for new search
        st.session_state.current_data = None
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            # Process the query immediately
            query_lower = prompt.lower()
            
            # Check if it's a search query
            if any(word in query_lower for word in ['search', 'find', 'show', 'get', 'fetch', 'data', '3c', 'm31', 'ngc', 'cygnus']):
                source_name = extract_source_name(prompt)
                
                if source_name:
                    st.markdown(f"🔍 Searching NRAO archives for **{source_name}**...")
                    
                    # Actually search for the data
                    df = search_nrao_data(source_name)
                    
                    if df is not None and not df.empty:
                        # Display results
                        display_data_results(df)
                        
                        # Add to messages
                        st.session_state.messages.append({"role": "user", "content": prompt})
                        st.session_state.messages.append({"role": "assistant", 
                                                         "content": f"Found {len(df)} observations of {source_name}"})
                    else:
                        st.warning(f"No observations found for {source_name}")
                        st.session_state.messages.append({"role": "user", "content": prompt})
                        st.session_state.messages.append({"role": "assistant", 
                                                         "content": f"No observations found for {source_name}"})
                else:
                    st.info("Please specify a source name like '3C 273', 'M31', or 'Cygnus A'")
                    st.session_state.messages.append({"role": "user", "content": prompt})
                    st.session_state.messages.append({"role": "assistant", 
                                                     "content": "Please specify a source name"})
            else:
                # General help message
                help_text = "I can search NRAO archives for any astronomical source. Try: 'Search for 3C 273' or 'Find M31 observations'"
                st.markdown(help_text)
                st.session_state.messages.append({"role": "user", "content": prompt})
                st.session_state.messages.append({"role": "assistant", "content": help_text})
        
        # Force a rerun to update the UI
        st.rerun()

if __name__ == "__main__":
    main()
