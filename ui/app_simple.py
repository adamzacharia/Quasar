"""
Quasar Web Interface - Simplified Working Version
Streamlit-based UI for radio astronomy data discovery and analysis
"""

import streamlit as st
import pandas as pd
import time
import os
from pathlib import Path
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from core.agent import QuasarAgent, AgentConfig
from integrations.tap import NRAOTapClient
from services.ads_service import ADSService
from services.code_generator import RadioCodeGenerator

# Page configuration
st.set_page_config(
    page_title="Quasar - Radio Astronomy Assistant",
    page_icon="🔭",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# CSS Styling
st.markdown("""
<style>
    /* Dark background */
    .stApp {
        background: linear-gradient(to bottom, #0a0e27 0%, #1a1e3a 100%);
    }
    
    /* Main header */
    .main-header {
        font-size: 2rem;
        font-weight: bold;
        color: #00d4ff;
        text-align: center;
        padding: 1rem 0;
    }
    
    /* Text visibility */
    .stMarkdown, p, span {
        color: #e0e0e0;
    }
    
    h1, h2, h3, h4, h5, h6 {
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'agent' not in st.session_state:
    st.session_state.agent = None
    st.session_state.messages = []
    st.session_state.current_data = None

def initialize_agent():
    """Initialize Quasar agent"""
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        st.error("OpenAI API key not found in .env file")
        return None
    
    config = AgentConfig(
        api_key=api_key,
        model='gpt-4o-mini',
        temperature=0.7
    )
    return QuasarAgent(config)

def render_header():
    """Render header"""
    st.markdown('<div class="main-header">🌌 QUASAR 🔭</div>', unsafe_allow_html=True)
    st.markdown('<center><p style="color: #a0a0a0;">Radio Astronomy Intelligence System</p></center>', unsafe_allow_html=True)

def stream_data_table(data):
    """Display data in table format"""
    if isinstance(data, pd.DataFrame) and not data.empty:
        st.markdown("### 📊 Observation Data")
        
        # Display the dataframe
        st.dataframe(data, use_container_width=True, height=300)
        
        # Quick stats
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Observations", len(data))
        with col2:
            if 'facility_name' in data.columns:
                st.metric("Facilities", data['facility_name'].nunique())
        with col3:
            if 'size_gb' in data.columns:
                st.metric("Total Size", f"{data['size_gb'].sum():.1f} GB")
        
        # Suggestion for next steps
        st.info("💡 I can help you find related papers or generate analysis code. Just ask!")

def search_papers_from_query(query, data):
    """Search for papers based on context"""
    ads_service = ADSService()
    papers = []
    
    if isinstance(data, pd.DataFrame) and 'target_name' in data.columns:
        target = data['target_name'].iloc[0]
        papers = ads_service.search_by_target(target, max_results=5)
    
    if papers:
        st.markdown("### 📚 Related Papers")
        for paper in papers[:5]:
            st.markdown(f"**{paper['title']}**")
            st.markdown(f"{paper['authors']} ({paper['year']})")
            st.markdown(f"[View Paper]({paper['link']})")
            st.markdown("---")

def generate_code_from_query(query):
    """Generate analysis code"""
    if 'casa' in query.lower():
        code = '''# CASA Data Reduction Script
import os
from casatasks import *

vis = 'data.ms'
listobs(vis=vis)
flagdata(vis=vis, mode='manual')
bandpass(vis=vis, caltable='bandpass.cal')
applycal(vis=vis, gaintable=['bandpass.cal'])
tclean(vis=vis, imagename='image')'''
    else:
        code = '''# Python Analysis Script
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def analyze_data(filename):
    data = pd.read_csv(filename)
    print(f"Loaded {len(data)} observations")
    return data

data = analyze_data('observations.csv')'''
    
    st.code(code, language='python')

def process_query(query):
    """Process user query"""
    if not st.session_state.agent:
        st.session_state.agent = initialize_agent()
        if not st.session_state.agent:
            return
    
    # Add to messages
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)
    
    with st.chat_message("assistant"):
        query_lower = query.lower()
        
        # Check for paper requests
        if 'paper' in query_lower or 'publication' in query_lower:
            if st.session_state.current_data is not None:
                search_papers_from_query(query, st.session_state.current_data)
            else:
                st.info("Please fetch some data first, then I can find related papers.")
            return
        
        # Check for code requests
        if 'code' in query_lower or 'script' in query_lower or 'casa' in query_lower:
            generate_code_from_query(query)
            return
        
        # Regular data search
        response = st.session_state.agent.process_query(query)
        st.markdown(response)
        
        # Simulate data fetching
        if any(word in query_lower for word in ['find', 'search', 'show', 'vla', 'alma']):
            # Create sample data
            sample_data = pd.DataFrame({
                'target_name': ['M31', 'Cygnus A', 'NGC 1234'],
                'facility_name': ['VLA', 'VLA', 'ALMA'],
                'obs_date': ['2024-01-15', '2024-02-20', '2024-03-10'],
                'freq_min_ghz': [1.4, 8.0, 230.0],
                'freq_max_ghz': [1.8, 12.0, 250.0],
                'size_gb': [12.5, 45.2, 125.8],
                'configuration': ['D', 'A', 'C43-6'],
                'duration_hours': [2.5, 4.0, 6.0]
            })
            
            st.session_state.current_data = sample_data
            stream_data_table(sample_data)
    
    st.session_state.messages.append({"role": "assistant", "content": ""})

def main():
    """Main application"""
    render_header()
    
    # Sidebar with examples
    with st.sidebar:
        st.markdown("### 💡 Examples")
        st.markdown("• Find VLA observations of M31")
        st.markdown("• Find papers about this data")
        st.markdown("• Generate CASA code")
    
    # Welcome message
    if not st.session_state.messages:
        st.markdown("""
        <center>
            <h3 style='color: #00d4ff;'>Welcome to Quasar!</h3>
            <p style='color: #a0a0a0;'>Ask me about radio astronomy observations</p>
        </center>
        """, unsafe_allow_html=True)
    
    # Chat interface
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Input
    if prompt := st.chat_input("Ask about observations..."):
        process_query(prompt)

if __name__ == "__main__":
    main()
