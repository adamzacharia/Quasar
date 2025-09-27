"""
Enhanced UI Functions for Quasar
Add these functions to your ui/app.py to enable real NRAO data fetching
"""

import re
from typing import Optional
import streamlit as st
import pandas as pd
import time

def extract_source_name(query: str) -> Optional[str]:
    """Extract astronomical source name from query"""
    
    # Common patterns for source names
    patterns = [
        r'(?:of|for|about|on)\s+([3][C]\s*\d+)',  # 3C sources
        r'(?:of|for|about|on)\s+([M]\d+)',  # Messier objects
        r'(?:of|for|about|on)\s+(NGC\s*\d+)',  # NGC objects
        r'(?:of|for|about|on)\s+(IC\s*\d+)',  # IC objects
        r'(?:of|for|about|on)\s+([A-Za-z]+\s*[A-Z])',  # Named sources like Cygnus A
        r'"([^"]+)"',  # Quoted names
        r"'([^']+)'",  # Single quoted names
    ]
    
    for pattern in patterns:
        match = re.search(pattern, query, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    
    # Fallback: look for known source names
    known_sources = [
        '3C 273', '3C 48', '3C 279', '3C 84', '3C 454.3',
        'M31', 'M51', 'M87', 'M82', 'M33', 'M101',
        'NGC 1275', 'NGC 4151', 'NGC 5128',
        'Cygnus A', 'Cas A', 'Sgr A*', 'Virgo A',
        'Crab Nebula', 'Orion Nebula', 'Crab Pulsar'
    ]
    
    query_upper = query.upper()
    for source in known_sources:
        if source.upper() in query_upper:
            return source
    
    # Last resort: extract potential source name
    words = query.split()
    for i, word in enumerate(words):
        if word.lower() in ['find', 'search', 'show', 'get'] and i < len(words) - 1:
            # Return the next 1-2 words as potential source
            if i < len(words) - 2:
                potential = f"{words[i+1]} {words[i+2]}"
                if not any(skip in potential.lower() for skip in ['data', 'observations', 'from', 'in', 'the']):
                    return potential
            if not any(skip in words[i+1].lower() for skip in ['data', 'observations', 'from', 'in', 'the']):
                return words[i+1]
    
    return None

def handle_data_search(query: str):
    """Handle real NRAO data searches with proper error handling"""
    from services.data_processor import DataProcessor
    from services.ads_service import ADSService
    
    # Parse the query to extract source name
    source_name = extract_source_name(query)
    
    if not source_name:
        st.warning("⚠️ Please specify a source name (e.g., '3C 273', 'M31', 'Cygnus A')")
        st.markdown("""
        **Example queries:**
        - Find VLA observations of 3C 273
        - Show me M31 data
        - Search for Cygnus A observations
        """)
        return
    
    # Show search progress
    with st.spinner(f"🔭 Searching NRAO archives for **{source_name}**..."):
        try:
            # Initialize services
            if not st.session_state.tap_client:
                st.session_state.tap_client = initialize_tap_client()
                if not st.session_state.tap_client:
                    st.error("Failed to connect to NRAO archives")
                    return
            
            ads_service = ADSService()
            processor = DataProcessor(st.session_state.tap_client, ads_service)
            
            # Fetch real data
            result = processor.fetch_source_data(source_name)
            
            # Display summary
            st.markdown(result['summary'])
            
            # Display observations table if found
            if result['observations'] is not None and not result['observations'].empty:
                st.markdown("### 📊 NRAO Observations")
                
                # Format the dataframe for display
                display_df = processor.format_observations_for_display(result['observations'])
                
                # Display the data with custom styling
                st.dataframe(
                    display_df, 
                    use_container_width=True, 
                    height=min(400, len(display_df) * 35 + 35)
                )
                
                # Display statistics in columns
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Observations", result['statistics']['total_observations'])
                with col2:
                    st.metric("Total Size", f"{result['statistics']['total_size_gb']:.1f} GB")
                with col3:
                    st.metric("Integration Time", f"{result['statistics']['total_integration_hours']:.1f} hrs")
                with col4:
                    st.metric("Facilities", len(result['statistics'].get('facilities', [])))
                
                # Add archive links in an expander
                with st.expander("🔗 Direct Archive Links"):
                    urls = processor.get_archive_urls(result['observations'])
                    if urls:
                        for i, url in enumerate(urls[:10], 1):  # Show first 10
                            st.markdown(f"{i}. [Observation Link]({url})")
                    else:
                        st.info("Archive links not available")
                
                # Store in session state for context
                st.session_state.current_data = result['observations']
                st.session_state.current_source = source_name
                
                # Display papers if found
                if result['papers']:
                    st.markdown("### 📚 Related Scientific Papers")
                    
                    # Display papers in a nice format
                    for i, paper in enumerate(result['papers'][:5], 1):
                        with st.expander(f"{i}. {paper['title'][:100]}... ({paper['year']})"):
                            col1, col2 = st.columns([3, 1])
                            with col1:
                                st.markdown(f"**Authors:** {paper['authors']}")
                                st.markdown(f"**Journal:** {paper['journal']}")
                                if paper.get('abstract'):
                                    st.markdown(f"**Abstract:** {paper['abstract'][:300]}...")
                            with col2:
                                st.metric("Citations", paper.get('citations', 0))
                                st.markdown(f"[View on NASA ADS]({paper['link']})")
                    
                    if len(result['papers']) > 5:
                        st.info(f"Showing 5 of {len(result['papers'])} papers. Ask me for more if needed!")
                
                # Suggest follow-up actions
                st.markdown("---")
                st.markdown("### 💡 What would you like to do next?")
                col1, col2, col3 = st.columns(3)
                with col1:
                    if st.button("📝 Generate CASA reduction script"):
                        st.session_state.next_action = "generate_casa"
                with col2:
                    if st.button("📊 Create analysis plots"):
                        st.session_state.next_action = "create_plots"
                with col3:
                    if st.button("🔍 Find more papers"):
                        st.session_state.next_action = "find_papers"
                
            else:
                # No data found - provide helpful suggestions
                st.markdown("### 💡 Try These Searches")
                
                suggestions = {
                    "Popular Quasars": ["3C 273", "3C 48", "3C 279"],
                    "Bright Galaxies": ["M31", "M87", "Cygnus A"],
                    "Pulsars": ["Crab Pulsar", "PSR J0437-4715"],
                    "Active Galaxies": ["NGC 1275", "Centaurus A", "Virgo A"]
                }
                
                for category, sources in suggestions.items():
                    st.markdown(f"**{category}:**")
                    cols = st.columns(len(sources))
                    for col, source in zip(cols, sources):
                        with col:
                            if st.button(source, key=f"search_{source}"):
                                st.session_state.next_search = source
                
        except Exception as e:
            st.error(f"❌ Error searching archives: {str(e)}")
            st.info("""
            **Possible issues:**
            - The NRAO TAP service might be temporarily unavailable
            - The source name might need a different format
            - Try using the full designation (e.g., 'PSR J0437-4715' instead of just 'J0437')
            """)
            
            # Show debug info in expander
            with st.expander("🔧 Debug Information"):
                st.code(str(e))

def handle_paper_search(query: str):
    """Enhanced paper search with context awareness"""
    from services.ads_service import ADSService
    
    ads_service = ADSService()
    
    # Check if we have current data context
    if hasattr(st.session_state, 'current_source') and st.session_state.current_source:
        source = st.session_state.current_source
        st.markdown(f"### 📚 Searching papers for {source}")
        
        with st.spinner("Searching NASA ADS..."):
            papers = ads_service.search_by_target(source, max_results=20)
            
            if papers:
                for i, paper in enumerate(papers[:10], 1):
                    with st.expander(f"{i}. {paper['title'][:80]}..."):
                        st.markdown(f"**{paper['authors']}** - {paper['year']}")
                        st.markdown(f"*{paper['journal']}*")
                        st.markdown(f"Citations: {paper.get('citations', 0)}")
                        st.markdown(f"[Read Paper]({paper['link']})")
            else:
                st.info("No papers found. The NASA ADS API key might not be configured.")
    else:
        st.info("Please search for astronomical data first, then I can find related papers.")

def handle_code_generation(query: str):
    """Generate analysis code based on context"""
    from services.code_generator import RadioCodeGenerator
    
    query_lower = query.lower()
    
    # Determine code type
    if 'casa' in query_lower or 'calibration' in query_lower:
        code_type = 'casa_calibration'
        language = 'python'
    elif 'python' in query_lower or 'analysis' in query_lower:
        code_type = 'python_analysis'
        language = 'python'
    else:
        code_type = 'python_analysis'
        language = 'python'
    
    # Check if we have context data
    has_context = hasattr(st.session_state, 'current_data') and st.session_state.current_data is not None
    
    if code_type == 'casa_calibration':
        st.markdown("### 🔧 CASA Calibration Script")
        
        code = '''# CASA Data Reduction Script
# Auto-generated by Quasar AI

import os
from casatools import table, msmetadata
from casatasks import *

# Configuration
vis = 'your_data.ms'  # Your measurement set
field = '0'  # Target field
refant = 'ea01'  # Reference antenna

# Step 1: Data inspection
print("Inspecting data...")
listobs(vis=vis, listfile=vis+'.listobs')

# Step 2: Initial flagging
print("Flagging bad data...")
flagdata(vis=vis, mode='manual', spw='0:0~5;60~63')
flagdata(vis=vis, mode='clip', clipminmax=[0,50])
flagdata(vis=vis, mode='tfcrop', datacolumn='data')

# Step 3: Set flux scale (for calibrators)
print("Setting flux scale...")
setjy(vis=vis, field=field, standard='Perley-Butler 2017', 
      model='3C286_C.im', usescratch=True)

# Step 4: Bandpass calibration
print("Bandpass calibration...")
bandpass(vis=vis, caltable='bandpass.cal', field=field, 
         refant=refant, solnorm=True, solint='inf', 
         combine='scan', bandtype='B')

# Step 5: Gain calibration (phase)
print("Phase calibration...")
gaincal(vis=vis, caltable='phase.cal', field=field, 
        refant=refant, calmode='p', solint='int', 
        minsnr=3, gaintable=['bandpass.cal'])

# Step 6: Gain calibration (amplitude)
print("Amplitude calibration...")
gaincal(vis=vis, caltable='amp.cal', field=field, 
        refant=refant, calmode='ap', solint='inf', 
        minsnr=3, gaintable=['bandpass.cal'])

# Step 7: Apply calibration
print("Applying calibration...")
applycal(vis=vis, field=field, 
         gaintable=['bandpass.cal', 'phase.cal', 'amp.cal'],
         calwt=True, flagbackup=True)

# Step 8: Image the calibrated data
print("Imaging...")
tclean(vis=vis, imagename='image', field=field,
       specmode='mfs', imsize=[2048, 2048], cell='0.5arcsec',
       deconvolver='hogbom', weighting='briggs', robust=0.5,
       niter=5000, threshold='0.1mJy', interactive=False,
       savemodel='modelcolumn', pbcor=True)

print("Calibration complete!")
print("Output image: image.image")
'''
        
    else:  # Python analysis
        st.markdown("### 📊 Python Analysis Script")
        
        if has_context and isinstance(st.session_state.current_data, pd.DataFrame):
            # Generate context-aware code
            df = st.session_state.current_data
            source = getattr(st.session_state, 'current_source', 'Unknown')
            
            code = f'''# Radio Astronomy Data Analysis
# Auto-generated by Quasar AI for {source}

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Load the data (already in memory from Quasar search)
# In practice, you would load from file:
# data = pd.read_csv('observations.csv')

# Data overview
print(f"Analyzing {len(data)} observations of {source}")
print(f"Date range: {{data['obs_date'].min()}} to {{data['obs_date'].max()}}")
print(f"Facilities: {{data['facility_name'].unique()}}")

# Frequency distribution
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(data['freq_min_ghz'], bins=20, alpha=0.7, edgecolor='black')
plt.xlabel('Frequency (GHz)')
plt.ylabel('Number of Observations')
plt.title('Frequency Distribution')

# Time coverage
plt.subplot(1, 2, 2)
dates = pd.to_datetime(data['obs_date'])
plt.scatter(dates, data['freq_min_ghz'], alpha=0.6)
plt.xlabel('Date')
plt.ylabel('Frequency (GHz)')
plt.title('Observations Over Time')
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

# Sky position analysis
if 's_ra' in data.columns and 's_dec' in data.columns:
    coords = SkyCoord(ra=data['s_ra']*u.deg, dec=data['s_dec']*u.deg)
    
    plt.figure(figsize=(10, 6))
    plt.scatter(coords.ra.deg, coords.dec.deg, 
                c=data['freq_min_ghz'], cmap='viridis',
                s=50, alpha=0.7)
    plt.colorbar(label='Frequency (GHz)')
    plt.xlabel('RA (degrees)')
    plt.ylabel('Dec (degrees)')
    plt.title(f'Sky Coverage for {source}')
    plt.grid(True, alpha=0.3)
    plt.show()

# Statistical summary
print("\\nStatistical Summary:")
print(f"Total integration time: {{data['duration_hours'].sum():.1f}} hours")
print(f"Total data volume: {{data['size_gb'].sum():.1f}} GB")
print(f"Frequency range: {{data['freq_min_ghz'].min():.1f}} - {{data['freq_max_ghz'].max():.1f}} GHz")

# Group by facility
facility_stats = data.groupby('facility_name').agg({{
    'duration_hours': 'sum',
    'size_gb': 'sum',
    'obs_date': 'count'
}}).rename(columns={{'obs_date': 'num_observations'}})

print("\\nPer-Facility Statistics:")
print(facility_stats)
'''
        else:
            # Generic code without context
            code = '''# Radio Astronomy Data Analysis Script
# Auto-generated by Quasar AI

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits

# Load observation data
def load_observations(filename):
    """Load observation data from CSV or FITS"""
    if filename.endswith('.csv'):
        return pd.read_csv(filename)
    elif filename.endswith('.fits'):
        with fits.open(filename) as hdul:
            return Table(hdul[1].data).to_pandas()
    else:
        raise ValueError(f"Unsupported format: {filename}")

# Analyze frequency coverage
def analyze_frequency_coverage(data):
    """Analyze frequency coverage of observations"""
    if 'freq_min' in data.columns and 'freq_max' in data.columns:
        freq_ranges = data[['freq_min', 'freq_max']].values
        
        plt.figure(figsize=(10, 6))
        for i, (fmin, fmax) in enumerate(freq_ranges):
            plt.barh(i, fmax - fmin, left=fmin, height=0.8)
        
        plt.xlabel('Frequency (GHz)')
        plt.ylabel('Observation')
        plt.title('Frequency Coverage')
        plt.show()
        
        return freq_ranges

# Plot sky coverage
def plot_sky_coverage(data):
    """Plot RA/Dec distribution"""
    if 's_ra' in data.columns and 's_dec' in data.columns:
        plt.figure(figsize=(10, 6))
        plt.scatter(data['s_ra'], data['s_dec'], alpha=0.6, s=50)
        plt.xlabel('RA (degrees)')
        plt.ylabel('Dec (degrees)')
        plt.title('Sky Coverage')
        plt.grid(True, alpha=0.3)
        plt.show()

# Main analysis
if __name__ == "__main__":
    # Load your data
    data = load_observations('observations.csv')
    
    print(f"Loaded {len(data)} observations")
    print(f"Columns: {list(data.columns)}")
    
    # Basic statistics
    print("\\nBasic Statistics:")
    print(data.describe())
    
    # Analyze frequency coverage
    analyze_frequency_coverage(data)
    
    # Plot sky distribution
    plot_sky_coverage(data)
    
    print("\\nAnalysis complete!")
'''
    
    # Display the code
    st.code(code, language=language)
    
    # Add download button
    st.download_button(
        label="📥 Download Script",
        data=code,
        file_name=f"{code_type}_script.py",
        mime="text/plain"
    )
    
    st.success("✅ Code generated successfully!")
    st.info("💡 Tip: Modify the script according to your specific data format and analysis needs.")

def process_query_enhanced(query: str):
    """Enhanced query processor that uses real NRAO data"""
    
    # Add user message
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)
    
    # Process the query
    with st.chat_message("assistant"):
        query_lower = query.lower()
        
        # Check for paper search
        if any(word in query_lower for word in ['paper', 'publication', 'article', 'literature']):
            handle_paper_search(query)
            response = "Searching for papers..."
            
        # Check for code generation
        elif any(word in query_lower for word in ['code', 'script', 'python', 'casa', 'generate']):
            handle_code_generation(query)
            response = "Generating code..."
            
        # Check for data search - this is the key part!
        elif any(word in query_lower for word in ['find', 'search', 'show', 'get', 'observations', 'data']):
            handle_data_search(query)
            response = f"Searching NRAO archives..."
            
        # Default: Use AI agent for general questions
        else:
            if not st.session_state.agent:
                st.session_state.agent = initialize_agent()
            
            if st.session_state.agent:
                with st.spinner("🤔 Thinking..."):
                    response = st.session_state.agent.process_query(query)
                    st.markdown(response)
            else:
                response = "Please configure your OpenAI API key to enable AI responses."
                st.warning(response)
        
        st.session_state.messages.append({"role": "assistant", "content": response})

# ADD THIS TO YOUR MAIN FUNCTION IN app.py:
# Replace the line: process_query(prompt)
# With: process_query_enhanced(prompt)
