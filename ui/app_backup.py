"""
Quasar Web Interface - WITH LLM INTELLIGENCE
The LLM understands queries, makes decisions, and generates summaries
"""

import streamlit as st
import pandas as pd
import os
import sys
from pathlib import Path
import time
from dotenv import load_dotenv
from astroquery.alma import Alma
from astropy.coordinates import SkyCoord
import astropy.units as u
import openai
import json

# Load environment variables
load_dotenv()

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Page configuration
st.set_page_config(
    page_title="Quasar - Radio Astronomy Assistant",
    page_icon="🔭",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# YOUR BEAUTIFUL CSS WITH STARS (keeping it exactly as you had)
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
            radial-gradient(2px 2px at 70% 40%, white, transparent),
            radial-gradient(1px 1px at 15% 15%, white, transparent),
            radial-gradient(1px 1px at 85% 85%, white, transparent);
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

<!-- Add shooting stars -->
<div class="shooting-star" style="top: 10%; left: 10%; animation-delay: 0s;"></div>
<div class="shooting-star" style="top: 30%; left: 80%; animation-delay: 1s;"></div>
<div class="shooting-star" style="top: 60%; left: 20%; animation-delay: 2s;"></div>
""", unsafe_allow_html=True)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'last_search_results' not in st.session_state:
    st.session_state.last_search_results = None
if 'last_source' not in st.session_state:
    st.session_state.last_source = None
if 'openai_client' not in st.session_state:
    api_key = os.getenv('OPENAI_API_KEY')
    if api_key:
        st.session_state.openai_client = openai.OpenAI(api_key=api_key)
    else:
        st.session_state.openai_client = None

def render_header():
    """Render application header"""
    header_html = """
    <div style='position: relative; z-index: 20; padding: 0.5rem 0;'>
        <div class="main-header">🌌 QUASAR 🔭</div>
        <center><p style='color: #00d4ff; font-size: 1rem; margin-top: -0.5rem;'>Radio Astronomy Intelligence System</p></center>
    </div>
    """
    st.markdown(header_html, unsafe_allow_html=True)

def search_alma_by_name(source_name):
    """Search ALMA archive for a source"""
    try:
        results = Alma.query_object(source_name, public=True, science=True)
        if results and len(results) > 0:
            df = results.to_pandas()
            # Store for context
            st.session_state.last_search_results = df
            st.session_state.last_source = source_name
            return df
        return None
    except Exception as e:
        st.error(f"ALMA search error: {e}")
        return None

def search_alma_by_coords(ra, dec, radius=0.1):
    """Search ALMA by coordinates"""
    try:
        coord = SkyCoord(ra, dec, unit='deg')
        results = Alma.query_region(coord, radius=radius*u.deg, public=True, science=True)
        if results and len(results) > 0:
            df = results.to_pandas()
            st.session_state.last_search_results = df
            st.session_state.last_source = f"RA={ra:.3f}, Dec={dec:.3f}"
            return df
        return None
    except Exception as e:
        st.error(f"Coordinate search error: {e}")
        return None

def generate_summary(df, source_name):
    """Generate intelligent summary of observations using LLM"""
    if st.session_state.openai_client is None:
        # Fallback summary without LLM
        summary = f"Found {len(df)} ALMA observations for {source_name}.\n"
        if 'Band' in df.columns:
            bands = df['Band'].value_counts()
            summary += f"Bands: {', '.join([f'Band {b} ({c} obs)' for b,c in bands.items()])}\n"
        if 'Project code' in df.columns:
            summary += f"Projects: {df['Project code'].nunique()} unique projects\n"
        return summary

    # Use LLM for intelligent summary
    try:
        # Prepare data for LLM
        data_summary = f"""
        Source: {source_name}
        Total observations: {len(df)}
        """

        if 'Band' in df.columns:
            data_summary += f"Bands: {df['Band'].value_counts().to_dict()}\n"
        if 'Integration' in df.columns:
            data_summary += f"Total integration time: {df['Integration'].sum()} seconds\n"
        if 'Project code' in df.columns:
            data_summary += f"Projects: {df['Project code'].nunique()} unique\n"
            data_summary += f"Recent projects: {df['Project code'].head(3).tolist()}\n"
        if 'Release date' in df.columns:
            data_summary += f"Date range: {df['Release date'].min()} to {df['Release date'].max()}\n"

        # Get LLM summary
        response = st.session_state.openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a radio astronomy expert. Provide concise, informative summaries of ALMA observations."},
                {"role": "user", "content": f"Summarize these ALMA observations:\n{data_summary}"}
            ],
            temperature=0.7
        )
        return response.choices[0].message.content
    except:
        # Fallback if LLM fails
        return f"Found {len(df)} ALMA observations for {source_name}."

def process_with_llm(user_query):
    """Process user query with LLM to understand intent and execute appropriate action"""

    if st.session_state.openai_client is None:
        # Fallback: simple keyword matching
        return process_without_llm(user_query)

    try:
        # System prompt that defines capabilities
        system_prompt = """You are an intelligent radio astronomy assistant. You can:
1. Search ALMA archives by source name or coordinates
2. Analyze search results
3. Generate summaries
4. Answer follow-up questions about the last search

When user asks about a source, extract the source name or coordinates.
When user asks about "largest quasar" or similar, identify the actual source name (e.g., TON 618, J0313-1806).
When user asks to analyze results, use the last search data.

Respond with a JSON object:
{
    "action": "search_name" | "search_coords" | "analyze" | "answer",
    "source_name": "source name if searching by name",
    "ra": RA in degrees if searching by coords,
    "dec": Dec in degrees if searching by coords,
    "radius": radius in degrees (default 0.1),
    "explanation": "brief explanation of what you're doing"
}
"""

        # Get LLM decision
        response = st.session_state.openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_query}
            ],
            temperature=0.3,
            response_format={ "type": "json_object" }
        )

        decision = json.loads(response.choices[0].message.content)

        # Execute the action
        if decision["action"] == "search_name":
            source = decision.get("source_name")
            if source:
                st.write(f"🔍 Searching ALMA for **{source}**...")
                df = search_alma_by_name(source)
                if df is not None:
                    return df, source, decision.get("explanation", "")
                else:
                    return None, source, f"No ALMA observations found for {source}"

        elif decision["action"] == "search_coords":
            ra = decision.get("ra")
            dec = decision.get("dec")
            radius = decision.get("radius", 0.1)
            if ra and dec:
                st.write(f"🔍 Searching ALMA at RA={ra:.3f}, Dec={dec:.3f}...")
                df = search_alma_by_coords(ra, dec, radius)
                if df is not None:
                    return df, f"RA={ra:.3f}, Dec={dec:.3f}", decision.get("explanation", "")
                else:
                    return None, None, f"No observations at those coordinates"

        elif decision["action"] == "analyze":
            if st.session_state.last_search_results is not None:
                return st.session_state.last_search_results, st.session_state.last_source, "Analyzing previous results..."
            else:
                return None, None, "No previous search results to analyze. Please search for a source first."

        else:
            return None, None, decision.get("explanation", "I can help you search ALMA archives.")

    except Exception as e:
        st.error(f"LLM error: {e}")
        return process_without_llm(user_query)

def process_without_llm(user_query):
    """Fallback processing without LLM - simple keyword matching"""
    query_lower = user_query.lower()

    # Check for coordinate search
    if "ra" in query_lower and "dec" in query_lower:
        # Try to extract numbers
        import re
        numbers = re.findall(r'[\d.]+', user_query)
        if len(numbers) >= 2:
            ra, dec = float(numbers[0]), float(numbers[1])
            radius = float(numbers[2]) if len(numbers) > 2 else 0.1
            df = search_alma_by_coords(ra, dec, radius)
            return df, f"RA={ra}, Dec={dec}", ""

    # Check for analyze command
    if "analyze" in query_lower or "last result" in query_lower:
        if st.session_state.last_search_results is not None:
            return st.session_state.last_search_results, st.session_state.last_source, "Analyzing previous results"
        else:
            return None, None, "No previous results to analyze"

    # Extract source name
    # Remove common words
    remove_words = ['search', 'find', 'show', 'get', 'observations', 'data', 'for', 'of', 'the',
                   'alma', 'nrao', 'archives', 'source', 'summarize', 'recent', 'most']

    words = query_lower.split()
    source_words = []
    for word in words:
        if word not in remove_words:
            # Check for known source patterns
            if any(x in word for x in ['3c', 'm31', 'ngc', 'cen', 'sgr', 'ton', 'j0']):
                source_words.append(word)
                # Get next word if it's a number (for sources like "3C 273")
                idx = words.index(word)
                if idx + 1 < len(words) and words[idx + 1].replace('.','').isdigit():
                    source_words.append(words[idx + 1])

    if source_words:
        source_name = ' '.join(source_words).upper()
        df = search_alma_by_name(source_name)
        return df, source_name, ""

    return None, None, "Please specify a source name or coordinates to search"

def display_results(df, source_name):
    """Display search results in table format"""
    if df is None or df.empty:
        return

    # Show statistics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Observations", len(df))

    with col2:
        if 'Band' in df.columns:
            st.metric("Bands", df['Band'].nunique())

    with col3:
        if 'Integration' in df.columns:
            st.metric("Total Integration", f"{df['Integration'].sum():.0f}s")

    with col4:
        if 'Project code' in df.columns:
            st.metric("Projects", df['Project code'].nunique())

    # Show data table
    st.markdown("### 📊 Observation Data")
    st.dataframe(df, use_container_width=True, height=300)

    # Archive links
    if 'Member ous id' in df.columns:
        with st.expander("📎 Archive Links"):
            unique_ids = df['Member ous id'].dropna().unique()[:5]
            for uid in unique_ids:
                if pd.notna(uid):
                    url = f"https://almascience.nrao.edu/aq/?member_ous_id={uid}"
                    st.markdown(f"[{uid}]({url})")

    # Download button
    csv = df.to_csv(index=False)
    st.download_button(
        label="📥 Download CSV",
        data=csv,
        file_name=f"ALMA_{source_name.replace(' ','_')}.csv",
        mime="text/csv"
    )

def render_chat_interface():
    """Chat interface that processes queries with LLM"""

    # Display message history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            # Display data if it's attached to the message
            if "data" in message and message["data"] is not None:
                display_results(message["data"], message.get("source", ""))

    # Chat input
    if prompt := st.chat_input("Ask about radio astronomy observations..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Process with LLM
        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking..."):
                df, source, explanation = process_with_llm(prompt)

            if df is not None and not df.empty:
                # Generate summary
                summary = generate_summary(df, source)
                st.markdown(summary)

                # Display the data
                display_results(df, source)

                # Save to messages
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": summary,
                    "data": df,
                    "source": source
                })
            else:
                # Just show the explanation
                response = explanation if explanation else "I couldn't find any data for that query."
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})

def main():
    """Main application"""
    render_header()

    # Sidebar with examples
    with st.sidebar:
        st.markdown("### 💡 Example Queries")
        st.markdown("""
        **Source searches:**
        • Search for 3C 273 and summarize
        • Find the largest quasar ever found
        • Look up Cygnus A observations

        **Coordinate searches:**
        • Search RA 187.7, Dec 12.4, radius 0.5
        • Find observations at RA 83.6, Dec 22.0

        **Follow-ups:**
        • Analyze the last results
        • Show frequency coverage
        • List counts by instrument
        """)

    # Welcome message
    if not st.session_state.messages:
        st.markdown("""
        <div style='text-align: center; padding: 1rem;'>
            <h3 style='color: #00d4ff;'>Welcome to Quasar! 🌟</h3>
            <p style='color: #a0a0a0;'>
                I understand complex queries like "find the largest quasar" or "search for Sgr A*".<br>
                I'll search ALMA, analyze the data, and provide summaries.
            </p>
        </div>
        """, unsafe_allow_html=True)

    render_chat_interface()

if __name__ == "__main__":
    main()