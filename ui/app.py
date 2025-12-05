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
from core.prompts import ENTITY_EXTRACTION_PROMPT
from services.rag_service import RAGService
from services.auth import AuthService

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Quasar - Radio Astronomy Assistant",
    page_icon="🔭",
    layout="wide",
    initial_sidebar_state="collapsed"
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
</style>
"""
st.markdown(hide_st_style, unsafe_allow_html=True)

# YOUR BEAUTIFUL CSS WITH ALL ANIMATIONS
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
            radial-gradient(1px 1px at 85% 85%, white, transparent),
            radial-gradient(2px 2px at 45% 25%, white, transparent),
            radial-gradient(1px 1px at 25% 65%, white, transparent),
            radial-gradient(3px 3px at 55% 45%, white, transparent),
            radial-gradient(1px 1px at 65% 90%, white, transparent),
            radial-gradient(2px 2px at 35% 15%, white, transparent),
            radial-gradient(1px 1px at 95% 35%, white, transparent);
        background-size: 200% 200%;
        animation: twinkle 20s ease-in-out infinite;
        pointer-events: none;
        z-index: -1;
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

    /* Force chat input to be on top */
    .stChatInput {
        z-index: 1000 !important;
        position: relative;
    }
    
    /* Fix for bottom container */
    .stBottom {
        z-index: 1000 !important;
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

    /* Paper type badges */
    .arxiv-badge {
        background: linear-gradient(135deg, #FF6B6B, #C44569);
        color: white;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.85em;
        display: inline-block;
        margin-right: 5px;
    }

    .published-badge {
        background: linear-gradient(135deg, #4ECDC4, #44A08D);
        color: white;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.85em;
        display: inline-block;
        margin-right: 5px;
    }

    /* Shooting stars */
    .shooting-star {
        position: fixed;
        width: 2px;
        height: 2px;
        background: white;
        box-shadow: 0 0 6px 2px white;
        animation: shoot 3s linear infinite;
        z-index: 0;
        pointer-events: none;
    }

    @keyframes shoot {
        0% {
            transform: translateX(0) translateY(0);
            opacity: 1;
        }
        100% {
            transform: translateX(300px) translateY(300px);
            opacity: 0;
        }
    }

    /* Supernova effect */
    .supernova {
        position: fixed;
        width: 10px;
        height: 10px;
        background: white;
        border-radius: 50%;
        animation: explode 10s ease-out infinite;
        z-index: 0;
        pointer-events: none;
    }

    @keyframes explode {
        0% {
            width: 10px;
            height: 10px;
            opacity: 0;
        }
        5% {
            width: 20px;
            height: 20px;
            opacity: 1;
            box-shadow: 0 0 30px 10px rgba(255, 255, 255, 0.8);
        }
        20% {
            width: 40px;
            height: 40px;
            opacity: 0.5;
            box-shadow: 0 0 50px 20px rgba(255, 200, 100, 0.5);
        }
        100% {
            width: 10px;
            height: 10px;
            opacity: 0;
        }
    }
    
    /* Command Tags */
    .cmd-tag {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.9em;
        font-weight: bold;
        color: white;
        margin-right: 5px;
    }
    
    .archive-tag {
        background: linear-gradient(135deg, #FF6B6B, #EE5253);
        box-shadow: 0 2px 4px rgba(255, 107, 107, 0.3);
    }
    
    .search-tag {
        background: linear-gradient(135deg, #4834d4, #686de0);
        box-shadow: 0 2px 4px rgba(72, 52, 212, 0.3);
    }
    
    .paper-tag {
        background: linear-gradient(135deg, #6ab04c, #badc58);
        text-shadow: 0 1px 1px rgba(0,0,0,0.2);
    }
</style>

<!-- Add shooting stars -->
<div class="shooting-star" style="top: 10%; left: 10%; animation-delay: 0s;"></div>
<div class="shooting-star" style="top: 30%; left: 80%; animation-delay: 1s;"></div>
<div class="shooting-star" style="top: 60%; left: 20%; animation-delay: 2s;"></div>

<!-- Add supernovae -->
<div class="supernova" style="top: 25%; left: 65%; animation-delay: 0s;"></div>
<div class="supernova" style="top: 70%; left: 35%; animation-delay: 5s;"></div>
""", unsafe_allow_html=True)

def render_header():
    """Render application header with animation"""
    header_html = """
    <div style='position: relative; z-index: 20; padding: 0.5rem 0;'>
        <div class="main-header">🌌 QUASAR 🔭</div>
        <center><p style='color: #00d4ff; font-size: 1rem; margin-top: -0.5rem;'>Radio Astronomy Intelligence System</p></center>
    </div>
    """
    st.markdown(header_html, unsafe_allow_html=True)

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

from core.agent import QuasarAgent, AgentConfig










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
                    response = st.session_state.agent.client.chat.completions.create(
                        model="gpt-4o",
                        messages=[{"role": "user", "content": extraction_prompt}],
                        response_format={"type": "json_object"},
                        temperature=0.1
                    )
                    result = json.loads(response.choices[0].message.content)
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
                # Stream general responses
                message_placeholder = st.empty()
                full_response = st.session_state.agent.stream_general_response(
                    prompt, 
                    message_placeholder,
                    user_id=st.session_state.get('user_id', 'user')
                )

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": full_response,
                    "type": "general"
                })
                
                # Check for side-effects (Tool results)
                # Use getattr to prevent AttributeError if agent is stale
                last_run_result = getattr(st.session_state.agent, 'last_run_result', None)
                if last_run_result:
                    result = last_run_result
                    
                    if result.get("type") == "image":
                        st.image(result["path"], caption=result.get("caption", "Generated Plot"))
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": "",
                            "image": result["path"],
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

    # Sidebar with user info and logout
    with st.sidebar:
        st.title("User Profile")
        st.write(f"Logged in as: **{st.session_state.username}**")
        if st.button("Logout", type="primary"):
            del st.session_state.user_id
            del st.session_state.username
            st.rerun()
            
    render_header()

    # Initialize session state
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # API key setup
    AGENT_VERSION = "3.6"  # Bump to force re-init (UI Text Update)
    
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
            
        # Initialize RAG service with caching
        @st.cache_resource
        def get_rag_service():
            return RAGService()
            
        rag_service = get_rag_service()
        
        try:
            # Try new signature with rag_service
            st.session_state.agent = QuasarAgent(config=config, rag_service=rag_service)
        except TypeError:
            # Fallback for stale cache (old signature)
            st.session_state.agent = QuasarAgent(config=config)
            
        st.session_state.agent_version = AGENT_VERSION

    # Sidebar removed as requested

    # Welcome message
    if not st.session_state.messages:
        st.markdown("""
        <div style='text-align: center; padding: 1rem;'>
            <h3 style='color: #00d4ff;'>Welcome to Quasar! 🌟</h3>
            <p style='color: #a0a0a0;'>
                Your Radio Astronomy Intelligence System<br><br>
                I can search data archives, find papers, generate code, and answer questions.<br><br>
                <b>Try:</b> <span class="cmd-tag archive-tag">@archive</span> Please make a table of the different (ALMA) datasets that exist for Sz65, so we can select the one with the best sensitivity and angular resolution.<br>
                <b>or</b> <span class="cmd-tag search-tag">@search</span> How long is the proprietary period for Principal Investigator data?
            </p>
        </div>
        """, unsafe_allow_html=True)

    render_chat_interface()

if __name__ == "__main__":
    main()