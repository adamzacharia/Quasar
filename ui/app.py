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
from astroquery.alma import Alma
import pyvo as vo
from astropy.coordinates import SkyCoord
from openai import OpenAI
import requests
from typing import Dict, List, Optional

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Quasar - Radio Astronomy Assistant",
    page_icon="🔭",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# NASA ADS Client - Fixed with simple queries that work
class NASAADSClient:
    """NASA ADS client optimized for radio astronomy papers"""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.adsabs.harvard.edu/v1/search/query"

        # Radio astronomy keywords
        self.radio_telescopes = [
            'VLA', 'ALMA', 'VLBI', 'VLBA', 'EVLA', 'JVLA',
            'Arecibo', 'Effelsberg', 'Parkes', 'ATCA',
            'GMRT', 'LOFAR', 'MeerKAT', 'SKA', 'FAST',
            'NRAO', 'JCMT', 'SMA', 'NOEMA', 'IRAM'
        ]

        self.radio_terms = [
            'radio', 'interferometry', 'synthesis', 'GHz', 'MHz',
            'Jansky', 'mJy', 'millimeter', 'submillimeter',
            'continuum', '21cm', '21-cm', 'HI line',
            'synchrotron', 'radio jet', 'radio galaxy'
        ]

    def search_general(self, query: str, max_results: int = 60) -> dict:
        """Simple NASA ADS search that actually works - just like the test app"""
        headers = {"Authorization": f"Bearer {self.api_key}"}

        # Clean the query - remove "papers on" etc.
        query_clean = query.replace("papers on", "").replace("Papers on", "").strip()

        # Just add "radio" if not already there - SIMPLE!
        if 'radio' not in query_clean.lower():
            search_query = f'{query_clean} radio'
        else:
            search_query = query_clean

        params = {
            "q": search_query,
            "fl": "title,author,year,abstract,bibcode,citation_count,pub,doi,keyword,aff",
            "rows": max_results,
            "sort": "citation_count desc"
        }

        try:
            response = requests.get(self.base_url, headers=headers, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            # Process and score papers
            processed_docs = self._process_and_score_papers(
                data.get('response', {}).get('docs', []),
                query_clean
            )

            data['response']['docs'] = processed_docs
            return data

        except Exception as e:
            st.error(f"NASA ADS error: {e}")
            return {"response": {"docs": []}}

    def _process_and_score_papers(self, docs: List[Dict], search_term: str) -> List[Dict]:
        """Score papers for radio astronomy relevance"""
        scored_docs = []
        search_lower = search_term.lower() if search_term else ""

        for doc in docs:
            title = doc.get('title', [''])[0].lower() if doc.get('title') else ''
            abstract = doc.get('abstract', '').lower()
            keywords = [k.lower() for k in doc.get('keyword', [])]
            bibcode = doc.get('bibcode', '')

            # Check if it's an arXiv paper
            is_arxiv = 'arXiv' in bibcode
            doc['is_arxiv'] = is_arxiv

            # Extract arXiv ID if present
            if is_arxiv:
                arxiv_id = self._extract_arxiv_id(bibcode)
                doc['arxiv_id'] = arxiv_id
                doc['pdf_url'] = f'https://arxiv.org/pdf/{arxiv_id}.pdf' if arxiv_id else None

            # Calculate relevance score
            score = 0
            radio_evidence = []

            # 1. Source name in title (highest weight)
            if search_lower and search_lower in title:
                score += 10
                radio_evidence.append("source_in_title")
            elif search_lower and search_lower in abstract:
                score += 3
                radio_evidence.append("source_in_abstract")

            # 2. Radio telescope names
            for telescope in self.radio_telescopes:
                telescope_lower = telescope.lower()
                if telescope_lower in title:
                    score += 5
                    radio_evidence.append(f"telescope_{telescope}")
                elif telescope_lower in abstract:
                    score += 2
                    radio_evidence.append(f"telescope_{telescope}_abstract")

            # 3. Radio terms
            for term in self.radio_terms:
                term_lower = term.lower()
                if term_lower in title:
                    score += 3
                    radio_evidence.append(f"term_{term}")
                elif term_lower in abstract[:1000]:  # Check first part of abstract
                    score += 1
                    radio_evidence.append(f"term_{term}_abstract")

            # 4. Keywords check
            radio_keywords = [
                'radio sources', 'radio astronomy', 'interferometry',
                'radio continuum', 'radio jets', 'synchrotron',
                'very large array', 'atacama large millimeter'
            ]

            for kw in keywords:
                for rk in radio_keywords:
                    if rk in kw:
                        score += 2
                        radio_evidence.append(f"keyword_{kw}")
                        break

            # Add processed fields
            doc['relevance_score'] = score
            doc['radio_evidence'] = radio_evidence
            doc['is_radio_paper'] = score >= 3  # Threshold for radio paper

            # Include all papers but mark their relevance
            scored_docs.append(doc)

        # Sort by relevance score, then by citations
        scored_docs.sort(
            key=lambda x: (x.get('relevance_score', 0), x.get('citation_count', 0)),
            reverse=True
        )

        return scored_docs[:40]

    def _extract_arxiv_id(self, bibcode: str) -> Optional[str]:
        """Extract arXiv ID from NASA ADS bibcode"""
        match = re.search(r'arXiv(\d{4})(\d{4,5})', bibcode)
        if match:
            year_month = match.group(1)
            paper_id = match.group(2)
            return f"{year_month}.{paper_id}"
        return None

    def _looks_like_object(self, text: str) -> bool:
        """Check if text looks like an astronomical object designation"""
        patterns = [
            r'^3C\s+\d+',      # 3C catalog
            r'^NGC\s+\d+',     # NGC catalog
            r'^M\s*\d+',       # Messier catalog
            r'^IC\s+\d+',      # IC catalog
            r'^UGC\s+\d+',     # UGC catalog
            r'^Arp\s+\d+',     # Arp catalog
            r'^PKS\s+[\d\+\-]+',  # PKS catalog
            r'^Cygnus\s+[A-Z]',   # Cygnus A, etc.
            r'^Sgr\s+[A-Z]\*?',   # Sgr A*, etc.
            r'^Cen\s+[A-Z]',      # Centaurus A, etc.
        ]

        for pattern in patterns:
            if re.match(pattern, text, re.IGNORECASE):
                return True

        # Short strings with numbers might be object names
        if len(text.split()) <= 3 and any(char.isdigit() for char in text):
            return True

        return False

    def format_authors(self, authors: List[str], max_display: int = 3) -> str:
        """Format author list for display"""
        if not authors:
            return "Unknown"

        if len(authors) <= max_display:
            return ', '.join(authors)
        else:
            return ', '.join(authors[:max_display]) + f' et al. ({len(authors)} authors)'

    def get_paper_stats(self, docs: List[Dict]) -> Dict:
        """Get statistics about the papers"""
        if not docs:
            return {
                'total': 0,
                'arxiv_count': 0,
                'published_count': 0,
                'radio_papers': 0,
                'total_citations': 0
            }

        arxiv_count = sum(1 for d in docs if d.get('is_arxiv', False))
        radio_count = sum(1 for d in docs if d.get('is_radio_paper', False))
        total_citations = sum(d.get('citation_count', 0) for d in docs)

        return {
            'total': len(docs),
            'arxiv_count': arxiv_count,
            'published_count': len(docs) - arxiv_count,
            'radio_papers': radio_count,
            'total_citations': total_citations,
            'avg_citations': total_citations / len(docs) if docs else 0,
            'max_relevance': max(d.get('relevance_score', 0) for d in docs) if docs else 0
        }

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

class HybridSearchClient:
    """Combines ALMA (fast) and TAP (comprehensive) searches"""

    def __init__(self):
        self.tap_service = vo.dal.TAPService("https://data-query.nrao.edu/tap")
        self.obscore_table = "ivoa.obscore"

    def search_source(self, source_name):
        """Hybrid search: ALMA via astroquery + VLA/VLBA via TAP"""
        all_results = []

        # 1. ALMA Search (FAST - 1-2 seconds)
        try:
            alma_results = Alma.query_object(source_name, public=True, science=True)
            if alma_results and len(alma_results) > 0:
                alma_df = alma_results.to_pandas()
                alma_df['telescope'] = 'ALMA'
                # Add URLs for ALMA
                if 'Member ous id' in alma_df.columns:
                    alma_df['access_url'] = alma_df['Member ous id'].apply(
                        lambda x: f"https://almascience.nrao.edu/aq/?member_ous_id={x}" if pd.notna(x) else ""
                    )
                all_results.append(alma_df)
        except Exception as e:
            pass  # Silently fail for ALMA

        # 2. VLA/VLBA Search via TAP (3-5 seconds)
        try:
            # Clean source name for SQL
            clean_name = source_name.replace("'", "''")

            query = f"""
            SELECT TOP 100
                target_name, obs_publisher_did, instrument_name, facility_name,
                t_min, t_max, freq_min, freq_max, access_url, s_ra, s_dec,
                dataproduct_type
            FROM {self.obscore_table}
            WHERE target_name LIKE '%{clean_name}%'
            ORDER BY t_min DESC
            """

            tap_results = self.tap_service.search(query)
            if tap_results:
                vla_df = tap_results.to_table().to_pandas()
                if not vla_df.empty:
                    vla_df['telescope'] = vla_df['facility_name'] if 'facility_name' in vla_df else 'VLA/VLBA'
                    all_results.append(vla_df)
        except Exception as e:
            # Fallback: resolve name to coordinates
            try:
                coord = SkyCoord.from_name(source_name)
                vla_df = self.search_by_position(coord.ra.deg, coord.dec.deg)
                if vla_df is not None and not vla_df.empty:
                    vla_df['telescope'] = vla_df.get('facility_name', 'VLA/VLBA')
                    all_results.append(vla_df)
            except:
                pass

        # Combine all results
        if all_results:
            combined = pd.concat(all_results, ignore_index=True)
            return combined
        return None

    def search_by_position(self, ra, dec, radius=0.5):
        """Search by coordinates for VLA/VLBA"""
        query = f"""
        SELECT TOP 100
            target_name, obs_publisher_did, instrument_name, facility_name,
            t_min, t_max, freq_min, freq_max, access_url, s_ra, s_dec
        FROM {self.obscore_table}
        WHERE CONTAINS(POINT('ICRS', s_ra, s_dec),
                      CIRCLE('ICRS', {ra}, {dec}, {radius})) = 1
        ORDER BY t_min DESC
        """
        try:
            results = self.tap_service.search(query)
            return results.to_table().to_pandas()
        except:
            return None

class QuasarAgent:
    """AI agent with enhanced paper search capabilities"""

    def __init__(self, openai_key, ads_api_key=None):
        self.client = OpenAI(api_key=openai_key)
        self.search_client = HybridSearchClient()
        self.last_results = None
        self.last_source = None

        # Initialize improved NASA ADS client
        self.ads_client = NASAADSClient(ads_api_key) if ads_api_key else None

    def extract_coordinates(self, text):
        """Extract RA/Dec from text"""
        patterns = [
            r'[Rr][Aa][\s:=]+?([\d.]+).*?[Dd][Ee][Cc][\s:=]+?([-\d.]+)',
            r'(\d+\.?\d*)[,\s]+?([-\d]+\.?\d*)',
            r'position[\s:]+(\d+\.?\d*)[,\s]+([-\d]+\.?\d*)'
        ]

        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                try:
                    ra = float(match.group(1))
                    dec = float(match.group(2))
                    return ra, dec
                except:
                    continue
        return None, None

    def resolve_coordinates_to_object(self, ra, dec, radius=0.1):
        """Given RA/Dec, find what object is there"""
        df = self.search_client.search_by_position(ra, dec, radius)

        if df is not None and not df.empty:
            if 'target_name' in df.columns:
                target_names = df['target_name'].value_counts()
                if not target_names.empty:
                    return target_names.index[0]

        return f"RA {ra:.3f} Dec {dec:.3f}"

    def analyze_query_intent(self, user_input):
        """Quick intent detection for immediate feedback"""

        prompt = """Quickly categorize this astronomy query:

        Categories:
        - "data": Telescope observations/measurements
        - "papers": Research papers/publications
        - "general": Questions, code generation, explanations, capabilities

        Return JSON:
        {
            "intent": "data" | "papers" | "general",
            "source_name": "object name if mentioned",
            "has_coordinates": true/false
        }
        """

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": user_input}
                ],
                response_format={"type": "json_object"},
                temperature=0.1
            )

            return json.loads(response.choices[0].message.content)
        except:
            user_lower = user_input.lower()
            if any(word in user_lower for word in ['paper', 'article', 'publication']):
                return {"intent": "papers", "source_name": None, "has_coordinates": False}
            elif any(word in user_lower for word in ['observation', 'data', 'vla', 'alma']):
                return {"intent": "data", "source_name": None, "has_coordinates": False}
            else:
                return {"intent": "general", "source_name": None, "has_coordinates": False}

    def search_papers(self, query):
        """Enhanced paper search with better radio astronomy filtering"""

        if not self.ads_client:
            st.warning("NASA ADS API key not configured")
            return None

        try:
            # Search NASA ADS with improved radio astronomy focus
            ads_results = self.ads_client.search_general(query)

            papers = []
            for doc in ads_results.get('response', {}).get('docs', []):
                # Format authors
                authors = doc.get('author', [])
                if len(authors) > 3:
                    author_str = ', '.join(authors[:3]) + f' et al. ({len(authors)} authors)'
                else:
                    author_str = ', '.join(authors)

                # Check paper type
                bibcode = doc.get('bibcode', '')
                is_arxiv = doc.get('is_arxiv', False)

                # Create paper entry
                paper_entry = {
                    'title': doc.get('title', [''])[0] if doc.get('title') else '',
                    'authors': author_str,
                    'year': doc.get('year', 'N/A'),
                    'citation_count': doc.get('citation_count', 0),
                    'journal': doc.get('pub', 'N/A'),
                    'bibcode': bibcode,
                    'doi': doc.get('doi', [''])[0] if doc.get('doi') else '',
                    'abstract': doc.get('abstract', 'No abstract')[:500] + '...' if len(doc.get('abstract', '')) > 500 else doc.get('abstract', 'No abstract'),
                    'relevance_score': doc.get('relevance_score', 0),
                    'is_radio_paper': doc.get('is_radio_paper', False),
                    'is_arxiv': is_arxiv,
                    'arxiv_id': doc.get('arxiv_id'),
                    'pdf_url': doc.get('pdf_url'),
                    'type_icon': '📄 arXiv' if is_arxiv else '📚 Published'
                }

                papers.append(paper_entry)

            if papers:
                df = pd.DataFrame(papers)
                # Sort by relevance and citation
                df = df.sort_values(['relevance_score', 'citation_count'],
                                   ascending=[False, False])
                return df

        except Exception as e:
            st.error(f"Error searching NASA ADS: {e}")
            return None

    def stream_general_response(self, user_input, message_placeholder):
        """Stream general responses token by token"""

        system_prompt = """You are Quasar, an advanced radio astronomy intelligence system.

        You help with:
        - Radio astronomy concepts and theory
        - Code generation for astronomy tasks (Python, CASA, etc.)
        - FITS file processing and analysis
        - Observation planning and strategy
        - Instrument specifications (VLA, ALMA, VLBA)
        - Data calibration and imaging techniques
        - General astronomy questions

        When generating code, make it practical and well-commented.
        Be helpful, accurate, and professional.
        Never mention specific AI model names or API implementation details."""

        full_response = ""

        try:
            stream = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_input}
                ],
                temperature=0.7,
                stream=True
            )

            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    full_response += chunk.choices[0].delta.content
                    message_placeholder.markdown(full_response + "▌")

            message_placeholder.markdown(full_response)

        except Exception as e:
            full_response = "I can help with radio astronomy questions, code generation, and data analysis. Please ask me anything!"
            message_placeholder.markdown(full_response)

        return full_response

    def process_query(self, user_input):
        """Process queries for data and papers (non-streaming)"""

        # Check for coordinates
        ra, dec = self.extract_coordinates(user_input)

        if ra is not None and dec is not None:
            if any(word in user_input.lower() for word in ['paper', 'publication', 'article']):
                object_name = self.resolve_coordinates_to_object(ra, dec)
                st.info(f"Object identified: {object_name}")
                papers_df = self.search_papers(object_name)
                return papers_df, object_name, "papers"
            else:
                df = self.search_client.search_by_position(ra, dec)
                if df is not None and not df.empty:
                    object_name = df['target_name'].iloc[0] if 'target_name' in df.columns else f"RA={ra:.3f}, Dec={dec:.3f}"
                    self.last_results = df
                    self.last_source = object_name
                    return df, object_name, "data"

        # Extract source name for data/paper searches
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Extract the astronomical object name. Return just the name or 'None'."},
                    {"role": "user", "content": user_input}
                ]
            )
            source_name = response.choices[0].message.content.strip()
            if source_name == 'None':
                source_name = None
        except:
            source_name = None

        # Use last source if not specified
        if not source_name and self.last_source:
            source_name = self.last_source

        if source_name:
            # Determine if searching for papers or data
            if any(word in user_input.lower() for word in ['paper', 'publication', 'article']):
                papers_df = self.search_papers(source_name)
                return papers_df, source_name, "papers"
            else:
                df = self.search_client.search_source(source_name)
                self.last_results = df
                self.last_source = source_name
                return df, source_name, "data"

        return None, None, "Could not identify source"

    def generate_summary(self, df, source_name):
        """Generate intelligent summary using LLM"""

        if df is None or df.empty:
            return "No observations found."

        stats = {
            "total": len(df),
            "source": source_name
        }

        if 'telescope' in df.columns:
            stats["telescopes"] = df['telescope'].value_counts().to_dict()

        if 'Band' in df.columns:
            alma_data = df[df['telescope'] == 'ALMA']
            if not alma_data.empty:
                stats["alma_bands"] = alma_data['Band'].value_counts().to_dict()

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Create a brief professional summary of these radio observations."},
                    {"role": "user", "content": f"Summarize observations of {source_name}: {json.dumps(stats)}"}
                ],
                temperature=0.7
            )

            return response.choices[0].message.content

        except:
            return f"Found {len(df)} observations of {source_name}"

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
    for col in ['obs_publisher_did', 'target_name', 'telescope', 'instrument_name',
                'Band', 't_min', 'freq_min', 'freq_max', 'access_url']:
        if col in df.columns:
            display_cols.append(col)

    if display_cols:
        st.dataframe(df[display_cols], use_container_width=True, height=400)
    else:
        st.dataframe(df, use_container_width=True, height=400)

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
                st.markdown(message["content"])
            else:
                if message.get("type") == "papers" and "data" in message:
                    display_papers(message["data"], message.get("source", "Unknown"))
                elif message.get("type") == "data" and "data" in message:
                    display_results_with_summary(
                        message["data"],
                        message.get("source", "Unknown"),
                        st.session_state.agent
                    )
                else:
                    st.markdown(message.get("content", ""))

    # Chat input
    if prompt := st.chat_input("Ask me anything about radio astronomy..."):
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            # Quick intent detection for immediate feedback
            with st.spinner("🤔 Thinking..."):
                intent_result = st.session_state.agent.analyze_query_intent(prompt)
                intent = intent_result.get('intent', 'general')

            if intent == "general":
                # Stream general responses
                message_placeholder = st.empty()
                full_response = st.session_state.agent.stream_general_response(prompt, message_placeholder)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": full_response,
                    "type": "general"
                })

            elif intent == "papers":
                # Search for papers
                with st.spinner("📚 Searching NASA ADS for radio astronomy papers..."):
                    result, source_name, result_type = st.session_state.agent.process_query(prompt)

                if result is not None and not result.empty:
                    display_papers(result, source_name)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "data": result,
                        "source": source_name,
                        "type": "papers"
                    })
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
                    result, source_name, result_type = st.session_state.agent.process_query(prompt)

                if result is not None and not result.empty:
                    display_results_with_summary(result, source_name, st.session_state.agent)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "data": result,
                        "source": source_name,
                        "type": "data"
                    })
                else:
                    response = f"No observations found for {source_name}" if source_name else "Could not identify what to search for"
                    st.markdown(response)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response,
                        "type": "general"
                    })

def main():
    """Main application"""
    render_header()

    # Initialize session state
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # API key setup
    if 'agent' not in st.session_state:
        openai_key = os.getenv('OPENAI_API_KEY')
        ads_key = os.getenv('NASA_ADS_API_KEY')

        if not openai_key:
            st.error("Set OPENAI_API_KEY in .env file")
            return

        if not ads_key:
            st.warning("NASA ADS API key not found. Paper search won't work.")
            st.info("Get a free API key at: https://ui.adsabs.harvard.edu/user/settings/token")

        st.session_state.agent = QuasarAgent(openai_key, ads_api_key=ads_key)

    # Sidebar
    with st.sidebar:
        st.markdown("### 💡 Example Queries")

        st.markdown("**📡 Data:**")
        for ex in ["Search for 3C 273", "RA 187.7 Dec 12.4", "Show me Cygnus A observations"]:
            st.markdown(f"• {ex}")

        st.markdown("\n**📚 Papers:**")
        for ex in ["Papers on 3C 273", "Find radio papers about Sagittarius A*", "Papers at RA 187.7 Dec 12.4"]:
            st.markdown(f"• {ex}")

        st.markdown("\n**💻 Code & Help:**")
        for ex in ["Generate FITS analysis code", "What is a pulsar?", "How to calibrate VLA data"]:
            st.markdown(f"• {ex}")

        st.markdown("\n---")
        st.markdown("### 🔍 Search Features")
        st.markdown("""
        **Paper Search includes:**
        - Published papers from journals
        - arXiv preprints
        - Radio telescope filtering (VLA, ALMA, VLBI, etc.)
        - Relevance scoring for radio astronomy
        - Direct PDF links for arXiv papers
        """)

    # Welcome message
    if not st.session_state.messages:
        st.markdown("""
        <div style='text-align: center; padding: 1rem;'>
            <h3 style='color: #00d4ff;'>Welcome to Quasar! 🌟</h3>
            <p style='color: #a0a0a0;'>
                Your Radio Astronomy Intelligence System<br><br>
                I can search data archives, find papers, generate code, and answer questions.<br>
                <b>Paper Search:</b> Automatically filters for radio astronomy papers and includes arXiv!<br>
                <b>Try:</b> "Papers on 3C 273" or "Search the NRAO archives for source "3C 273" and summarize the most recent observations.
"
            </p>
        </div>
        """, unsafe_allow_html=True)

    render_chat_interface()

if __name__ == "__main__":
    main()