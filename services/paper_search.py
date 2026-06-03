"""
Paper Search Service - NASA ADS
Fixed version for radio astronomy papers
"""
import requests
import pandas as pd
import re
from typing import Dict, List, Optional

class NASAADSClient:
    """NASA ADS client - working version"""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.adsabs.harvard.edu/v1/search/query"

    def search(self, query: str, max_results: int = 30) -> Dict:
        """Just pass through to search_general"""
        return self.search_general(query, max_results)

    def search_general(self, query: str, max_results: int = 60) -> Dict:
        """
        Simple NASA ADS search for radio astronomy papers
        """
        headers = {"Authorization": f"Bearer {self.api_key}"}

        # Clean the query
        query_clean = query.replace("papers on", "").replace("Papers on", "").strip()

        # Build query based on what it looks like
        if self._looks_like_object(query_clean):
            # For astronomical objects, use AND to combine with radio
            search_query = f'("{query_clean}" OR title:"{query_clean}" OR abstract:"{query_clean}") AND radio'
        else:
            # For other searches, just search for the terms
            if 'radio' not in query_clean.lower():
                search_query = f'{query_clean} AND radio'
            else:
                search_query = query_clean

        params = {
            "q": search_query,
            "fl": "title,author,year,abstract,bibcode,citation_count,pub,doi,keyword",
            "rows": max_results,
            "sort": "citation_count desc"
        }

        try:
            # Increased timeout for slow responses
            response = requests.get(self.base_url, headers=headers, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            # Process results to identify arXiv papers and radio relevance
            docs = data.get('response', {}).get('docs', [])
            for doc in docs:
                bibcode = doc.get('bibcode', '')
                doc['is_arxiv'] = 'arXiv' in bibcode

                # Extract arXiv ID if present
                if doc['is_arxiv']:
                    arxiv_id = self._extract_arxiv_id(bibcode)
                    doc['arxiv_id'] = arxiv_id
                    if arxiv_id:
                        doc['pdf_url'] = f'https://arxiv.org/pdf/{arxiv_id}.pdf'

                # Simple radio relevance check
                abstract = doc.get('abstract', '').lower()
                title = doc.get('title', [''])[0].lower() if doc.get('title') else ''

                # Score relevance
                score = 0
                if query_clean.lower() in title:
                    score += 10
                if 'radio' in title or 'vla' in title or 'alma' in title or 'vlbi' in title:
                    score += 5
                if 'radio' in abstract or 'vla' in abstract or 'alma' in abstract:
                    score += 2

                doc['relevance_score'] = score
                doc['is_radio_paper'] = score > 0 or any(term in abstract + title for term in
                                           ['radio', 'vla', 'alma', 'vlbi', 'ghz', 'mhz'])

            return data

        except Exception as e:
            print(f"NASA ADS error: {e}")
            return {"response": {"docs": []}}

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

    def _extract_arxiv_id(self, bibcode: str) -> Optional[str]:
        """Extract arXiv ID from bibcode"""
        match = re.search(r'arXiv(\d{4})(\d{4,5})', bibcode)
        if match:
            return f"{match.group(1)}.{match.group(2)}"
        return None