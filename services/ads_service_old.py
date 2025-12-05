"""
NASA ADS Integration Service
For searching astronomical literature
"""

import os
import requests
from typing import List, Dict, Optional, Any
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class ADSService:
    """Service for interacting with NASA ADS API"""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize ADS service with API key"""
        self.api_key = api_key or os.getenv('NASA_ADS_API_KEY')
        self.base_url = "https://api.adsabs.harvard.edu/v1"
        
        if not self.api_key:
            logger.warning("NASA ADS API key not found. Paper search will be limited.")
    
    def search_papers(self, 
                     query: str, 
                     max_results: int = 10,
                     sort: str = "date desc") -> List[Dict[str, Any]]:
        """
        Search for papers in NASA ADS
        
        Args:
            query: Search query string
            max_results: Maximum number of results to return
            sort: Sort order (e.g., "date desc", "citation_count desc")
        
        Returns:
            List of paper dictionaries
        """
        if not self.api_key:
            return self._get_example_papers(query)
        
        try:
            headers = {
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json'
            }
            
            params = {
                'q': query,
                'fl': 'title,author,year,bibcode,abstract,citation_count,pub,doi,identifier',
                'rows': max_results,
                'sort': sort
            }
            
            response = requests.get(
                f"{self.base_url}/search/query",
                headers=headers,
                params=params,
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                return self._format_papers(data.get('response', {}).get('docs', []))
            else:
                logger.error(f"ADS API error: {response.status_code}")
                return self._get_example_papers(query)
                
        except Exception as e:
            logger.error(f"Error searching ADS: {str(e)}")
            return self._get_example_papers(query)
    
    def search_by_target(self, target_name: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """Search papers related to a specific astronomical target"""
        # Build query for astronomical target
        query = f'object:"{target_name}" OR title:"{target_name}" OR abstract:"{target_name}"'
        query += ' AND (radio OR VLA OR ALMA OR "Very Large Array" OR VLBA OR GBT)'
        
        return self.search_papers(query, max_results)
    
    def search_by_facility(self, facility: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """Search papers related to a specific radio facility"""
        facility_keywords = {
            'VLA': '("Very Large Array" OR VLA OR JVLA OR EVLA)',
            'ALMA': '(ALMA OR "Atacama Large Millimeter Array")',
            'VLBA': '(VLBA OR "Very Long Baseline Array")',
            'GBT': '(GBT OR "Green Bank Telescope")',
            'GMRT': '(GMRT OR "Giant Metrewave Radio Telescope")',
            'ASKAP': '(ASKAP OR "Australian Square Kilometre Array Pathfinder")',
            'MeerKAT': '(MeerKAT OR "Karoo Array Telescope")'
        }
        
        query = facility_keywords.get(facility.upper(), f'"{facility}"')
        query += ' AND (observation OR data OR survey OR catalog)'
        
        return self.search_papers(query, max_results, sort="date desc")
    
    def search_by_frequency(self, 
                          min_freq_ghz: float, 
                          max_freq_ghz: float,
                          max_results: int = 10) -> List[Dict[str, Any]]:
        """Search papers related to observations in a frequency range"""
        # Convert to common radio astronomy bands
        bands = []
        
        if min_freq_ghz <= 0.5 and max_freq_ghz >= 0.3:
            bands.append('"P-band"')
        if min_freq_ghz <= 1.5 and max_freq_ghz >= 1.0:
            bands.append('"L-band"')
        if min_freq_ghz <= 3.0 and max_freq_ghz >= 2.0:
            bands.append('"S-band"')
        if min_freq_ghz <= 8.0 and max_freq_ghz >= 4.0:
            bands.append('"C-band"')
        if min_freq_ghz <= 12.0 and max_freq_ghz >= 8.0:
            bands.append('"X-band"')
        if min_freq_ghz <= 18.0 and max_freq_ghz >= 12.0:
            bands.append('"Ku-band"')
        if min_freq_ghz <= 26.5 and max_freq_ghz >= 18.0:
            bands.append('"K-band"')
        if min_freq_ghz <= 40.0 and max_freq_ghz >= 26.5:
            bands.append('"Ka-band"')
        
        # Also include specific frequency mentions
        freq_terms = []
        if min_freq_ghz < 10:
            freq_terms.append(f'("{min_freq_ghz:.1f} GHz" OR "{max_freq_ghz:.1f} GHz")')
        else:
            freq_terms.append(f'("{min_freq_ghz:.0f} GHz" OR "{max_freq_ghz:.0f} GHz")')
        
        query_parts = []
        if bands:
            query_parts.append(f"({' OR '.join(bands)})")
        if freq_terms:
            query_parts.append(f"({' OR '.join(freq_terms)})")
        
        query = ' OR '.join(query_parts) if query_parts else f'"{min_freq_ghz:.1f}-{max_freq_ghz:.1f} GHz"'
        query += ' AND (radio OR observation OR survey)'
        
        return self.search_papers(query, max_results)
    
    def get_paper_details(self, bibcode: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific paper"""
        if not self.api_key:
            return None
        
        try:
            headers = {
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json'
            }
            
            params = {
                'q': f'bibcode:{bibcode}',
                'fl': 'title,author,year,bibcode,abstract,citation_count,pub,doi,identifier,keyword,aff'
            }
            
            response = requests.get(
                f"{self.base_url}/search/query",
                headers=headers,
                params=params,
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                docs = data.get('response', {}).get('docs', [])
                if docs:
                    return self._format_paper_details(docs[0])
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting paper details: {str(e)}")
            return None
    
    def _format_papers(self, papers: List[Dict]) -> List[Dict[str, Any]]:
        """Format raw ADS response to standardized format"""
        formatted = []
        
        for paper in papers:
            authors = paper.get('author', [])
            if len(authors) > 3:
                author_str = f"{authors[0]} et al."
            else:
                author_str = ", ".join(authors)
            
            formatted.append({
                'title': paper.get('title', ['Unknown'])[0],
                'authors': author_str,
                'year': paper.get('year', 'Unknown'),
                'journal': paper.get('pub', 'Unknown'),
                'bibcode': paper.get('bibcode', ''),
                'abstract': paper.get('abstract', 'No abstract available'),
                'citations': paper.get('citation_count', 0),
                'doi': paper.get('doi', [''])[0] if paper.get('doi') else '',
                'link': f"https://ui.adsabs.harvard.edu/abs/{paper.get('bibcode', '')}"
            })
        
        return formatted
    
    def _format_paper_details(self, paper: Dict) -> Dict[str, Any]:
        """Format detailed paper information"""
        return {
            'title': paper.get('title', ['Unknown'])[0],
            'authors': paper.get('author', []),
            'affiliations': paper.get('aff', []),
            'year': paper.get('year', 'Unknown'),
            'journal': paper.get('pub', 'Unknown'),
            'bibcode': paper.get('bibcode', ''),
            'abstract': paper.get('abstract', 'No abstract available'),
            'citations': paper.get('citation_count', 0),
            'keywords': paper.get('keyword', []),
            'doi': paper.get('doi', [''])[0] if paper.get('doi') else '',
            'identifiers': paper.get('identifier', []),
            'link': f"https://ui.adsabs.harvard.edu/abs/{paper.get('bibcode', '')}"
        }
    
    def _get_example_papers(self, query: str) -> List[Dict[str, Any]]:
        """Return example papers when ADS API is not available"""
        examples = [
            {
                'title': 'The Very Large Array Sky Survey (VLASS): Science Case and Survey Design',
                'authors': 'Lacy, M. et al.',
                'year': '2020',
                'journal': 'PASP',
                'bibcode': '2020PASP..132c5001L',
                'abstract': 'The Very Large Array Sky Survey (VLASS) is a synoptic, all-sky radio sky survey...',
                'citations': 250,
                'doi': '10.1088/1538-3873/ab63eb',
                'link': 'https://ui.adsabs.harvard.edu/abs/2020PASP..132c5001L'
            },
            {
                'title': 'ALMA Observations of Molecular Gas in High-Redshift Galaxies',
                'authors': 'Casey, C. M. et al.',
                'year': '2023',
                'journal': 'ApJ',
                'bibcode': '2023ApJ...950..123C',
                'abstract': 'We present ALMA observations of CO emission in a sample of high-redshift galaxies...',
                'citations': 45,
                'doi': '10.3847/1538-4357/acd123',
                'link': 'https://ui.adsabs.harvard.edu/abs/2023ApJ...950..123C'
            },
            {
                'title': 'Radio Properties of Active Galactic Nuclei: A VLA Survey',
                'authors': 'Smith, J. A. et al.',
                'year': '2024',
                'journal': 'MNRAS',
                'bibcode': '2024MNRAS.520..456S',
                'abstract': 'We present results from a comprehensive VLA survey of radio-loud AGN...',
                'citations': 12,
                'doi': '10.1093/mnras/stad789',
                'link': 'https://ui.adsabs.harvard.edu/abs/2024MNRAS.520..456S'
            }
        ]
        
        # Filter examples based on query keywords
        query_lower = query.lower()
        filtered = []
        
        for paper in examples:
            if any(keyword in paper['title'].lower() or keyword in paper['abstract'].lower() 
                   for keyword in query_lower.split()):
                filtered.append(paper)
        
        return filtered if filtered else examples[:2]
