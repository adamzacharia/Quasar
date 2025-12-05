"""
QuasarAgent - Core AI agent for radio astronomy operations
Handles natural language processing and tool orchestration
"""

import os
import json
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, field
import openai
from openai import OpenAI


from core.memory import ConversationMemory
from core.tools import ToolRegistry, Tool
from core.prompts import (
    INTENT_CLASSIFICATION_PROMPT,
    ENTITY_EXTRACTION_PROMPT,
    RESPONSE_GENERATION_PROMPT
)
# from integrations.tap import NRAOTapClient
from integrations.datalink import DataLinkClient
from integrations.ads_client import ADSService
from services.search import SearchService
from services.analysis import RadioAnalysisService
from services.rag_service import RAGService
from services.memory_service import MemoryService



@dataclass
class AgentConfig:
    """Configuration for QuasarAgent"""
    api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    ads_api_key: str = field(default_factory=lambda: os.getenv("NASA_ADS_API_KEY", ""))
    model: str = "gpt-4o"  # Changed to gpt-4o for better reasoning
    temperature: float = 0.7
    max_tokens: int = 2000
    max_memory_turns: int = 10
    verbose: bool = False

class QuasarAgent:
    """Main AI agent for radio astronomy operations"""

    def __init__(self, config: Optional[AgentConfig] = None, rag_service: Optional[RAGService] = None):
        """Initialize the Quasar agent"""
        print("DEBUG: Agent init start - VERSION 2")
        self.config = config or AgentConfig()

        if not self.config.api_key:
            raise ValueError("OpenAI API key is required")

        # Initialize OpenAI client
        print("DEBUG: Init OpenAI")
        self.client = OpenAI(api_key=self.config.api_key)

        # Initialize components
        print("DEBUG: Init Memory")
        self.memory = ConversationMemory(max_turns=self.config.max_memory_turns)
        print("DEBUG: Init ToolRegistry")
        self.tool_registry = ToolRegistry()
        # TAP Client removed for ALMA-only scope
        print("DEBUG: Init Search Service")
        self.search_service = SearchService()
        print("DEBUG: Init Analysis Service")
        self.analysis_service = RadioAnalysisService()
        print("DEBUG: Init RAG Service")
        self.rag_service = rag_service or RAGService()
        print("DEBUG: Init Memory Service (Long-Term)")
        self.memory_service = MemoryService()
        print("DEBUG: Init ADS Client")
        ads_key = getattr(self.config, 'ads_api_key', None)
        self.ads_client = ADSService(ads_key) if ads_key else None

        # Register tools
        print("DEBUG: Register Tools")
        self._register_tools()

        # System prompt
        print("DEBUG: Build Prompt")
        self.system_prompt = self._build_system_prompt()

        if self.config.verbose:
            print("[green]QuasarAgent initialized successfully[/green]")
        
        self.last_run_result = None
        self.last_search_results = None
        print("DEBUG: Agent init done")

    def _build_system_prompt(self) -> str:
        """Build the system prompt for the agent"""
        return """You are Quasar, an expert AI assistant for radio astronomy.

You have access to the ALMA Science Archive via the 'alminer' library.
Your goal is to help users find, visualize, and analyze ALMA data.

GUIDELINES:
- **ACTION OVER CHATTER**: If the user asks for data/search/plots, **IMMEDIATELY** call the appropriate tool.
- **NO HALLUCINATIONS**: Only cite data you have retrieved using tools.
- After a tool runs, summarize the output concisely.
- If a search returns many results, offer to plot them (but execute the search first).
- If the user says "yes/proceed" to a previous suggestion, ACT on it immediately.
- **DO NOT** output raw tool usage strings like `[TOOL: ...]` or JSON. Just use the Native Tool Calling feature.

- If the user says "yes/proceed" to a previous suggestion, ACT on it immediately.

Current Context:
Date: {date}
""".format(date=datetime.now().strftime("%Y-%m-%d"))


    def _register_tools(self):
        """Register available tools with the agent using OpenAI Schemas"""

        # Search tools
        self.tool_registry.register(Tool(
            name="search_by_position",
            description="Search NRAO archives by sky position (cone search)",
            function=self._search_by_position,
            parameters={
                "type": "object",
                "properties": {
                    "ra": {"type": "number", "description": "Right ascension in degrees"},
                    "dec": {"type": "number", "description": "Declination in degrees"},
                    "radius": {"type": "number", "description": "Search radius in degrees (default 0.5)"},
                    "facility": {"type": "string", "enum": ["VLA", "VLBA", "ALMA", "GBT"], "description": "Observatory facility. Default to ALMA."},
                    "max_results": {"type": "integer", "description": "Maximum results to return"}
                },
                "required": ["ra", "dec"]
            }
        ))

        self.tool_registry.register(Tool(
            name="search_by_target",
            description="Search NRAO/ALMA archives by target name",
            function=self._search_by_target,
            parameters={
                "type": "object",
                "properties": {
                    "target_name": {"type": "string", "description": "Name of the astronomical target (e.g., 'HL Tau', 'M87')"},
                    "facility": {"type": "string", "enum": ["VLA", "VLBA", "ALMA", "GBT"], "description": "Observatory facility. Default to ALMA."},
                    "max_results": {"type": "integer", "description": "Maximum results to return"}
                },
                "required": ["target_name"]
            }
        ))

        self.tool_registry.register(Tool(
            name="search_by_frequency",
            description="Search archives by frequency range",
            function=self._search_by_frequency,
            parameters={
                "type": "object",
                "properties": {
                    "min_freq_ghz": {"type": "number", "description": "Minimum frequency in GHz"},
                    "max_freq_ghz": {"type": "number", "description": "Maximum frequency in GHz"},
                    "facility": {"type": "string", "description": "Facility name"},
                    "max_results": {"type": "integer", "description": "Max results"}
                },
                "required": ["min_freq_ghz", "max_freq_ghz"]
            }
        ))

        self.tool_registry.register(Tool(
            name="get_observation_details",
            description="Get detailed information about a specific observation",
            function=self._get_observation_details,
            parameters={
                "type": "object",
                "properties": {
                    "obs_id": {"type": "string", "description": "Observation ID or execution block ID"}
                },
                "required": ["obs_id"]
            }
        ))

        # Basic download (legacy)
        self.tool_registry.register(Tool(
            name="download_data",
            description="Legacy download tool (Use download_alma_data instead)",
            function=self._download_data,
            parameters={
                "type": "object",
                "properties": {
                     "obs_id": {"type": "string", "description": "Observation ID"}
                },
                "required": ["obs_id"]
            }
        ))

        self.tool_registry.register(Tool(
            name="analyze_uv_coverage",
            description="Analyze UV coverage for an observation",
            function=self._analyze_uv_coverage,
            parameters={
                "type": "object",
                "properties": {
                    "ms_path": {"type": "string", "description": "Path to measurement set"}
                },
                "required": ["ms_path"]
            }
        ))

        # NEW: ALminer Tools
        self.tool_registry.register(Tool(
            name="search_alma_with_keywords",
            description="Search ALMA archives using specific keywords (pi_name, project_code, etc.)",
            function=self._search_alma_with_keywords,
            parameters={
                "type": "object",
                "properties": {
                    "keywords": {
                        "type": "object", 
                        "description": "Dictionary of keywords e.g. {'pi_name': 'Smith', 'project_code': '2017.1...'}",
                        "additionalProperties": True 
                    }
                },
                "required": ["keywords"]
            }
        ))

        self.tool_registry.register(Tool(
            name="advanced_search",
            description="Execute a custom SQL/TAP query on ALMA archive",
            function=self._advanced_search,
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "ADQL/TAP query string"}
                },
                "required": ["query"]
            }
        ))

        self.tool_registry.register(Tool(
            name="plot_alma_results",
            description="Generate visualization for valid ALMA search results. Must have searched first.",
            function=self._plot_alma_results,
            parameters={
                "type": "object",
                "properties": {
                    "plot_type": {"type": "string", "enum": ["sky", "frequency", "overview"], "description": "Type of plot"}
                },
                 "required": ["plot_type"]
            }
        ))
        
        self.tool_registry.register(Tool(
            name="download_alma_data",
            description="Download ALMA data (FITS) for current results",
            function=self._download_alma_data,
            parameters={
                "type": "object",
                "properties": {
                    "dry_run": {"type": "boolean", "description": "If true, only simulates download. Default True."}
                },
                "required": []
            }
        ))

        self.tool_registry.register(Tool(
            name="search_papers",
            description="Search NASA ADS for research papers",
            function=self._search_papers,
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query keywords or object name"},
                    "sort": {"type": "string", "enum": ["date", "relevance", "citation_count"], "description": "Sort order"}
                },
                "required": ["query"]
            }
        ))

        # NEW: Advanced ALminer Tools
        self.tool_registry.register(Tool(
            name="check_line_coverage",
            description="Check if specific lines are covered in the LAST search results.",
            function=self._check_line_coverage,
            parameters={
                "type": "object",
                "properties": {
                    "line_freq_ghz": {"type": "number", "description": "Frequency in GHz"},
                    "z": {"type": "number", "description": "Redshift (default 0.0)"},
                    "line_name": {"type": "string", "description": "Name of the line"}
                },
                "required": ["line_freq_ghz"]
            }
        ))

        self.tool_registry.register(Tool(
            name="check_co_lines",
            description="Check for CO, 13CO, and C18O lines in the LAST search results.",
            function=self._check_co_lines,
            parameters={
                "type": "object",
                "properties": {
                    "z": {"type": "number", "description": "Redshift (default 0.0)"}
                },
                "required": []
            }
        ))

        self.tool_registry.register(Tool(
            name="search_catalog",
            description="Search for a catalog of objects (Name, RA, Dec)",
            function=self._search_catalog,
            parameters={
                "type": "object",
                "properties": {
                    "objects": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "Name": {"type": "string"},
                                "RAJ2000": {"type": "number"},
                                "DEJ2000": {"type": "number"}
                            },
                             "required": ["Name"]
                        },
                        "description": "List of objects with Name, RA, Dec"
                    }
                },
                "required": ["objects"]
            }
        ))

    def _search_by_position(self, ra: float, dec: float, radius: float = 0.5,
                           facility: Optional[str] = None,
                           max_results: int = 100) -> Dict[str, Any]:
        """Search archives by sky position"""
        try:
            results = self.search_service.cone_search(
                ra, dec, radius, facility, max_results
            )
            # Cache for plotting/download
            self.last_search_results = results
            self.last_run_result = {"type": "data", "data": results, "source": f"Position Search: {ra}, {dec}"}
            
            return {
                "success": True,
                "count": len(results),
                "results": results.to_dict("records") if not results.empty else []
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _search_by_target(self, target_name: str, facility: Optional[str] = None,
                          date_range: Optional[str] = None,
                          max_results: int = 100) -> Dict[str, Any]:
        """Search archives by target name"""
        try:
            results = self.search_service.search_by_target(
                target_name, facility, date_range, max_results
            )
            self.last_search_results = results
            self.last_run_result = {"type": "data", "data": results, "source": f"Target Search: {target_name}"}
            
            return {
                "success": True,
                "count": len(results),
                "results": results.to_dict("records") if not results.empty else []
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _search_by_frequency(self, min_freq_ghz: float, max_freq_ghz: float,
                            facility: Optional[str] = None,
                            max_results: int = 100) -> Dict[str, Any]:
        """Search archives by frequency range"""
        try:
            results = self.search_service.search_by_frequency(
                min_freq_ghz, max_freq_ghz, facility, max_results
            )
            self.last_search_results = results
            self.last_run_result = {"type": "data", "data": results, "source": f"Freq Search: {min_freq_ghz}-{max_freq_ghz} GHz"}
            
            return {
                "success": True,
                "count": len(results),
                "results": results.to_dict("records") if not results.empty else []
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _get_observation_details(self, obs_id: str) -> Dict[str, Any]:
        """Get detailed observation information"""
        try:
            details = self.search_service.get_observation_details(obs_id)
            return {"success": True, "details": details}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _download_data(self, obs_id: str, output_dir: str = "./data") -> Dict[str, Any]:
        """Legacy download implementation for non-ALMA or general use"""
        # This one is the original stub. We won't modify it to avoid breaking older tests/tools
        # But we added _download_alma_data which is the real one.
        try:
            return {
                "success": False,
                "error": "Basic download not supported. Use download_alma_data for ALMA observations.",
                "obs_id": obs_id
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _analyze_uv_coverage(self, ms_path: str) -> Dict[str, Any]:
        """Analyze UV coverage of a measurement set"""
        try:
            analysis = self.analysis_service.analyze_uv_coverage(ms_path)
            # Store analysis result potentially?
            self.last_run_result = {"type": "analysis", "data": analysis}
            return {"success": True, "analysis": analysis}
        except Exception as e:
            return {"success": False, "error": str(e)}
            
    # NEW METHODS implemented from SearchService enhancements
    
    def _search_alma_with_keywords(self, keywords: Dict[str, Any]) -> Dict[str, Any]:
        try:
            # Handle string input if LLM passed JSON string
            if isinstance(keywords, str):
                keywords = json.loads(keywords)
                
            results = self.search_service.search_alma_with_keywords(keywords)
            self.last_search_results = results
            self.last_run_result = {"type": "data", "data": results, "source": f"Keywords: {keywords}"}
            
            return {
                "success": True,
                "count": len(results),
                "results": results.to_dict("records") if not results.empty else []
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _advanced_search(self, query: str) -> Dict[str, Any]:
        try:
            results = self.search_service.advanced_search(query)
            self.last_search_results = results
            self.last_run_result = {"type": "data", "data": results, "source": f"SQL: {query}"}
            
            return {
                "success": True,
                "count": len(results),
                "results": results.to_dict("records") if not results.empty else []
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _plot_alma_results(self, plot_type: str = "sky") -> Dict[str, Any]:
        """Generate plots. Uses the LAST search results."""
        try:
            if not hasattr(self, 'last_search_results') or self.last_search_results is None or self.last_search_results.empty:
                 return {"success": False, "error": "No results available to plot. Please run a search first."}
            
            path = self.search_service.plot_alma_results(self.last_search_results, plot_type)
            if path:
                self.last_run_result = {"type": "image", "path": path, "caption": f"ALMA {plot_type.capitalize()} Plot"}
                return {"success": True, "path": path}
            return {"success": False, "error": "Plot generation returned empty path"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _search_papers(self, query: str, sort: str = "date desc") -> Dict[str, Any]:
        """Search NASA ADS for papers"""
        try:
            if not self.ads_client:
                 return {"success": False, "error": "NASA ADS Client not initialized (check API Key)"}
            
            # Use the smart natural language search from the new client
            # This uses LLM to build the optimal ADS query
            result_pkg = self.ads_client.search_natural_language(query, max_results=20, sort=sort)
            results = pd.DataFrame(result_pkg['papers'])
            
            # Standardize output for UI
            self.last_run_result = {
                "type": "papers", 
                "data": results, 
                "source": f"ADS: {result_pkg['query']}"
            }
            return {
                "success": True, 
                "count": len(results), 
                "top_title": results.iloc[0]['title'] if not results.empty else "No results"
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _download_alma_data(self, dry_run: bool = True) -> Dict[str, Any]:
        """Download ALMA data for observations in current context"""
        try:
            if not hasattr(self, 'last_search_results') or self.last_search_results is None or self.last_search_results.empty:
                 return {"success": False, "error": "No results available to download."}
            
            msg = self.search_service.download_alma_data(self.last_search_results, dry_run=dry_run)
            return {"success": True, "message": msg}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _check_line_coverage(self, line_freq_ghz: float, z: float = 0.0, line_name: str = "Line") -> Dict[str, Any]:
        """Check line coverage on cache"""
        if self.last_search_results is None or self.last_search_results.empty:
             return {"success": False, "error": "No previous search results to check. Run a search first."}
        
        try:
            results = self.search_service.check_line_coverage_on_last(
                self.last_search_results, line_freq_ghz, z, line_name
            )
            # Don't overwrite last_search_results, just return analysis? 
            # Or do we overwrite context? Let's overwrite so we can plot THIS result.
            self.last_search_results = results
            self.last_run_result = {"type": "data", "data": results, "source": f"Line Check: {line_name} @ {line_freq_ghz}GHz"}
            
            return {
                "success": True, 
                "count": len(results), 
                "results": results.to_dict("records")
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _check_co_lines(self, z: float = 0.0) -> Dict[str, Any]:
        """Check CO lines on cache"""
        if self.last_search_results is None or self.last_search_results.empty:
             return {"success": False, "error": "No previous search results to check. Run a search first."}
             
        try:
            results = self.search_service.check_co_lines_on_last(self.last_search_results, z)
            self.last_search_results = results
            self.last_run_result = {"type": "data", "data": results, "source": "CO Lines Check"}
            
            return {
                "success": True, 
                "count": len(results), 
                "results": results.to_dict("records")
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _search_catalog(self, objects: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Search by catalog"""
        try:
            # restructure for service: dict of lists
            # Input: [{"Name": "A", "RA": 1}, {"Name": "B"}]
            # Output needed: {"Name": ["A", "B"], ...}
            
            # Simple pivot
            catalog = {}
            if objects:
                keys = objects[0].keys()
                for k in keys:
                    catalog[k] = [o.get(k) for o in objects]
            
            results = self.search_service.search_catalog(catalog)
            self.last_search_results = results
            self.last_run_result = {"type": "data", "data": results, "source": "Catalog Search"}
            
            return {
                "success": True, 
                "count": len(results), 
                "results": results.to_dict("records")
            }
        except Exception as e:
            return {"success": False, "error": str(e)}



    def determine_intent(self, query: str) -> Dict[str, Any]:
        """Determine the user's intent from the query"""
        try:
            prompt = INTENT_CLASSIFICATION_PROMPT.format(query=query)
            
            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            if self.config.verbose:
                print(f"[cyan]Intent Detected: {result.get('intent')} ({result.get('confidence')})[/cyan]")
            return result
        except Exception as e:
            print(f"[red]Intent classification failed: {e}[/red]")
            return {"intent": "QUESTION", "confidence": 0.0}

    def extract_entities(self, query: str) -> Dict[str, Any]:
        """Extract search entities from the query"""
        try:
            prompt = ENTITY_EXTRACTION_PROMPT.format(query=query)
            
            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            if self.config.verbose:
                print(f"[cyan]Entities Extracted: {result}[/cyan]")
            return result
        except Exception as e:
            print(f"[red]Entity extraction failed: {e}[/red]")
            return {}

    def analyze_query_intent(self, query: str) -> Dict[str, Any]:
        """Alias for determine_intent to match UI expectation"""
        return self.determine_intent(query)

    def process_query(self, query: str, user_id: str = "user") -> Tuple[Optional[Any], str, str]:
        """Process a user query and return (result, source_name, result_type)"""
        self.memory.add_message("user", query)

        # 1. Determine Intent
        intent_data = self.determine_intent(query)
        intent = intent_data.get("intent", "QUESTION")
        
        # 2. Extract Entities
        entities = self.extract_entities(query)
        source_name = entities.get("source_name", "")
        
        # Handle specific intents
        if intent == "SEARCH" or "search" in query.lower() and "paper" not in query.lower():
            # Data Search
            if not source_name:
                # Try to extract from query if entity extraction failed
                # Simple heuristic: look for capitalized words that aren't keywords
                pass

            if source_name:
                # Use search service (which uses TAP)
                # For now, we'll use the search service's target search
                # In the future, we might want to use the HybridSearchClient logic if we port it
                try:
                    results = self.search_service.search_by_target(
                        target_name=source_name,
                        facility="ALMA",
                        max_results=50
                    )
                    return results, source_name, "data"
                except Exception as e:
                    print(f"Search failed: {e}")
                    return None, source_name, "data"
            else:
                 return None, "", "clarification"

        elif intent == "PAPERS" or "paper" in query.lower():
            # Paper Search
            if source_name:
                results = self.search_papers(source_name)
                return results, source_name, "papers"
            else:
                return None, "", "clarification"

        else:
            # General Question - Handled by stream_general_response usually, 
            # but if process_query is called, we return None to signal no structured data
            return None, "", "general"

    def stream_general_response(self, query: str, message_placeholder=None, user_id: str = "user") -> str:
        """Stream a general response using RAG and LLM with Native Tool Support"""
        
        # Helper to check for simple confirmations
        def is_simple_confirmation(q):
            clean = q.strip().lower()
            return clean in ["yes", "proceed", "ok", "go ahead", "sure", "please", "confirm"] or len(clean.split()) < 3
            
        # Skip RAG for simple confirmations OR direct commands that don't need manual lookup
        clean_q = query.strip().lower()
        skip_rag = is_simple_confirmation(query) or clean_q.startswith(("@archive", "@paper"))
        
        # Retrieve context from RAG service
        rag_context = ""
        if not skip_rag:
            try:
                docs = self.rag_service.search(query)
                if docs:
                    # Enrich context with citation metadata
                    context_pieces = []
                    for d in docs:
                        source = d.metadata.get('source', 'ALMA Manual')
                        page = d.metadata.get('page', 'Unknown')
                        context_pieces.append(f"Content: {d.page_content}\n[Citation: {source}, Page: {page}]")
                    
                    rag_context = "\n\nRelevant Information from ALMA Manual:\n" + "\n---\n".join(context_pieces)
                    
                    if message_placeholder:
                        message_placeholder.markdown("📘 Consulting ALMA Manual...")
            except Exception as e:
                print(f"RAG search failed: {e}")

        # Retrieve Long-Term Memory
        ltm_context = ""
        if not skip_rag:
            try:
                memories = self.memory_service.search_memories(query, user_id=user_id)
                if memories:
                    ltm_context = "\nRelevant User Facts:\n" + "\n".join([f"- {m}" for m in memories])
            except Exception as e:
                print(f"Memory search failed: {e}")

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "system", "content": "You are answering the user's LATEST question. Do not assume you need to continue a previous task unless explicitly asked."},
            {"role": "system", "content": RESPONSE_GENERATION_PROMPT.format(context=f"Relevant Technical Context:\n{rag_context}\n{ltm_context}", query=query)}
        ]
        
        # COMMAND HANDLING
        tool_choice = "auto"
        clean_query = query.strip().lower()
        
        if clean_query.startswith("@archive"):
            messages.append({"role": "system", "content": "USER COMMAND: @Archive detected. You MUST use a search tool (search_by_target, etc.) IMMEDIATELY. DO NOT OUTPUT ANY TEXT. DO NOT SAY 'I will search'. ARTIFACTS OR TABLES WILL APPEAR AUTOMATICALLY. JUST CALL THE FUNCTION."})
            tool_choice = "required" 
            
        if clean_query.startswith("@search"):
            messages.append({"role": "system", "content": "USER COMMAND: @search detected. FORCE RAG/LLM ONLY. Do NOT use any external search tools. Answer strictly from the provided 'Relevant Technical Context'. You MUST cite the Source and Page Number for every claim. Example: (ALMA Cycle 10 Handbook, Page 42)."})
            tool_choice = "none"

        if clean_query.startswith("@paper"):
             messages.append({"role": "system", "content": "USER COMMAND: @paper detected. You MUST use the 'search_papers' tool. Do not use ALMA data search."})
             tool_choice = "required" # Force a tool, and prompt implies search_papers preference
        
        # Add conversation history
        # Add conversation history only if NOT in strict search mode
        # This prevents "memory leakage" where previous data searches confuse the RAG answer
        if not clean_query.startswith("@search"):
            for msg in self.memory.get_history():
                messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
            
        try:
            # Prepare tools and choice
            tools = self.tool_registry.get_openai_tools()
            
            # If command forced "none", explicitly remove tools to avoid API errors
            # and prevent any chance of tool usage (Pure RAG)
            if tool_choice == "none":
                tools = None
                tool_choice = None
            
            # Safety check: if registry empty, cannot use tools
            if not tools:
                tools = None
                tool_choice = None 

            if tools:
                # Standard call WITH tools
                stream = self.client.chat.completions.create(
                    model=self.config.model,
                    messages=messages,
                    tools=tools,
                    tool_choice=tool_choice,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                    stream=True
                )
            else:
                # Clean call WITHOUT tools (Prevention of Error 400)
                # Used for @search or when no tools are available.
                stream = self.client.chat.completions.create(
                    model=self.config.model,
                    messages=messages,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                    stream=True
                )
            
            full_response = ""
            tool_calls = [] # List of dicts to build up tool calls
            
            for chunk in stream:
                delta = chunk.choices[0].delta
                
                # 1. Content Streaming
                if delta.content is not None:
                    full_response += delta.content
                    if message_placeholder:
                        message_placeholder.markdown(full_response + "▌")
                        
                # 2. Tool Call Streaming
                if delta.tool_calls:
                    for tc in delta.tool_calls:
                        if len(tool_calls) <= tc.index:
                            tool_calls.append({"id": tc.id, "function": {"name": "", "arguments": ""}})
                        
                        if tc.function.name:
                            tool_calls[tc.index]["function"]["name"] += tc.function.name
                        if tc.function.arguments:
                            tool_calls[tc.index]["function"]["arguments"] += tc.function.arguments
            
            if message_placeholder:
                message_placeholder.markdown(full_response)
                
            self.memory.add_message("assistant", full_response)
            
            # EXECUTE TOOLS
            for tc in tool_calls:
                tool_name = tc["function"]["name"]
                args_str = tc["function"]["arguments"]
                
                try:
                    params = json.loads(args_str)
                    if self.config.verbose:
                        print(f"Executing Tool: {tool_name} with {params}")
                    
                    tool_def = self.tool_registry.get_tool(tool_name)
                    if tool_def:
                        result = tool_def.function(**params)
                        
                        if self.config.verbose:
                            print(f"Tool Result: {result}")
                        
                        # Feed result back into memory
                        summary = self._summarize_tool_output(tool_name, result)
                        self.memory.add_message("system", f"Tool '{tool_name}' executed. Output: {summary}")
                        
                        # Optionally: append tool output to UI if needed, or rely on next turn
                        # For now, we just update memory so the next turn knows.
                        
                except json.JSONDecodeError:
                    print(f"Failed to parse arguments for {tool_name}: {args_str}")
                except Exception as tool_err:
                    print(f"Tool Execution Failed: {tool_err}")
                    self.memory.add_message("system", f"Tool execution failed: {tool_err}")

            # Async Memory Update
            self._update_memory(query, user_id=user_id)
            
            return full_response
            
        except Exception as e:
            error_msg = f"Error generating response: {str(e)}"
            if message_placeholder:
                message_placeholder.error(error_msg)
            return error_msg

    def _summarize_tool_output(self, tool_name: str, result: Any) -> str:
        """Summarize tool output for context window efficiency"""
        import pandas as pd
        
        if isinstance(result, dict):
            # Handle specific result types
            if "results" in result:
                data = result["results"]
                if isinstance(data, list) and len(data) > 0:
                    count = len(data)
                    # Convert to DF for easy peeking. 
                    # Note: data is list of dicts here from the wrapper
                    df = pd.DataFrame(data)
                    
                    # Extract key info for summary
                    columns = df.columns.tolist()
                    sample = df.head(3).to_dict('records')
                    
                    summary = f"Found {count} records. Columns: {columns}. Sample: {sample}"
                    return summary
            
            if "path" in result:
                return f"Image generated at {result['path']}"
                
            # If generic dict, simplified str
            return str(result)[:500] + "..." if len(str(result)) > 500 else str(result)
            
        return str(result)[:500]

    def generate_summary(self, df: pd.DataFrame, source_name: str) -> str:
        """Generate a natural language summary of the search results"""
        try:
            if df.empty:
                return f"No observations found for {source_name}."

            # Prepare a summary of the data for the LLM
            total_obs = len(df)
            unique_projects = df['project_code'].nunique() if 'project_code' in df.columns else 0
            
            bands = []
            # Check for standardized 'Band' column first, then fallback
            if 'Band' in df.columns:
                bands = sorted(df['Band'].dropna().unique().tolist())
            elif 'band_number' in df.columns:
                bands = sorted(df['band_number'].dropna().unique().tolist())
            elif 'band' in df.columns:
                bands = sorted(df['band'].dropna().unique().tolist())
                
            # Check for standardized freq columns
            min_freq = 0
            max_freq = 0
            
            if 'freq_min' in df.columns:
                min_freq = df['freq_min'].min()
            elif 'freq_min_ghz' in df.columns:
                min_freq = df['freq_min_ghz'].min()
                
            if 'freq_max' in df.columns:
                max_freq = df['freq_max'].max()
            elif 'freq_max_ghz' in df.columns:
                max_freq = df['freq_max_ghz'].max()
            
            context = f"""
            Search Results for {source_name}:
            - Total Observations: {total_obs}
            - Unique Projects: {unique_projects}
            - Bands Covered: {bands}
            - Frequency Range: {min_freq:.1f} - {max_freq:.1f} GHz
            """
            
            prompt = f"""
            Summarize these ALMA observation results for the source {source_name}. 
            Highlight the key available data, such as the bands, frequency coverage, and the volume of data.
            Keep it concise (2-3 sentences).
            
            Data Context:
            {context}
            """
            
            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"[red]Summary generation failed: {e}[/red]")
            return f"Found {len(df)} observations for {source_name}."

    def reset_conversation(self):
        """Reset the conversation memory"""
        self.memory.clear()
        if self.config.verbose:
            print("[yellow]Conversation memory cleared[/yellow]")

    def get_conversation_history(self) -> List[Dict[str, str]]:
        """Get the current conversation history"""
        return self.memory.get_history()

    def set_model(self, model: str):
        """Change the LLM model"""
        self.config.model = model
        if self.config.verbose:
            print(f"[cyan]Model changed to: {model}[/cyan]")

    def _update_memory(self, query: str, user_id: str = "user"):
        """Extract and save new memories from user interaction"""
        try:
            # Simple extraction prompt
            prompt = f"""
            Extract any personal facts, preferences, or research interests from the user's message.
            If there are none, return "NONE".
            
            User Message: "{query}"
            
            Output format: Just the fact string or "NONE".
            Example: "User is interested in protoplanetary disks."
            """
            
            response = self.client.chat.completions.create(
                model="gpt-4o-mini", # Use cheaper model for background tasks
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=50
            )
            
            fact = response.choices[0].message.content.strip()
            
            if fact != "NONE" and len(fact) > 5:
                if self.config.verbose:
                    print(f"[magenta]New Memory: {fact}[/magenta]")
                self.memory_service.add_memory(fact, user_id=user_id)
                
        except Exception as e:
            if self.config.verbose:
                print(f"[red]Memory update failed: {e}[/red]")

    def search_papers(self, source_name: str) -> Optional[pd.DataFrame]:
        """Search for papers using NASA ADS or arXiv fallback"""
        if not source_name:
            return None

        # Use NASA ADS if available
        if self.ads_client:
            papers = self.ads_client.search_radio_papers(source_name)
            if papers:
                return pd.DataFrame(papers)
        
        return None
