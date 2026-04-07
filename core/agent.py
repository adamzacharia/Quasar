"""
QuasarAgent — Central orchestrator for Quasar AI.

CALLED BY: ui/app.py (Streamlit), ui-pro/api/main.py (FastAPI SSE),
           core/cli.py (terminal REPL), telegram.py (webhook)
CALLS:     All services/* modules, integrations/*, core/rlm.py,
           OpenAI API (GPT-4o), mem0 (long-term memory)

This is the heart of Quasar. The QuasarAgent class:
  1. Registers 27+ tools as OpenAI function-calling schemas
  2. Routes user queries through RLM complexity detection
  3. Manages RAG context, conversation memory, and long-term memory
  4. Streams responses via Chat Completions API or Responses API
  5. Caches search results (DataFrames) for follow-up operations
"""

import os
import json
import re
import threading
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, field
import openai
from openai import OpenAI
from core.llm_client import LLMClient, detect_provider

from core.logger import logger, log_tool


from core.memory import ConversationMemory
from core.tools import ToolRegistry, Tool
from core.prompts import (
    INTENT_CLASSIFICATION_PROMPT,
    ENTITY_EXTRACTION_PROMPT,
    RESPONSE_GENERATION_PROMPT,
    ALMA_TAP_SCHEMA
)
# from integrations.tap import NRAOTapClient
from integrations.datalink import DataLinkClient
from integrations.ads_client import ADSService
from services.search import SearchService
from services.analysis import RadioAnalysisService
from services.rag_service import RAGService
from services.memory_service import MemoryService
from core.rlm import RecursiveLanguageModel
from services.browser import BrowserService
from services.plotting import PlottingService
from services.splatalogue import SplatalogueTool
from services.multi_archive import MultiArchiveMatcher
from services.casa_generator import CASAScriptGenerator
from services.gcn_monitor import GCNAlertMonitor
from services.notebook_gen import generate_analysis_notebook
from services.pdf_processing import PDFProcessingService
from services.fits_processing import FITSProcessingService
from core.prompts.lit_to_code import LIT_TO_CODE_PROMPT

# Phase 1-4: Multi-Agent Workforce modules
from core.conductor import Conductor
from core.model_router import ModelRouter
from core.recovery import RecoveryEngine
from core.observability import QueryTracer
from core.agent_pool import AgentPool

# Phase 5: OpenClaude-inspired reliability & context management
from core.context_manager import ContextManager
from core.session_memory import SessionMemory
from core.token_budget import TokenBudget
from core.tool_budget import apply_tool_result_budget
from core.health_monitor import HealthMonitor

# mem0 DISABLED — it pulls in sentence-transformers + PyTorch (~1–1.5GB RAM),
# which causes OOM on 2GB Render instances. The app already uses Qdrant + RAG
# for knowledge persistence. Set ENABLE_MEM0=1 to re-enable if you have ≥4GB.
import os as _os_mem0
if _os_mem0.getenv("ENABLE_MEM0", "").strip() in ("1", "true", "yes"):
    try:
        from mem0 import Memory as Mem0Memory
        MEM0_AVAILABLE = True
        print("[INFO] mem0 long-term memory enabled (ENABLE_MEM0=1)")
    except ImportError:
        MEM0_AVAILABLE = False
        print("[WARNING] mem0 not installed. Long-term memory disabled.")
else:
    MEM0_AVAILABLE = False
    print("[INFO] mem0 disabled to save memory. Set ENABLE_MEM0=1 to enable.")



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
    # Responses API configuration
    use_responses_api: bool = False  # Toggle for Responses API vs Chat Completions
    mcp_server_url: str = "http://localhost:8000/sse"  # MCP server SSE endpoint
    enable_mcp: bool = False  # Enable MCP tool connection
    # User context — needed to load custom tools on startup
    user_id: str = ""

class QuasarAgent:
    """Main AI agent for radio astronomy operations"""

    def __init__(self, config: Optional[AgentConfig] = None, rag_service: Optional[RAGService] = None):
        """Initialize the Quasar agent"""
        print("DEBUG: Agent init start - VERSION 2")
        self.config = config or AgentConfig()

        # API key validation — only required for cloud providers
        provider = detect_provider(self.config.model)
        if provider == "openai" and not self.config.api_key:
            raise ValueError(
                "OPENAI_API_KEY is required for OpenAI models. "
                "Set it in .env or use a local model (prefix with 'local/')."
            )

        # Initialize unified LLM client (routes to OpenAI/Claude/Gemini/Local)
        print(f"DEBUG: Init LLMClient (provider={provider}, model={self.config.model})")
        self.client = LLMClient(model=self.config.model)

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
        import os as _os
        ads_key = getattr(self.config, 'ads_api_key', None) or _os.getenv("NASA_ADS_API_KEY")
        self.ads_client = ADSService(ads_key) if ads_key else ADSService()  # ADSService handles missing key gracefully

        # Register tools
        print("DEBUG: Register Tools")
        self._register_tools()

        # System prompt
        print("DEBUG: Build Prompt")
        self.system_prompt = self._build_system_prompt()

        if self.config.verbose:
            print("[green]QuasarAgent initialized successfully[/green]")
        
        self.last_run_result = None
        self._accumulated_run_results = []  # All tool results in a session (multi-target support)
        self.last_search_results = None
        
        # Responses API state tracking
        self.last_response_id = None  # For conversation continuity
        self._session_token_estimate = 0  # Running token count estimate (legacy — kept for compat)
        self._session_token_limit = 90000  # Legacy threshold (superseded by ContextManager)
        
        # Initialize mem0 long-term memory (if available)
        self.long_term_memory = None
        if MEM0_AVAILABLE:
            try:
                print("DEBUG: Init mem0 Long-Term Memory")
                self.long_term_memory = Mem0Memory()
            except Exception as e:
                print(f"[WARNING] mem0 initialization failed: {e}")
        
        # Initialize BrowserService for web browsing capabilities
        print("DEBUG: Init BrowserService")
        self.browser_service = BrowserService()

        # Initialize new science services
        print("DEBUG: Init PlottingService")
        self.plotting_service = PlottingService()
        print("DEBUG: Init SplatalogueTool")
        self.splatalogue_tool = SplatalogueTool()
        print("DEBUG: Init MultiArchiveMatcher")
        self.multi_archive = MultiArchiveMatcher()
        print("DEBUG: Init CASAScriptGenerator")
        self.casa_generator = CASAScriptGenerator()
        print("DEBUG: Init GCNAlertMonitor")
        self.gcn_monitor = GCNAlertMonitor()
        print("DEBUG: Init PDFProcessingService")
        self.pdf_service = PDFProcessingService(self.config.api_key)

        # Phase 0: DataLink + FITS remote header
        print("DEBUG: Init DataLinkClient")
        self.datalink_client = DataLinkClient()
        print("DEBUG: Init FITSProcessingService")
        self.fits_service = FITSProcessingService()

        # Phase 1-4: Multi-Agent Workforce infrastructure
        print("DEBUG: Init ModelRouter")
        self.model_router = ModelRouter(default_model=self.config.model)
        print("DEBUG: Init RecoveryEngine")
        self.recovery_engine = RecoveryEngine(
            client=self.client, model=self.config.model, verbose=True
        )
        print("DEBUG: Init QueryTracer")
        self.query_tracer = QueryTracer()
        print("DEBUG: Init AgentPool")

        # Phase 5: OpenClaude-inspired modules
        print("DEBUG: Init ContextManager")
        self.context_manager = ContextManager(
            client=self.client, model=self.config.model,
        )
        print("DEBUG: Init SessionMemory")
        self.session_memory = SessionMemory(client=self.client)
        print("DEBUG: Init HealthMonitor")
        self.health_monitor = HealthMonitor()
        # Wire health monitor into model router
        self.model_router.health_monitor = self.health_monitor
        self.agent_pool = AgentPool()
        print("DEBUG: Init Conductor")
        self.conductor = Conductor(
            client=self.client,
            model=self.config.model,
            tool_executor=self._conductor_tool_executor,
            recovery_engine=self.recovery_engine,
            model_router=self.model_router,
            agent_pool=self.agent_pool,
            verbose=True,
        )

        # Initialize RLM (Recursive Language Model) for complex queries
        print("DEBUG: Init RLM")
        self.rlm = RecursiveLanguageModel(
            client=self.client,
            model=self.config.model,
            tool_executor=self._rlm_tool_executor,
            verbose=self.config.verbose,
        )
        # Wire all 27 registered tools into the REPL executor so the LLM can
        # call any Quasar tool from within REPL Python code via call_tool()
        self.rlm.repl_executor.tool_executor = self._rlm_tool_bridge

        # Load per-user custom tools (if user_id is set)
        if self.config.user_id:
            print(f"DEBUG: Loading user tools for {self.config.user_id}")
            self._load_user_tools(self.config.user_id)
            print(f"DEBUG: Loading MCP servers for {self.config.user_id}")
            self._load_mcp_servers(self.config.user_id)
        
        print("DEBUG: Agent init done")

    # ── Custom User Tools ──────────────────────────────────────────────────

    def _load_user_tools(self, user_id: str):
        """
        Load per-user custom tool definitions from user_tools/<user_id>/tools.json,
        inject their API key secrets into os.environ, then register each callable
        into the live ToolRegistry so the LLM can call them immediately.
        """
        try:
            from services.user_tools_service import UserToolsService
            svc = UserToolsService()

            # Inject API key secrets before building callables
            svc.inject_secrets_to_env(user_id)

            tools_loaded = 0
            for tool_def in svc.load_tools(user_id):
                try:
                    fn = svc.build_callable(tool_def)
                    self.tool_registry.register(Tool(
                        name=tool_def["name"],
                        description=tool_def["description"],
                        function=fn,
                        parameters=tool_def["parameters"],
                        category="custom",
                    ))
                    tools_loaded += 1
                    print(f"[UserTools] Registered custom tool: {tool_def['name']}")
                except Exception as e:
                    print(f"[UserTools] Skipping '{tool_def.get('name', '?')}': {e}")

            if tools_loaded:
                print(f"[UserTools] {tools_loaded} custom tool(s) loaded for user '{user_id}'")
        except Exception as e:
            print(f"[UserTools] Failed to load user tools: {e}")

    def _load_mcp_servers(self, user_id: str):
        """
        Load per-user MCP server definitions from user_tools/<user_id>/mcp_servers.json,
        spawn them via stdio, and register their exported tools into ToolRegistry.
        """
        try:
            from services.mcp_server_service import MCPServerService
            import asyncio
            import threading
            from mcp.client.stdio import stdio_client, StdioServerParameters
            from mcp.client.sse import sse_client
            from mcp.client.session import ClientSession
            
            svc = MCPServerService()
            configs = svc.load_servers(user_id)
            if not configs:
                return
                
            # We must maintain sessions for the lifetime of the agent.
            if not hasattr(self, "_mcp_exit_stacks"):
                self._mcp_exit_stacks = []
                
            def _start_mcp_bridge(config):
                # We need to run the async MCP client bridging in a background thread/event loop
                # because the stdio client is fully async, but Quasar's tool_registry expects sync callables.
                async def _run_client():
                    from contextlib import AsyncExitStack
                    stack = AsyncExitStack()
                    
                    try:
                        transport = config.get("transport", "stdio")
                        
                        if transport == "http":
                            url = config.get("url")
                            if not url:
                                raise ValueError("HTTP transport requires a URL")
                            # sse_client requires the URL
                            read, write = await stack.enter_async_context(sse_client(url))
                        else:
                            # Default to stdio
                            env = os.environ.copy()
                            env.update(config.get("env", {}))
                            
                            server_params = StdioServerParameters(
                                command=config["command"],
                                args=config.get("args", []),
                                env=env
                            )
                            read, write = await stack.enter_async_context(stdio_client(server_params))
                    
                        session = await stack.enter_async_context(ClientSession(read, write))
                        await session.initialize()
                        
                        # List exported tools
                        tools_resp = await session.list_tools()
                        
                        for mcp_tool in tools_resp.tools:
                            t_name = f"{config['name']}__{mcp_tool.name}"
                            t_desc = mcp_tool.description or f"Tool {mcp_tool.name} from {config['name']}"
                            t_params = mcp_tool.inputSchema
                            
                            # Build a sync wrapper that calls the async session.call_tool
                            def make_wrapper(session_ref, orig_name):
                                def wrapper(**kwargs):
                                    # Since we're in a sync context (LLM Tool call), we need to run the 
                                    # async call_tool in the background loop. 
                                    # Using asyncio.run directly here might clash with Streamlit's loop,
                                    # but since tools run in their own thread in _rlm_tool_executor or 
                                    # the main loop, we'll spawn a quick loop just for the tool call.
                                    import asyncio
                                    async def _do_call():
                                        res = await session_ref.call_tool(orig_name, arguments=kwargs)
                                        # mcp result format usually has .content array
                                        if getattr(res, "content", None):
                                            return [c.text for c in res.content if getattr(c, 'type', '') == 'text']
                                        elif getattr(res, "isError", False):
                                            return {"error": "Tool execution failed"}
                                        return {"status": "success"}
                                    try:
                                        # If there's a running loop, this might fail, but streamlit usually
                                        # runs event handlers in dummy threads without loops.
                                        loop = asyncio.new_event_loop()
                                        return loop.run_until_complete(_do_call())
                                    except Exception as e:
                                        return {"error": str(e)}
                                return wrapper
                            
                            sync_fn = make_wrapper(session, mcp_tool.name)
                            
                            self.tool_registry.register(Tool(
                                name=t_name,
                                description=t_desc,
                                function=sync_fn,
                                parameters=t_params,
                                category="mcp"
                            ))
                            print(f"[MCPServers] Registered bridged tool: {t_name}")
                            
                        self._mcp_exit_stacks.append(stack)
                        
                        # We must keep the event loop alive so the stdio pipes don't close.
                        # This is a bit of a hack: just sleep forever in this thread.
                        while True:
                            await asyncio.sleep(3600)
                            
                    except Exception as e:
                        print(f"[MCPServers] Failed to bridge {config['name']}: {e}")
                
                # Run the bridge setup in a dedicated background thread per server
                # This ensures the async context manager stays alive and connected.
                t = threading.Thread(target=lambda: asyncio.run(_run_client()), daemon=True)
                t.start()
                
            for cfg in configs:
                _start_mcp_bridge(cfg)
                
        except Exception as e:
            print(f"[MCPServers] Failed to set up MCP clients: {e}")

    # ── Token helpers ──────────────────────────────────────────────────────

    def _estimate_tokens(self, text: str) -> int:
        """Rough token estimate: ~4 chars per token for English text."""
        return len(text) // 4

    def _prune_session_if_needed(self, query: str, user_id: str):
        """
        Smart context management — replaces the old session nuke with
        LLM-powered summarization via ContextManager.

        Legacy: used to reset last_response_id and lose ALL context.
        New: summarizes old turns, keeps recent context intact.
        """
        # Note: For the Responses API path (previous_response_id chaining),
        # context management is handled server-side by OpenAI. This method
        # updates the session_memory and tracks token growth for diagnostics.
        self._session_token_estimate += self._estimate_tokens(query)

        # Fire session memory extraction (runs in background thread)
        try:
            self.session_memory.extract_if_needed(
                self.memory.get_history(),
            )
        except Exception as e:
            print(f"[SESSION MEMORY] Background extraction failed: {e}")

    def _build_system_prompt(self) -> str:
        """Build the system prompt for the agent"""
        return f"""You are Quasar, an expert AI assistant for radio astronomy.

You have access to the ALMA Science Archive via the 'alminer' library,
and the Canadian Astronomy Data Centre (CADC) for multi-wavelength data
from JWST, HST, JCMT, CFHT, and Gemini telescopes.
Your goal is to help users find, visualize, and analyze astronomical data.

GUIDELINES:
- **ACTION OVER CHATTER**: If the user asks for data/search/plots, **IMMEDIATELY** call the appropriate tool.
- **NO HALLUCINATIONS**: Only cite data you have retrieved using tools.
- **MULTI-STEP RULE**: When asked to do multiple steps (e.g. "Do the following: 1. Search... 2. Filter... 3. Check..."), you MUST call the appropriate tool for EACH numbered step — do NOT describe what you would do. If there are 8 steps, make 8+ tool calls before writing your final summary. NEVER write "Access ALMA Archive: ..." — instead CALL search_by_target(). NEVER write "Use Splatalogue to..." — instead CALL search_lines_by_molecule().
- **PAPER SEARCH**: When you use the `search_papers` tool, do NOT write any text listing the papers. Output NOTHING after the tool call. The UI renders the papers as interactive cards automatically.
- **WEB SEARCH**: Use the `web_search` tool for real-time queries: current telescope schedules, recent arXiv preprints, observatory news, instrument specs, call-for-proposals, or anything not in the ALMA archive or NASA ADS.
- After a tool runs (except search_papers), summarize the output concisely.
- If a search returns many results, offer to plot them (but execute the search first).
- If the user says "yes/proceed" to a previous suggestion, ACT on it immediately.
- **DO NOT** output raw tool usage strings like `[TOOL: ...]` or JSON. Just use the Native Tool Calling feature.
- **NAME RESOLUTION**: If search_by_target returns empty for a valid target, use the resolve_target tool to get RA/Dec, then use search_by_position.
- **MINIMAL PARAMETERS**: When calling search_by_target, ONLY include optional parameters (band, max_resolution, min_freq_ghz, etc.) if the user EXPLICITLY requested them. For example, if the user says "Find ALMA data of M87", call search_by_target(target_name="M87") with NO other parameters. Do NOT pass band=0, min_freq_ghz=0, max_resolution=100 etc. Leaving them out returns ALL data.
- **MULTI-TARGET**: If the user mentions multiple targets (e.g. "M87 and Sz65"), pass them as a single comma-separated string: search_by_target(target_name="M87, Sz65"). The tool handles splitting and searching each target.
- **MULTI-BAND**: If the user mentions multiple bands (e.g. "Band 6 and Band 7"), pass them as comma-separated: search_by_target(target_name="M87", band="6,7"). The tool handles searching each band separately and shows a data card for each. NEVER make separate tool calls for each band — use comma-separated bands in ONE call.
- **MULTI-WAVELENGTH**: When users ask about JWST, HST, Hubble, or optical/infrared data, use `search_cadc_archive` instead of or in addition to ALMA search. For comprehensive coverage, search both ALMA and CADC.
- **FILTERING**: If the user asks for constraints like "resolution < 0.05", use the filter_results tool AFTER a search.
- **TAP QUERIES**: When generating SQL/ADQL queries, use the column names in the schema below.

FORMATTING RULES:
- **ALWAYS use proper Markdown** for your final output. The UI renders full GitHub-Flavored Markdown.
- **TABLES**: Whenever you present tabular data (types, descriptions, comparisons, summaries), you MUST use proper Markdown table syntax with pipe characters and header separators. Example:
  | Type | Description | Format |
  | --- | --- | --- |
  | Science data | Final calibrated data | FITS |
  Never use space-aligned plain text for tabular data — it will not render correctly.
- **HEADINGS**: Use ## and ### for sections, not numbered lists for top-level categories.
- **BOLD** key values, observatory names, and important findings.
- **BULLET LISTS**: Use - for lists of items or key points.
- Keep your response well-structured, scannable, and visually organized.

Current Context:
Date: {datetime.now().strftime("%Y-%m-%d")}

{ALMA_TAP_SCHEMA}
"""


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
                    "band": {"type": "string", "description": "ALMA band number(s) to filter (3-10). Pass a single band like '6' or multiple like '6,7'."},
                    "max_results": {"type": "integer", "description": "Maximum results to return"}
                },
                "required": ["ra", "dec"]
            }
        ))

        self.tool_registry.register(Tool(
            name="search_by_target",
            description=(
                "Search ALMA archive by target name. Supports multiple targets separated by "
                "'and' or comma (e.g. 'M87 and Sz65' or 'M87, NGC 1068').\n"
                "CRITICAL: ONLY pass optional filter parameters (band, resolution, frequency) "
                "if the user EXPLICITLY mentions them. Do NOT invent default values. "
                "If the user just says 'Find ALMA data of M87', pass ONLY target_name='M87' "
                "with NO other parameters — this returns ALL observations across all bands.\n"
                "Only pass band=6 if the user says 'Band 6'. Only pass max_resolution if "
                "the user specifies a resolution constraint. Omitting a filter means 'no filter'."
            ),
            function=self._search_by_target,
            parameters={
                "type": "object",
                "properties": {
                    "target_name":    {"type": "string",  "description": "Astronomical target name (e.g. 'TW Hya', 'HL Tau'). For multiple targets use comma or 'and': 'M87, NGC 1068'."},
                    "facility":       {"type": "string",  "enum": ["ALMA", "VLA", "VLBA", "GBT"], "description": "Observatory. Default ALMA."},
                    "band":           {"type": "string", "description": "ALMA band number(s) to filter (3-10). For a single band pass '6'. For multiple bands pass comma-separated like '6,7'. ONLY pass if user explicitly asks for a specific band."},
                    "max_resolution": {"type": "number",  "description": "Maximum angular resolution in arcsec. ONLY pass if user specifies."},
                    "min_resolution": {"type": "number",  "description": "Minimum angular resolution in arcsec. ONLY pass if user specifies."},
                    "min_freq_ghz":   {"type": "number",  "description": "Minimum frequency in GHz. ONLY pass if user specifies."},
                    "max_freq_ghz":   {"type": "number",  "description": "Maximum frequency in GHz. ONLY pass if user specifies."},
                    "min_exp_s":      {"type": "number",  "description": "Minimum integration time in seconds. ONLY pass if user specifies."},
                    "public_only":    {"type": "boolean", "description": "Only return publicly available data."},
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

        # ── CADC Multi-wavelength Archive Search ──────────────────
        self.tool_registry.register(Tool(
            name="search_cadc_archive",
            description=(
                "Search the Canadian Astronomy Data Centre (CADC) for multi-wavelength observations "
                "from JWST, HST, JCMT, CFHT, Gemini, and other telescopes. Uses the IVOA ObsCore TAP "
                "service. Complements ALMA searches with optical/infrared/submm data."
            ),
            function=self._search_cadc,
            parameters={
                "type": "object",
                "properties": {
                    "target_name": {"type": "string", "description": "Target name to resolve via SIMBAD (e.g. 'M31', 'TW Hya')"},
                    "ra":          {"type": "number", "description": "RA in decimal degrees (ICRS). Used if target_name is not given."},
                    "dec":         {"type": "number", "description": "Dec in decimal degrees (ICRS). Used if target_name is not given."},
                    "radius":      {"type": "number", "description": "Search radius in degrees. Default 0.05 (~3 arcmin).", "default": 0.05},
                    "collection":  {"type": "string", "description": "Filter by telescope collection (e.g. 'JWST', 'HST', 'JCMT', 'CFHT', 'Gemini'). Leave empty for all."},
                    "max_results": {"type": "integer", "description": "Maximum results to return. Default 100.", "default": 100},
                },
                "required": []
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
            description=(
                "Execute a custom ADQL/TAP query directly on the ALMA Science Archive (ivoa.obscore table).\n"
                "IMPORTANT: The obscore table has NO 'redshift' column. Use frequency/bandwidth containment instead.\n"
                "To find observations covering a specific frequency nu_ghz:\n"
                "  WHERE (frequency - 0.5*bandwidth/1e9) < {nu_ghz} AND (frequency + 0.5*bandwidth/1e9) > {nu_ghz}\n"
                "Key columns: target_name, s_ra, s_dec, frequency (GHz), bandwidth (Hz), scientific_category,\n"
                "  science_keyword, proposal_id, member_ous_uid, t_exptime, s_resolution, band_list.\n"
                "Add OFFSET 0 ROWS FETCH NEXT 500 ROWS ONLY to limit results."
            ),
            function=self._advanced_search,
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "ADQL TAP query string for ivoa.obscore"}
                },
                "required": ["query"]
            }
        ))

        self.tool_registry.register(Tool(
            name="search_alma_co_in_redshift_range",
            description=(
                "Search the ALMA archive for observations that cover CO emission lines "
                "for galaxies at a given redshift range. Handles the CO rest-frequency → "
                "observed-frequency conversion and TAP frequency-containment query automatically. "
                "Use this for any query like 'galaxies at z=1-2 with CO coverage' or "
                "'ALMA CO detections at high redshift'.\n"
                "CO transitions checked: J=1-0 (115.3 GHz), J=2-1 (230.5), J=3-2 (345.8), "
                "J=4-3 (461.0), J=5-4 (576.3), J=6-5 (691.5), J=7-6 (806.7).\n"
                "Returns: target_name, proposal_id, CO_transition, obs_frequency_ghz, bandwidth_ghz."
            ),
            function=self._search_alma_co_in_redshift_range,
            parameters={
                "type": "object",
                "properties": {
                    "z_min": {"type": "number", "description": "Minimum redshift (e.g. 1.0)"},
                    "z_max": {"type": "number", "description": "Maximum redshift (e.g. 2.0)"},
                    "science_category": {
                        "type": "string",
                        "description": "Optional ALMA science category filter (e.g. 'Galaxy evolution', 'Cosmology', 'Active galaxies'). Leave empty for all.",
                        "default": ""
                    },
                    "max_results": {"type": "integer", "description": "Maximum rows to return (default 500)", "default": 500}
                },
                "required": ["z_min", "z_max"]
            }
        ))

        # plot_alma_results is registered once below under "Publication Plotting Tools"
        
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

        # NEW: Fix 3 - Target name resolution using SIMBAD
        self.tool_registry.register(Tool(
            name="resolve_target",
            description="Resolve a target name to RA/Dec coordinates using SIMBAD. Use this if search_by_target returns empty for a valid target name.",
            function=self._resolve_target,
            parameters={
                "type": "object",
                "properties": {
                    "target_name": {"type": "string", "description": "The astronomical target name to resolve (e.g., RXJ1347-1145, M31)"}
                },
                "required": ["target_name"]
            }
        ))

        # NEW: Fix 2 - Deterministic filtering tool
        self.tool_registry.register(Tool(
            name="filter_results",
            description="Apply numeric filters to the LAST search results. Use this when user asks for specific constraints like 'resolution < 0.05 arcsec' or 'sensitivity > 10 mJy'.",
            function=self._filter_results,
            parameters={
                "type": "object",
                "properties": {
                    "column": {"type": "string", "description": "Column name to filter (e.g., 'resolution', 'sensitivity', 'Band')"},
                    "operator": {"type": "string", "enum": ["<", ">", "<=", ">=", "==", "!="], "description": "Comparison operator"},
                    "value": {"type": "number", "description": "Numeric value to compare against"}
                },
                "required": ["column", "operator", "value"]
            }
        ))

        # ── Browser Control Tools ──────────────────────────────────
        self.tool_registry.register(Tool(
            name="web_search",
            description=(
                "Search the web for real-time information: astronomy news, telescope schedules, "
                "arXiv preprints, observatory announcements, instrument specs, or any live web content. "
                "Uses Tavily for grounded, source-cited results. "
                "Examples: 'latest JWST observations 2024', 'ALMA call for proposals 2025', "
                "'what is the VLA sensitivity at 1.4 GHz'."
            ),
            function=self._tavily_web_search,
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Web search query string"},
                    "max_results": {"type": "integer", "description": "Number of results to return (default 5, max 10)"},
                    "search_depth": {"type": "string", "enum": ["basic", "advanced"], "description": "'basic' for quick answers, 'advanced' for comprehensive research (default: basic)"},
                },
                "required": ["query"]
            }
        ))

        self.tool_registry.register(Tool(
            name="navigate_to_url",
            description="Navigate to a specific URL and return its page content. Use for ALMA archive, NASA ADS, ESO portal, VizieR, arXiv paper pages, etc.",
            function=lambda **kw: self.browser_service.navigate_to_url(**kw),
            parameters={
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "Full URL to navigate to (must start with http:// or https://)"}
                },
                "required": ["url"]
            }
        ))

        self.tool_registry.register(Tool(
            name="read_page",
            description="Read the text content of the currently open browser page. Call after navigate_to_url to get the full page content.",
            function=lambda **kw: self.browser_service.read_page(**kw),
            parameters={
                "type": "object",
                "properties": {},
                "required": []
            }
        ))

        self.tool_registry.register(Tool(
            name="click_element",
            description="Click a button, link, or other element on the current browser page by CSS selector.",
            function=lambda **kw: self.browser_service.click_element(**kw),
            parameters={
                "type": "object",
                "properties": {
                    "selector": {"type": "string", "description": "CSS selector for the element to click (e.g., 'button.search', '#submit', 'a.download-link')"},
                    "wait_after_ms": {"type": "integer", "description": "Milliseconds to wait after clicking (default: 1500)"}
                },
                "required": ["selector"]
            }
        ))

        # ── Publication Plotting Tools ─────────────────────────────
        self.tool_registry.register(Tool(
            name="plot_alma_results",
            description="Generate a publication-quality scatter plot (ApJ/MNRAS style, 300 DPI, colorblind-safe) from the last ALMA search results. Use after any search to visualize data.",
            function=lambda **kw: self.plotting_service.plot_alma_results(
                data_records=(self.last_search_results.to_dict("records") if self.last_search_results is not None and not self.last_search_results.empty else []),
                **kw
            ),
            parameters={
                "type": "object",
                "properties": {
                    "x_column": {"type": "string", "description": "Column for x-axis (e.g. 'frequency', 'spatial_resolution', 'band')"},
                    "y_column": {"type": "string", "description": "Column for y-axis"},
                    "color_by": {"type": "string", "description": "Column to color-code points by (e.g. 'band', 'facility')"},
                    "title": {"type": "string", "description": "Plot title"},
                    "dark_mode": {"type": "boolean", "description": "Use dark background for presentations/posters"},
                },
                "required": []
            }
        ))

        self.tool_registry.register(Tool(
            name="plot_sky_map",
            description="Generate a publication-quality RA/Dec sky distribution map from the last ALMA search results.",
            function=lambda **kw: self.plotting_service.plot_sky_map(
                data_records=(self.last_search_results.to_dict("records") if self.last_search_results is not None and not self.last_search_results.empty else []),
                **kw
            ),
            parameters={
                "type": "object",
                "properties": {
                    "ra_col": {"type": "string", "description": "Column name for Right Ascension"},
                    "dec_col": {"type": "string", "description": "Column name for Declination"},
                    "color_by": {"type": "string", "description": "Column for color coding"},
                    "title": {"type": "string", "description": "Plot title"},
                },
                "required": []
            }
        ))

        self.tool_registry.register(Tool(
            name="plot_spectrum",
            description="Generate a publication-quality 1D spectral line profile plot with optional error bars and molecular line ID labels.",
            function=lambda **kw: self.plotting_service.plot_spectrum(**kw),
            parameters={
                "type": "object",
                "properties": {
                    "frequencies": {"type": "array", "items": {"type": "number"}, "description": "Frequency values (GHz)"},
                    "fluxes": {"type": "array", "items": {"type": "number"}, "description": "Flux density values (Jy)"},
                    "title": {"type": "string", "description": "Plot title"},
                    "errors": {"type": "array", "items": {"type": "number"}, "description": "Optional error bars (same length as fluxes)"},
                },
                "required": ["frequencies", "fluxes"]
            }
        ))

        # ── Splatalogue Line ID Tools ───────────────────────────────
        self.tool_registry.register(Tool(
            name="identify_spectral_line",
            description="Identify molecular spectral lines near a given rest frequency using the Splatalogue database. Essential for ALMA/VLA spectral line identification.",
            function=lambda **kw: self.splatalogue_tool.identify_spectral_line(**kw),
            parameters={
                "type": "object",
                "properties": {
                    "frequency_ghz": {"type": "number", "description": "Rest frequency to search around (GHz), e.g. 230.538"},
                    "tolerance_ghz": {"type": "number", "description": "Search window ± around the frequency in GHz (default: 0.01 = 10 MHz)"},
                    "top_n": {"type": "integer", "description": "Max candidate lines to return (default: 5)"},
                },
                "required": ["frequency_ghz"]
            }
        ))

        self.tool_registry.register(Tool(
            name="search_lines_by_molecule",
            description="Search Splatalogue for all known spectral line transitions of a specific molecule (e.g., 'CO', 'HCN', 'CH3OH', 'H2O').",
            function=lambda **kw: self.splatalogue_tool.search_lines_by_molecule(**kw),
            parameters={
                "type": "object",
                "properties": {
                    "molecule_name": {"type": "string", "description": "Molecule name (e.g., 'CO', 'HCN', 'CH3OH', 'SiO')"},
                    "freq_min_ghz": {"type": "number", "description": "Minimum frequency filter (GHz)"},
                    "freq_max_ghz": {"type": "number", "description": "Maximum frequency filter (GHz)"},
                },
                "required": ["molecule_name"]
            }
        ))

        # ── Multi-archive Cross-matcher Tools ─────────────────────
        self.tool_registry.register(Tool(
            name="cross_match_source",
            description="Query multiple astronomical archives (Simbad, NED, MAST, VizieR, Fermi 4FGL) in parallel for a source. Returns a unified multi-wavelength summary.",
            function=lambda **kw: self.multi_archive.cross_match_source(**kw),
            parameters={
                "type": "object",
                "properties": {
                    "target_name": {"type": "string", "description": "Source name recognized by Simbad (e.g., 'M87', 'HL Tau', 'NGC 1275')"},
                    "archives": {"type": "array", "items": {"type": "string"}, "description": "Specific archives to query, e.g. ['simbad', 'ned', 'mast']. Defaults to all."},
                },
                "required": ["target_name"]
            }
        ))

        # ── CASA Script Generator Tools ────────────────────────────
        self.tool_registry.register(Tool(
            name="generate_casa_imaging_script",
            description="Generate a complete CASA tclean imaging script for ALMA/VLA data. Returns ready-to-run Python code for radio interferometry imaging.",
            function=lambda **kw: self.casa_generator.generate_casa_imaging_script(**kw),
            parameters={
                "type": "object",
                "properties": {
                    "target": {"type": "string", "description": "Target source field name (as in the MS)"},
                    "vis": {"type": "string", "description": "Path to calibrated Measurement Set (.ms)"},
                    "band": {"type": "string", "description": "ALMA band number (e.g. '6', '3', '7')"},
                    "cell": {"type": "string", "description": "Cell size, e.g. '0.02arcsec'"},
                    "imsize": {"type": "integer", "description": "Square image size in pixels"},
                    "weighting": {"type": "string", "enum": ["briggs", "natural", "uniform"], "description": "Visibility weighting scheme"},
                    "robust": {"type": "number", "description": "Briggs robust parameter (-2 to +2)"},
                    "threshold": {"type": "string", "description": "Clean stopping threshold (e.g. '0.1mJy')"},
                    "specmode": {"type": "string", "enum": ["mfs", "cube"], "description": "'mfs' for continuum, 'cube' for spectral line"},
                },
                "required": ["target", "vis"]
            }
        ))

        self.tool_registry.register(Tool(
            name="generate_casa_calibration_script",
            description="Generate a CASA manual calibration script for ALMA/VLA data reduction. Returns ready-to-run Python code for bandpass, gain, and flux calibration.",
            function=lambda **kw: self.casa_generator.generate_casa_calibration_script(**kw),
            parameters={
                "type": "object",
                "properties": {
                    "target": {"type": "string", "description": "Science target field name"},
                    "vis": {"type": "string", "description": "Input Measurement Set path"},
                    "flux_cal": {"type": "string", "description": "Flux calibrator field name"},
                    "phase_cal": {"type": "string", "description": "Phase calibrator field name"},
                    "refant": {"type": "string", "description": "Reference antenna name (e.g. 'DA41')"},
                },
                "required": ["target", "vis", "flux_cal", "phase_cal", "refant"]
            }
        ))

        # ── GCN / GW Alert Tools ───────────────────────────────────
        self.tool_registry.register(Tool(
            name="get_latest_gw_events",
            description="Get the latest gravitational wave events from the GWOSC (Gravitational Wave Open Science Center) catalog. Use for multi-messenger astronomy queries.",
            function=lambda **kw: self.gcn_monitor.get_latest_gcn_alerts(**kw),
            parameters={
                "type": "object",
                "properties": {
                    "n": {"type": "integer", "description": "Number of recent events to return (default: 10)"},
                    "event_type": {"type": "string", "description": "Filter by type: 'BBH', 'BNS', 'NSBH', or None for all"},
                },
                "required": []
            }
        ))

        self.tool_registry.register(Tool(
            name="search_gwtc_catalog",
            description="Search the GWTC gravitational wave transient catalog with mass, distance, and type filters.",
            function=lambda **kw: self.gcn_monitor.search_gwtc_catalog(**kw),
            parameters={
                "type": "object",
                "properties": {
                    "mass_min_solar": {"type": "number", "description": "Minimum total mass in solar masses"},
                    "mass_max_solar": {"type": "number", "description": "Maximum total mass in solar masses"},
                    "distance_max_mpc": {"type": "number", "description": "Maximum luminosity distance (Mpc)"},
                    "event_type": {"type": "string", "description": "Event type: 'BBH', 'BNS', 'NSBH'"},
                },
                "required": []
            }
        ))

        self.tool_registry.register(Tool(
            name="summarize_gcn_circular",
            description="Fetch and parse a NASA GCN (Gamma-ray Coordinates Network) circular by number. Extracts key parameters: event name, trigger time, coordinates, and summary.",
            function=lambda **kw: self.gcn_monitor.summarize_gcn_circular(**kw),
            parameters={
                "type": "object",
                "properties": {
                    "circular_number": {"type": "integer", "description": "GCN circular number (e.g., 33000)"},
                },
                "required": ["circular_number"]
            }
        ))

        # ── NASA ADS Literature Tools ──────────────────────────────────────
        _ads = self.ads_client  # may be None if no key

        self.tool_registry.register(Tool(
            name="search_papers",
            description=(
                "Search the NASA ADS database for astronomical papers. "
                "Returns titles, authors, abstracts, citation counts, DOIs, and a direct link to each paper on NASA ADS. "
                "IMPORTANT: Use valid ADS field syntax in the query:\n"
                "- For telescope/instrument names (ALMA, VLA, JWST, Hubble, VLBI, etc): use abstract:\"ALMA\" AND abstract:\"topic\"\n"
                "- For astronomical objects: object:\"M87\" or title:\"black hole\"\n"
                "- For authors: author:\"Last, First\"\n"
                "- For year range: year:2020-2024\n"
                "- Combine with AND, OR operators\n"
                "Examples:\n"
                "- 'recent ALMA molecular cloud papers' → query: abstract:\"ALMA\" AND abstract:\"molecular cloud\"\n"
                "- 'VLA AGN observations 2022-2024' → query: abstract:\"VLA\" AND abstract:\"AGN\" AND year:2022-2024\n"
                "- 'black hole accretion disk' → query: title:\"black hole\" AND abstract:\"accretion disk\""
            ),
            function=lambda query, max_results=10, sort="date desc", **kw: (
                self._search_papers(query, max_results=max_results, sort=sort)
                if self.ads_client else {"error": "ADS client not configured"}
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "ADS search query using field syntax (abstract:, title:, author:, object:, year:)"},
                    "max_results": {"type": "integer", "description": "Number of results to return (default 10, max 50)"},
                    "sort": {"type": "string", "description": "Sort order: 'date desc' (newest first), 'citation_count desc' (most cited), 'score desc' (relevance). Default: 'date desc'"},
                },
                "required": ["query"]
            },
            category="literature"
        ))

        self.tool_registry.register(Tool(
            name="get_author_papers",
            description="Find all papers published by a specific author. Use 'Last, First' format for best results.",
            function=lambda author, max_results=20, **kw: (
                self.ads_client.get_author_papers(author, max_results=max_results)
                if self.ads_client else {"error": "ADS client not configured"}
            ),
            parameters={
                "type": "object",
                "properties": {
                    "author": {"type": "string", "description": "Author name, e.g. 'Accomazzi, Alberto'"},
                    "max_results": {"type": "integer", "description": "Max papers to return (default 20)"},
                },
                "required": ["author"]
            },
            category="literature"
        ))

        self.tool_registry.register(Tool(
            name="get_paper_metrics",
            description="Get citation count, read statistics, and impact metrics for a specific paper by its ADS bibcode.",
            function=lambda bibcode, **kw: (
                self.ads_client.get_paper_metrics(bibcode)
                if self.ads_client else {"error": "ADS client not configured"}
            ),
            parameters={
                "type": "object",
                "properties": {
                    "bibcode": {"type": "string", "description": "ADS bibcode, e.g. '2020PASP..132c5001L'"},
                },
                "required": ["bibcode"]
            },
            category="literature"
        ))

        self.tool_registry.register(Tool(
            name="get_author_metrics",
            description="Calculate scholarly impact metrics for an author: h-index, i10-index, total citations, refereed citations.",
            function=lambda author, **kw: (
                self.ads_client.get_author_metrics(author)
                if self.ads_client else {"error": "ADS client not configured"}
            ),
            parameters={
                "type": "object",
                "properties": {
                    "author": {"type": "string", "description": "Author name, e.g. 'Accomazzi, Alberto'"},
                },
                "required": ["author"]
            },
            category="literature"
        ))

        self.tool_registry.register(Tool(
            name="export_bibtex",
            description="Export properly formatted BibTeX citations for one or more papers given their ADS bibcodes. Use when the user asks to cite papers or needs BibTeX.",
            function=lambda bibcodes, **kw: (
                self.ads_client.export_bibtex(bibcodes if isinstance(bibcodes, list) else [bibcodes])
                if self.ads_client else "% ADS client not configured"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "bibcodes": {
                        "oneOf": [
                            {"type": "string", "description": "Single bibcode"},
                            {"type": "array", "items": {"type": "string"}, "description": "List of bibcodes"}
                        ],
                        "description": "ADS bibcode(s) to export"
                    },
                },
                "required": ["bibcodes"]
            },
            category="literature"
        ))

        self.tool_registry.register(Tool(
            name="get_paper_abstract",
            description="Get the full abstract and detailed metadata (keywords, affiliations) for a specific paper by bibcode.",
            function=lambda bibcode, **kw: (
                self.ads_client.get_paper_details(bibcode)
                if self.ads_client else {"error": "ADS client not configured"}
            ),
            parameters={
                "type": "object",
                "properties": {
                    "bibcode": {"type": "string", "description": "ADS bibcode of the paper"},
                },
                "required": ["bibcode"]
            },
            category="literature"
        ))

        self.tool_registry.register(Tool(
            name="list_ads_libraries",
            description="List all personal ADS paper libraries for the authenticated ADS user.",
            function=lambda **kw: (
                self.ads_client.list_libraries()
                if self.ads_client else {"error": "ADS client not configured"}
            ),
            parameters={
                "type": "object",
                "properties": {},
                "required": []
            },
            category="literature"
        ))

        self.tool_registry.register(Tool(
            name="reproduce_paper_methods",
            description="Extract the methodology from a published paper (by arXiv ID or ADS bibcode) and construct a Python/CASA data reduction script that replicates its steps. Use when a user asks 'how did they reduce the data for this paper' or 'reproduce this paper'.",
            function=self._reproduce_paper_methods,
            parameters={
                "type": "object",
                "properties": {
                    "identifier": {"type": "string", "description": "arXiv ID (e.g. '1812.04040') or ADS bibcode (e.g. '2018ApJ...869L..41A')"}
                },
                "required": ["identifier"]
            },
            category="literature"
        ))

        self.tool_registry.register(Tool(
            name="get_ads_library_papers",
            description="Get papers stored in a specific personal ADS library by library ID.",
            function=lambda library_id, max_results=50, **kw: (
                self.ads_client.get_library_papers(library_id, max_results=max_results)
                if self.ads_client else {"error": "ADS client not configured"}
            ),
            parameters={
                "type": "object",
                "properties": {
                    "library_id": {"type": "string", "description": "ADS library ID"},
                    "max_results": {"type": "integer", "description": "Max papers to return"},
                },
                "required": ["library_id"]
            },
            category="literature"
        ))

        self.tool_registry.register(Tool(
            name="create_ads_library",
            description="Create a new personal ADS paper library to organize papers by topic or project.",
            function=lambda name, description="", public=False, **kw: (
                self.ads_client.create_library(name, description=description, public=public)
                if self.ads_client else {"error": "ADS client not configured"}
            ),
            parameters={
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Library name"},
                    "description": {"type": "string", "description": "Library description"},
                    "public": {"type": "boolean", "description": "Whether the library is public"},
                },
                "required": ["name"]
            },
            category="literature"
        ))

        self.tool_registry.register(Tool(
            name="add_to_ads_library",
            description="Add papers to an existing personal ADS library by their bibcodes.",
            function=lambda library_id, bibcodes, **kw: (
                self.ads_client.add_to_library(library_id, bibcodes if isinstance(bibcodes, list) else [bibcodes])
                if self.ads_client else {"error": "ADS client not configured"}
            ),
            parameters={
                "type": "object",
                "properties": {
                    "library_id": {"type": "string", "description": "ADS library ID to add to"},
                    "bibcodes": {
                        "oneOf": [
                            {"type": "string"},
                            {"type": "array", "items": {"type": "string"}}
                        ],
                        "description": "Bibcode(s) to add"
                    },
                },
                "required": ["library_id", "bibcodes"]
            },
            category="literature"
        ))

        self.tool_registry.register(Tool(
            name="generate_jupyter_notebook",
            description="Generate a runnable Jupyter Notebook (.ipynb) for data analysis workflows. Use this when the user asks for code to analyze data, make maps, or perform reductions.",
            function=lambda title, steps, **kw: self._generate_notebook(title, steps),
            parameters={
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "Title of the notebook"},
                    "steps": {
                        "type": "array",
                        "description": "List of notebook cells (markdown or code)",
                        "items": {
                            "type": "object",
                            "properties": {
                                "type": {"type": "string", "enum": ["markdown", "code"]},
                                "content": {"type": "string", "description": "The exact cell content. Use complete astropy/spectral-cube/numpy code for code cells."}
                            },
                            "required": ["type", "content"]
                        }
                    }
                },
                "required": ["title", "steps"]
            },
            category="analysis"
        ))

        # ── FITS Image Rendering Tools ─────────────────────────────
        self.tool_registry.register(Tool(
            name="render_fits_image",
            description=(
                "Download a FITS file from an archive URL and render it as a "
                "publication-quality image displayed inline in the chat. Use this "
                "when the user asks to SEE or VISUALIZE data. The URL should come "
                "from a prior archive search (access_url or datalink URL)."
            ),
            function=self._render_fits_image,
            parameters={
                "type": "object",
                "properties": {
                    "url":      {"type": "string", "description": "Direct URL to the FITS file (from access_url, DataLink, or CADC cutout service)."},
                    "title":    {"type": "string", "description": "Title for the rendered image (e.g., 'JWST NIRCam F200W — Hubble Ultra Deep Field')."},
                    "colormap": {"type": "string", "description": "Matplotlib colormap. Default 'inferno'. Options: 'viridis', 'plasma', 'magma', 'gray', 'hot'."},
                    "stretch":  {"type": "string", "enum": ["sqrt", "log", "linear", "asinh"], "description": "Pixel stretch. Default 'sqrt'."},
                },
                "required": ["url"]
            },
            category="analysis"
        ))

        self.tool_registry.register(Tool(
            name="overlay_fits_images",
            description=(
                "Download two FITS files and create an overlay composite: one rendered "
                "as a colorscale background, the other as contours on top. Uses WCS "
                "reprojection to align them. Perfect for showing ALMA contours on "
                "JWST/HST colorscale images."
            ),
            function=self._overlay_fits_images,
            parameters={
                "type": "object",
                "properties": {
                    "base_url":       {"type": "string", "description": "URL to the base/background FITS file (rendered as colorscale)."},
                    "contour_url":    {"type": "string", "description": "URL to the FITS file rendered as contours on top."},
                    "base_label":     {"type": "string", "description": "Label for the base image (e.g., 'JWST NIRCam'). Default 'JWST'."},
                    "contour_label":  {"type": "string", "description": "Label for the contour image (e.g., 'ALMA Band 6'). Default 'ALMA'."},
                    "base_cmap":      {"type": "string", "description": "Colormap for base image. Default 'inferno'."},
                    "contour_levels": {"type": "integer", "description": "Number of contour levels. Default 8."},
                },
                "required": ["base_url", "contour_url"]
            },
            category="analysis"
        ))

        self.tool_registry.register(Tool(
            name="compute_moment_map",
            description=(
                "Download a FITS spectral cube and compute a moment map. "
                "Moment 0 = integrated intensity (total emission). "
                "Moment 1 = velocity field (mean velocity). "
                "Moment 2 = velocity dispersion (turbulence). "
                "Use this for ALMA cubes when the user asks about emission maps, "
                "velocity fields, or line intensity maps."
            ),
            function=self._compute_moment_map,
            parameters={
                "type": "object",
                "properties": {
                    "url":          {"type": "string",  "description": "Direct URL to the FITS spectral cube."},
                    "order":        {"type": "integer", "description": "Moment order: 0 (intensity), 1 (velocity), 2 (dispersion). Default 0."},
                    "title":        {"type": "string",  "description": "Title for the rendered image."},
                    "colormap":     {"type": "string",  "description": "Colormap for moment 0. Moment 1 uses RdBu_r, moment 2 uses magma. Default 'inferno'."},
                    "freq_min_ghz": {"type": "number",  "description": "Optional: only use channels above this frequency (GHz) for the moment."},
                    "freq_max_ghz": {"type": "number",  "description": "Optional: only use channels below this frequency (GHz) for the moment."},
                },
                "required": ["url"]
            },
            category="analysis"
        ))

        self.tool_registry.register(Tool(
            name="extract_spectrum",
            description=(
                "Download a FITS spectral cube and extract a 1D spectrum at a given "
                "sky position (RA/Dec) or pixel coordinate. If no position is given, "
                "extracts at the peak emission pixel. The spectrum is plotted as "
                "flux vs frequency/velocity and displayed inline."
            ),
            function=self._extract_spectrum,
            parameters={
                "type": "object",
                "properties": {
                    "url":     {"type": "string", "description": "Direct URL to the FITS spectral cube."},
                    "ra_deg":  {"type": "number", "description": "RA in decimal degrees (ICRS). Optional."},
                    "dec_deg": {"type": "number", "description": "Dec in decimal degrees (ICRS). Optional."},
                    "x_pixel": {"type": "integer", "description": "X pixel coordinate. Optional. Use if RA/Dec not available."},
                    "y_pixel": {"type": "integer", "description": "Y pixel coordinate. Optional."},
                    "title":   {"type": "string",  "description": "Title for the spectrum plot."},
                },
                "required": ["url"]
            },
            category="analysis"
        ))

        # ── DataLink + FITS Remote Header Tools (Phase 0) ──────────
        self.tool_registry.register(Tool(
            name="list_alma_files",
            description=(
                "List all deliverable files (images, cubes, continuum maps) for a "
                "given ALMA MOUS UID via the DataLink protocol. Returns filenames, "
                "sizes, and direct access URLs.  Use after a search to discover "
                "which data products (e.g. *pbcor.fits) are available for download "
                "or remote FITS header inspection."
            ),
            function=self._list_alma_files,
            parameters={
                "type": "object",
                "properties": {
                    "mous_uid": {
                        "type": "string",
                        "description": "MOUS UID, e.g. 'uid://A001/X1590/X30ae'"
                    },
                    "filename_pattern": {
                        "type": "string",
                        "description": "Optional glob filter, e.g. '*.pbcor.fits' to only show primary-beam-corrected images"
                    },
                },
                "required": ["mous_uid"]
            },
            category="archive"
        ))

        self.tool_registry.register(Tool(
            name="inspect_fits_header",
            description=(
                "Read key metadata from a remote FITS file header WITHOUT downloading "
                "the full file. Returns beam size (BMAJ/BMIN in arcsec), RMS noise/sensitivity, "
                "rest frequency, target name, pixel scale, and image dimensions. "
                "Use after list_alma_files to inspect the data quality of each product."
            ),
            function=self._inspect_fits_header,
            parameters={
                "type": "object",
                "properties": {
                    "access_url": {
                        "type": "string",
                        "description": "Direct HTTPS URL to the FITS file (from list_alma_files access_url)"
                    },
                },
                "required": ["access_url"]
            },
            category="archive"
        ))

    # ── Phase 0: DataLink + FITS backing methods ──────────────────────────

    @log_tool
    def _list_alma_files(self, mous_uid: str, filename_pattern: str = None) -> Dict[str, Any]:
        """List deliverable files for a MOUS via the ALMA DataLink protocol."""
        try:
            files = self.datalink_client.list_files(
                mous_uid=mous_uid,
                pattern=filename_pattern,
            )
            if not files:
                return {
                    "success": True,
                    "mous_uid": mous_uid,
                    "file_count": 0,
                    "message": "No files found. The MOUS UID might be invalid or the data is not yet public.",
                }
            return {
                "success": True,
                "mous_uid": mous_uid,
                "file_count": len(files),
                "files": files[:50],  # Cap at 50 for LLM context
                "note": f"Found {len(files)} file(s)." + (
                    f" Showing first 50." if len(files) > 50 else ""
                ),
            }
        except Exception as e:
            return {"success": False, "error": str(e), "mous_uid": mous_uid}

    @log_tool
    def _inspect_fits_header(self, access_url: str) -> Dict[str, Any]:
        """Read FITS header from a remote URL without downloading the full file."""
        try:
            metadata = self.fits_service.extract_metadata_from_url(access_url)
            if "error" in metadata:
                return {"success": False, **metadata}
            return {"success": True, **metadata}
        except Exception as e:
            return {"success": False, "error": str(e), "url": access_url}

    @log_tool
    def _tavily_web_search(self, query: str, max_results: int = 5, search_depth: str = "basic") -> Dict[str, Any]:
        """
        Real-time web search powered by Tavily.
        Falls back to BrowserService if Tavily key is unavailable.
        """
        tavily_key = os.getenv("TAVILY_API_KEY", "")
        if tavily_key:
            try:
                from tavily import TavilyClient
                client = TavilyClient(api_key=tavily_key)
                max_results = min(int(max_results), 10)
                response = client.search(
                    query=query,
                    max_results=max_results,
                    search_depth=search_depth,
                    include_answer=True,          # brief AI-synthesised answer
                    include_raw_content=False,
                )
                # Build a clean, LLM-friendly result
                results = []
                for r in response.get("results", []):
                    results.append({
                        "title":   r.get("title", ""),
                        "url":     r.get("url", ""),
                        "snippet": r.get("content", "")[:500],
                    })
                return {
                    "success":      True,
                    "provider":     "Tavily",
                    "query":        query,
                    "answer":       response.get("answer", ""),   # synthesised answer
                    "results":      results,
                    "result_count": len(results),
                }
            except Exception as e:
                print(f"[WARN] Tavily search failed: {e}. Falling back to BrowserService.")

        # ── Fallback: BrowserService ──────────────────────────────
        try:
            fallback = self.browser_service.web_search(query=query)
            return {"success": True, "provider": "BrowserService (fallback)", "results": fallback}
        except Exception as e2:
            return {"success": False, "error": str(e2)}

    @log_tool
    def _search_by_position(self, ra: float, dec: float, radius: float = 0.5,
                           facility: Optional[str] = None,
                           max_results: int = 100) -> Dict[str, Any]:
        """Search archives by sky position"""
        try:
            results = self.search_service.cone_search(
                ra, dec, radius, facility, max_results
            )
            self.last_search_results = results
            self.last_run_result = {"type": "data", "data": results, "source": "ALMA", "tool_name": "search_by_position"}

            # Include top MOUS UIDs + access URLs so Conductor subtasks
            # can use them for list_alma_files / render_fits_image.
            top_mous = []
            if hasattr(results, 'columns') and 'member_ous_uid' in results.columns:
                top_mous = results['member_ous_uid'].dropna().unique()[:5].tolist()
            top_urls = []
            if hasattr(results, 'columns') and 'access_url' in results.columns:
                top_urls = results['access_url'].dropna().head(5).tolist()

            return {
                "success": True,
                "total_results": len(results),
                "ra": ra, "dec": dec, "radius_deg": radius,
                "top_mous_uids": top_mous,
                "top_access_urls": top_urls,
                "note": f"Found {len(results)} observations. Full dataset with sky previews shown in UI table. Do NOT render a table — the UI already displays one."
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    @log_tool
    def _search_by_target(self, target_name: str, facility: Optional[str] = None,
                          date_range: Optional[str] = None, max_results: int = 100,
                          band = None,
                          max_resolution: Optional[float] = None,
                          min_resolution: Optional[float] = None,
                          min_freq_ghz: Optional[float] = None,
                          max_freq_ghz: Optional[float] = None,
                          min_exp_s: Optional[float] = None,
                          public_only: bool = False) -> Dict[str, Any]:
        """Search ALMA by target name, with optional native post-filters."""
        # ── Normalize band to a list (multi-band support) ──
        band_list_input: List[int] = []
        if band is not None:
            if isinstance(band, list):
                # LLM passed a list (e.g. [6, 7])
                band_list_input = [int(b) for b in band if str(b).strip().isdigit() and 1 <= int(b) <= 10]
            elif isinstance(band, (int, float)):
                # Single integer
                b = int(band)
                if 1 <= b <= 10:
                    band_list_input = [b]
                else:
                    print(f"[FILTER] Ignoring invalid band={band} (must be 3-10)")
            elif isinstance(band, str):
                # String like "6" or "6,7" or "6 and 7"
                import re as _re_band
                parts = _re_band.split(r'[,\s]+and[\s]+|[,\s]+', band.strip())
                for p in parts:
                    p = p.strip()
                    if p.isdigit():
                        b = int(p)
                        if 1 <= b <= 10:
                            band_list_input.append(b)
            else:
                print(f"[FILTER] Ignoring unrecognized band={band}")

        # ── Safety: ignore near-zero min values (LLM default filling) ──
        if min_resolution is not None and min_resolution <= 0:
            min_resolution = None
        if min_freq_ghz is not None and min_freq_ghz <= 0:
            min_freq_ghz = None
        if min_exp_s is not None and min_exp_s <= 0:
            min_exp_s = None
        # Ignore absurdly wide max values (LLM defaults)
        if max_resolution is not None and max_resolution >= 100:
            max_resolution = None
        if max_freq_ghz is not None and max_freq_ghz >= 5000:
            max_freq_ghz = None

        try:
            # ── Multi-target support ──────────────────────────────────
            # Detect "M87 and Sz65" or "M87, NGC 1068" patterns
            import re as _re
            raw_names = [n.strip() for n in _re.split(r'\s+and\s+|\s*,\s*', target_name) if n.strip()]

            if len(raw_names) > 1:
                # Search each target independently, concatenate results
                all_frames = []
                searched_names = []
                for name in raw_names[:5]:  # Cap at 5 targets
                    try:
                        df = self.search_service.search_by_target(
                            name, facility, date_range, max_results
                        )
                        if not df.empty:
                            all_frames.append(df)
                            searched_names.append(name)
                            print(f"[MULTI] '{name}' → {len(df)} results")
                        else:
                            print(f"[MULTI] '{name}' → 0 results")
                    except Exception as e:
                        print(f"[MULTI] '{name}' failed: {e}")

                if all_frames:
                    results = pd.concat(all_frames, ignore_index=True)
                    target_name = " + ".join(searched_names)  # update label
                else:
                    results = pd.DataFrame()
            else:
                results = self.search_service.search_by_target(
                    target_name, facility, date_range, max_results
                )

            if results.empty:
                self.last_run_result = {"type": "data", "data": results, "source": "ALMA", "tool_name": "search_by_target"}
                return {"success": True, "total_results": 0, "target": target_name, "note": "No results found."}

            # ── Tier 2: Pandas post-filters (non-band) ─────────────────
            filter_parts = []

            # Resolution filter
            res_col = next((c for c in ["spatial_resolution", "s_resolution", "resolution"] if c in results.columns), None)
            if res_col:
                if max_resolution is not None:
                    before = len(results)
                    results = results[pd.to_numeric(results[res_col], errors="coerce") <= max_resolution]
                    filter_parts.append(f"res ≤ {max_resolution}\"")
                    print(f"[FILTER] max_resolution {max_resolution}: {before} → {len(results)} rows")
                if min_resolution is not None:
                    before = len(results)
                    results = results[pd.to_numeric(results[res_col], errors="coerce") >= min_resolution]
                    filter_parts.append(f"res ≥ {min_resolution}\"")

            # Frequency filter
            freq_col = next((c for c in ["frequency", "min_frequency", "freq_min"] if c in results.columns), None)
            if freq_col:
                if min_freq_ghz is not None:
                    results = results[pd.to_numeric(results[freq_col], errors="coerce") >= min_freq_ghz]
                    filter_parts.append(f"freq ≥ {min_freq_ghz} GHz")
                if max_freq_ghz is not None:
                    results = results[pd.to_numeric(results[freq_col], errors="coerce") <= max_freq_ghz]
                    filter_parts.append(f"freq ≤ {max_freq_ghz} GHz")

            # Integration time filter
            exp_col = next((c for c in ["t_exptime", "integration"] if c in results.columns), None)
            if exp_col and min_exp_s is not None:
                results = results[pd.to_numeric(results[exp_col], errors="coerce") >= min_exp_s]
                filter_parts.append(f"exp ≥ {min_exp_s}s")

            # ── Multi-band handling ────────────────────────────────────
            # When multiple bands are requested (e.g. [6, 7]), produce a
            # separate data card for each band via _accumulated_run_results.
            band_col = next((c for c in ["band_list", "Band", "band"] if c in results.columns), None)

            if len(band_list_input) > 1 and band_col:
                # Multi-band: emit a separate result per band
                any_found = False
                for b in band_list_input:
                    band_filtered = results[results[band_col].astype(str).str.split(",").apply(
                        lambda bands, _b=b: any(str(_b).strip() == x.strip() for x in bands)
                    )]
                    print(f"[MULTI-BAND] Band {b}: {len(results)} → {len(band_filtered)} rows")
                    if not band_filtered.empty:
                        any_found = True
                        b_filter_label = f"ALMA › {target_name} [Band {b}" + (", ".join([""] + filter_parts) if filter_parts else "") + "]"
                        self.last_search_results = band_filtered
                        self.last_run_result = {
                            "type": "data", "data": band_filtered,
                            "source": "ALMA", "filter_label": b_filter_label,
                            "tool_name": "search_by_target"
                        }
                        # Accumulate each band result for multi-card display
                        self._accumulated_run_results.append(self.last_run_result.copy())

                if not any_found:
                    # None of the bands had data — still show the unfiltered results
                    filter_label = f"ALMA › {target_name}"
                    if filter_parts:
                        filter_label += " [" + ", ".join(filter_parts) + "]"
                    self.last_search_results = results
                    self.last_run_result = {
                        "type": "data", "data": results,
                        "source": "ALMA", "filter_label": filter_label,
                        "tool_name": "search_by_target"
                    }
            elif len(band_list_input) == 1 and band_col:
                # Single band filter
                b = band_list_input[0]
                before = len(results)
                results = results[results[band_col].astype(str).str.split(",").apply(
                    lambda bands, _b=b: any(str(_b).strip() == x.strip() for x in bands)
                )]
                filter_parts.append(f"Band {b}")
                print(f"[FILTER] Band {b}: {before} → {len(results)} rows")

                filter_label = f"ALMA › {target_name}"
                if filter_parts:
                    filter_label += " [" + ", ".join(filter_parts) + "]"
                self.last_search_results = results
                self.last_run_result = {
                    "type": "data", "data": results,
                    "source": "ALMA", "filter_label": filter_label,
                    "tool_name": "search_by_target"
                }
            else:
                # No band filter
                filter_label = f"ALMA › {target_name}"
                if filter_parts:
                    filter_label += " [" + ", ".join(filter_parts) + "]"
                self.last_search_results = results
                self.last_run_result = {
                    "type": "data", "data": results,
                    "source": "ALMA", "filter_label": filter_label,
                    "tool_name": "search_by_target"
                }

            # Compact summary + top MOUS UIDs & URLs for Conductor subtask chaining
            top_mous = []
            if 'member_ous_uid' in results.columns:
                top_mous = results['member_ous_uid'].dropna().unique()[:5].tolist()
            top_urls = []
            if 'access_url' in results.columns:
                top_urls = results['access_url'].dropna().head(5).tolist()

            return {
                "success": True,
                "total_results": len(results),
                "filters_applied": filter_parts,
                "target": target_name,
                "top_mous_uids": top_mous,
                "top_access_urls": top_urls,
                "note": f"Found {len(results)} observations matching your constraints. Full data with sky previews shown in UI table. Do NOT render a table — the UI already displays one."
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    @log_tool
    def _search_by_frequency(self, min_freq_ghz: float, max_freq_ghz: float,
                            facility: Optional[str] = None,
                            max_results: int = 100) -> Dict[str, Any]:
        """Search archives by frequency range"""
        try:
            results = self.search_service.search_by_frequency(
                min_freq_ghz, max_freq_ghz, facility, max_results
            )
            self.last_search_results = results
            self.last_run_result = {"type": "data", "data": results,
                                    "source": "ALMA",
                                    "filter_label": f"ALMA › {min_freq_ghz}–{max_freq_ghz} GHz",
                                    "tool_name": "search_by_frequency"}
            return {
                "success": True,
                "total_results": len(results),
                "note": f"Found {len(results)} observations at {min_freq_ghz}–{max_freq_ghz} GHz. Full data with sky previews shown in UI table. Do NOT render a table — the UI already displays one."
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    @log_tool
    def _search_cadc(self, target_name: Optional[str] = None,
                     ra: Optional[float] = None, dec: Optional[float] = None,
                     radius: float = 0.02, collection: Optional[str] = None,
                     max_results: int = 100) -> Dict[str, Any]:
        """Search CADC archive for multi-wavelength observations (JWST, HST, JCMT, etc.)."""
        try:
            import pyvo

            # Resolve target name to coordinates if needed
            if target_name and (ra is None or dec is None):
                try:
                    from astroquery.simbad import Simbad
                    result = Simbad.query_object(target_name)
                    if result is not None and len(result) > 0:
                        ra = float(result["RA"][0].replace(" ", ":").split(":")[0]) * 15 + \
                             float(result["RA"][0].replace(" ", ":").split(":")[1]) * 15/60 + \
                             float(result["RA"][0].replace(" ", ":").split(":")[2]) * 15/3600
                        dec_parts = result["DEC"][0].replace(" ", ":").split(":")
                        dec_sign = -1 if dec_parts[0].startswith("-") else 1
                        dec = dec_sign * (abs(float(dec_parts[0])) + float(dec_parts[1])/60 + float(dec_parts[2])/3600)
                    else:
                        return {"success": False, "error": f"Could not resolve target '{target_name}' via SIMBAD."}
                except Exception as e:
                    # Fallback: try using our existing resolve_target
                    try:
                        resolved = self._resolve_target(target_name)
                        if resolved.get("success") and resolved.get("ra") is not None:
                            ra = resolved["ra"]
                            dec = resolved["dec"]
                        else:
                            return {"success": False, "error": f"Could not resolve '{target_name}': {e}"}
                    except Exception:
                        return {"success": False, "error": f"Could not resolve '{target_name}': {e}"}

            if ra is None or dec is None:
                return {"success": False, "error": "Provide target_name or (ra, dec) coordinates."}

            # Build ADQL query against CADC's IVOA ObsCore
            collection_filter = ""
            if collection:
                collection_filter = f"AND obs_collection = '{collection.upper()}'"

            adql = f"""
            SELECT TOP {max_results}
                obs_collection, target_name, s_ra, s_dec,
                instrument_name, dataproduct_type, calib_level,
                t_exptime, em_min, em_max,
                obs_id, obs_publisher_did, access_url, access_format
            FROM ivoa.ObsCore
            WHERE CONTAINS(POINT('ICRS', s_ra, s_dec),
                           CIRCLE('ICRS', {ra:.6f}, {dec:.6f}, {radius})) = 1
            {collection_filter}
            """

            # Query CADC TAP
            tap_url = "https://ws.cadc-ccda.hia-iha.nrc-cnrc.gc.ca/argus"
            tap = pyvo.dal.TAPService(tap_url)
            result = tap.search(adql)
            df = result.to_table().to_pandas()

            if df.empty:
                self.last_run_result = {"type": "data", "data": df, "source": "CADC", "tool_name": "search_cadc_archive"}
                note = f"No CADC observations found near "
                note += f"'{target_name}'" if target_name else f"RA={ra:.4f}, Dec={dec:.4f}"
                note += f" (radius={radius}°)"
                if collection:
                    note += f" for {collection}"
                return {"success": True, "total_results": 0, "note": note}

            # Build source label
            telescopes = df["obs_collection"].unique().tolist() if "obs_collection" in df.columns else []
            filter_label = "CADC"
            if target_name:
                filter_label += f" › {target_name}"
            if collection:
                filter_label += f" [{collection}]"
            elif telescopes:
                filter_label += f" [{', '.join(telescopes[:3])}]"

            self.last_search_results = df
            self.last_run_result = {
                "type": "data", "data": df,
                "source": "CADC", "filter_label": filter_label,
                "tool_name": "search_cadc_archive"
            }

            # Build compact summary for LLM
            # Count by telescope
            tel_summary = df["obs_collection"].value_counts().to_dict() if "obs_collection" in df.columns else {}

            # Include top access_urls so Conductor subtasks can download/render
            top_urls = []
            if "access_url" in df.columns:
                top_urls = df["access_url"].dropna().head(5).tolist()

            return {
                "success": True,
                "total_results": len(df),
                "telescopes": tel_summary,
                "top_access_urls": top_urls,
                "note": (
                    f"Found {len(df)} observations from {len(telescopes)} telescope(s): "
                    f"{', '.join(f'{t} ({c})' for t, c in tel_summary.items())}. "
                    f"Full data with sky previews shown in UI table. Do NOT render a table — the UI already displays one."
                )
            }
        except ImportError:
            return {"success": False, "error": "pyvo library not installed. Run: pip install pyvo"}
        except Exception as e:
            return {"success": False, "error": f"CADC TAP query failed: {str(e)}"}

    def _filter_results(self, column: str, operator: str, value: float) -> Dict[str, Any]:
        """
        Apply a numeric or string filter to the LAST search results.
        Friendly column aliases are supported (e.g. 'Band', 'resolution', 'frequency').
        Updates last_run_result so the filtered table is shown in the UI.
        """
        try:
            if self.last_search_results is None or self.last_search_results.empty:
                return {"success": False, "error": "No search results to filter. Run a search first."}

            df = self.last_search_results.copy()

            # ── Friendly column alias mapping ──────────────────────────────────
            ALIAS = {
                "band":       ["band_list", "Band", "band"],
                "resolution": ["spatial_resolution", "s_resolution", "resolution"],
                "frequency":  ["frequency", "min_frequency", "freq_min", "freq"],
                "freq":       ["frequency", "min_frequency", "freq_min"],
                "exp":        ["t_exptime", "integration"],
                "exptime":    ["t_exptime", "integration"],
                "pi":         ["pi_name"],
                "project":    ["project_code"],
            }
            # Resolve column name
            col_lower = column.lower()
            candidates = ALIAS.get(col_lower, [column])
            real_col = next((c for c in candidates if c in df.columns), None)
            if real_col is None:
                # Try direct match (case-insensitive)
                real_col = next((c for c in df.columns if c.lower() == col_lower), None)
            if real_col is None:
                available = [c for c in df.columns][:15]
                return {"success": False, "error": f"Column '{column}' not found. Available: {available}"}

            # ── Apply filter ──────────────────────────────────────────────────
            before = len(df)
            numeric_series = pd.to_numeric(df[real_col], errors="coerce")

            OPS = {"<": lambda s, v: s < v, "<=": lambda s, v: s <= v,
                   ">": lambda s, v: s > v, ">=": lambda s, v: s >= v,
                   "==": lambda s, v: s == v, "!=": lambda s, v: s != v}
            if operator not in OPS:
                return {"success": False, "error": f"Invalid operator '{operator}'. Use: < <= > >= == !="}

            mask = OPS[operator](numeric_series, value)
            df = df[mask]

            print(f"[FILTER] {real_col} {operator} {value}: {before} → {len(df)} rows")

            # Update agent state so UI shows filtered table
            self.last_search_results = df
            prev_label = (self.last_run_result or {}).get("filter_label", "ALMA")
            filter_label = f"{prev_label} | {real_col} {operator} {value}"
            self.last_run_result = {
                **(self.last_run_result or {}),
                "data": df,
                "filter_label": filter_label,
                "tool_name": "filter_results",
            }

            return {
                "success": True,
                "rows_before": before,
                "rows_after": len(df),
                "filter": f"{real_col} {operator} {value}",
                "note": f"Filtered from {before} to {len(df)} rows. Updated table shown in UI. Do NOT render a table — the UI already displays one."
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

    def _search_alma_co_in_redshift_range(
        self,
        z_min: float,
        z_max: float,
        science_category: str = "",
        max_results: int = 500,
    ) -> Dict[str, Any]:
        """
        Search ALMA archive for observations covering CO emission lines at a given
        redshift range.  Fully stateless — no prior search needed.

        Method (from ALMA archive notebook nb6):
          1. For each CO transition, compute the observed frequency at z_min and z_max.
          2. Issue a TAP ADQL query with the frequency-containment WHERE clause.
          3. Union results across all transitions; deduplicate; return summary table.

        The ALMA obscore table has NO 'redshift' column — this tool implements the
        correct indirect approach (ν_obs = ν_rest / (1+z)).
        """
        import pandas as pd

        # CO rotational transitions: (J_upper, rest_freq_GHz)
        CO_TRANSITIONS = [
            ("CO(1-0)",  115.2712018),
            ("CO(2-1)",  230.5380000),
            ("CO(3-2)",  345.7959899),
            ("CO(4-3)",  461.0407682),
            ("CO(5-4)",  576.2679305),
            ("CO(6-5)",  691.4730763),
            ("CO(7-6)",  806.6518060),
        ]

        # Build ADQL frequency-range OR conditions for all transitions
        # Each transition covers a *range* of frequencies depending on the z range.
        # ν_obs_min (at z_max) to ν_obs_max (at z_min)
        freq_conditions = []
        transition_map = {}  # (freq_min, freq_max) -> transition label

        for label, nu_rest in CO_TRANSITIONS:
            nu_at_z_max = nu_rest / (1.0 + z_max)   # lower observed freq (higher z)
            nu_at_z_min = nu_rest / (1.0 + z_min)   # higher observed freq (lower z)

            # We want any observation whose spectral window overlaps [nu_at_z_max, nu_at_z_min]
            # (frequency - 0.5*bw/1e9) < nu_at_z_min  AND  (frequency + 0.5*bw/1e9) > nu_at_z_max
            cond = (
                f"((frequency - 0.5*bandwidth/1e9) < {nu_at_z_min:.4f} "
                f"AND (frequency + 0.5*bandwidth/1e9) > {nu_at_z_max:.4f})"
            )
            freq_conditions.append(cond)
            transition_map[(round(nu_at_z_max, 4), round(nu_at_z_min, 4))] = label

        freq_where = " OR ".join(freq_conditions)

        # Optional science category filter
        cat_clause = ""
        if science_category:
            cat_clause = f" AND scientific_category LIKE '%{science_category}%'"
        else:
            # Default: restrict to extragalactic categories
            cat_clause = (
                " AND (scientific_category LIKE '%Galaxy%' "
                "OR scientific_category LIKE '%Cosmology%' "
                "OR scientific_category LIKE '%Active%')"
            )

        adql_query = f"""
SELECT TOP {max_results}
       target_name, proposal_id, member_ous_uid,
       frequency, bandwidth, scientific_category, science_keyword,
       s_ra, s_dec, t_exptime, s_resolution
FROM ivoa.obscore
WHERE ({freq_where})
{cat_clause}
ORDER BY target_name
"""

        logger.info("CO redshift search ADQL:\n%s", adql_query.strip())

        try:
            import pyvo
            tap_url = "https://almascience.eso.org/tap"
            service = pyvo.dal.TAPService(tap_url)
            res = service.search(adql_query)
            df = res.to_table().to_pandas()

            if df.empty:
                return {
                    "success": True,
                    "count": 0,
                    "message": (
                        f"No ALMA observations found covering CO lines at z={z_min}–{z_max}. "
                        "The archive may not have public data for this parameter space, or "
                        "the frequency range falls outside ALMA's standard bands."
                    ),
                    "z_range": [z_min, z_max],
                    "co_obs_freq_ranges_ghz": {
                        label: {
                            "nu_min_ghz": round(nu_rest / (1 + z_max), 2),
                            "nu_max_ghz": round(nu_rest / (1 + z_min), 2),
                        }
                        for label, nu_rest in CO_TRANSITIONS
                    },
                }

            # Annotate which CO transition each observation covers
            def _which_co(row):
                obs_nu = float(row.get("frequency", 0))
                bw_ghz = float(row.get("bandwidth", 0)) / 1e9
                covered = []
                for label, nu_rest in CO_TRANSITIONS:
                    nu_lo = nu_rest / (1 + z_max)
                    nu_hi = nu_rest / (1 + z_min)
                    obs_lo = obs_nu - 0.5 * bw_ghz
                    obs_hi = obs_nu + 0.5 * bw_ghz
                    if obs_lo < nu_hi and obs_hi > nu_lo:
                        covered.append(label)
                return ", ".join(covered) if covered else "unknown"

            df["CO_transitions_covered"] = df.apply(_which_co, axis=1)
            df["obs_freq_ghz"] = df["frequency"].round(3)
            df["bandwidth_ghz"] = (df["bandwidth"] / 1e9).round(3)

            # Store in agent cache for follow-up plotting
            self.last_search_results = df
            self.last_run_result = {"type": "data", "data": df, "source": f"CO z={z_min}-{z_max}"}

            # Build summary
            summary_cols = ["target_name", "proposal_id", "CO_transitions_covered",
                            "obs_freq_ghz", "bandwidth_ghz", "scientific_category"]
            available_cols = [c for c in summary_cols if c in df.columns]
            summary = df[available_cols].drop_duplicates().to_dict("records")

            return {
                "success": True,
                "count": len(df),
                "unique_targets": int(df["target_name"].nunique()) if "target_name" in df.columns else None,
                "z_range": [z_min, z_max],
                "co_transitions_searched": [t[0] for t in CO_TRANSITIONS],
                "results": summary[:100],  # cap at 100 for LLM context
                "note": (
                    "Observations found where CO line at given z falls inside the ALMA spectral window. "
                    "Use plot_alma_results() to visualize sky distribution."
                )
            }

        except ImportError:
            return {
                "success": False,
                "error": "pyvo is not installed. Run: pip install pyvo",
            }
        except Exception as e:
            import traceback
            logger.error("CO redshift search failed: %s\n%s", e, traceback.format_exc())
            return {"success": False, "error": str(e)}

    def _plot_alma_results(self, plot_type: str = "sky") -> Dict[str, Any]:
        """Generate plots. Uses the LAST search results. Returns image as bytes (Fix 5)."""
        try:
            if not hasattr(self, 'last_search_results') or self.last_search_results is None or self.last_search_results.empty:
                 return {"success": False, "error": "No results available to plot. Please run a search first."}
            
            image_bytes = self.search_service.plot_alma_results(self.last_search_results, plot_type)
            if image_bytes:
                self.last_run_result = {
                    "type": "image", 
                    "image_bytes": image_bytes, 
                    "caption": f"ALMA {plot_type.capitalize()} Plot"
                }
                return {"success": True, "message": f"Generated {plot_type} plot successfully"}
            return {"success": False, "error": "Plot generation returned empty"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _render_fits_image(self, url: str, title: str = "", colormap: str = "inferno",
                           stretch: str = "sqrt") -> Dict[str, Any]:
        """Download a FITS file and render it as a PNG image displayed in chat."""
        try:
            from services.fits_service import render_fits_image
            result = render_fits_image(url, title=title, colormap=colormap, stretch=stretch)
            if result.get("success"):
                self.last_run_result = {
                    "type": "image",
                    "image_url": result["image_path"],
                    "caption": result.get("caption", ""),
                }
            return result
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _overlay_fits_images(self, base_url: str, contour_url: str,
                             base_label: str = "JWST", contour_label: str = "ALMA",
                             base_cmap: str = "inferno",
                             contour_levels: int = 8) -> Dict[str, Any]:
        """Create an overlay composite of two FITS images and display it in chat."""
        try:
            from services.fits_service import overlay_fits_images
            result = overlay_fits_images(
                base_url, contour_url,
                base_label=base_label, contour_label=contour_label,
                base_cmap=base_cmap, contour_levels=contour_levels,
            )
            if result.get("success"):
                self.last_run_result = {
                    "type": "image",
                    "image_url": result["image_path"],
                    "caption": result.get("caption", ""),
                }
            return result
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _compute_moment_map(self, url: str, order: int = 0, title: str = "",
                            colormap: str = "inferno", freq_min_ghz: float = None,
                            freq_max_ghz: float = None) -> Dict[str, Any]:
        """Compute a moment map from a FITS spectral cube and display it in chat."""
        try:
            from services.fits_service import compute_moment_map
            result = compute_moment_map(
                url, order=order, title=title, colormap=colormap,
                freq_min_ghz=freq_min_ghz, freq_max_ghz=freq_max_ghz,
            )
            if result.get("success"):
                self.last_run_result = {
                    "type": "image",
                    "image_url": result["image_path"],
                    "caption": result.get("caption", ""),
                }
            return result
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _extract_spectrum(self, url: str, ra_deg: float = None, dec_deg: float = None,
                          x_pixel: int = None, y_pixel: int = None,
                          title: str = "") -> Dict[str, Any]:
        """Extract a 1D spectrum from a FITS spectral cube and display it in chat."""
        try:
            from services.fits_service import extract_spectrum
            result = extract_spectrum(
                url, ra_deg=ra_deg, dec_deg=dec_deg,
                x_pixel=x_pixel, y_pixel=y_pixel, title=title,
            )
            if result.get("success"):
                self.last_run_result = {
                    "type": "image",
                    "image_url": result["image_path"],
                    "caption": result.get("caption", ""),
                }
            return result
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _generate_notebook(self, title: str, steps: list) -> Dict[str, Any]:
        """Generate a Jupyter Notebook and store in last_run_result for UI delivery."""
        try:
            nb_dict = generate_analysis_notebook(title, steps)
            self.last_run_result = {
                "type": "notebook",
                "notebook_data": nb_dict,
                "title": title
            }
            return {"success": True, "message": "Notebook generated successfully. Let the user know it is ready to download."}
        except Exception as e:
            return {"success": False, "error": str(e)}

    @log_tool
    def _search_papers(self, query: str, max_results: int = 10, sort: str = "date desc") -> Dict[str, Any]:
        """Search NASA ADS for papers.
        
        The calling LLM already produces valid ADS fielded-syntax queries
        (e.g. author:"Torrey" AND abstract:"galaxy formation"), so we call
        search_papers() directly — no extra LLM query builder needed.
        """
        try:
            if not self.ads_client:
                return {"success": False, "error": "NASA ADS Client not initialized (check API Key)"}

            # Call ADS API directly with the query the agent already formatted
            papers_list = self.ads_client.search_papers(
                query, max_results=max_results, sort=sort
            )

            # Store result for the UI backend to pick up
            self.last_run_result = {
                "type": "papers",
                "papers": papers_list,
                "source": f"ADS: {query}",
            }
            return {
                "success": True,
                "count": len(papers_list),
                "ads_query": query,
                "papers": papers_list,
                "top_title": papers_list[0]["title"] if papers_list else "No results",
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _reproduce_paper_methods(self, identifier: str) -> Dict[str, Any]:
        """Extract methodology from a paper and generate a reproduction script.
        
        Args:
            identifier: arXiv ID (e.g. '1812.04040') or ADS bibcode 
                        (e.g. '2018ApJ...869L..41A')
        """
        try:
            # Step 1: Resolve identifier to a PDF URL
            # Try arXiv first (most common for astro papers)
            arxiv_id = identifier.strip()
            # If it looks like a bibcode, try to get the arXiv ID from ADS
            if '.' not in arxiv_id or len(arxiv_id) > 20:
                # Likely an ADS bibcode — try to resolve via ADS
                if self.ads_client:
                    try:
                        details = self.ads_client.get_paper_details(arxiv_id)
                        if isinstance(details, dict) and details.get("arxiv_id"):
                            arxiv_id = details["arxiv_id"]
                        elif isinstance(details, dict) and details.get("doi"):
                            return {
                                "success": False,
                                "error": f"Paper has DOI ({details['doi']}) but no arXiv ID. "
                                         "PDF download is only supported for arXiv papers currently."
                            }
                    except Exception:
                        pass  # Fall through and try the identifier as-is

            pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"

            # Step 2: Download and extract methodology
            result = self.pdf_service.get_paper_methodology_from_url(pdf_url)

            if not result.get("success"):
                return {
                    "success": False,
                    "error": f"Could not extract methodology: {result.get('error', 'Unknown error')}",
                    "identifier": identifier
                }

            methodology_text = result["methodology"]

            # Step 3: Generate reproduction script using LIT_TO_CODE_PROMPT
            prompt = LIT_TO_CODE_PROMPT.format(methodology_text=methodology_text)

            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {"role": "system", "content": "You are an expert radio astronomy data reduction specialist."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=4000
            )

            script = response.choices[0].message.content.strip()

            self.last_run_result = {
                "type": "code",
                "code": script,
                "source": f"Reproduce: {identifier}"
            }

            return {
                "success": True,
                "identifier": identifier,
                "methodology_summary": methodology_text[:500] + "..." if len(methodology_text) > 500 else methodology_text,
                "generated_script": script
            }

        except Exception as e:
            return {"success": False, "error": str(e), "identifier": identifier}

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

    def _resolve_target(self, target_name: str) -> Dict[str, Any]:
        """Resolve target name to RA/Dec using SIMBAD (Fix 3)"""
        try:
            from astroquery.simbad import Simbad
            from astropy.coordinates import SkyCoord
            import astropy.units as u
            
            result = Simbad.query_object(target_name)
            
            if result is None or len(result) == 0:
                return {
                    "success": False, 
                    "error": f"SIMBAD could not resolve '{target_name}'. Check spelling or try alternate designation."
                }
            
            # Get coordinates from first match
            ra_str = result['RA'][0]   # Format: "HH MM SS.ss"
            dec_str = result['DEC'][0]  # Format: "+DD MM SS.s"
            
            # Convert to decimal degrees
            coord = SkyCoord(ra_str, dec_str, unit=(u.hourangle, u.deg))
            
            return {
                "success": True,
                "target_name": target_name,
                "ra_deg": round(coord.ra.deg, 6),
                "dec_deg": round(coord.dec.deg, 6),
                "message": f"Resolved '{target_name}' to RA={coord.ra.deg:.4f}°, Dec={coord.dec.deg:.4f}°. Use search_by_position with these coordinates."
            }
        except ImportError:
            return {"success": False, "error": "astroquery not installed. Cannot resolve target names."}
        except Exception as e:
            return {"success": False, "error": f"Resolution failed: {str(e)}"}

    def _filter_results(self, column: str, operator: str, value: float) -> Dict[str, Any]:
        """Apply deterministic numeric filter to last search results (Fix 2)"""
        if self.last_search_results is None or self.last_search_results.empty:
            return {"success": False, "error": "No search results to filter. Run a search first."}
        
        try:
            df = self.last_search_results.copy()
            
            # Find the actual column name (case-insensitive match)
            actual_column = None
            for col in df.columns:
                if col.lower() == column.lower():
                    actual_column = col
                    break
            
            if actual_column is None:
                # Try common aliases
                aliases = {
                    'resolution': ['resolution', 's_resolution', 'angular_resolution'],
                    'sensitivity': ['sensitivity', 'sensitivity_10kms', 'cont_sens_bandwidth'],
                    'frequency': ['freq_min', 'freq_max', 'frequency', 'min_freq_ghz', 'max_freq_ghz'],
                    'band': ['Band', 'band_number', 'band_list']
                }
                for alias_key, alias_list in aliases.items():
                    if column.lower() == alias_key:
                        for alias in alias_list:
                            if alias in df.columns:
                                actual_column = alias
                                break
                        break
            
            if actual_column is None:
                return {
                    "success": False, 
                    "error": f"Column '{column}' not found. Available: {list(df.columns)}"
                }
            
            # Apply filter using operator
            original_count = len(df)
            if operator == "<":
                df = df[df[actual_column] < value]
            elif operator == ">":
                df = df[df[actual_column] > value]
            elif operator == "<=":
                df = df[df[actual_column] <= value]
            elif operator == ">=":
                df = df[df[actual_column] >= value]
            elif operator == "==":
                df = df[df[actual_column] == value]
            elif operator == "!=":
                df = df[df[actual_column] != value]
            else:
                return {"success": False, "error": f"Unknown operator: {operator}"}
            
            # Update cached results
            self.last_search_results = df
            self.last_run_result = {
                "type": "data", 
                "data": df, 
                "source": f"Filtered: {actual_column} {operator} {value}"
            }
            
            return {
                "success": True,
                "original_count": original_count,
                "filtered_count": len(df),
                "filter_applied": f"{actual_column} {operator} {value}",
                "results": df.to_dict("records") if len(df) < 100 else f"[{len(df)} rows - too large to display]"
            }
        except Exception as e:
            return {"success": False, "error": f"Filter failed: {str(e)}"}


    def determine_intent(self, query: str) -> Dict[str, Any]:
        """Determine the user's intent from the query using Responses API"""
        try:
            prompt = INTENT_CLASSIFICATION_PROMPT.format(query=query)
            
            response = self.client.responses.create(
                model=self.config.model,
                input=prompt,
                instructions="You are an intent classification system. Always respond with valid JSON.",
                temperature=0.1,
                text={"format": {"type": "json_object"}}
            )
            
            # Extract text from Responses API output
            output_text = self._extract_response_text(response)
            result = json.loads(output_text)
            if self.config.verbose:
                print(f"[cyan]Intent Detected: {result.get('intent')} ({result.get('confidence')})[/cyan]")
            return result
        except Exception as e:
            print(f"[red]Intent classification failed: {e}[/red]")
            return {"intent": "QUESTION", "confidence": 0.0}

    def extract_entities(self, query: str) -> Dict[str, Any]:
        """Extract search entities from the query using Responses API"""
        try:
            prompt = ENTITY_EXTRACTION_PROMPT.format(query=query)
            
            response = self.client.responses.create(
                model=self.config.model,
                input=prompt,
                instructions="You are an entity extraction system. Always respond with valid JSON.",
                temperature=0.1,
                text={"format": {"type": "json_object"}}
            )
            
            output_text = self._extract_response_text(response)
            result = json.loads(output_text)
            if self.config.verbose:
                print(f"[cyan]Entities Extracted: {result}[/cyan]")
            return result
        except Exception as e:
            print(f"[red]Entity extraction failed: {e}[/red]")
            return {}
    
    def _extract_response_text(self, response) -> str:
        """Helper to extract text from Responses API response object"""
        if hasattr(response, 'output_text') and response.output_text:
            return response.output_text
        elif hasattr(response, 'output'):
            for item in response.output:
                if hasattr(item, 'content'):
                    content = item.content
                    # content can be a list of content-part objects or a plain string
                    if isinstance(content, list):
                        return " ".join(
                            c.text for c in content if hasattr(c, 'text') and c.text
                        )
                    return str(content)
        return str(response)

    def analyze_query_intent(self, query: str) -> Dict[str, Any]:
        """Alias for determine_intent to match UI expectation"""
        return self.determine_intent(query)

    # --- RLM helpers -------------------------------------------------------

    def _rlm_tool_bridge(self, tool_name: str, kwargs: dict) -> Any:
        """
        Bridge from REPL call_tool(name, **kwargs) to registered Quasar tools.

        This is the key paper-standard addition: any of the 27+ registered
        Quasar tools can be called from inside the REPL Python environment:
          call_tool("search_by_target", target_name="Elias 2-27")
          call_tool("check_line_coverage", line_freq_ghz=230.538, z=0.0)
          call_tool("generate_casa_imaging_script", target="Elias 2-27", vis="x.ms")
        """
        tool = self.tool_registry.get_tool(tool_name)
        if tool:
            try:
                result = tool.execute(**kwargs)
                return result
            except Exception as e:
                return {"error": f"Tool '{tool_name}' execution failed: {e}"}
        return {"error": f"Unknown tool: '{tool_name}'. Available: {[t.name for t in self.tool_registry.list_tools()]}"}

    def _rlm_tool_executor(self, subtask: str, context: str) -> Optional[str]:
        """
        Callback used by the RLM engine to execute domain-specific sub-tasks.
        Routes sub-task descriptions to the appropriate agent tools.
        Returns a string answer or None to let the RLM fall back to LLM.
        """
        st = subtask.lower()

        # Search archive for a target
        if any(kw in st for kw in ["search", "find observations", "query archive"]):
            # Try to extract target name from the subtask
            entities = self.extract_entities(subtask)
            target = entities.get("source_name")
            if target:
                try:
                    results = self.search_service.search_by_target(
                        target_name=target, facility="ALMA", max_results=20
                    )
                    if results is not None and not results.empty:
                        self.last_search_results = results
                        return f"Found {len(results)} ALMA observations for {target}."
                    return f"No ALMA observations found for {target}."
                except Exception as e:
                    return f"Search error: {e}"

        # Resolve target coordinates
        if "resolve" in st or "coordinates" in st:
            entities = self.extract_entities(subtask)
            target = entities.get("source_name")
            if target:
                result = self._resolve_target(target)
                return str(result)

        # Frequency / sensitivity calculation
        if "sensitivity" in st or "noise" in st:
            return None  # Let LLM answer from the accumulated context

        # Default: let RLM LLM handle it
        return None

    def process_query(self, query: str, user_id: str = "user") -> Tuple[Optional[Any], str, str]:
        """Process a user query and return (result, source_name, result_type)"""
        self.memory.add_message("user", query)

        # --- RLM: check if this is a complex multi-hop query ---
        try:
            should_rlm, complexity = self.rlm.should_use_rlm(query)
            if should_rlm:
                if self.config.verbose:
                    print(f"[RLM] Complex query detected (score={complexity.score:.2f}): {complexity.reasoning}")
                answer = self.rlm.execute(query)
                self.memory.add_message("assistant", answer)
                return None, answer, "rlm"  # answer goes in response_text slot
        except Exception as e:
            if self.config.verbose:
                print(f"[RLM] Failed, falling back to normal flow: {e}")

        # 1. Determine Intent
        intent_data = self.determine_intent(query)
        intent = intent_data.get("intent", "QUESTION")
        
        # 2. Extract Entities
        entities = self.extract_entities(query)
        source_name = entities.get("source_name", "")
        
        # Handle specific intents
        if intent == "SEARCH" or ("search" in query.lower() and "paper" not in query.lower()):
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
        """
        DEPRECATED: This method now redirects to stream_response_api().
        Kept for backward compatibility with existing callers.
        """
        return self.stream_response_api(query, message_placeholder, user_id)

    # ═══════════════════════════════════════════════════════════════════════════════
    # CONDUCTOR TOOL EXECUTOR — Bridges sub-agents to the full tool set
    # ═══════════════════════════════════════════════════════════════════════════════

    def _conductor_tool_executor(self, task_description: str, dep_context: str = "") -> str:
        """
        Execute a sub-task using a mini Responses API call with full tool access.

        This is the callback passed to Conductor so each DAG node can use
        all 28+ registered tools (ALMA search, ADS, Splatalogue, etc.).
        """
        system_instructions = (
            "You are a radio astronomy specialist executing one step of a larger analysis. "
            "Use the available tools to complete this specific task. "
            "Be concise — return only the relevant data/findings, no preamble.\n\n"
            "CRITICAL — You have tools to DOWNLOAD and RENDER FITS images directly:\n"
            "- render_fits_image(url, title, colormap): Download a FITS file and render as PNG\n"
            "- overlay_fits_images(base_url, contour_url, ...): Overlay contours from one FITS on another\n"
            "- compute_moment_map(url, order): Compute moment 0/1/2 from a spectral cube\n"
            "- extract_spectrum(url, ra_deg, dec_deg): Extract 1D spectrum at a position\n"
            "- search_cadc_archive: Search JWST/HST/etc. (returns access_url for download)\n"
            "- list_alma_files: List FITS files for an ALMA observation (returns file URLs)\n\n"
            "When the task says visualize, render, show, or display — you MUST call the rendering "
            "tools with actual URLs. Do NOT just describe steps or give recommendations. "
            "ACTUALLY call the tools to produce the result."
        )

        user_input = task_description
        if dep_context:
            user_input = f"Context from prior steps:\n{dep_context}\n\nYour task: {task_description}"

        try:
            # ── Use _build_tools_for_responses_api() — reads from self.tool_registry
            # (self.tools does not exist; _build_tool_definitions() would crash)
            tools = self._build_tools_for_responses_api()

            # Single Responses API call with tool access
            response = self.client.responses.create(
                model=self.config.model,
                instructions=system_instructions,
                input=user_input,
                tools=tools,
                temperature=0.1,
                max_output_tokens=2000,
            )

            # Process tool calls (up to 4 rounds)
            output_text = ""
            for _round in range(4):
                # Look for function_call items in response.output
                tool_calls = [
                    item for item in (response.output or [])
                    if getattr(item, "type", "") == "function_call"
                ]

                if not tool_calls:
                    # No more tool calls — extract text from every possible location
                    # 1. Convenience property (works for OpenAI and LLMResponse shim)
                    if hasattr(response, "output_text") and response.output_text:
                        output_text = response.output_text
                    else:
                        # 2. Walk output items
                        for item in (response.output or []):
                            item_type = getattr(item, "type", "")
                            if item_type == "output_text":
                                output_text += getattr(item, "text", "")
                            elif item_type == "message":
                                for chunk in getattr(item, "content", []):
                                    output_text += getattr(chunk, "text", "")
                    break

                # Execute each tool call
                tool_results = []
                tool_summaries = []   # Track what tools did for fallback text
                for tc in tool_calls:
                    fn_name = getattr(tc, "name", "")
                    fn_args = getattr(tc, "arguments", "{}")
                    call_id = getattr(tc, "call_id", "")
                    if on_status := getattr(self, "_last_on_status", None):
                        on_status(f"Calling tool: {fn_name}", "running")
                    result = self._dispatch_tool_call(fn_name, fn_args)
                    if on_status:
                        on_status(f"Calling tool: {fn_name}", "completed")

                    # ── Capture image results IMMEDIATELY after each tool call ──
                    if self.last_run_result and self.last_run_result.get("type") == "image":
                        if hasattr(self, '_conductor_images'):
                            lock = getattr(self, '_conductor_images_lock', None)
                            if lock:
                                with lock:
                                    self._conductor_images.append(self.last_run_result.copy())
                            else:
                                self._conductor_images.append(self.last_run_result.copy())
                        img_url = self.last_run_result.get("image_url", "")
                        caption = self.last_run_result.get("caption", "")
                        tool_summaries.append(
                            f"✅ Image rendered via {fn_name}: {caption} "
                            f"[image_url: {img_url}]"
                        )
                    else:
                        # Track non-image tool results too
                        result_str = str(result)[:500]
                        tool_summaries.append(f"Tool {fn_name}: {result_str}")

                    tool_results.append({
                        "type": "function_call_output",
                        "call_id": call_id,
                        "output": str(result)[:4000],
                    })

                # Continue conversation with tool results
                response = self.client.responses.create(
                    model=self.config.model,
                    previous_response_id=response.id,
                    input=tool_results,
                    tools=tools,
                    temperature=0.1,
                    max_output_tokens=2000,
                )

            # If the LLM produced text, use it. Otherwise fall back to tool summaries.
            if output_text.strip():
                return output_text.strip()
            elif tool_summaries:
                return "\n".join(tool_summaries)
            else:
                return "[No output from subtask]"

        except Exception as e:
            import traceback
            logger.error(
                "Conductor tool executor failed for task '%s': %s\n%s",
                task_description[:80], e, traceback.format_exc()
            )
            return f"[Tool execution error: {e}]"
        finally:
            # Images already captured in the loop above — just clean up.
            self.last_search_results = None
            self.last_run_result = None
            self._accumulated_run_results = []
            import gc; gc.collect()

    def _dispatch_tool_call(self, tool_name: str, arguments_json: str) -> str:
        """
        Dispatch a tool call by name using the tool registry.

        Parameters
        ----------
        tool_name : str
            Name of the registered tool function.
        arguments_json : str
            JSON string of the tool arguments.

        Returns
        -------
        str
            JSON-encoded result from the tool.
        """
        import json as _json
        try:
            args = _json.loads(arguments_json) if arguments_json else {}
        except _json.JSONDecodeError:
            args = {}

        tool = self.tool_registry.get_tool(tool_name)
        if tool:
            try:
                result = tool.execute(**args)
                return _json.dumps(result, default=str)[:8000]
            except Exception as e:
                return _json.dumps({"error": str(e)})
        else:
            return _json.dumps({"error": f"Unknown tool: {tool_name}"})

    # ═══════════════════════════════════════════════════════════════════════════════
    # RESPONSES API METHOD (New Architecture)
    # ═══════════════════════════════════════════════════════════════════════════════
    
    # ── Knowledge cutoff detection ──────────────────────────────────────────

    # LLM training knowledge cutoff — GPT-4o data ends ~Oct 2024
    _LLM_CUTOFF_YEAR = 2024
    _LLM_CUTOFF_MONTH = 10   # October 2024

    def _detect_beyond_cutoff(self, query: str) -> Optional[str]:
        """
        Check if a query references dates or time periods beyond the LLM's
        training data cutoff.  Returns a web-search query string if detected,
        else None.

        Triggers on:
        - Explicit future years:  "as of March 2025", "in 2026"
        - Freshness keywords:    "latest", "current", "recent", "now",
                                  "today", "this year", "this month"
        """
        _q = query.lower()

        # 1. Explicit year mentions beyond cutoff
        year_matches = re.findall(r'\b(20[2-9]\d)\b', query)
        for ym in year_matches:
            y = int(ym)
            if y > self._LLM_CUTOFF_YEAR:
                return query  # whole query is the search string

        # 2. Month+Year combos in cutoff year but after cutoff month
        month_year = re.findall(
            r'(?:january|february|march|april|may|june|july|august|'
            r'september|october|november|december)\s+(20[2-9]\d)',
            _q,
        )
        month_names = {
            'january': 1, 'february': 2, 'march': 3, 'april': 4,
            'may': 5, 'june': 6, 'july': 7, 'august': 8,
            'september': 9, 'october': 10, 'november': 11, 'december': 12,
        }
        for match in re.finditer(
            r'(january|february|march|april|may|june|july|august|'
            r'september|october|november|december)\s+(20[2-9]\d)',
            _q,
        ):
            m_name, m_year = match.group(1), int(match.group(2))
            if m_year > self._LLM_CUTOFF_YEAR:
                return query
            if m_year == self._LLM_CUTOFF_YEAR and month_names[m_name] > self._LLM_CUTOFF_MONTH:
                return query

        # 3. Freshness keywords — REMOVED.
        #    Web search now only triggers on explicit dates beyond the cutoff.
        #    Words like "latest", "current", "recent" are too common and caused
        #    false positives on nearly every query.

        return None

    def stream_response_api(
        self,
        query: str,
        message_placeholder=None,
        user_id: str = "user",
        on_token=None,
        on_status=None,
        attachments: Optional[List[Dict[str, Any]]] = None,
        raw_query: Optional[str] = None,
    ) -> str:
        # Reset accumulated results for this new request
        self._accumulated_run_results = []
        """
        Stream a response using OpenAI Responses API with:
        - Native conversation state (via previous_response_id)
        - Optional MCP tool connection
        - mem0 long-term memory for cross-session facts
        - Token streaming via `on_token` callback
        - Parallel web search for queries beyond LLM knowledge cutoff

        Parameters
        ----------
        raw_query : str, optional
            The original, un-enriched user message.  When the caller wraps
            the user's question inside personal-RAG context (e.g.
            ``"The user has …\n---\nUser's question: …"``), the enriched
            text should go in *query* (so the LLM sees everything) while
            the bare question goes in *raw_query* (used for complexity
            detection, cutoff checks, and conductor routing).  If omitted,
            *query* is used for everything.
        """
        # Derive the bare user question for routing / classification, and strip UI tags
        _user_query = raw_query or query
        for tag in ["@archive", "@paper", "@search"]:
            if _user_query.lower().strip().startswith(tag):
                _user_query = _user_query.strip()[len(tag):].strip()
            # Also strip from the enriched query so the LLM doesn't get confused
            pattern = re.compile(re.escape(tag), re.IGNORECASE)
            query = pattern.sub("", query).strip()

        # 0. Session Management — smart context handling + session memory
        self._prune_session_if_needed(query, user_id)

        # Emit connecting step
        if on_status:
            on_status("Connecting to QUASAR engine", "running")
            on_status("Connecting to QUASAR engine", "completed")

        # 0a. Knowledge-cutoff detection — launch parallel web search
        _web_search_query = self._detect_beyond_cutoff(_user_query)
        _web_result_holder = {}   # will be filled by background thread

        if _web_search_query:
            if on_status:
                on_status(
                    "⚡ Time period beyond training knowledge cutoff detected — "
                    "searching the web in parallel",
                    "running",
                )

            def _bg_web_search():
                try:
                    _web_result_holder["data"] = self._tavily_web_search(
                        query=_web_search_query,
                        max_results=5,
                        search_depth="advanced",
                    )
                except Exception as _e:
                    _web_result_holder["error"] = str(_e)

            _web_thread = threading.Thread(target=_bg_web_search, daemon=True)
            _web_thread.start()
        else:
            _web_thread = None

        # 1. Smart RAG — only search documentation for queries that likely
        #    relate to ALMA/radio astronomy/technical documentation.
        rag_context = ""
        _rag_keywords = {
            "alma", "band", "frequency", "resolution", "calibration",
            "observation", "correlator", "antenna", "baseline", "uv",
            "spectral", "continuum", "imaging", "pipeline", "casa",
            "interferometry", "interferometer", "receiver", "sensitivity",
            "proposal", "proprietary", "archive", "data reduction",
            "cycle", "configuration", "mosaic", "polarization",
            "flux", "beam", "synthesized", "primary beam", "fov",
            "spectral window", "spw", "channel", "bandwidth",
            "integration", "scheduling", "phase", "amplitude",
            "manual", "documentation", "technical handbook",
            "vla", "vlba", "gbt", "radio", "submillimeter",
            "millimeter", "ghz", "mhz", "jy", "arcsec",
            "fits", "measurement set", "uvfits", "clean", "tclean",
        }
        _query_lower = query.lower()
        _should_rag = any(kw in _query_lower for kw in _rag_keywords)

        if _should_rag:
            try:
                if on_status:
                    on_status("Searching ALMA Manuals & Documentation", "running")
                docs = self.rag_service.search(query)
                if docs:
                    context_pieces = []
                    for d in docs[:3]:
                        src = d.metadata.get("source", d.metadata.get("source_file", "Unknown"))
                        if "/" in src or "\\" in src:
                            src = src.replace("\\", "/").split("/")[-1]
                        page = d.metadata.get("page", "?")
                        context_pieces.append(f"[Source: {src}, Page {page}]\n{d.page_content}")
                    rag_context = "\n\nRelevant Technical Context (from ALMA documentation):\n" + "\n---\n".join(context_pieces)
                if on_status:
                    on_status("Searching ALMA Manuals & Documentation", "completed")
            except Exception as e:
                if on_status:
                    on_status("Searching ALMA Manuals & Documentation", "completed")
                print(f"[WARNING] RAG search failed: {e}")
        
        # 2. Retrieve long-term memories (from mem0) — only for authenticated users
        memory_context = ""
        _is_anonymous = (not user_id or user_id == "anonymous")
        if self.long_term_memory and not _is_anonymous:
            try:
                memories = self.long_term_memory.search(query=query, user_id=user_id, limit=5)
                if memories and memories.get("results"):
                    memory_pieces = [f"- {m['memory']}" for m in memories["results"]]
                    memory_context = "\n\nUser Memories:\n" + "\n".join(memory_pieces)
            except Exception as e:
                print(f"[WARNING] mem0 search failed: {e}")
        
        # 3. Build tools list
        tools = self._build_tools_for_responses_api()

        # Emit model step
        if on_status:
            on_status(f"Calling {self.config.model}", "running")
            on_status(f"Calling {self.config.model}", "completed")
        
        # 4. Build the full input
        citation_note = ""
        if rag_context:
            citation_note = (
                "\n\nIMPORTANT CITATION RULES: When your answer uses information from the "
                "Relevant Technical Context above, you MUST cite the source at the end "
                "of the relevant sentence using this exact format: "
                "[Source: filename, Page X]. For example: "
                "[Source: ALMA_Technical_Handbook.pdf, Page 42]. "
                "Always include the page number. This is critical for traceability."
            )
        full_input = f"{memory_context}{rag_context}{citation_note}\n\nUser: {query}"

        # 4a. Conductor check — delegate complex queries to DAG orchestration
        try:
            complexity = self.rlm.detector.assess(_user_query).score if hasattr(self, 'rlm') else 0.0
            if complexity > Conductor.COMPLEXITY_THRESHOLD:
                import asyncio, json as _json
                trace_id = self.query_tracer.new_trace(query, user_id=user_id)

                # ① Mark detection as COMPLETED immediately so the UI shows a ✓
                if on_status:
                    on_status(f"Complex query detected (score={complexity:.2f}) — activating multi-agent workforce", "completed")

                # ② Build on_event emitter — forwards task_group / task_update / task_list
                #    events through the SSE queue in api/main.py
                def on_event(evt: dict):
                    if on_status:
                        on_status(f"__event__{_json.dumps(evt)}", evt.get('type', 'status'))

                # Store on_status so _conductor_tool_executor can emit tool-call steps
                self._last_on_status = on_status

                # ③ Run the async Conductor in a dedicated thread with its own event loop.
                #    We CANNOT use pool.submit(asyncio.run, coro) here because this
                #    function already runs inside a ThreadPoolExecutor thread, and some
                #    Python/asyncio combinations deadlock when nesting executors that way.
                conductor_answer = None
                conductor_exc = None
                self._conductor_images = []  # Accumulate images from sub-agents
                self._conductor_images_lock = threading.Lock()  # Thread-safe — subtasks run in parallel
                _done = threading.Event()

                def _run_conductor():
                    nonlocal conductor_answer, conductor_exc
                    try:
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        try:
                            conductor_answer = loop.run_until_complete(
                                self.conductor.orchestrate(
                                    query,
                                    context=rag_context,
                                    on_status=on_status,
                                    on_token=on_token,
                                    on_event=on_event,
                                )
                            )
                        finally:
                            loop.close()
                    except Exception as _ce:
                        conductor_exc = _ce
                    finally:
                        _done.set()

                t = threading.Thread(target=_run_conductor, daemon=True)
                t.start()
                _done.wait(timeout=300)   # wait up to 5 min for complex queries

                if conductor_exc:
                    print(f"[WARNING] Conductor orchestration failed: {conductor_exc}. Falling back to standard path.")
                    self.query_tracer.end_trace(trace_id, "failed")
                elif conductor_answer is not None:
                    # Conductor ran — use its answer (even if some subtasks failed).
                    # Guard against empty synthesis: if the model returned blank,
                    # build a minimal fallback from the DAG status.
                    if not conductor_answer.strip():
                        dag_summary = self.conductor.dag.get_execution_summary()
                        completed = dag_summary.get("completed", 0)
                        total = dag_summary.get("total_tasks", 0)
                        conductor_answer = (
                            f"The multi-agent analysis completed {completed}/{total} tasks "
                            f"but failed to synthesize a final response. Please try rephrasing "
                            f"your question or breaking it into simpler parts."
                        )
                        print(f"[WARNING] Conductor synthesis returned empty. DAG summary: {dag_summary}")
                        if on_token:
                            on_token(conductor_answer)

                    self.query_tracer.end_trace(trace_id, "completed")
                    # Append parallel web search results to conductor answer
                    if _web_thread is not None:
                        _web_thread.join(timeout=15)
                        if on_status:
                            on_status(
                                "⚡ Time period beyond training knowledge cutoff detected — "
                                "searching the web in parallel",
                                "completed",
                            )
                        web_data = _web_result_holder.get("data")
                        if web_data and web_data.get("success"):
                            web_section = "\n\n---\n\n## 🌐 Web Search Results\n\n"
                            web_section += "*The following information was retrieved from the web "
                            web_section += "because your query references a time period beyond "
                            web_section += "the model's training data cutoff.*\n\n"
                            if web_data.get("answer"):
                                web_section += f"**Summary:** {web_data['answer']}\n\n"
                            for i, r in enumerate(web_data.get("results", []), 1):
                                title = r.get("title", "Untitled")
                                url = r.get("url", "")
                                snippet = r.get("snippet", "")
                                web_section += f"{i}. **[{title}]({url})**\n"
                                web_section += f"   {snippet[:300]}\n\n"
                            conductor_answer += web_section
                            if on_token:
                                on_token(web_section)
                            print(f"[WEB SEARCH] Appended {len(web_data.get('results', []))} web results to conductor response")
                    # Re-emit accumulated images via last_run_result so the SSE
                    # loop in main.py can emit them as inline image events.
                    if hasattr(self, '_conductor_images') and self._conductor_images:
                        self.last_run_result = {
                            "type": "conductor_result",
                            "images": self._conductor_images,
                        }

                    # Attach companion notebook if the Conductor generated one
                    notebook = getattr(self.conductor, '_notebook', None)
                    if notebook:
                        if self.last_run_result is None:
                            self.last_run_result = {"type": "conductor_result"}
                        else:
                            self.last_run_result["type"] = "conductor_result"
                        # Store notebook data so main.py can emit it for history
                        self.last_run_result["notebook_data"] = notebook
                        
                        words = [w for w in re.split(r'\W+', query) if w]
                        short_title = "_".join(words[:2]) if words else "Analysis"
                        self.last_run_result["title"] = short_title

                    return conductor_answer
                # Conductor returned None → not complex enough, fall through to standard path
        except Exception as e:
            print(f"[WARNING] Complexity detection failed: {e}. Using standard path.")
        
        try:
            
            # Smart token budget replaces hard MAX_TOOL_ROUNDS = 12
            _token_budget = TokenBudget(max_budget=100_000)
            last_id = self.last_response_id
            output_text = ""
            
            # 5. Call Responses API with manual streaming loop
            for _round in range(_token_budget.HARD_MAX_ITERATIONS if hasattr(_token_budget, 'HARD_MAX_ITERATIONS') else 25):
                request_kwargs = {
                    "model": self.config.model,
                    "input": full_input if _round == 0 else tool_results,
                    "instructions": self.system_prompt,
                    "previous_response_id": last_id,
                    "tools": tools,
                    "temperature": self.config.temperature,
                    "max_output_tokens": self.config.max_tokens,
                    "stream": True,
                }
                if _round == 0 and attachments:
                    request_kwargs["attachments"] = attachments

                response_stream = self.client.responses.create(**request_kwargs)
                
                function_calls = {} # call_id -> dict
                item_id_to_call_id = {}  # item.id -> call_id mapping
                
                for event in response_stream:
                    if event.type == "response.created":
                        last_id = event.response.id
                        self.last_response_id = last_id
                    elif event.type == "response.output_text.delta":
                        output_text += event.delta
                        if on_token:
                            on_token(event.delta)
                    elif event.type == "response.output_item.added":
                        # Check if it's a function_call item
                        item = event.item
                        if getattr(item, 'type', None) == 'function_call':
                            call_id = getattr(item, 'call_id', None)
                            item_id = getattr(item, 'id', None)
                            name = getattr(item, 'name', 'unknown')
                            cid = call_id or item_id
                            if cid:
                                function_calls[cid] = {"name": name, "arguments": "", "call_id": cid, "_item_id": item_id}
                                # Map item_id to call_id so argument deltas can find the right entry
                                if item_id and item_id != cid:
                                    item_id_to_call_id[item_id] = cid
                                if call_id and call_id != item_id:
                                    item_id_to_call_id[call_id] = cid
                    elif event.type == "response.output_item.done":
                        # CRITICAL FIX: The real call_id may only be available
                        # when the function_call item is DONE, not when it is
                        # first added.  Update our records with the canonical
                        # call_id so the follow-up request matches what the
                        # API expects.
                        item = event.item
                        if getattr(item, 'type', None) == 'function_call':
                            final_call_id = getattr(item, 'call_id', None)
                            item_id = getattr(item, 'id', None)
                            if final_call_id:
                                # Find the entry we stored (keyed by item_id or preliminary call_id)
                                old_cid = item_id_to_call_id.get(item_id, item_id)
                                if old_cid in function_calls:
                                    function_calls[old_cid]["call_id"] = final_call_id
                                # Also try direct item_id lookup
                                elif item_id in function_calls:
                                    function_calls[item_id]["call_id"] = final_call_id
                    elif event.type == "response.function_call_arguments.delta":
                        # Try all possible ID fields the API might use
                        raw_id = getattr(event, 'call_id', None) or getattr(event, 'item_id', None)
                        # Resolve to the canonical call_id we stored
                        cid = item_id_to_call_id.get(raw_id, raw_id)
                        if cid and cid in function_calls:
                            function_calls[cid]["arguments"] += event.delta
                
                if not function_calls:
                    break  # No tool calls — we have the final text

                # Track output growth for smart budget
                _token_budget.record_output(len(output_text))
                if not _token_budget.should_continue():
                    print(f"[TOKEN BUDGET] Stopping — {_token_budget.get_stats()}")
                    break
                
                # Execute each function call and collect results
                tool_results = []
                for fc in function_calls.values():
                    tool_name = fc["name"]
                    try:
                        args_str = fc["arguments"]
                        args = json.loads(args_str) if args_str else {}
                    except json.JSONDecodeError:
                        args = {}
                    
                    print(f"[TOOL CALL] {tool_name}({args})")

                    # Emit tool call status to Processing Pipeline
                    query_hint = args.get("query", args.get("author", args.get("bibcode", "")))
                    step_label = f"Calling tool: {tool_name}"
                    if query_hint:
                        step_label += f' ("{str(query_hint)[:60]}")'
                    if on_status:
                        on_status(step_label, "running")

                    tool = self.tool_registry.get_tool(tool_name)
                    if tool:
                        try:
                            result = tool.execute(**args)
                            result_str = json.dumps(result, default=str)[:8000]  # increased for multi-step chains
                            # Capture each tool's run result for multi-target support
                            if self.last_run_result is not None:
                                self._accumulated_run_results.append(self.last_run_result.copy())
                            # Record tool calls for session memory
                            self.session_memory.record_tool_calls(1)
                        except Exception as te:
                            result_str = json.dumps({"error": str(te)})
                    else:
                        result_str = json.dumps({"error": f"Unknown tool: {tool_name}"})

                    if on_status:
                        on_status(step_label, "completed")
                    
                    tool_results.append({
                        "type": "function_call_output",
                        "call_id": fc["call_id"],
                        "output": result_str,
                    })

                # Apply tool result budget — truncate oversized old results
                tool_results = apply_tool_result_budget(tool_results)
            
            # Emit final step
            if on_status:
                on_status("Generating response", "running")
                on_status("Generating response", "completed")

            if not output_text:
                output_text = "I processed your query but didn't generate a text response. Please try rephrasing."
                if on_token:
                    on_token(output_text)
            
            print(f"[DEBUG] Response text length: {len(output_text)}")

            # 7a. Append parallel web search results if available
            if _web_thread is not None:
                _web_thread.join(timeout=15)  # wait up to 15s for web results
                if on_status:
                    on_status(
                        "⚡ Time period beyond training knowledge cutoff detected — "
                        "searching the web in parallel",
                        "completed",
                    )
                web_data = _web_result_holder.get("data")
                if web_data and web_data.get("success"):
                    web_section = "\n\n---\n\n## 🌐 Web Search Results\n\n"
                    web_section += "*The following information was retrieved from the web "
                    web_section += "because your query references a time period beyond "
                    web_section += "the model's training data cutoff.*\n\n"

                    # Synthesised answer from Tavily
                    if web_data.get("answer"):
                        web_section += f"**Summary:** {web_data['answer']}\n\n"

                    # Individual sources
                    for i, r in enumerate(web_data.get("results", []), 1):
                        title = r.get("title", "Untitled")
                        url = r.get("url", "")
                        snippet = r.get("snippet", "")
                        web_section += f"{i}. **[{title}]({url})**\n"
                        web_section += f"   {snippet[:300]}\n\n"

                    output_text += web_section
                    if on_token:
                        on_token(web_section)
                    print(f"[WEB SEARCH] Appended {len(web_data.get('results', []))} web results to response")
                elif _web_result_holder.get("error"):
                    print(f"[WEB SEARCH] Parallel web search failed: {_web_result_holder['error']}")
            

            # 8. Update long-term memory — only for authenticated users
            if self.long_term_memory and not _is_anonymous:
                try:
                    messages = [
                        {"role": "user", "content": query},
                        {"role": "assistant", "content": output_text}
                    ]
                    self.long_term_memory.add(messages, user_id=user_id)
                except Exception as e:
                    print(f"[WARNING] mem0 memory add failed: {e}")
            
            return output_text
            
        except AttributeError as ae:
            # Responses API not available in this OpenAI version
            error_msg = f"Responses API not available: {ae}. Please upgrade the openai package."
            print(f"[ERROR] {error_msg}")
            return error_msg
            
        except Exception as e:
            error_msg = f"Error with Responses API: {str(e)}"
            print(f"[ERROR] {error_msg}")
            return error_msg
    
    def _build_tools_for_responses_api(self) -> list:
        """
        Build tools list for Responses API.
        Includes function tools and optionally MCP server connection.
        
        NOTE: Responses API uses a flat tool schema:
          {"type": "function", "name": "...", "description": "...", "parameters": {...}}
        NOT the nested Chat Completions format:
          {"type": "function", "function": {"name": "...", ...}}
        """
        tools = []
        
        # Convert from Chat Completions format to Responses API format
        for tool in self.tool_registry.list_tools():
            tools.append({
                "type": "function",
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            })
        
        # Add MCP server connection if enabled
        if self.config.enable_mcp and self.config.mcp_server_url:
            tools.append({
                "type": "mcp",
                "server_label": "alma",
                "server_url": self.config.mcp_server_url,
                "require_approval": "never"
            })
        
        return tools if tools else None
    
    def reset_conversation_state(self):
        """Reset conversation state for new chat session"""
        self.last_response_id = None
        self.memory.clear()
        self.session_memory.clear()
        self._session_token_estimate = 0
        if self.config.verbose:
            print("[yellow]Conversation state reset[/yellow]")

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
        """Generate a DETERMINISTIC summary of the search results using Pandas stats.
        
        This avoids LLM hallucination of 'empty' data by computing actual values.
        """
        try:
            if df.empty:
                return f"No observations found for {source_name}."

            # === DETERMINISTIC STATS ===
            total_rows = len(df)
            
            # Unique counts (addressing ambiguous terminology issue)
            unique_mous = df['member_ous_uid'].nunique() if 'member_ous_uid' in df.columns else None
            unique_projects = df['project_code'].nunique() if 'project_code' in df.columns else None
            
            # Band information - check multiple possible column names
            bands = []
            for band_col in ['Band', 'band_number', 'band', 'band_list']:
                if band_col in df.columns:
                    band_values = df[band_col].dropna().unique().tolist()
                    if band_values:
                        bands = sorted([str(b) for b in band_values])
                        break
            
            # Frequency range - check multiple possible column names
            min_freq, max_freq = None, None
            for min_col in ['freq_min', 'freq_min_ghz', 'min_freq_ghz', 'frequency']:
                if min_col in df.columns:
                    min_freq = df[min_col].min()
                    break
            for max_col in ['freq_max', 'freq_max_ghz', 'max_freq_ghz', 'frequency']:
                if max_col in df.columns:
                    max_freq = df[max_col].max()
                    break
            
            # Resolution range
            resolution_col = None
            for res_col in ['resolution', 's_resolution', 'angular_resolution']:
                if res_col in df.columns:
                    resolution_col = res_col
                    break
            
            min_res, max_res = None, None
            if resolution_col:
                min_res = df[resolution_col].min()
                max_res = df[resolution_col].max()
            
            # Sensitivity range
            sens_col = None
            for s_col in ['sensitivity', 'sensitivity_10kms', 'cont_sens_bandwidth']:
                if s_col in df.columns:
                    sens_col = s_col
                    break
            
            # === BUILD SUMMARY STRING ===
            summary_parts = [f"**{source_name} - ALMA Archive Summary**"]
            summary_parts.append(f"- **Total Execution Blocks**: {total_rows}")
            
            if unique_mous is not None:
                summary_parts.append(f"- **Unique MOUS UIDs**: {unique_mous}")
            if unique_projects is not None:
                summary_parts.append(f"- **Unique Projects**: {unique_projects}")
            
            if bands:
                summary_parts.append(f"- **Bands**: {', '.join(bands)}")
            
            if min_freq is not None and max_freq is not None:
                summary_parts.append(f"- **Frequency Range**: {min_freq:.2f} - {max_freq:.2f} GHz")
            
            if min_res is not None and max_res is not None:
                summary_parts.append(f"- **Resolution Range**: {min_res:.4f}\" - {max_res:.4f}\"")
            
            return "\n".join(summary_parts)
            
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
        """Change the LLM model and update the LLMClient default."""
        old_provider = detect_provider(self.config.model)
        self.config.model = model
        # Update the LLMClient's default model so provider routing stays in sync
        self.client.default_model = model
        new_provider = detect_provider(model)
        if self.config.verbose:
            print(f"[cyan]Model changed to: {model} (provider: {new_provider})[/cyan]")
        if old_provider != new_provider:
            logger.info(f"Provider switch: {old_provider} → {new_provider}")

    def _update_memory(self, query: str, user_id: str = "user"):
        """Extract and save new memories from user interaction using Responses API"""
        try:
            # Simple extraction prompt
            prompt = f"""
            Extract any personal facts, preferences, or research interests from the user's message.
            If there are none, return "NONE".
            
            User Message: "{query}"
            
            Output format: Just the fact string or "NONE".
            Example: "User is interested in protoplanetary disks."
            """
            
            response = self.client.responses.create(
                model="gpt-4o-mini",  # Use cheaper model for background tasks
                input=prompt,
                temperature=0.1,
                max_output_tokens=50
            )
            
            fact = self._extract_response_text(response).strip()
            
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
            papers = self.ads_client.search_by_target(source_name)
            if papers:
                return pd.DataFrame(papers)
        
        return None
