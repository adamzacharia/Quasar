"""
QuasarAgent — Central orchestrator for Quasar AI.

CALLED BY: ui/app.py (Streamlit), ui-pro/api/main.py (FastAPI SSE),
           core/cli.py (terminal REPL), telegram.py (webhook)
CALLS:     All services/* modules, integrations/*, core/complexity.py,
           core/sandbox.py, OpenAI API (GPT-4o), mem0 (long-term memory)

This is the heart of Quasar. The QuasarAgent class:
  1. Registers 27+ tools as OpenAI function-calling schemas
  2. Routes user queries through complexity detection → Conductor DAG
  3. Manages RAG context, conversation memory, and long-term memory
  4. Streams responses via Chat Completions API or Responses API
  5. Caches search results (DataFrames) for follow-up operations
"""

import os
import json
import re
import uuid
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
from integrations.openalex_client import OpenAlexService
from services.search import SearchService
from services.analysis import RadioAnalysisService
from services.rag_service import RAGService
from services.memory_service import MemoryService
from core.complexity import ComplexityDetector
from core.sandbox import SandboxExecutor
from services.browser import BrowserService
from services.plotting import PlottingService
from services.splatalogue import SplatalogueTool
from services.multi_archive import MultiArchiveMatcher
from integrations.mast_client import MASTClient
from integrations.eso_tap_client import ESOTAPClient
from integrations.irsa_client import IRSAClient
from integrations.skyview_client import SkyViewClient
from services.casa_generator import CASAScriptGenerator
from services.gcn_monitor import GCNAlertMonitor
from services.notebook_gen import generate_analysis_notebook
from services.pdf_processing import PDFProcessingService
from services.fits_processing import FITSProcessingService
from core.prompts.lit_to_code import LIT_TO_CODE_PROMPT
from services.astro_calculators import (
    calculate_redshift, convert_coordinates, calculate_beam,
    calculate_alma_sensitivity,
)

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
        try:
            self.rag_service = rag_service or RAGService()
        except Exception as _rag_err:
            print(f"[WARN] RAG service unavailable (Qdrant timeout?): {_rag_err}")
            self.rag_service = None
        print("DEBUG: Init Memory Service (Long-Term)")
        self.memory_service = MemoryService()
        print("DEBUG: Init ADS Client")
        import os as _os
        ads_key = getattr(self.config, 'ads_api_key', None) or _os.getenv("NASA_ADS_API_KEY")
        self.ads_client = ADSService(ads_key) if ads_key else ADSService()  # ADSService handles missing key gracefully
        print("DEBUG: Init OpenAlex Client")
        self.openalex_client = OpenAlexService()  # handles missing key gracefully

        # Register tools
        print("DEBUG: Register Tools")
        self._register_tools()

        # System prompt
        print("DEBUG: Build Prompt")
        self.system_prompt = self._build_system_prompt()

        if self.config.verbose:
            print("[green]QuasarAgent initialized successfully[/green]")
        
        # ── Per-request thread-local storage ─────────────────────────
        # These attributes are accessed via @property so each concurrent
        # request (running in its own ThreadPoolExecutor thread) gets
        # isolated state.  Tools still write `self.last_run_result = {...}`
        # — the property setter transparently redirects to thread-local.
        self._tls = threading.local()

        # ── Per-conversation OpenAI response-ID tracking ──────────────
        # Maps conversation_id → last OpenAI response_id.  Thread-safe.
        # Replaces the old singleton `self.last_response_id` which caused
        # cross-user state poisoning on the shared agent instance.
        self._conv_response_ids: Dict[str, str] = {}
        self._conv_ids_lock = threading.Lock()

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
        print("DEBUG: Init MASTClient")
        self.mast_client = MASTClient()
        print("DEBUG: Init ESOTAPClient")
        self.eso_client = ESOTAPClient()
        print("DEBUG: Init IRSAClient")
        self.irsa_client = IRSAClient()
        print("DEBUG: Init SkyViewClient")
        self.skyview_client = SkyViewClient()
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
        # Initialize complexity detector (gates Conductor activation)
        print("DEBUG: Init ComplexityDetector")
        self.complexity_detector = ComplexityDetector(
            client=self.client, model="gpt-4o-mini"
        )

        # Initialize sandbox executor (for Conductor "compute" agent type)
        print("DEBUG: Init SandboxExecutor")
        self.sandbox_executor = SandboxExecutor(
            client=self.client,
            model=self.config.model,
            tool_executor=self._sandbox_tool_bridge,
            verbose=self.config.verbose,
        )

        self.conductor = Conductor(
            client=self.client,
            model=self.config.model,
            conductor_model="gpt-5.4",  # Stronger model for planning/synthesis
            tool_executor=self._conductor_tool_executor,
            recovery_engine=self.recovery_engine,
            model_router=self.model_router,
            agent_pool=self.agent_pool,
            sandbox_executor=self.sandbox_executor,
            verbose=True,
        )

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

    # ── Thread-local properties ───────────────────────────────────────
    # These let 40+ tool methods keep writing `self.last_run_result = {}`
    # while each concurrent request thread sees its own isolated value.

    @property
    def last_run_result(self):
        return getattr(self._tls, 'last_run_result', None)

    @last_run_result.setter
    def last_run_result(self, value):
        self._tls.last_run_result = value

    @property
    def _accumulated_run_results(self):
        if not hasattr(self._tls, 'accumulated_run_results'):
            self._tls.accumulated_run_results = []
        return self._tls.accumulated_run_results

    @_accumulated_run_results.setter
    def _accumulated_run_results(self, value):
        self._tls.accumulated_run_results = value

    @property
    def last_search_results(self):
        return getattr(self._tls, 'last_search_results', None)

    @last_search_results.setter
    def last_search_results(self, value):
        self._tls.last_search_results = value

    # ── Per-conversation response ID helpers ────────────────────────

    def _get_response_id(self, conversation_id: str) -> Optional[str]:
        """Get the OpenAI previous_response_id for a specific conversation."""
        with self._conv_ids_lock:
            return self._conv_response_ids.get(conversation_id)

    def _set_response_id(self, conversation_id: str, response_id: Optional[str]):
        """Set (or clear) the OpenAI response_id for a conversation."""
        with self._conv_ids_lock:
            if response_id is None:
                self._conv_response_ids.pop(conversation_id, None)
            else:
                self._conv_response_ids[conversation_id] = response_id

    def _cleanup_conv_states(self, max_entries: int = 500):
        """Prevent memory leak — evict oldest conversation entries."""
        with self._conv_ids_lock:
            if len(self._conv_response_ids) > max_entries:
                keys = list(self._conv_response_ids.keys())
                for k in keys[:len(keys) // 2]:
                    del self._conv_response_ids[k]

    # Backward-compatible property so legacy code (e.g. reset_conversation_state)
    # still works.  In production, prefer _get/_set_response_id with a conv_id.
    @property
    def last_response_id(self):
        return self._get_response_id("__global__")

    @last_response_id.setter
    def last_response_id(self, value):
        self._set_response_id("__global__", value)

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
the Canadian Astronomy Data Centre (CADC) for multi-wavelength data
from JWST, HST, JCMT, CFHT, and Gemini telescopes,
the MAST archive (JWST, HST, TESS, Kepler) via `search_mast` and `search_mast_by_criteria`,
the ESO Science Archive (VLT/MUSE, KMOS, X-Shooter, FORS2) via `search_eso_archive`,
and the IRSA infrared catalog services (WISE, 2MASS, Spitzer catalogs) via `search_irsa`.
Your goal is to help users find, visualize, and analyze astronomical data.

GUIDELINES:
- **ACTION OVER CHATTER**: If the user asks for data/search/plots, **IMMEDIATELY** call the appropriate tool.
- **FRESH DATA ALWAYS**: NEVER answer archive/data queries from conversation memory or prior tool results. ALWAYS make a fresh tool call, even if you already called the same tool earlier in this conversation. Every data request MUST trigger a new search_by_target, search_by_position, search_cadc_archive, or search_papers call. The user expects live data with a data card in the UI — text-only answers without a tool call are UNACCEPTABLE for data queries.
- **KNOWLEDGE vs DATA**: For factual/conceptual/how-to questions, answer directly from the documentation context (RAG) and your knowledge. Do NOT call search_papers or any data tool — these are NOT data queries, they are knowledge queries. Only call search_papers when the user EXPLICITLY asks for papers, articles, publications, or literature (e.g. "find papers about...", "show me recent publications on...").
  KNOWLEDGE QUERY EXAMPLES (answer from RAG, NEVER call search_papers):
  - "What is the ALMA proprietary period?" → RAG answer
  - "What are the Cycle 13 proposal submission deadlines?" → RAG answer (the word "proposal" does NOT mean "find papers")
  - "How do I access archival ALMA data?" → RAG answer (the word "archival" means ALMA archive, NOT research articles)
  - "How does the ALMA proposal review process work?" → RAG answer
  - "How do I calibrate ALMA Band 6 data?" → RAG answer
  - "What receiver bands are available on ALMA?" → RAG answer
  - "What file formats does ALMA deliver?" → RAG answer
  PAPER QUERY EXAMPLES (call search_papers):
  - "Find recent papers on protoplanetary disks" → search_papers
  - "Show me publications about ALMA observations of M87" → search_papers
  - "What are the latest studies on galaxy mergers?" → search_papers
  If the query is asking HOW something works, WHAT something is, or about ALMA procedures/policies/deadlines — it is a KNOWLEDGE query. NEVER call search_papers for these.
- **NO HALLUCINATIONS**: Only cite data you have retrieved using tools.
- **MULTI-STEP RULE**: When asked to do multiple steps (e.g. "Do the following: 1. Search... 2. Filter... 3. Check..."), you MUST call the appropriate tool for EACH numbered step — do NOT describe what you would do. If there are 8 steps, make 8+ tool calls before writing your final summary. NEVER write "Access ALMA Archive: ..." — instead CALL search_by_target(). NEVER write "Use Splatalogue to..." — instead CALL search_lines_by_molecule().
- **PAPER SEARCH (MANDATORY TOOL)**: When the user asks for papers, publications, articles, literature, or studies — you MUST call the `search_papers` tool. Pass the user's request as NATURAL LANGUAGE (e.g. "recent papers on protoplanetary disks", "best ALMA papers on disk gaps", "foundational papers on planet formation"). Do NOT try to construct ADS field syntax — the tool has an internal AI query builder that translates natural language into optimal ADS queries using keyword searches, bibgroup filters, SIMBAD object linking, and second-order discovery operators. NEVER use `web_search` for paper requests. After the tool runs, do NOT write any text listing the papers — output NOTHING. The UI renders the papers as interactive cards automatically.
- **RESEARCHER LOOKUP**: When the user asks about a person, scientist, astronomer — "Who is X?", "Tell me about X", "Where does X work?" — call `lookup_researcher`. ALWAYS present the profile using this EXACT format:
  1. **Header**: "## Profile: [Full Name]" with email and personal webpage (from web search if available)
  2. **Identity**: ORCID, alternative name forms, current institution(s)
  3. **Academic Metrics** — ALWAYS as a markdown table:
     | Metric | Value |
     |--------|-------|
     | Publications | N |
     | Citations | N |
     | h-index | N |
     | i10-index | N |
     | 2yr Mean Citedness | N |
  4. **Research Focus**: Bulleted list of top topics
  5. **Affiliation History**: Chronological list of past institutions with year ranges
  6. **Recent Research Activity** — ALWAYS as a markdown table with Year / Works / Citations columns (last 5–10 years)
  7. **Summary**: A brief narrative paragraph about the researcher
  If web search results are also available, extract and include their email address, personal webpage, and recent news/awards at the top.
- **RESEARCH TRENDS**: When the user asks about publication volume, field growth, or funding landscape — "How much research on FRBs?", "Is interest in X growing?", "Who funds research on Y?" — call `get_research_trends`. Returns papers-per-year breakdown and top funders.
- **NEVER MENTION DATA SOURCES**: NEVER mention "OpenAlex", "OpenAlex profile", "OpenAlex database", or any internal data source by name in your response. Present all researcher/trend/enrichment data as if it is native QUASAR knowledge. Do NOT include links to OpenAlex pages or API URLs.
- **WEB SEARCH**: Use the `web_search` tool ONLY for non-paper, non-archive real-time queries: current telescope schedules, observatory news, instrument specs, call-for-proposals, or operational status. NEVER use web_search when the user asks for papers/publications — use `search_papers` instead.
- After a tool runs (except search_papers), summarize the output concisely.
- If a search returns many results, offer to plot them (but execute the search first).
- If the user says "yes/proceed" to a previous suggestion, ACT on it immediately.
- **DO NOT** output raw tool usage strings like `[TOOL: ...]` or JSON. Just use the Native Tool Calling feature.
- **NAME RESOLUTION**: If search_by_target returns empty for a valid target, use the resolve_target tool to get RA/Dec, then use search_by_position.
- **MINIMAL PARAMETERS**: When calling search_by_target, ONLY include optional parameters (band, max_resolution, min_freq_ghz, etc.) if the user EXPLICITLY requested them. For example, if the user says "Find ALMA data of M87", call search_by_target(target_name="M87") with NO other parameters. Do NOT pass band=0, min_freq_ghz=0, max_resolution=100 etc. Leaving them out returns ALL data.
- **MULTI-TARGET (SAME CONSTRAINTS)**: If the user mentions multiple targets with the SAME constraints (e.g. "M87 and Sz65", or "M87, Sz65, NGC23 and M83"), pass them as a single comma-separated string: search_by_target(target_name="M87, Sz65"). The tool handles splitting and searching each target.
- **MULTI-BAND**: If the user mentions multiple bands (e.g. "Band 6 and Band 7"), pass them as comma-separated: search_by_target(target_name="M87", band="6,7"). The tool handles searching each band separately and shows a data card for each. NEVER make separate tool calls for each band — use comma-separated bands in ONE call.
- **PER-TARGET CONSTRAINTS**: If different targets have DIFFERENT band/constraint requirements (e.g. "M87 in Band 6 and Sz65 in Band 7"), make SEPARATE tool calls for each target-constraint pair: first search_by_target(target_name="M87", band="6"), then search_by_target(target_name="Sz65", band="7"). Each call produces its own data card. You MAY also pass them in one call as search_by_target(target_name="M87 in band 6, Sz65 in band 7") — the tool can parse per-target bands.
- **MULTI-WAVELENGTH / MIXED SOURCES**: For JWST/HST data with rich filtering (instrument, program, filter), prefer `search_mast` or `search_mast_by_criteria` — they provide deeper queries than search_cadc_archive. For ESO/VLT data (MUSE, KMOS, X-Shooter, FORS2), use `search_eso_archive`. For infrared catalog data (WISE, 2MASS, Spitzer), use `search_irsa`. Use `search_cadc_archive` for general multi-wavelength cone searches or telescopes like Gemini, JCMT, and CFHT. When the user asks for data from DIFFERENT archives (e.g. "ALMA data of M87 and JWST data of NGC23"), make SEPARATE tool calls for each: search_by_target(target_name="M87") for ALMA, then search_mast(target_name="NGC23", mission="JWST") for JWST. Each produces its own data card in the UI.
- **RESPECT EXCLUSIONS**: If the user explicitly excludes a source (e.g. "non-ALMA", "not from ALMA", "only CADC"), do NOT call the excluded tool. Only call the tools the user actually wants.
- **FILTERING**: If the user asks for constraints like "resolution < 0.05", use the filter_results tool AFTER a search.
- **LINE COVERAGE**: When the user asks about line coverage (e.g. "Check CO(2-1) line coverage for M87"), follow this exact 2-step workflow:
  1. **Step 1**: Search the ALMA archive for the target using search_by_target(target_name="M87"). Do NOT search VLA or other archives unless the user explicitly asks.
  2. **Step 2**: Check line coverage on the results using check_co_lines(z=<target_redshift>) for CO lines, or check_line_coverage(line_freq_ghz=<freq>, line_name="<name>") for a specific line.
  That's it — just 2 tool calls. Do NOT add extra analysis tasks, do NOT search multiple archives unless asked, and do NOT search for papers. The check_line_coverage and check_co_lines tools automatically work on the LAST search results.
- **TAP QUERIES**: When generating SQL/ADQL queries, use the column names in the schema below.

DUAL-SOURCE RESPONSE STRUCTURE (RAG + Web):
When your answer draws on BOTH the documentation context provided below AND web search results, you MUST structure your response in this EXACT order:

**SECTION 1 — Documentation Answer** (from ALMA docs/tutorials):
Present the main answer using the documentation context. Citations MUST be placed INLINE right after the sentence that uses the information — NEVER collect citations into a "References" or "Sources" section at the bottom. Each documentation chunk has a CITE_AS tag — copy that EXACT string verbatim as your citation. Do NOT modify, rephrase, or invent any fields in the citation.
- CORRECT inline citation: "The proprietary period is 12 months. [Source: alma-proposers-guide-cycle13.pdf, Page 36, Date: February 2026, Relevance: 0.88]" — copied from the CITE_AS tag.
- WRONG (bottom-grouped): Putting a "References:" section at the end listing all sources — NEVER do this.
- WRONG (missing fields): "[Source: alma-proposers-guide.pdf]" — NEVER omit Page, Date, or Relevance.
- WRONG (unknown): "[Source: file.pdf, Page unknown, Date: unknown]" — the CITE_AS tag always has the correct values. Copy them. NEVER write "unknown" or invent scores.

**SECTION 2 — Documentation Disclaimer** (immediately after Section 1, BEFORE any web content):
Add this line right after your documentation answer, before the web section:
"*📚 The above information is sourced from ALMA Documentation, tutorials, and community notebooks and may not reflect the very latest policies or changes.*"

**SECTION 3 — Web Search Updates** (after the disclaimer):
Start with the heading "🌐 Updated Information from the Web:" and then present a DETAILED summary of what the web search results contain. This section MUST:
- Be at least one full paragraph (3-5 sentences minimum) with specific facts extracted from the web results
- Include clickable Markdown links to the source URLs
- Present the web findings as-is, regardless of whether they overlap with the documentation above — the user wants to see what the web says independently
- NEVER write "No additional web updates were found" or similar dismissals. If web results are provided to you, there IS content to present — extract and summarize it.

IMPORTANT: The disclaimer (Section 2) MUST appear BETWEEN the documentation content and the web content. Never place the disclaimer after the web section.

If ONLY documentation context is available (no web results), still cite sources inline and add the documentation disclaimer.
If ONLY web results are available (no documentation), present them with links and note they are from the web.

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
- **NO IMAGE URLS**: NEVER use markdown image syntax ![alt](url) in your responses. You cannot verify image URLs and they will often be broken or incorrect. Describe visuals in text instead. The system has its own image retrieval tools -- do not embed external URLs.
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
                    "max_results": {"type": "integer", "description": "Maximum results to return. Default 500.", "default": 500},
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
                "observatory announcements, instrument specs, call-for-proposals, or operational status updates. "
                "NEVER use this for finding papers or publications — use search_papers (NASA ADS) instead. "
                "Uses Tavily for grounded, source-cited results. "
                "Use the user's query as-is — do NOT add years or dates unless the user explicitly mentioned them. "
                "Examples: 'ALMA proprietary period policy', 'JWST cycle 4 call for proposals', "
                "'VLA sensitivity at 1.4 GHz'."
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
            description=(
                "Generate a plot from the last ALMA search results. Two modes:\n"
                "1. Quick overview: pass plot_type='sky', 'frequency', or 'overview' for pre-built plots.\n"
                "2. Publication-quality scatter: pass x_column and y_column for a custom ApJ/MNRAS-style "
                "scatter plot (300 DPI, colorblind-safe).\n"
                "If plot_type is given, x_column/y_column are ignored. Use after any search."
            ),
            function=self._merged_plot_alma_results,
            parameters={
                "type": "object",
                "properties": {
                    "plot_type": {"type": "string", "enum": ["sky", "frequency", "overview"], "description": "Quick overview plot type. If provided, x_column/y_column are ignored."},
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
        # ── MAST Archive Tools (JWST, HST, TESS, Kepler) ──────────
        self.tool_registry.register(Tool(
            name="search_mast",
            description=(
                "Search the MAST archive for observations from JWST, HST, TESS, Kepler, "
                "and other space telescopes. Use this for any JWST or HST data queries. "
                "Returns observation metadata including target, instrument, filters, "
                "exposure time, and data product type."
            ),
            function=self._search_mast,
            parameters={
                "type": "object",
                "properties": {
                    "target_name": {"type": "string", "description": "Astronomical target name (e.g., 'M87', 'Carina Nebula', 'TRAPPIST-1')"},
                    "mission": {"type": "string", "description": "Filter by mission: 'JWST', 'HST', 'TESS', 'Kepler'. Leave empty for all missions."},
                    "instrument": {"type": "string", "description": "Filter by instrument (e.g., 'NIRCAM', 'NIRSPEC', 'MIRI', 'ACS', 'WFC3'). Leave empty for all."},
                    "radius": {"type": "string", "description": "Search radius (e.g., '30s' for 30 arcsec, '1m' for 1 arcmin). Default '30s'."},
                    "ra": {"type": "number", "description": "RA in degrees (use instead of target_name for positional search)"},
                    "dec": {"type": "number", "description": "Dec in degrees (use instead of target_name for positional search)"},
                },
                "required": []
            }
        ))

        self.tool_registry.register(Tool(
            name="search_mast_by_criteria",
            description=(
                "Advanced MAST search with rich filtering: program ID, date range, "
                "filter name, data product type, etc. Use this when users ask for "
                "specific JWST/HST programs, particular filters (F200W, F444W), "
                "or time-constrained searches."
            ),
            function=self._search_mast_by_criteria,
            parameters={
                "type": "object",
                "properties": {
                    "mission": {"type": "string", "description": "Mission name (JWST, HST, TESS, Kepler)"},
                    "instrument": {"type": "string", "description": "Instrument name (NIRCAM, NIRSPEC, MIRI, ACS, WFC3)"},
                    "proposal_id": {"type": "string", "description": "Specific proposal/program ID (e.g., '1345' for JADES)"},
                    "filters": {"type": "string", "description": "Filter name (e.g., 'F200W', 'F444W', 'F115W')"},
                    "target_name": {"type": "string", "description": "Target name for the search"},
                    "dataproduct_type": {"type": "string", "description": "'image', 'spectrum', 'cube', 'timeseries'"},
                    "start_date": {"type": "string", "description": "Start date for time filter (ISO format, e.g., '2022-07-01')"},
                    "end_date": {"type": "string", "description": "End date for time filter (ISO format, e.g., '2023-07-01')"},
                },
                "required": []
            }
        ))

        self.tool_registry.register(Tool(
            name="get_mast_products",
            description=(
                "Get file-level product list for the LAST MAST search results. "
                "Shows individual data files available for download (filenames, sizes, URLs). "
                "Call this AFTER a search_mast or search_mast_by_criteria call."
            ),
            function=self._get_mast_products,
            parameters={
                "type": "object",
                "properties": {
                    "product_type": {"type": "string", "description": "Filter by type: 'SCIENCE', 'CALIBRATION', 'PREVIEW'. Default all."},
                    "extension": {"type": "string", "description": "Filter by file extension: 'fits', 'jpg', etc."},
                },
                "required": []
            }
        ))

        # ── ESO Science Archive Tools (VLT instruments) ───────────
        self.tool_registry.register(Tool(
            name="search_eso_archive",
            description=(
                "Search the ESO Science Archive for VLT instrument observations. "
                "Supports instruments: MUSE, KMOS, X-Shooter, FORS2, HAWK-I, UVES, "
                "SPHERE, GRAVITY, ESPRESSO, and more. Uses TAP/ADQL queries against "
                "the ESO ObsCore table."
            ),
            function=self._search_eso,
            parameters={
                "type": "object",
                "properties": {
                    "target_name": {"type": "string", "description": "Astronomical target name (e.g., 'NGC 1068', 'Eta Carinae')"},
                    "instrument": {"type": "string", "description": "ESO instrument (e.g., 'MUSE', 'KMOS', 'XSHOOTER', 'FORS2', 'HAWK-I', 'UVES', 'SPHERE')"},
                    "ra": {"type": "number", "description": "RA in degrees (use instead of target_name)"},
                    "dec": {"type": "number", "description": "Dec in degrees (use instead of target_name)"},
                    "radius_arcmin": {"type": "number", "description": "Search radius in arcminutes. Default 1.0."},
                },
                "required": []
            }
        ))

        # ── IRSA Infrared Archive Tools (WISE, 2MASS, Spitzer) ────
        self.tool_registry.register(Tool(
            name="search_irsa",
            description=(
                "Search the IRSA (Infrared Science Archive) catalog services for "
                "infrared source catalogs. Catalogs include AllWISE, 2MASS Point "
                "Source, 2MASS Extended Source, and Spitzer SEIP."
            ),
            function=self._search_irsa,
            parameters={
                "type": "object",
                "properties": {
                    "target_name": {"type": "string", "description": "Astronomical target name (e.g., 'M31', 'NGC 253')"},
                    "catalog": {"type": "string", "description": "IRSA catalog: 'allwise' (default), '2mass', '2mass_xsc', 'seip'. Or a specific IRSA catalog ID."},
                    "radius_arcsec": {"type": "number", "description": "Search radius in arcseconds. Default 30."},
                    "ra": {"type": "number", "description": "RA in degrees (use instead of target_name)"},
                    "dec": {"type": "number", "description": "Dec in degrees (use instead of target_name)"},
                },
                "required": []
            }
        ))

        # ── Sky Survey Image Tools ────────────────────────────────
        self.tool_registry.register(Tool(
            name="get_sky_image",
            description=(
                "Fetch a sky survey cutout image for a target. Returns a FITS file "
                "and PNG preview from surveys like DSS2 (optical), 2MASS (near-IR), "
                "SDSS (optical), WISE (mid-IR), NVSS/FIRST (radio). "
                "Use this when users ask for 'an image of', 'show me', 'DSS image', "
                "or 'what does X look like'."
            ),
            function=self._get_sky_image,
            parameters={
                "type": "object",
                "properties": {
                    "target_name": {"type": "string", "description": "Astronomical target name (e.g., 'M87', 'Carina Nebula')"},
                    "survey": {"type": "string", "description": "Survey name: 'dss2' (default optical), '2mass', 'sdss', 'wise', 'nvss', 'first'. Or specific like 'DSS2 Red', '2MASS-J'."},
                    "radius_arcmin": {"type": "number", "description": "Image radius in arcminutes. Default 5."},
                    "ra": {"type": "number", "description": "RA in degrees (alternative to target_name)"},
                    "dec": {"type": "number", "description": "Dec in degrees (alternative to target_name)"},
                },
                "required": []
            }
        ))

        # ── Data Download Tools ───────────────────────────────────
        self.tool_registry.register(Tool(
            name="download_mast_data",
            description=(
                "Download FITS files from the MAST archive (JWST/HST data). "
                "Call this AFTER search_mast to download actual science data files. "
                "Downloads to ~/quasar_data/mast/ by default. Has a safety limit of 10 files."
            ),
            function=self._download_mast_data,
            parameters={
                "type": "object",
                "properties": {
                    "product_type": {"type": "string", "description": "Filter: 'SCIENCE' (default), 'CALIBRATION', 'PREVIEW'"},
                    "extension": {"type": "string", "description": "File extension filter: 'fits' (default), 'jpg', etc."},
                    "max_files": {"type": "integer", "description": "Max files to download (default 10, safety limit)"},
                },
                "required": []
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
                "IMPORTANT: Pass the user's request as natural language — an internal AI query builder will "
                "automatically translate it into optimal ADS syntax using keyword searches, bibgroup filters, "
                "SIMBAD object linking, second-order discovery operators (trending, similar, useful), and more.\n"
                "Examples of what to pass as query:\n"
                "- 'recent papers on protoplanetary disks'\n"
                "- 'best ALMA papers on disk gaps'\n"
                "- 'papers about HL Tau'\n"
                "- 'what are people reading about FRBs right now'\n"
                "- 'foundational papers on planet formation'\n"
                "- 'review articles on AGN feedback'\n"
                "- 'papers by Sean Andrews on disk surveys'\n"
                "Do NOT try to construct ADS field syntax yourself — just pass the natural language query."
            ),
            function=lambda query, max_results=15, sort="date desc", **kw: (
                self._search_papers(query, max_results=max_results, sort=sort)
                if self.ads_client else {"error": "ADS client not configured"}
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Natural language search query describing what papers the user wants (e.g. 'recent ALMA papers on protoplanetary disk gaps')"},
                    "max_results": {"type": "integer", "description": "Number of results to return (default 15, max 50)"},
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
            name="extract_paper_details",
            description="Download a scientific paper by its arXiv ID or ADS bibcode and extract specific details (e.g. beam size, flux density, telescope configuration) using an LLM QA pass over the full text. Use when the user asks specific questions about the contents of a published paper.",
            function=self._extract_paper_details,
            parameters={
                "type": "object",
                "properties": {
                    "identifier": {"type": "string", "description": "arXiv ID (e.g. '1812.04040') or ADS bibcode (e.g. '2018ApJ...869L..41A')"},
                    "query": {"type": "string", "description": "The specific question to ask about the paper's contents (e.g. 'What was the exact angular resolution achieved for AS 209?')"}
                },
                "required": ["identifier", "query"]
            },
            category="literature"
        ))

        self.tool_registry.register(Tool(
            name="evaluate_consensus",
            description=(
                "Evaluate the scientific consensus on a research question by searching NASA ADS for the most-cited "
                "papers on the topic, reading all their abstracts, and producing a structured analysis. "
                "The output includes: overall consensus level (Strong Agreement / Divided / etc.), "
                "which specific papers agree vs disagree, WHY they disagree (different methods, data, assumptions), "
                "key evidence from each side with proper citations, how the consensus has evolved over time, "
                "and what open questions remain. "
                "Use this when the user asks questions like: 'Do scientists agree on X?', 'What does the field think about X?', "
                "'Is there consensus on X?', 'What's the current understanding of X?', or any question where "
                "a literature-wide summary would be more useful than individual paper results."
            ),
            function=self._evaluate_consensus,
            parameters={
                "type": "object",
                "properties": {
                    "question": {"type": "string", "description": "The scientific question to evaluate consensus on (e.g. 'Is planet migration necessary for hot Jupiter formation?')"},
                    "max_papers": {"type": "integer", "description": "Number of top-cited papers to analyze (default 20, max 50)"},
                },
                "required": ["question"]
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

        # ── OpenAlex Researcher & Bibliometric Tools ──────────────────────
        _oalex = self.openalex_client

        self.tool_registry.register(Tool(
            name="lookup_researcher",
            description=(
                "Look up a researcher/scientist by name or ORCID to get their full academic profile: "
                "current institution, h-index, i10-index, total publications, total citations, "
                "ORCID, Scopus ID, research topics, affiliation history, and publication trend "
                "over the last 10 years.  Powered by OpenAlex (90M+ disambiguated authors).\n"
                "Use this when the user asks about a person, wants to know who someone is, "
                "or wants contact/institutional information about a researcher.\n"
                "Examples: 'Who is Andrea Isella?', 'Tell me about Crystal Brogan', "
                "'Look up ORCID 0000-0001-2345-6789'"
            ),
            function=self._lookup_researcher,
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "Researcher name (e.g. 'Andrea Isella') or ORCID "
                            "(e.g. '0000-0001-2345-6789')"
                        ),
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Max author matches to return (default 3)",
                    },
                },
                "required": ["query"]
            },
            category="literature"
        ))

        self.tool_registry.register(Tool(
            name="get_research_trends",
            description=(
                "Get a bibliometric trend showing papers-per-year for a given topic or search "
                "query.  Returns total paper count and yearly breakdown.\n"
                "Use when the user asks 'How much research is being done on X?', "
                "'Is interest in X growing?', 'Publication trends for FRBs'.\n"
                "Also returns the funding landscape — which funders (NSF, NASA, ESA, etc.) "
                "have funded research on the topic."
            ),
            function=self._get_research_trends,
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Topic or search query (e.g. 'fast radio bursts', 'ALMA protoplanetary disks')",
                    },
                    "year_from": {
                        "type": "integer",
                        "description": "Start year for the trend (default 2015)",
                    },
                    "year_to": {
                        "type": "integer",
                        "description": "End year for the trend (default 2026)",
                    },
                },
                "required": ["query"]
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

        # ── Spectral Line Profile Fitter (R3) ──────────────────────
        self.tool_registry.register(Tool(
            name="fit_spectral_line",
            description=(
                "Download a FITS spectral cube, extract a 1D spectrum at a given "
                "position, and fit a Gaussian profile to the strongest line. "
                "Returns peak flux, FWHM (in frequency and velocity), center "
                "frequency, and integrated flux. The fit is overlaid on the "
                "spectrum plot."
            ),
            function=self._fit_spectral_line,
            parameters={
                "type": "object",
                "properties": {
                    "url":     {"type": "string", "description": "Direct URL to the FITS spectral cube."},
                    "ra_deg":  {"type": "number", "description": "RA in decimal degrees (ICRS). Optional."},
                    "dec_deg": {"type": "number", "description": "Dec in decimal degrees (ICRS). Optional."},
                    "x_pixel": {"type": "integer", "description": "X pixel coordinate. Optional."},
                    "y_pixel": {"type": "integer", "description": "Y pixel coordinate. Optional."},
                    "title":   {"type": "string",  "description": "Title for the spectrum plot."},
                },
                "required": ["url"]
            },
            category="analysis"
        ))

        # ── Astronomy Calculators (U9, U10, R6, R4) ───────────────
        self.tool_registry.register(Tool(
            name="calculate_redshift",
            description=(
                "Compute cosmological quantities for a given redshift z using "
                "Planck18 cosmology. Returns luminosity distance, angular diameter "
                "distance, comoving distance, lookback time, age of the universe "
                "at that epoch, and the physical scale (kpc per arcsecond). "
                "Use for any question about distances, ages, or scales at a "
                "given redshift."
            ),
            function=self._calculate_redshift,
            parameters={
                "type": "object",
                "properties": {
                    "z": {"type": "number", "description": "Cosmological redshift (must be >= 0)."},
                },
                "required": ["z"]
            },
            category="analysis"
        ))

        self.tool_registry.register(Tool(
            name="convert_coordinates",
            description=(
                "Convert sky coordinates between ICRS (RA/Dec), Galactic (l/b), "
                "Ecliptic (lon/lat), FK5 (J2000), and FK4 (B1950) frames. "
                "Returns the position in ALL frames at once. "
                "Use for coordinate transformations, epoch precession, or when "
                "the user gives Galactic coordinates and needs RA/Dec."
            ),
            function=self._convert_coordinates,
            parameters={
                "type": "object",
                "properties": {
                    "ra":           {"type": "number", "description": "RA or longitude in degrees (for ICRS/Ecliptic/FK5 input)."},
                    "dec":          {"type": "number", "description": "Dec or latitude in degrees."},
                    "l":            {"type": "number", "description": "Galactic longitude in degrees (for Galactic input)."},
                    "b":            {"type": "number", "description": "Galactic latitude in degrees (for Galactic input)."},
                    "input_frame":  {"type": "string", "description": "Source frame: 'icrs', 'galactic', 'ecliptic', 'fk5', 'fk4'. Default 'icrs'."},
                    "output_frame": {"type": "string", "description": "Target frame (all frames are always returned). Default 'galactic'."},
                },
                "required": []
            },
            category="analysis"
        ))

        self.tool_registry.register(Tool(
            name="calculate_beam",
            description=(
                "Calculate the synthesized beam size for a radio interferometer "
                "given the maximum baseline and observing frequency. For ALMA, "
                "you can specify an array configuration name (C-1 through C-10) "
                "instead of a raw baseline length. Returns beam size in arcsec "
                "and milliarcsec."
            ),
            function=self._calculate_beam,
            parameters={
                "type": "object",
                "properties": {
                    "frequency_ghz":  {"type": "number", "description": "Observing frequency in GHz."},
                    "max_baseline_m": {"type": "number", "description": "Maximum baseline in meters. Optional if array_config is given."},
                    "array_config":   {"type": "string", "description": "ALMA config name: C-1 through C-10. Overrides max_baseline_m."},
                },
                "required": ["frequency_ghz"]
            },
            category="analysis"
        ))

        self.tool_registry.register(Tool(
            name="calculate_alma_sensitivity",
            description=(
                "Estimate ALMA continuum and spectral line sensitivity using "
                "the radiometer equation. Returns noise level in mJy/beam and "
                "uJy/beam for given band, bandwidth, and integration time. "
                "Includes Tsys scaling for weather (PWV). Use when the user "
                "asks about ALMA sensitivity, noise levels, or integration "
                "time estimates."
            ),
            function=self._calculate_alma_sensitivity,
            parameters={
                "type": "object",
                "properties": {
                    "band":             {"type": "integer", "description": "ALMA band number (3-10)."},
                    "bandwidth_ghz":    {"type": "number",  "description": "Total continuum bandwidth in GHz. Default 7.5."},
                    "t_integration_s":  {"type": "number",  "description": "On-source integration time in seconds. Default 60."},
                    "n_antennas":       {"type": "integer", "description": "Number of antennas. Default 50."},
                    "n_polarizations":  {"type": "integer", "description": "Number of polarizations (1 or 2). Default 2."},
                    "channel_width_khz":{"type": "number",  "description": "Spectral channel width in kHz (for line sensitivity). Optional."},
                    "pwv_mm":           {"type": "number",  "description": "Precipitable water vapor in mm. Default 1.0."},
                },
                "required": ["band"]
            },
            category="analysis"
        ))

        # ── Finding Chart Generator (O6) ──────────────────────────
        self.tool_registry.register(Tool(
            name="generate_finding_chart",
            description=(
                "Generate a publication-quality finding chart for a target. "
                "Creates a DSS2 or 2MASS image with WCS axes, a target "
                "crosshair marker, N/E compass arrows, and an angular scale "
                "bar. Use when the user needs a finding chart for observations "
                "or proposals."
            ),
            function=self._generate_finding_chart,
            parameters={
                "type": "object",
                "properties": {
                    "target":      {"type": "string", "description": "Target name (e.g., 'M87', 'NGC 1068')."},
                    "ra":          {"type": "number", "description": "RA in degrees (alternative to target)."},
                    "dec":         {"type": "number", "description": "Dec in degrees (alternative to target)."},
                    "survey":      {"type": "string", "description": "Sky survey: 'DSS2 Red', '2MASS-J', 'WISE 3.4', etc. Default 'DSS2 Red'."},
                    "fov_arcmin":  {"type": "number", "description": "Field of view in arcminutes. Default 5."},
                    "title":       {"type": "string", "description": "Custom chart title."},
                },
                "required": []
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
            result = self.datalink_client.list_files(
                mous_uid=mous_uid,
                pattern=filename_pattern,
            )
            # list_files() returns a dict: {success, mous_uid, total_files, files, error?}
            # Unpack the file list from the dict
            if isinstance(result, dict):
                if not result.get("success", False):
                    return {
                        "success": False,
                        "mous_uid": mous_uid,
                        "error": result.get("error", "DataLink query failed"),
                    }
                file_list = result.get("files", [])
            elif isinstance(result, list):
                # Legacy path: if list_files ever returns a raw list
                file_list = result
            else:
                file_list = []

            if not file_list:
                return {
                    "success": True,
                    "mous_uid": mous_uid,
                    "file_count": 0,
                    "message": "No files found. The MOUS UID might be invalid or the data is not yet public.",
                }
            return {
                "success": True,
                "mous_uid": mous_uid,
                "file_count": len(file_list),
                "files": file_list[:50],  # Cap at 50 for LLM context
                "note": f"Found {len(file_list)} file(s)." + (
                    f" Showing first 50." if len(file_list) > 50 else ""
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

    # ── Astronomy acronym dictionary for web search disambiguation ─────
    _ASTRO_ACRONYMS: dict = {
        # Telescopes & Observatories
        "ALMA":   "ALMA (Atacama Large Millimeter/submillimeter Array)",
        "VLA":    "VLA (Very Large Array radio telescope)",
        "VLBA":   "VLBA (Very Long Baseline Array)",
        "VLBI":   "VLBI (Very Long Baseline Interferometry)",
        "GBT":    "GBT (Green Bank Telescope)",
        "JWST":   "JWST (James Webb Space Telescope)",
        "HST":    "HST (Hubble Space Telescope)",
        "JCMT":   "JCMT (James Clerk Maxwell Telescope)",
        "CFHT":   "CFHT (Canada-France-Hawaii Telescope)",
        "SKA":    "SKA (Square Kilometre Array)",
        "ELT":    "ELT (Extremely Large Telescope)",
        "TMT":    "TMT (Thirty Meter Telescope)",
        "GMT":    "GMT (Giant Magellan Telescope)",
        "NOEMA":  "NOEMA (NOrthern Extended Millimeter Array)",
        "IRAM":   "IRAM (Institut de Radioastronomie Millimétrique)",
        "LOFAR":  "LOFAR (Low-Frequency Array)",
        "MeerKAT":"MeerKAT (Karoo Array Telescope)",
        "ASKAP":  "ASKAP (Australian Square Kilometre Array Pathfinder)",
        "FAST":   "FAST (Five-hundred-meter Aperture Spherical Telescope)",
        "CTA":    "CTA (Cherenkov Telescope Array)",
        "LSST":   "LSST (Legacy Survey of Space and Time, Vera C. Rubin Observatory)",
        # Space missions
        "TESS":   "TESS (Transiting Exoplanet Survey Satellite)",
        "WMAP":   "WMAP (Wilkinson Microwave Anisotropy Probe)",
        "XMM":    "XMM-Newton (X-ray Multi-Mirror Mission)",
        "NuSTAR": "NuSTAR (Nuclear Spectroscopic Telescope Array)",
        "SOFIA":  "SOFIA (Stratospheric Observatory for Infrared Astronomy)",
        "NICER":  "NICER (Neutron star Interior Composition Explorer)",
        # Data archives & services
        "ADS":    "ADS (NASA Astrophysics Data System)",
        "CADC":   "CADC (Canadian Astronomy Data Centre)",
        "ESO":    "ESO (European Southern Observatory)",
        "MAST":   "MAST (Mikulski Archive for Space Telescopes)",
        "NRAO":   "NRAO (National Radio Astronomy Observatory)",
        "SIMBAD": "SIMBAD (Set of Identifications, Measurements, and Bibliography for Astronomical Data)",
        "NED":    "NED (NASA/IPAC Extragalactic Database)",
        "CDS":    "CDS (Centre de Données astronomiques de Strasbourg)",
        # Software & pipelines
        "CASA":   "CASA (Common Astronomy Software Applications)",
        "CARTA":  "CARTA (Cube Analysis and Rendering Tool for Astronomy)",
        # Concepts / techniques
        "AGN":    "AGN (Active Galactic Nucleus)",
        "ISM":    "ISM (interstellar medium)",
        "IGM":    "IGM (intergalactic medium)",
        "CMB":    "CMB (Cosmic Microwave Background)",
        "GRB":    "GRB (Gamma-Ray Burst)",
        "SNR":    "SNR (Supernova Remnant)",
        "HII":    "HII region (ionized hydrogen region)",
        "FRB":    "FRB (Fast Radio Burst)",
        "SED":    "SED (Spectral Energy Distribution)",
        "RFI":    "RFI (Radio Frequency Interference)",
        "RA":     None,  # skip — too common
        "DEC":    None,  # skip — too common
    }

    def _expand_astro_query(self, query: str) -> str:
        """Expand astronomy acronyms in a web search query so generic
        search engines return domain-relevant results instead of
        irrelevant hits (e.g., 'ALMA' → 'ALMA (Atacama Large Millimeter Array)')."""
        import re as _re
        words = query.split()
        expanded = False
        for i, w in enumerate(words):
            clean = _re.sub(r'[^A-Za-z]', '', w)
            upper = clean.upper()
            if upper in self._ASTRO_ACRONYMS and self._ASTRO_ACRONYMS[upper] is not None:
                # Only expand if the word appears to be an acronym (all-caps or title-case)
                if clean.isupper() or (len(clean) >= 2 and clean[0].isupper()):
                    # Preserve any trailing punctuation
                    trail = w[len(clean):]
                    words[i] = self._ASTRO_ACRONYMS[upper] + trail
                    expanded = True
        result = " ".join(words)
        # If we expanded something, add astronomy context hint
        if expanded and "astronomy" not in result.lower() and "astrophysic" not in result.lower():
            result += " astronomy"
        return result

    def _synthesize_web_summary(self, query: str, web_data: Dict[str, Any], reason: str) -> str:
        """
        Use a fast LLM to synthesize a concise, query-relevant summary from
        web search snippets.  This replaces the raw Tavily `answer` field which
        is often generic and unrelated to the user's actual question.
        """
        snippets = []
        for r in web_data.get("results", [])[:5]:
            title = r.get("title", "")
            snippet = r.get("snippet", r.get("content", ""))[:400]
            url = r.get("url", "")
            if snippet:
                snippets.append(f"[{title}]({url}): {snippet}")

        if not snippets:
            return web_data.get("answer", "").strip()

        context_block = "\n\n".join(snippets)

        system_prompt = (
            "You are a concise research assistant. Given web search snippets, "
            "write a SHORT (2-4 sentence) summary that DIRECTLY answers the "
            "user's question using ONLY information from the snippets. "
            "Include specific facts, numbers, or dates when available. "
            "If the snippets don't contain relevant information, say so briefly. "
            "Do NOT repeat background context the user already knows. "
            "Do NOT include generic descriptions of organizations or telescopes."
        )

        user_prompt = (
            f"User question: {query}\n\n"
            f"Web search snippets:\n{context_block}\n\n"
            f"Write a concise, directly relevant summary:"
        )

        try:
            from openai import OpenAI
            client = OpenAI()
            resp = client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=200,
                temperature=0.2,
            )
            summary = resp.choices[0].message.content.strip()
            if summary:
                return summary
        except Exception as e:
            print(f"[WEB SEARCH] Summary synthesis failed: {e}")

        # Fallback to raw Tavily answer
        return web_data.get("answer", "").strip()

    @log_tool
    def _tavily_web_search(self, query: str, max_results: int = 10, search_depth: str = "basic") -> Dict[str, Any]:
        """
        Real-time web search powered by Tavily.
        Returns source URLs + related images for ChatGPT-style inline display.
        Falls back to BrowserService if Tavily key is unavailable.
        """
        # Expand astronomy acronyms — only needed for dumb keyword search
        # engines (BrowserService fallback). Tavily is AI-powered and handles
        # acronyms natively; expanding pollutes the query and returns generic
        # results instead of matching the user's specific intent.
        search_query = self._expand_astro_query(query)
        if search_query != query:
            print(f"[WEB SEARCH] Expanded query (for fallback): {query!r} → {search_query!r}")

        tavily_key = os.getenv("TAVILY_API_KEY", "")
        if tavily_key:
            try:
                from tavily import TavilyClient
                client = TavilyClient(api_key=tavily_key)
                max_results = min(int(max_results), 10)
                # Use the RAW query — Tavily's AI understands acronyms
                response = client.search(
                    query=query,
                    max_results=max_results,
                    search_depth=search_depth,
                    include_answer=True,
                    include_images=True,
                    include_image_descriptions=True,
                    include_raw_content=False,
                )
                # Build source results with full URLs for citation cards
                results = []
                for r in response.get("results", []):
                    results.append({
                        "title":   r.get("title", ""),
                        "url":     r.get("url", ""),
                        "snippet": r.get("content", "")[:500],
                    })

                # Extract images (top-level query images from Tavily)
                images = []
                for img in response.get("images", []):
                    if isinstance(img, dict):
                        images.append({
                            "url": img.get("url", ""),
                            "description": img.get("description", ""),
                        })
                    elif isinstance(img, str):
                        images.append({"url": img, "description": ""})

                return {
                    "success":      True,
                    "provider":     "Tavily",
                    "query":        query,
                    "answer":       response.get("answer", ""),
                    "results":      results,
                    "images":       images[:6],  # cap at 6 images
                    "result_count": len(results),
                }
            except Exception as e:
                print(f"[WARN] Tavily search failed: {e}. Falling back to BrowserService.")

        # -- Fallback: BrowserService --
        try:
            fallback = self.browser_service.web_search(query=search_query)
            return {"success": True, "provider": "BrowserService (fallback)", "results": fallback, "images": []}
        except Exception as e2:
            return {"success": False, "error": str(e2)}

    @log_tool
    def _search_by_position(self, ra: float, dec: float, radius: float = 0.5,
                           facility: Optional[str] = None,
                           band: Optional[str] = None,
                           max_results: int = 100, **kwargs) -> Dict[str, Any]:
        """Search archives by sky position.
        
        Accepts `band` and extra kwargs so the LLM can pass them without
        crashing, even though the underlying cone_search doesn't use them.
        Band filtering is applied as a post-filter on the results.
        """
        try:
            results = self.search_service.cone_search(
                ra, dec, radius, facility, max_results
            )

            # Post-filter by band if specified
            if band is not None and not results.empty:
                import re as _re_band
                band_vals = []
                if isinstance(band, str):
                    parts = _re_band.split(r'[,\s]+and[\s]+|[,\s]+', band.strip())
                    band_vals = [int(p) for p in parts if p.strip().isdigit()]
                elif isinstance(band, (int, float)):
                    band_vals = [int(band)]
                if band_vals:
                    b_col = next((c for c in ['band_list', 'Band', 'band'] if c in results.columns), None)
                    if b_col:
                        before = len(results)
                        results = results[results[b_col].astype(str).apply(
                            lambda x: any(str(b) in [v.strip() for v in x.split(',')] for b in band_vals)
                        )]
                        print(f"[FILTER] Band {band}: {before} → {len(results)} rows")

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

        # ── Safety: ignore near-zero/zero min values (LLM default filling) ──
        # The LLM often fills 0 for optional params despite being told not to.
        # A value of 0 for resolution/frequency/exptime means "no filter".
        if min_resolution is not None and min_resolution <= 0:
            min_resolution = None
        if max_resolution is not None and max_resolution <= 0:
            max_resolution = None  # 0 arcsec = impossible, treat as no filter
        if min_freq_ghz is not None and min_freq_ghz <= 0:
            min_freq_ghz = None
        if max_freq_ghz is not None and max_freq_ghz <= 0:
            max_freq_ghz = None
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
            # Also handles per-target band specs like:
            #   "M87 in band 6, Sz65 in band 7"  →  per-target bands
            #   "M87, Sz65, NGC23"               →  shared bands (from band= param)
            import re as _re
            raw_names = [n.strip() for n in _re.split(r'\s+and\s+|\s*,\s*', target_name) if n.strip()]

            # ── Parse per-target band specifications ──────────────────
            # If target_name contains inline band specs (e.g. "M87 in band 6"),
            # extract them so each target gets its own filter.
            _per_target_specs = []  # list of (name, [bands]) tuples
            _has_per_target_bands = False
            _inline_band_pattern = _re.compile(
                r'^(.+?)\s+(?:in\s+)?band\s*([\d,\s]+(?:\s*(?:and|,)\s*\d+)*)$',
                _re.IGNORECASE,
            )
            for raw in raw_names:
                m = _inline_band_pattern.match(raw.strip())
                if m:
                    _tgt_name = m.group(1).strip()
                    _band_str = m.group(2)
                    _bands = [int(b.strip()) for b in _re.split(r'[,\s]+and[\s]+|[,\s]+', _band_str) if b.strip().isdigit()]
                    _bands = [b for b in _bands if 1 <= b <= 10]
                    _per_target_specs.append((_tgt_name, _bands))
                    if _bands:
                        _has_per_target_bands = True
                else:
                    _per_target_specs.append((raw.strip(), []))

            if _has_per_target_bands:
                # Per-target band mode: search each target with its own band filter
                all_frames = []
                searched_names = []
                for _tgt, _tgt_bands in _per_target_specs:
                    if not _tgt:
                        continue
                    try:
                        df = self.search_service.search_by_target(
                            _tgt, facility, date_range, max_results
                        )
                        if not df.empty and _tgt_bands:
                            # Apply per-target band filter
                            b_col = next((c for c in ["band_list", "Band", "band"] if c in df.columns), None)
                            if b_col:
                                df = df[df[b_col].astype(str).str.split(",").apply(
                                    lambda bands: any(str(b).strip() == x.strip() for x in bands for b in _tgt_bands)
                                )]
                        if not df.empty:
                            all_frames.append(df)
                            band_label = ",".join(str(b) for b in _tgt_bands) if _tgt_bands else "all"
                            searched_names.append(_tgt)
                            print(f"[PER-TARGET] '{_tgt}' band={band_label} → {len(df)} results")
                        else:
                            print(f"[PER-TARGET] '{_tgt}' → 0 results")
                    except Exception as e:
                        print(f"[PER-TARGET] '{_tgt}' failed: {e}")

                if all_frames:
                    results = pd.concat(all_frames, ignore_index=True)
                    target_name = " + ".join(searched_names)
                else:
                    results = pd.DataFrame()
                # Skip shared band filtering below — bands already applied per target
                band_list_input = []

            elif len(raw_names) > 1:
                # Search each target independently, concatenate results
                all_frames = []
                searched_names = []
                for name in raw_names[:10]:  # Cap at 10 targets
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
                # ── Automatic positional fallback ─────────────────────
                # The name-based search uses a tight radius (0.05°).
                # Many ALMA observations have offset pointing centers, so
                # retry with a wider cone search to avoid losing results
                # (and critically, to keep band/filter params applied).
                print(f"[FALLBACK] Name search empty for '{target_name}', trying positional fallback...")
                try:
                    resolved = self._resolve_target(target_name)
                    if resolved.get("success") and resolved.get("ra") is not None:
                        _fb_ra, _fb_dec = resolved["ra"], resolved["dec"]
                        print(f"[FALLBACK] Resolved to RA={_fb_ra:.4f}, Dec={_fb_dec:.4f} — cone search 0.14°")
                        results = self.search_service.cone_search(
                            _fb_ra, _fb_dec, radius=0.14,
                            facility=facility, max_results=max_results
                        )
                        if not results.empty:
                            print(f"[FALLBACK] Cone search found {len(results)} results — continuing with filters")
                except Exception as _fb_err:
                    print(f"[FALLBACK] Positional fallback failed: {_fb_err}")

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
                # Multi-band: combine into a single result with all matching bands
                band_str_list = [str(b) for b in band_list_input]
                combined = results[results[band_col].astype(str).str.split(",").apply(
                    lambda bands: any(x.strip() in band_str_list for x in bands)
                )]
                for b in band_list_input:
                    ct = len(results[results[band_col].astype(str).str.split(",").apply(
                        lambda bands, _b=b: any(str(_b).strip() == x.strip() for x in bands)
                    )])
                    print(f"[MULTI-BAND] Band {b}: {ct} rows")

                if not combined.empty:
                    results = combined
                    band_label = ", ".join(f"Band {b}" for b in band_list_input)
                    filter_label = f"ALMA › {target_name} [{band_label}" + (", ".join([""] + filter_parts) if filter_parts else "") + "]"
                else:
                    # None of the bands matched — show unfiltered
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
                     max_results: int = 500) -> Dict[str, Any]:
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
            SELECT TOP {max_results} *
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

    # ── MAST Archive Handlers ──────────────────────────────────────────

    def _search_mast(self, target_name: Optional[str] = None,
                     mission: Optional[str] = None,
                     instrument: Optional[str] = None,
                     radius: str = "30s",
                     ra: Optional[float] = None,
                     dec: Optional[float] = None) -> Dict[str, Any]:
        """Search MAST archive for JWST/HST/TESS observations."""
        try:
            if target_name:
                df = self.mast_client.search_by_target(
                    target=target_name, mission=mission,
                    instrument=instrument, radius=radius
                )
            elif ra is not None and dec is not None:
                df = self.mast_client.search_by_position(
                    ra=ra, dec=dec,
                    radius_arcmin=self.mast_client._parse_radius(radius).to("arcmin").value,
                    mission=mission, instrument=instrument
                )
            else:
                return {"success": False, "error": "Provide target_name or (ra, dec) coordinates."}

            if df.empty:
                note = f"No MAST observations found"
                if target_name:
                    note += f" for '{target_name}'"
                if mission:
                    note += f" [{mission}]"
                if instrument:
                    note += f" [{instrument}]"
                self.last_run_result = {"type": "data", "data": df, "source": "MAST", "tool_name": "search_mast"}
                return {"success": True, "total_results": 0, "note": note}

            # Build source label
            filter_label = "MAST"
            if target_name:
                filter_label += f" › {target_name}"
            if mission:
                filter_label += f" [{mission}]"
            if instrument:
                filter_label += f" [{instrument}]"

            self.last_search_results = df
            self.last_run_result = {
                "type": "data", "data": df,
                "source": "MAST", "filter_label": filter_label,
                "tool_name": "search_mast"
            }

            # Build summary for LLM
            mission_summary = df["telescope"].value_counts().to_dict() if "telescope" in df.columns else {}
            instr_summary = df["instrument_name"].value_counts().to_dict() if "instrument_name" in df.columns else {}

            return {
                "success": True,
                "total_results": len(df),
                "missions": mission_summary,
                "instruments": instr_summary,
                "unique_targets": int(df["target_name"].nunique()) if "target_name" in df.columns else 0,
                "unique_programs": int(df["project_code"].nunique()) if "project_code" in df.columns else 0,
                "note": (
                    f"Found {len(df)} MAST observations. "
                    f"Missions: {', '.join(f'{m} ({c})' for m, c in mission_summary.items())}. "
                    f"Full data shown in UI table. Do NOT render a table — the UI already displays one."
                )
            }
        except Exception as e:
            return {"success": False, "error": f"MAST search failed: {str(e)}"}

    def _search_mast_by_criteria(self, mission: Optional[str] = None,
                                  instrument: Optional[str] = None,
                                  proposal_id: Optional[str] = None,
                                  filters: Optional[str] = None,
                                  target_name: Optional[str] = None,
                                  dataproduct_type: Optional[str] = None,
                                  start_date: Optional[str] = None,
                                  end_date: Optional[str] = None) -> Dict[str, Any]:
        """Advanced MAST criteria search with rich filtering."""
        try:
            date_range = None
            if start_date and end_date:
                date_range = (start_date, end_date)

            df = self.mast_client.search_by_criteria(
                mission=mission, instrument=instrument,
                proposal_id=proposal_id, filters=filters,
                target_name=target_name,
                dataproduct_type=dataproduct_type,
                date_range=date_range
            )

            if df.empty:
                criteria_parts = []
                if mission: criteria_parts.append(f"mission={mission}")
                if instrument: criteria_parts.append(f"instrument={instrument}")
                if proposal_id: criteria_parts.append(f"program={proposal_id}")
                if filters: criteria_parts.append(f"filter={filters}")
                note = f"No MAST observations found for criteria: {', '.join(criteria_parts) or 'unspecified'}"
                self.last_run_result = {"type": "data", "data": df, "source": "MAST", "tool_name": "search_mast_by_criteria"}
                return {"success": True, "total_results": 0, "note": note}

            # Build label
            filter_label = "MAST Criteria"
            if mission: filter_label += f" [{mission}]"
            if proposal_id: filter_label += f" Program {proposal_id}"
            if target_name: filter_label += f" › {target_name}"

            self.last_search_results = df
            self.last_run_result = {
                "type": "data", "data": df,
                "source": "MAST", "filter_label": filter_label,
                "tool_name": "search_mast_by_criteria"
            }

            mission_summary = df["telescope"].value_counts().to_dict() if "telescope" in df.columns else {}
            instr_summary = df["instrument_name"].value_counts().to_dict() if "instrument_name" in df.columns else {}
            filter_summary = df["filters"].value_counts().head(10).to_dict() if "filters" in df.columns else {}

            return {
                "success": True,
                "total_results": len(df),
                "missions": mission_summary,
                "instruments": instr_summary,
                "filters_used": filter_summary,
                "unique_targets": int(df["target_name"].nunique()) if "target_name" in df.columns else 0,
                "note": (
                    f"Found {len(df)} observations. "
                    f"Instruments: {', '.join(f'{i} ({c})' for i, c in instr_summary.items())}. "
                    f"Full data shown in UI table."
                )
            }
        except Exception as e:
            return {"success": False, "error": f"MAST criteria search failed: {str(e)}"}

    def _get_mast_products(self, product_type: Optional[str] = None,
                           extension: Optional[str] = None) -> Dict[str, Any]:
        """Get file-level product list for last MAST search results."""
        try:
            if self.last_search_results is None or self.last_search_results.empty:
                return {
                    "success": False,
                    "error": "No MAST search results to get products for. Run search_mast or search_mast_by_criteria first.",
                }
            if "obsid" not in self.last_search_results.columns:
                return {
                    "success": False,
                    "error": "Last cached results are not MAST observation results. Run search_mast or search_mast_by_criteria first.",
                }

            df = self.mast_client.get_product_list(
                self.last_search_results,
                productType=product_type,
                extension=extension
            )

            if df.empty:
                return {"success": True, "total_products": 0, "note": "No data products found."}

            self.last_search_results = df
            self.last_run_result = {
                "type": "data", "data": df,
                "source": "MAST Products",
                "filter_label": "MAST Products",
                "tool_name": "get_mast_products"
            }

            type_summary = df["productType"].value_counts().to_dict() if "productType" in df.columns else {}

            return {
                "success": True,
                "total_products": len(df),
                "product_types": type_summary,
                "note": (
                    f"Found {len(df)} data products. "
                    f"Types: {', '.join(f'{t} ({c})' for t, c in type_summary.items())}."
                )
            }
        except Exception as e:
            return {"success": False, "error": f"MAST product listing failed: {str(e)}"}

    # ── ESO Archive Handler ────────────────────────────────────────────

    def _search_eso(self, target_name: Optional[str] = None,
                    instrument: Optional[str] = None,
                    ra: Optional[float] = None,
                    dec: Optional[float] = None,
                    radius_arcmin: float = 1.0) -> Dict[str, Any]:
        """Search ESO Science Archive for VLT instrument observations."""
        try:
            if target_name:
                df = self.eso_client.search_by_target(
                    target=target_name, instrument=instrument,
                    radius_arcmin=radius_arcmin
                )
            elif ra is not None and dec is not None:
                df = self.eso_client.search_by_position(
                    ra=ra, dec=dec, radius_arcmin=radius_arcmin,
                    instrument=instrument
                )
            else:
                return {"success": False, "error": "Provide target_name or (ra, dec) coordinates."}

            if df.empty:
                note = f"No ESO observations found"
                if target_name:
                    note += f" for '{target_name}'"
                if instrument:
                    note += f" [{instrument}]"
                self.last_run_result = {"type": "data", "data": df, "source": "ESO", "tool_name": "search_eso_archive"}
                return {"success": True, "total_results": 0, "note": note}

            # Build source label
            filter_label = "ESO"
            if target_name:
                filter_label += f" › {target_name}"
            if instrument:
                filter_label += f" [{instrument}]"

            self.last_search_results = df
            self.last_run_result = {
                "type": "data", "data": df,
                "source": "ESO", "filter_label": filter_label,
                "tool_name": "search_eso_archive"
            }

            instr_summary = df["instrument_name"].value_counts().to_dict() if "instrument_name" in df.columns else {}
            dptype_summary = df["dataproduct_type"].value_counts().to_dict() if "dataproduct_type" in df.columns else {}

            return {
                "success": True,
                "total_results": len(df),
                "instruments": instr_summary,
                "data_types": dptype_summary,
                "note": (
                    f"Found {len(df)} ESO observations. "
                    f"Instruments: {', '.join(f'{i} ({c})' for i, c in instr_summary.items())}. "
                    f"Full data shown in UI table. Do NOT render a table — the UI already displays one."
                )
            }
        except Exception as e:
            return {"success": False, "error": f"ESO archive search failed: {str(e)}"}

    # ── IRSA Archive Handler ───────────────────────────────────────────

    def _search_irsa(self, target_name: Optional[str] = None,
                     catalog: Optional[str] = None,
                     radius_arcsec: float = 30.0,
                     ra: Optional[float] = None,
                     dec: Optional[float] = None) -> Dict[str, Any]:
        """Search IRSA archive for infrared survey data."""
        try:
            catalog = catalog or "allwise"

            if target_name:
                df = self.irsa_client.search_by_target(
                    target=target_name, catalog=catalog,
                    radius_arcsec=radius_arcsec
                )
            elif ra is not None and dec is not None:
                df = self.irsa_client.search_by_position(
                    ra=ra, dec=dec,
                    radius_arcsec=radius_arcsec, catalog=catalog
                )
            else:
                return {"success": False, "error": "Provide target_name or (ra, dec) coordinates."}

            if df.empty:
                note = f"No IRSA sources found"
                if target_name:
                    note += f" for '{target_name}'"
                note += f" in catalog '{catalog}'"
                self.last_run_result = {"type": "data", "data": df, "source": "IRSA", "tool_name": "search_irsa"}
                return {"success": True, "total_results": 0, "note": note}

            filter_label = f"IRSA › {catalog.upper()}"
            if target_name:
                filter_label += f" › {target_name}"

            self.last_search_results = df
            self.last_run_result = {
                "type": "data", "data": df,
                "source": "IRSA", "filter_label": filter_label,
                "tool_name": "search_irsa"
            }

            return {
                "success": True,
                "total_results": len(df),
                "catalog": catalog,
                "columns": list(df.columns[:15]),  # First 15 columns for LLM context
                "note": (
                    f"Found {len(df)} sources in IRSA {catalog.upper()} catalog. "
                    f"Full data shown in UI table. Do NOT render a table — the UI already displays one."
                )
            }
        except Exception as e:
            return {"success": False, "error": f"IRSA search failed: {str(e)}"}

    # ── Sky Survey Image Handler ───────────────────────────────────────

    def _get_sky_image(self, target_name: Optional[str] = None,
                       survey: str = "dss2",
                       radius_arcmin: float = 5.0,
                       ra: Optional[float] = None,
                       dec: Optional[float] = None) -> Dict[str, Any]:
        """Fetch a sky survey cutout image (DSS2, 2MASS, SDSS, WISE, etc.)."""
        try:
            result = self.skyview_client.get_image(
                target=target_name, survey=survey,
                ra=ra, dec=dec,
                radius_arcmin=radius_arcmin, save=True
            )

            if not result.get("success"):
                return result

            # If we have a preview PNG, set it as the run result for UI display
            if "preview_path" in result:
                self.last_run_result = {
                    "type": "image",
                    "image_path": result["preview_path"],
                    "source": f"SkyView ({result.get('survey', 'DSS2')})",
                    "tool_name": "get_sky_image"
                }

            label = target_name or f"RA={ra:.3f}, Dec={dec:.3f}"
            return {
                "success": True,
                "target": label,
                "survey": result.get("survey"),
                "fits_path": result.get("fits_path", ""),
                "preview_path": result.get("preview_path", ""),
                "image_shape": result.get("image_shape"),
                "note": (
                    f"Fetched {result.get('survey')} image for {label}. "
                    f"FITS saved to: {result.get('fits_path', 'N/A')}. "
                    f"Preview shown in UI."
                )
            }
        except Exception as e:
            return {"success": False, "error": f"Sky image fetch failed: {str(e)}"}

    # ── MAST Data Download Handler ─────────────────────────────────────

    def _download_mast_data(self, product_type: str = "SCIENCE",
                             extension: str = "fits",
                             max_files: int = 10) -> Dict[str, Any]:
        """Download FITS files from MAST for the last search results."""
        try:
            if self.last_search_results is None or self.last_search_results.empty:
                return {"success": False, "error": "No search results. Run search_mast first."}

            result = self.mast_client.download_products(
                observations=self.last_search_results,
                productType=product_type,
                extension=extension,
                max_files=max_files
            )

            return result
        except Exception as e:
            return {"success": False, "error": f"MAST download failed: {str(e)}"}

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

    def _merged_plot_alma_results(self, plot_type: str = None,
                                    x_column: str = None, y_column: str = None,
                                    color_by: str = None, title: str = None,
                                    dark_mode: bool = False) -> Dict[str, Any]:
        """Unified plot handler — supports quick overview and publication scatter modes."""
        try:
            if not hasattr(self, 'last_search_results') or self.last_search_results is None or self.last_search_results.empty:
                return {"success": False, "error": "No results available to plot. Please run a search first."}

            # Mode 1: Quick overview plot (sky, frequency, overview)
            if plot_type:
                image_bytes = self.search_service.plot_alma_results(self.last_search_results, plot_type)
                if image_bytes:
                    self.last_run_result = {
                        "type": "image",
                        "image_bytes": image_bytes,
                        "caption": f"ALMA {plot_type.capitalize()} Plot"
                    }
                    return {"success": True, "message": f"Generated {plot_type} plot successfully"}
                return {"success": False, "error": "Plot generation returned empty"}

            # Mode 2: Publication-quality scatter plot
            data_records = self.last_search_results.to_dict("records")
            kw = {}
            if x_column: kw["x_column"] = x_column
            if y_column: kw["y_column"] = y_column
            if color_by: kw["color_by"] = color_by
            if title: kw["title"] = title
            if dark_mode: kw["dark_mode"] = dark_mode
            return self.plotting_service.plot_alma_results(data_records=data_records, **kw)

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

    # ── Phase 1 Tool Handlers: Astronomy Calculators ───────────────────

    def _fit_spectral_line(self, url: str, ra_deg: float = None, dec_deg: float = None,
                           x_pixel: int = None, y_pixel: int = None,
                           title: str = "") -> Dict[str, Any]:
        """Extract spectrum and fit a Gaussian line profile."""
        try:
            from services.fits_service import fit_spectral_line
            result = fit_spectral_line(
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

    def _calculate_redshift(self, z: float) -> Dict[str, Any]:
        """Compute cosmological quantities for a given redshift."""
        try:
            return calculate_redshift(z)
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _convert_coordinates(self, ra: float = None, dec: float = None,
                             l: float = None, b: float = None,
                             input_frame: str = "icrs",
                             output_frame: str = "galactic") -> Dict[str, Any]:
        """Convert coordinates between frames."""
        try:
            return convert_coordinates(
                ra=ra, dec=dec, l=l, b=b,
                input_frame=input_frame, output_frame=output_frame,
            )
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _calculate_beam(self, frequency_ghz: float,
                        max_baseline_m: float = None,
                        array_config: str = None) -> Dict[str, Any]:
        """Calculate synthesized beam size."""
        try:
            return calculate_beam(
                frequency_ghz=frequency_ghz,
                max_baseline_m=max_baseline_m,
                array_config=array_config,
            )
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _calculate_alma_sensitivity(self, band: int, bandwidth_ghz: float = 7.5,
                                    t_integration_s: float = 60.0,
                                    n_antennas: int = None,
                                    n_polarizations: int = 2,
                                    channel_width_khz: float = None,
                                    pwv_mm: float = 1.0) -> Dict[str, Any]:
        """Estimate ALMA sensitivity using the radiometer equation."""
        try:
            return calculate_alma_sensitivity(
                band=band, bandwidth_ghz=bandwidth_ghz,
                t_integration_s=t_integration_s, n_antennas=n_antennas,
                n_polarizations=n_polarizations,
                channel_width_khz=channel_width_khz, pwv_mm=pwv_mm,
            )
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _generate_finding_chart(self, target: str = None, ra: float = None,
                                dec: float = None, survey: str = "DSS2 Red",
                                fov_arcmin: float = 5.0,
                                title: str = None) -> Dict[str, Any]:
        """Generate a finding chart with WCS, crosshair, and compass."""
        try:
            result = self.skyview_client.generate_finding_chart(
                target=target, ra=ra, dec=dec,
                survey=survey, fov_arcmin=fov_arcmin, title=title,
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
        """Search NASA ADS for papers using intelligent query translation.
        
        The user's natural-language question is passed through an LLM-backed
        ADSQueryBuilder that translates it into optimal ADS syntax using
        keyword:, bibgroup:, object:, trending(), useful(), similar(), etc.
        Falls back to direct query if the builder fails.
        
        After ADS returns results, papers are silently enriched with OpenAlex
        data: FWCI, citation percentile, top-1% flag, funders, and OA PDF URLs.
        """
        try:
            if not self.ads_client:
                return {"success": False, "error": "NASA ADS Client not initialized (check API Key)"}

            # Use the smart NL→ADS query builder for rich query translation
            result = self.ads_client.search_natural_language(
                question=query,
                max_results=max_results,
                sort=sort,
            )
            
            papers_list = result.get("papers", [])
            ads_query = result.get("query", query)

            # ── Silent OpenAlex enrichment ────────────────────────────
            # Batch-enrich papers with funding data, FWCI scores, citation
            # percentiles, and OA PDF URLs that ADS doesn't provide.
            # Failures are silently swallowed — enrichment is best-effort.
            try:
                oalex = self.openalex_client
                dois = [p.get("doi", "") for p in papers_list if p.get("doi")]
                if dois and oalex:
                    enrichments = oalex.enrich_batch_dois(dois)
                    if enrichments:
                        _enriched_count = 0
                        for paper in papers_list:
                            doi = paper.get("doi", "")
                            if doi and doi in enrichments:
                                e = enrichments[doi]
                                paper["fwci"] = e.get("fwci")
                                paper["citation_percentile"] = e.get("citation_percentile")
                                paper["is_top_1_percent"] = e.get("is_top_1_percent", False)
                                paper["is_top_10_percent"] = e.get("is_top_10_percent", False)
                                paper["funders"] = e.get("funders", [])
                                paper["oa_pdf_url"] = e.get("oa_pdf_url", "")
                                paper["openalex_topics"] = e.get("topics", [])
                                _enriched_count += 1
                        print(f"[OpenAlex] Enriched {_enriched_count}/{len(papers_list)} papers")
            except Exception as _enrich_err:
                print(f"[OpenAlex] Enrichment failed (non-fatal): {_enrich_err}")

            # Store result for the UI backend to pick up
            self.last_run_result = {
                "type": "papers",
                "papers": papers_list,
                "source": f"ADS: {ads_query}",
            }
            return {
                "success": True,
                "count": len(papers_list),
                "ads_query": ads_query,
                "papers": papers_list,
                "top_title": papers_list[0]["title"] if papers_list else "No results",
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    # ------------------------------------------------------------------
    # OpenAlex tool handlers
    # ------------------------------------------------------------------

    @log_tool
    def _lookup_researcher(self, query: str, max_results: int = 3) -> Dict[str, Any]:
        """Look up a researcher by name or ORCID via OpenAlex."""
        try:
            oalex = self.openalex_client

            # Check if the query looks like an ORCID
            clean = query.strip().replace("https://orcid.org/", "")
            is_orcid = (
                clean.replace("-", "").isdigit() and len(clean) >= 16
            )

            if is_orcid:
                author = oalex.get_author(clean)
                if author:
                    return {
                        "success": True,
                        "count": 1,
                        "researchers": [author],
                    }
                return {"success": False, "error": f"No author found for ORCID {clean}"}

            # Name search
            authors = oalex.search_authors(query, max_results=max_results)
            if not authors:
                return {"success": False, "error": f"No researchers found matching '{query}'"}

            return {
                "success": True,
                "count": len(authors),
                "researchers": authors,
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    @log_tool
    def _get_research_trends(
        self, query: str, year_from: int = 2015, year_to: int = 2026
    ) -> Dict[str, Any]:
        """Get bibliometric trends and funding landscape for a topic via OpenAlex."""
        try:
            oalex = self.openalex_client

            # Publication trends
            trends = oalex.get_topic_trends(
                query, year_from=year_from, year_to=year_to,
            )

            # Funding landscape (top funded works)
            funded = oalex.get_funding_landscape(query, max_results=15)

            return {
                "success": True,
                "trends": trends,
                "funded_works_count": len(funded),
                "top_funded_works": funded[:8],
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    @log_tool
    def _evaluate_consensus(self, question: str, max_papers: int = 20) -> Dict[str, Any]:
        """Search for papers on a scientific question and evaluate the field's consensus.
        
        Searches ADS, reads all abstracts, and produces a structured analysis of
        whether the field agrees or disagrees, citing specific papers and explaining
        the reasoning behind different positions.
        """
        try:
            if not self.ads_client:
                return {"success": False, "error": "NASA ADS Client not initialized"}

            # Step 1: Search for highly-cited papers on the topic for authoritative consensus
            result = self.ads_client.search_natural_language(
                question=question,
                max_results=max_papers,
                sort="citation_count desc",
            )
            papers_list = result.get("papers", [])
            
            if not papers_list:
                return {"success": False, "error": "No papers found for this question."}
            
            # Step 2: Build a structured context with all abstracts + metadata
            paper_contexts = []
            for i, p in enumerate(papers_list, 1):
                ctx = (
                    f"[{i}] {p.get('title', 'Unknown')} "
                    f"({p.get('authors', 'Unknown')}, {p.get('year', '?')})\n"
                    f"    Journal: {p.get('journal', 'Unknown')} | "
                    f"Citations: {p.get('citations', 0)} | "
                    f"Bibcode: {p.get('bibcode', '')}\n"
                    f"    Abstract: {p.get('abstract', 'No abstract')}\n"
                )
                paper_contexts.append(ctx)
            
            all_papers_text = "\n".join(paper_contexts)
            
            # Step 3: Ask the LLM to evaluate consensus with detailed citations
            consensus_prompt = f"""\
You are an expert scientific literature analyst. You have been given {len(papers_list)} \
peer-reviewed papers retrieved from NASA ADS on the following question:

QUESTION: "{question}"

YOUR TASK: Analyze all the abstracts below and produce a detailed CONSENSUS EVALUATION.

PAPERS:
{all_papers_text}

PRODUCE YOUR ANALYSIS IN THIS EXACT FORMAT:

## 📊 Field Consensus: [Strong Agreement / Moderate Agreement / Divided / Strong Disagreement]
**Confidence:** [High / Medium / Low] (based on {len(papers_list)} papers, weighted by citation count)
**Papers Analyzed:** {len(papers_list)}

### Majority Position
State the dominant view clearly in 2-3 sentences. Cite the specific papers that support it using their [number] references.

Example: "The majority of the literature ([1], [3], [5], [7], [8], [12]) concludes that..."

### Dissenting/Alternative Views  
If papers disagree, group them by their position. For EACH dissenting view:
- State the alternative conclusion
- List which papers support it (by [number])
- Explain WHY they reach a different conclusion (different methodology? different data? different assumptions? different telescope?)
- Cite the specific evidence or reasoning from their abstracts

If there are no dissenting views, state "No significant dissent found in the analyzed literature."

### Key Evidence Summary
Bullet-point the strongest pieces of evidence from both sides, citing specific papers:
- "[1] found that..." 
- "[5] measured X and concluded..."
- "[9] used ALMA data showing..."

### Evolution Over Time
If the consensus has shifted over time (older papers say X, newer papers say Y), note this trend.

### Open Questions
What does the literature identify as unresolved? What would settle the debate?

IMPORTANT RULES:
- ALWAYS cite papers by their [number] reference
- Include the author name and year when first citing a paper
- Be specific about evidence — don't just say "some papers agree"
- Weight highly-cited papers more heavily in your assessment
- If a question is too narrow or the papers don't directly address it, say so honestly
"""

            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {"role": "system", "content": "You are a meticulous scientific literature analyst who produces rigorous, well-cited consensus evaluations."},
                    {"role": "user", "content": consensus_prompt}
                ],
                temperature=0.1,
                max_tokens=4000,
            )
            
            analysis = response.choices[0].message.content.strip()
            
            # Store for UI rendering
            self.last_run_result = {
                "type": "consensus",
                "text": analysis,
                "papers": papers_list,
                "question": question,
                "source": f"Consensus Analysis: {len(papers_list)} papers",
            }
            
            return {
                "success": True,
                "question": question,
                "papers_analyzed": len(papers_list),
                "analysis": analysis,
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _extract_paper_details(self, identifier: str, query: str) -> Dict[str, Any]:
        """Download a paper and extract specific details via QA."""
        try:
            arxiv_id = identifier.strip()
            if '.' not in arxiv_id or len(arxiv_id) > 20:
                if self.ads_client:
                    try:
                        details = self.ads_client.get_paper_details(arxiv_id)
                        if isinstance(details, dict) and details.get("arxiv_id"):
                            arxiv_id = details["arxiv_id"]
                        elif isinstance(details, dict) and details.get("doi"):
                            return {"success": False, "error": f"Paper has DOI ({details['doi']}) but no arXiv ID. PDF download requires an open access arXiv ID."}
                    except Exception:
                        pass

            pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
            result = self.pdf_service.query_paper_pdf(pdf_url, query)

            if not result.get("success"):
                return {"success": False, "error": f"Could not extract details: {result.get('error')}", "identifier": identifier}

            self.last_run_result = {"type": "text", "text": result["answer"], "source": f"Paper Extractor: {identifier}"}
            return {"success": True, "identifier": identifier, "extracted_answer": result["answer"]}
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

    def _sandbox_tool_bridge(self, tool_name: str, kwargs: dict) -> Any:
        """
        Bridge from sandbox call_tool(name, **kwargs) to registered Quasar tools.

        Any of the 27+ registered Quasar tools can be called from inside the
        sandbox Python environment:
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

    def _conductor_tool_executor(self, task_description: str, dep_context: str = "", subtask_model: str = "") -> str:
        """
        Execute a sub-task using a mini Responses API call with full tool access.

        This is the callback passed to Conductor so each DAG node can use
        all 28+ registered tools (ALMA search, ADS, Splatalogue, etc.).

        Parameters
        ----------
        subtask_model : str, optional
            The model to use for this subtask, as determined by ModelRouter.
            If empty, falls back to conductor_model (GPT-5.4).
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

            # ── Model selection: use the routed model if provided,
            # otherwise fall back to conductor_model (GPT-5.4)
            if subtask_model:
                model_to_use = subtask_model
            else:
                model_to_use = getattr(self.conductor, 'conductor_model', self.config.model)
            print(f"[CONDUCTOR] Subtask model: {model_to_use} for: {task_description[:80]}")

            response = self.client.responses.create(
                model=model_to_use,
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

        Does NOT trigger on archive search queries — those hit live databases
        (ALMA, CADC, etc.) directly and don't need web search augmentation.
        """
        _q = query.lower()

        # ── Skip web search for archive queries ────────────────────────
        # Archive searches already query live databases — web search adds
        # nothing and just wastes time / clutters the response.
        _archive_keywords = re.search(
            r'\b(?:observation|observations|data|archive|band\s*\d|'
            r'search_by|search_cadc|member_ous|mous|project_code|fits)\b',
            _q,
        )
        _archive_verbs = re.search(
            r'\b(?:find|search|show|get|list|query|look\s*up)\b.*'
            r'\b(?:observation|observations|data|archive)\b',
            _q,
        )
        _telescope_query = re.search(
            r'\b(?:alma|vla|vlba|gbt|jwst|hst|gemini|jcmt|cfht|chandra|xmm)\b.*'
            r'\b(?:observation|observations|data|of)\b',
            _q,
        )
        if _archive_verbs or _telescope_query:
            return None  # skip web search for archive queries

        # ── Skip web search for paper/literature queries ───────────────
        # Papers come from NASA ADS (search_papers tool), not web search.
        # "recent papers on X" should NOT trigger web search just because
        # of the word "recent".
        _paper_query = re.search(
            r'\b(?:papers?|publications?|articles?|literature|studies)\b',
            _q,
        )
        if _paper_query:
            return None  # skip web search for paper queries

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

        # 3. Selective freshness keywords — only high-confidence temporal phrases
        #    that strongly imply the user wants current-year information.
        #    Avoids broad terms like "latest", "current", "recent" which cause
        #    false positives on nearly every query.
        _freshness_pattern = re.search(
            r'\b(?:this\s+year|this\s+month|today|right\s+now|happening\s+now)\b',
            _q,
        )
        if _freshness_pattern:
            return query

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
        conversation_id: Optional[str] = None,
        plan_feedback_queue=None,
    ) -> str:
        # Assign a unique conversation_id if none provided (isolates anonymous
        # concurrent requests so they never share OpenAI response state).
        if not conversation_id:
            conversation_id = f"anon_{uuid.uuid4().hex[:12]}"

        # Periodic cleanup to prevent unbounded memory growth
        self._cleanup_conv_states()

        # Reset per-request thread-local state
        self._accumulated_run_results = []
        self.last_run_result = None
        self.last_search_results = None
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

        # 0. Session Management -- smart context handling + session memory
        self._prune_session_if_needed(query, user_id)

        # Helper: emit structured events (web_sources, etc.) through the SSE queue.
        # Uses the __event__ prefix that main.py already handles.
        _json = json  # local alias for use in closures
        def on_event(evt: dict):
            print(f"[DEBUG on_event] type={evt.get('type')}, images={len(evt.get('images', []))}, sources={len(evt.get('sources', []))}")
            if on_status:
                on_status(f"__event__{_json.dumps(evt)}", evt.get('type', 'status'))

        # Emit connecting step
        if on_status:
            on_status("Connecting to QUASAR engine", "running")
            on_status("Connecting to QUASAR engine", "completed")

        # 0a. Knowledge-cutoff detection — launch parallel web search
        _web_search_query = self._detect_beyond_cutoff(_user_query)
        _web_result_holder = {}   # will be filled by background thread
        _web_search_reason = None
        _email_result_holder = {}  # dedicated email search for researcher queries
        _email_thread = None

        if _web_search_query:
            _web_search_reason = "cutoff"
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
        #    SKIP for archive data-fetch queries (e.g. "find ALMA observations of M87")
        #    because those hit the live archive, not documentation.
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

        # Skip RAG for pure archive data-fetch queries — the user wants data
        # from the live archive, not ALMA technical documentation.
        _is_archive_fetch = bool(re.search(
            r'\b(?:find|search|show|get|list|query|look\s*up|fetch)\b.*'
            r'\b(?:observation|observations|data)\b',
            _query_lower,
        )) or bool(re.search(
            r'\b(?:alma|vla|vlba|gbt|jwst|hst|gemini|jcmt|cfht|chandra|xmm)\b.*'
            r'\b(?:observation|observations|data)\s+(?:of|for|from)\b',
            _query_lower,
        )) or bool(re.search(
            r'\b(?:what|any|available)\b.*'
            r'\b(?:observation|observations|data)\b.*\b(?:of|for|from|available)\b',
            _query_lower,
        ))
        if _is_archive_fetch:
            _should_rag = False

        # Skip RAG for paper/literature queries — these go through NASA ADS
        # (search_papers tool), NOT ALMA documentation. Words like "disk",
        # "spectral", "radio" in "recent papers on protoplanetary disks" would
        # false-positive on _rag_keywords and waste time searching manuals.
        #
        # Simple rule: if ANY paper/literature word appears → skip RAG.
        # Exception: "summarize this paper" / "explain this article" with an
        # attachment is a DOCUMENT query, not a search — handled separately.
        _has_paper_word = bool(re.search(
            r'\b(?:papers?|publications?|articles?|literature|studies|preprints?)\b',
            _query_lower,
        ))
        _is_document_analysis = bool(re.search(
            r'\b(?:summarize|summarise|explain|describe|extract|read|analyze|analyse|'
            r'review|translate|what does|tell me about)\b.*'
            r'\b(?:this|the|attached|uploaded)\b.*'
            r'\b(?:paper|article|document|pdf)\b',
            _query_lower,
        )) or bool(attachments)

        # Determine if this is a paper search query.
        # When paper keywords are detected but the intent is ambiguous,
        # use a fast LLM call to verify before committing to search_papers.
        _is_paper_query = False
        if _has_paper_word and not _is_document_analysis:
            # Fast LLM intent verification (~200ms with gpt-4o-mini)
            try:
                from openai import OpenAI as _OAI
                _mini = _OAI()
                _intent_resp = _mini.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{
                        "role": "user",
                        "content": (
                            f"Classify this astronomy query into exactly one category.\n\n"
                            f"Query: \"{_user_query}\"\n\n"
                            f"PAPERS = The user wants to FIND scientific papers, publications, or literature from NASA ADS or arXiv. "
                            f"Example: 'Find papers about protoplanetary disks', 'Recent publications on galaxy mergers'.\n"
                            f"KNOWLEDGE = The user wants general information, how-to guides, ALMA policies, procedures, or technical details. "
                            f"Words like 'proposal', 'archival', 'access', 'deadline', 'review process' in context of ALMA operations are KNOWLEDGE, not PAPERS.\n"
                            f"Example: 'What are the proposal submission deadlines?', 'How do I access archival data?'\n\n"
                            f"Reply with ONLY one word: PAPERS or KNOWLEDGE"
                        ),
                    }],
                    temperature=0,
                    max_tokens=5,
                )
                _intent = _intent_resp.choices[0].message.content.strip().upper()
                _is_paper_query = "PAPERS" in _intent
                print(f"[INTENT] Query: '{_user_query[:60]}...' → {_intent} (paper_query={_is_paper_query})")
            except Exception as e:
                # Fallback to keyword-based detection if LLM call fails
                print(f"[INTENT] LLM verification failed, falling back to keyword: {e}")
                _is_paper_query = True  # Preserve original behavior on failure

        if _is_paper_query:
            _should_rag = False

        # Detect OpenAlex-targeted queries — researcher lookups, funding,
        # metrics, popularity, and trend questions. These ALWAYS trigger the
        # OpenAlex tool (even if RAG also runs — the two are additive).
        # IMPORTANT: use _user_query (bare question) not _query_lower
        # (enriched query) because the enrichment wrapper may contain ALMA
        # terms that would falsely trigger the exclusion regex.
        _bare_lower = _user_query.lower()
        _is_researcher_query = bool(re.search(
            r'\b(?:who is|who\'s|tell me about|look up|profile of|'
            r'where does .+ work|what does .+ (?:research|study|work on)|'
            r'what (?:topics?|areas?|fields?) does .+ (?:research|study|work)|'
            r'how many papers has .+ (?:published|written|authored)|'
            r'which institution|h-index|orcid|'
            r'.+\'s research|.+\'s h.index|.+\'s publications?)\b',
            _bare_lower,
        )) and not bool(re.search(
            # Exclude ALMA instrument/process questions (only in the BARE query)
            r'\b(?:correlator|band\s?\d|pipeline|calibrat|antenna|baseline)\b',
            _bare_lower,
        ))
        _is_trend_query = bool(re.search(
            r'\b(?:interest in .+ growing|publication trend|research trend|'
            r'how much research|how many papers on|papers per year|'
            r'publication volume|publication rate|research output|'
            r'is .+ growing|field growth|who funds|funding landscape|'
            r'bibliometric|citation metrics?|impact factor)\b',
            _bare_lower,
        ))
        _is_openalex_query = _is_researcher_query or _is_trend_query
        if _is_openalex_query:
            # Do NOT set _should_rag = False — OpenAlex is additive, not
            # exclusive. RAG may still provide useful ALMA-related context.
            # The OpenAlex directive (injected later) ensures the tool is called
            # regardless of whether RAG context is present.
            print(f"[ROUTE] OpenAlex query detected (researcher={_is_researcher_query}, "
                  f"trend={_is_trend_query}): '{_user_query[:60]}'")

            # For researcher queries, run TWO parallel web searches:
            #   1. General context search (bio, news, awards, personal page)
            #   2. Targeted email/contact search (faculty page, directory)
            # Both run concurrently with zero extra latency.
            if _is_researcher_query and _web_thread is None and os.getenv("TAVILY_API_KEY", ""):
                _web_search_reason = "researcher_supplement"
                if on_status:
                    on_status("Searching the web for researcher profile", "running")

                # Extract the person's name from the query for targeted searches
                _person_name = re.sub(
                    r'\b(?:who is|who\'s|tell me about|look up|profile of)\b',
                    '', _user_query, flags=re.IGNORECASE,
                ).strip().strip('?').strip()

                # Thread 1: General context search
                def _bg_web_search_researcher():
                    try:
                        _web_result_holder["data"] = self._tavily_web_search(
                            query=_user_query,
                            max_results=10,
                            search_depth="basic",
                        )
                    except Exception as _e:
                        _web_result_holder["error"] = str(_e)

                _web_thread = threading.Thread(target=_bg_web_search_researcher, daemon=True)
                _web_thread.start()

                # Thread 2: Targeted email/contact search
                def _bg_email_search():
                    try:
                        _email_result_holder["data"] = self._tavily_web_search(
                            query=f"{_person_name} email contact professor astronomy",
                            max_results=3,
                            search_depth="basic",
                        )
                    except Exception as _e:
                        _email_result_holder["error"] = str(_e)

                _email_thread = threading.Thread(target=_bg_email_search, daemon=True)
                _email_thread.start()

        if _should_rag:
            try:
                if on_status:
                    on_status("Searching ALMA Manuals & Documentation", "running")

                # ── Year-aware filtering ──────────────────────────────
                # Auto-detect year references in the query so we prefer
                # the most relevant version of the documentation.
                _rag_min_year = None
                _rag_year_matches = re.findall(r'\b(20[1-3]\d)\b', _user_query)
                if _rag_year_matches:
                    _rag_min_year = max(int(y) for y in _rag_year_matches)

                # Also detect "Cycle N" → year mapping
                _cycle_year_map = {
                    "1": 2013, "2": 2014, "3": 2015, "4": 2016,
                    "5": 2017, "6": 2018, "7": 2019, "8": 2020,
                    "9": 2021, "10": 2022, "11": 2023, "12": 2024,
                    "13": 2025, "14": 2026,
                }
                _cycle_match = re.search(r'Cycle\s+(\d{1,2})', _user_query, re.IGNORECASE)
                if _cycle_match and _cycle_match.group(1) in _cycle_year_map:
                    _cycle_yr = _cycle_year_map[_cycle_match.group(1)]
                    if _rag_min_year is None or _cycle_yr > _rag_min_year:
                        _rag_min_year = _cycle_yr

                docs = self.rag_service.search(
                    query,
                    min_year=_rag_min_year,
                )
                if docs:
                    context_pieces = []
                    for chunk_idx, d in enumerate(docs[:3], 1):
                        # Prefer source_file (clean filename) over source (full path)
                        src = d.metadata.get("source_file", d.metadata.get("source", "Unknown"))
                        if "/" in src or "\\" in src:
                            src = src.replace("\\", "/").split("/")[-1]
                        page = d.metadata.get("page", "?")
                        doc_year = d.metadata.get("doc_year", "?")
                        doc_month = d.metadata.get("doc_month")
                        score = d.metadata.get("_score", d.metadata.get("_semantic_score", "?"))
                        if isinstance(score, float):
                            score = round(score, 2)
                        category = d.metadata.get("doc_category", "")

                        # Build human-readable date string
                        month_names = {
                            1: "January", 2: "February", 3: "March",
                            4: "April", 5: "May", 6: "June",
                            7: "July", 8: "August", 9: "September",
                            10: "October", 11: "November", 12: "December",
                        }
                        if doc_month and isinstance(doc_month, int) and doc_month in month_names:
                            date_str = f"{month_names[doc_month]} {doc_year}"
                        elif doc_year and doc_year != "?":
                            date_str = str(doc_year)
                        else:
                            date_str = "N/A"

                        # Build page string
                        if page and page != "?" and str(page) != "?":
                            page_str = str(page)
                        else:
                            page_str = "N/A"

                        # Pre-build the EXACT citation the LLM must copy verbatim
                        cite_tag = f"[Source: {src}, Page {page_str}, Date: {date_str}, Relevance: {score}]"

                        # Build structured chunk with explicit CITE_AS tag
                        context_pieces.append(
                            f"--- DOCUMENT CHUNK {chunk_idx} ---\n"
                            f"CITE_AS: {cite_tag}\n"
                            f"{d.page_content}"
                        )
                    rag_context = (
                        "\n\n📚 DOCUMENTATION CONTEXT (from ALMA Technical Documentation):\n"
                        + "\n---\n".join(context_pieces)
                    )

                    # Launch a parallel web search to supplement RAG with fresh data.
                    # Only if a web thread isn't already running (from cutoff detection)
                    # AND the query actually benefits from web search.
                    #
                    # SKIP web search for:
                    # - Attachment/document queries (summarize PDF, explain this doc)
                    # - Conversational/follow-up queries (thanks, yes, explain more)
                    # - Data analysis queries (plot, analyze, compare these results)
                    # - General knowledge the LLM can answer from training data
                    #
                    # ALLOW web search for:
                    # - Queries about current observatory status/schedules/deadlines
                    # - Policy/proposal questions that change over time
                    # - Queries explicitly asking for latest/recent/current info
                    _needs_web_supplement = False
                    if _web_thread is None and os.getenv("TAVILY_API_KEY", ""):
                        _uq = _user_query.lower()

                        # ── Blocklist: never web-search for these patterns ──
                        _is_attachment_query = bool(re.search(
                            r'\b(?:summarize|summarise|summary|explain|describe|extract|'
                            r'read|analyze|analyse|parse|review|translate|what does|'
                            r'tell me about)\b.*'
                            r'\b(?:pdf|document|file|paper|thesis|article|attachment|'
                            r'uploaded|attached|this)\b',
                            _uq,
                        )) or bool(re.search(
                            r'\b(?:pdf|document|file|paper|thesis|attachment)\b.*'
                            r'\b(?:summarize|summarise|summary|explain|about|says?|'
                            r'contain|content|mean)\b',
                            _uq,
                        ))
                        _has_attachments = bool(attachments)

                        _is_conversational = bool(re.search(
                            r'^(?:thank|thanks|yes|no|ok|okay|sure|got it|'
                            r'explain more|continue|go on|what do you mean|'
                            r'can you elaborate|tell me more|great)\b',
                            _uq.strip(),
                        ))

                        # ── Allowlist: web-search IS useful here ──
                        _wants_current_info = bool(re.search(
                            r'\b(?:current|latest|recent|upcoming|deadline|'
                            r'schedule|status|call for|cfp|cycle \d|'
                            r'when is|when does|how to apply|'
                            r'policy|policies|regulation)\b',
                            _uq,
                        ))

                        # Paper/literature queries should NEVER trigger web
                        # search — papers come from NASA ADS, not the web.
                        # "recent papers on X" matches _wants_current_info
                        # because of "recent", but that's a false positive.
                        _is_paper_query_web = bool(re.search(
                            r'\b(?:papers?|publications?|articles?|literature|studies)\b',
                            _uq,
                        ))

                        if _is_attachment_query or _has_attachments or _is_conversational or _is_paper_query_web:
                            _needs_web_supplement = False
                        else:
                            # Always supplement RAG with web search for fresh context
                            _needs_web_supplement = True

                    if _needs_web_supplement:
                        _web_search_reason = "rag_supplement"
                        if on_status:
                            on_status("Searching the web for updated information", "running")

                        def _bg_web_search_rag():
                            try:
                                _web_result_holder["data"] = self._tavily_web_search(
                                    query=_user_query,
                                    max_results=10,
                                    search_depth="basic",
                                )
                            except Exception as _e:
                                _web_result_holder["error"] = str(_e)

                        _web_thread = threading.Thread(target=_bg_web_search_rag, daemon=True)
                        _web_thread.start()

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
                "\n\nIMPORTANT CITATION & STRUCTURE RULES:\n"
                "1. INLINE CITATIONS: Place each citation IMMEDIATELY after the sentence that uses the information. "
                "NEVER create a 'References:' or 'Sources:' section at the bottom. "
                "Each document chunk above has a CITE_AS tag — you MUST copy that EXACT string verbatim as your citation. "
                "Do NOT modify, rephrase, or invent citation fields. The CITE_AS tag already contains the correct "
                "Page number, Date, and Relevance score. Example: if CITE_AS says "
                "'[Source: alma-proposers-guide-cycle13.pdf, Page 36, Date: February 2026, Relevance: 0.88]', "
                "write EXACTLY that string after the sentence that uses info from that chunk. "
                "NEVER write 'Page unknown', 'Date: unknown', or make up your own Relevance scores.\n"
                "2. After presenting the documentation-based answer, add a disclaimer line: "
                "'*📚 The above is sourced from ALMA Documentation, tutorials, and community notebooks and may not reflect the very latest policies.*'\n"
                "3. WEB SECTION: If web search results are available in your context, you MUST present them under "
                "'🌐 Updated Information from the Web:' with a detailed paragraph. NEVER say 'No additional updates were found'. "
                "Always extract and present the actual content from the web results, even if it overlaps with the documentation."
            )
        full_input = f"{memory_context}{rag_context}{citation_note}\n\nUser: {query}"

        # 4b. Paper query safety net — even if RAG context leaked in above,
        #     force the LLM to call search_papers (ADS) for paper queries.
        #     Reuse the _has_paper_word / _is_paper_query flags from step 1.
        # OpenAlex queries take priority over paper queries when both match.
        # E.g. "How many papers has Paola Caselli published?" matches both
        # _is_paper_query (contains 'papers') and _is_researcher_query,
        # but should route to lookup_researcher, not search_papers.
        if _is_openalex_query:
            pass  # handled below
        elif _is_paper_query:
            paper_directive = (
                "\n\nMANDATORY INSTRUCTION: The user is asking for papers/publications. "
                "You MUST call the `search_papers` tool to query NASA ADS. "
                "Do NOT answer from memory, web search results, or documentation context. "
                "Do NOT use `web_search`. Call `search_papers` NOW."
            )
            full_input += paper_directive
        if _is_openalex_query:
            # Force the LLM to call the OpenAlex tool
            if _is_researcher_query:
                openalex_directive = (
                    "\n\nMANDATORY INSTRUCTION: The user is asking about a researcher/scientist. "
                    "You MUST call the `lookup_researcher` tool with their name or ORCID to get "
                    "their academic profile (institution, h-index, ORCID, research topics, publication history). "
                    "Present the OpenAlex profile data as the core of your answer. "
                    "If web search results are also available, combine them with the OpenAlex profile "
                    "to provide additional context (e.g. recent news, awards, personal webpage). "
                    "Do NOT skip calling `lookup_researcher` — always call it first."
                )

                # Join the email search thread and inject contact info into
                # the prompt so the LLM can include it in the profile header.
                if _email_thread is not None:
                    _email_thread.join(timeout=8)
                    _email_data = _email_result_holder.get("data")
                    if _email_data and _email_data.get("success"):
                        _email_snippets = []
                        for _r in _email_data.get("results", [])[:3]:
                            _snippet = _r.get("content", "").strip()
                            if _snippet:
                                _email_snippets.append(_snippet)
                        if _email_snippets:
                            _email_context = "\n".join(_email_snippets)
                            openalex_directive += (
                                "\n\nCONTACT INFORMATION FROM WEB (include email "
                                "and webpage in the profile header if found):\n"
                                + _email_context
                            )
                            print(f"[EMAIL SEARCH] Injected {len(_email_snippets)} contact snippets into prompt")
                    elif _email_result_holder.get("error"):
                        print(f"[EMAIL SEARCH] Failed: {_email_result_holder['error']}")
            else:
                openalex_directive = (
                    "\n\nMANDATORY INSTRUCTION: The user is asking about research trends or funding. "
                    "You MUST call the `get_research_trends` tool with the topic query. "
                    "Do NOT answer from memory or web search results. "
                    "Do NOT use `web_search`. Call `get_research_trends` NOW."
                )
            full_input += openalex_directive
        elif rag_context:
            # Knowledge query with RAG context — explicitly prevent search_papers
            knowledge_directive = (
                "\n\n[SYSTEM NOTE: This is a KNOWLEDGE query answered from ALMA documentation. "
                "Do NOT call `search_papers` — the user is asking about ALMA procedures, policies, "
                "or technical details, NOT requesting scientific papers or publications. "
                "Answer using the DOCUMENTATION CONTEXT provided above.]"
            )
            full_input += knowledge_directive

        # 4c. Prevent duplicate web searches — when the parallel cutoff search
        #     is already running, tell the LLM not to call web_search itself.
        #     This eliminates redundant Tavily calls and speeds up response time.
        if _web_search_query is not None:
            full_input += (
                "\n\n[SYSTEM NOTE: A web search is already running in parallel for this query. "
                "Do NOT call the `web_search` tool yourself — the results will be appended "
                "automatically after your response. Focus on answering from your knowledge.]"
            )

        # 4a. Conductor check — delegate complex queries to DAG orchestration
        #     IMPORTANT: The Conductor receives `_user_query` (the bare user question),
        #     NOT `query` (which may be wrapped with personal RAG context, mem0 memories,
        #     citation instructions, etc.).  The ALMA documentation RAG context is passed
        #     separately via `context=rag_context` so the planner can use it for planning
        #     without it polluting the DAG decomposition or synthesis prompts.
        try:
            complexity = self.complexity_detector.assess(_user_query).score
            if complexity > Conductor.COMPLEXITY_THRESHOLD:
                import asyncio, json as _json
                trace_id = self.query_tracer.new_trace(_user_query, user_id=user_id)

                # Classify complexity tier → controls max subtasks
                _tier_name, _tier_max = Conductor.classify_tier(complexity)

                # ① Mark detection as COMPLETED immediately so the UI shows a ✓
                if on_status:
                    on_status(
                        f"Complex query detected (score={complexity:.2f}, tier={_tier_name}) "
                        f"— activating multi-agent workforce",
                        "completed",
                    )

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
                                    _user_query,
                                    context=rag_context,
                                    max_subtasks=_tier_max,
                                    complexity_tier=_tier_name,
                                    on_status=on_status,
                                    on_token=on_token,
                                    on_event=on_event,
                                    plan_feedback_queue=plan_feedback_queue,
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
                            _web_status_label = (
                                "Searching the web for updated information"
                                if _web_search_reason == "rag_supplement"
                                else "⚡ Time period beyond training knowledge cutoff detected — searching the web in parallel"
                            )
                            on_status(_web_status_label, "completed")
                        web_data = _web_result_holder.get("data")
                        if web_data and web_data.get("success"):
                            # Synthesize a query-relevant summary instead of using
                            # the raw Tavily answer which is often generic.
                            tavily_answer = self._synthesize_web_summary(
                                _user_query, web_data, _web_search_reason or ""
                            )
                            if tavily_answer:
                                web_section = "\n\n---\n\n"
                                if _web_search_reason == "rag_supplement":
                                    web_section += "🌐 **Updated Information from the Web:** "
                                else:
                                    web_section += "🌐 **From the Web:** "
                                web_section += tavily_answer + "\n"
                                conductor_answer += web_section
                                if on_token:
                                    on_token(web_section)
                            print(f"[WEB SEARCH] Appended synthesized web summary ({_web_search_reason}) to conductor response")
                            print(f"[WEB SEARCH] Images from Tavily: {len(web_data.get('images', []))}, Sources: {len(web_data.get('results', []))}")

                            # Emit web_sources event for frontend source cards + image grid
                            if on_event:
                                on_event({
                                    "type": "web_sources",
                                    "sources": web_data.get("results", []),
                                    "images": web_data.get("images", []),
                                    "query": web_data.get("query", ""),
                                })
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
            last_id = self._get_response_id(conversation_id)
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

                # Force tool call on first round for data-fetch queries.
                # This prevents the LLM from answering from conversation
                # memory and ensures a fresh data card is always shown.
                if _round == 0 and _is_archive_fetch:
                    request_kwargs["tool_choice"] = "required"

                try:
                    response_stream = self.client.responses.create(**request_kwargs)
                except Exception as e:
                    if "No tool output found for function call" in str(e) and "previous_response_id" in request_kwargs:
                        # Recover from hanging tool call in a previous interrupted turn
                        print(f"[WARNING] Recovering from hanging tool call state for conv={conversation_id}. Dropping previous_response_id.")
                        del request_kwargs["previous_response_id"]
                        last_id = None
                        self._set_response_id(conversation_id, None)
                        response_stream = self.client.responses.create(**request_kwargs)
                    else:
                        raise e
                
                function_calls = {} # call_id -> dict
                item_id_to_call_id = {}  # item.id -> call_id mapping
                
                for event in response_stream:
                    if event.type == "response.created":
                        last_id = event.response.id
                        self._set_response_id(conversation_id, last_id)
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
                    elif event.type == "response.completed":
                        # Final sweep: reconcile call_ids from the completed response
                        # The streaming events may miss the final call_id assignment.
                        completed_resp = getattr(event, 'response', None)
                        if completed_resp and hasattr(completed_resp, 'output'):
                            for out_item in completed_resp.output:
                                if getattr(out_item, 'type', None) == 'function_call':
                                    final_cid = getattr(out_item, 'call_id', None)
                                    item_id = getattr(out_item, 'id', None)
                                    fn_name = getattr(out_item, 'name', '')
                                    fn_args = getattr(out_item, 'arguments', '')
                                    if final_cid:
                                        # Find the matching entry by item_id or name
                                        old_key = item_id_to_call_id.get(item_id, item_id)
                                        if old_key in function_calls:
                                            function_calls[old_key]["call_id"] = final_cid
                                        elif item_id in function_calls:
                                            function_calls[item_id]["call_id"] = final_cid
                                        elif final_cid not in function_calls:
                                            # Entirely new — create the entry
                                            function_calls[final_cid] = {
                                                "name": fn_name,
                                                "arguments": fn_args,
                                                "call_id": final_cid,
                                                "_item_id": item_id,
                                            }
                
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
                            _acc_len_before = len(self._accumulated_run_results)
                            result = tool.execute(**args)
                            result_str = json.dumps(result, default=str)[:8000]  # increased for multi-step chains
                            _acc_len_after = len(self._accumulated_run_results)

                            # If the tool itself already accumulated results
                            # (e.g. multi-target search appends per-target results),
                            # eagerly emit each new result for parallel data card rendering.
                            if _acc_len_after > _acc_len_before:
                                # Tool accumulated its own results — emit each new one
                                if on_status:
                                    for _new_idx in range(_acc_len_before, _acc_len_after):
                                        _new_rc = self._accumulated_run_results[_new_idx]
                                        if _new_rc.get("type") in ("data", "papers"):
                                            # Send the result directly (not just index) so the
                                            # event-loop thread doesn't read thread-local state.
                                            _payload = json.dumps({"_eager_result": True, "_idx": _new_idx, "_inline": True})
                                            on_status(f"__data_ready__{_payload}", "ready")
                                            # Stash inline data for the SSE handler to pick up
                                            on_status(f"__eager_data__{json.dumps(_new_rc, default=str)}", "data")
                            elif self.last_run_result is not None:
                                # Tool didn't accumulate — add last_run_result ourselves
                                _rc = self.last_run_result.copy()
                                _rc["_result_id"] = id(self.last_run_result)
                                self._accumulated_run_results.append(_rc)
                                if on_status and _rc.get("type") in ("data", "papers"):
                                    _payload = json.dumps({"_eager_result": True, "_idx": len(self._accumulated_run_results) - 1, "_inline": True})
                                    on_status(f"__data_ready__{_payload}", "ready")
                                    on_status(f"__eager_data__{json.dumps(_rc, default=str)}", "data")
                            # Record tool calls for session memory
                            self.session_memory.record_tool_calls(1)

                            # Emit web_sources event for LLM-initiated web searches
                            # so source cards + images always appear in the UI.
                            if tool_name == "web_search" and isinstance(result, dict) and result.get("success"):
                                on_event({
                                    "type": "web_sources",
                                    "sources": result.get("results", []),
                                    "images": result.get("images", []),
                                    "query": result.get("query", ""),
                                })
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
                _web_thread.join(timeout=30)  # wait up to 30s for web results
                # Close off the web status indicator
                if on_status:
                    _web_status_label = (
                        "Searching the web for updated information"
                        if _web_search_reason == "rag_supplement"
                        else "⚡ Time period beyond training knowledge cutoff detected — searching the web in parallel"
                    )
                    on_status(_web_status_label, "completed")
                web_data = _web_result_holder.get("data")
                if web_data and web_data.get("success"):
                    # Synthesize a query-relevant summary instead of using
                    # the raw Tavily answer which is often generic.
                    tavily_answer = self._synthesize_web_summary(
                        _user_query, web_data, _web_search_reason or ""
                    )
                    if tavily_answer:
                        web_section = "\n\n---\n\n"
                        if _web_search_reason == "rag_supplement":
                            web_section += "🌐 **Updated Information from the Web:** "
                        else:
                            web_section += "🌐 **From the Web:** "
                        web_section += tavily_answer + "\n"
                        output_text += web_section
                        if on_token:
                            on_token(web_section)
                    print(f"[WEB SEARCH] Appended synthesized web summary ({_web_search_reason}) to response")
                    print(f"[WEB SEARCH] Images from Tavily: {len(web_data.get('images', []))}, Sources: {len(web_data.get('results', []))}")

                    # Emit web_sources event for frontend source cards + image grid
                    if on_event:
                        on_event({
                            "type": "web_sources",
                            "sources": web_data.get("results", []),
                            "images": web_data.get("images", []),
                            "query": web_data.get("query", ""),
                        })
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
            # If the error is about a hanging tool call, clear the poisoned
            # response ID for THIS conversation so it doesn't keep failing.
            if "No tool output found for function call" in str(e):
                print(f"[WARNING] Clearing poisoned response_id for conv={conversation_id} to break error loop.")
                self._set_response_id(conversation_id, None)
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
    
    def reset_conversation_state(self, conversation_id: Optional[str] = None):
        """Reset conversation state for a specific or all chat sessions."""
        if conversation_id:
            self._set_response_id(conversation_id, None)
        else:
            # Clear ALL conversation states (full reset)
            with self._conv_ids_lock:
                self._conv_response_ids.clear()
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
