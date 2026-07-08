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
import time
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, field
import openai
from openai import OpenAI
from core.llm_client import LLMClient, detect_provider

from core.logger import logger, log_tool
from services.ads_auto_link import build_exact_project_paper_links
from services.citation_verifier import append_citation_warning
from services.evidence_quality import annotate_web_source_evidence, rank_web_sources
from services.content_safety import (
    FILTER_NOTICE,
    is_explicit_query,
    is_safe_web_image,
    is_safe_web_source,
    safe_assistant_text,
    sanitize_web_payload,
)


from core.memory import ConversationMemory
from core.tools import ToolRegistry, Tool
from core.prompts import (
    INTENT_CLASSIFICATION_PROMPT,
    ENTITY_EXTRACTION_PROMPT,
    RESPONSE_GENERATION_PROMPT,
    ALMA_TAP_SCHEMA
)
# NRAO TAP (VLA/VLBA/GBT) routing lives in services/search.py (SearchService.nrao_client)
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
from services.data_product_triage import (
    build_product_row,
    classify_alma_product_request,
    filter_observations_by_band,
    is_fits_product,
    product_rank,
    summarize_project_options,
    unique_values,
)
from services.alma_science_queries import (
    LINE_REST_FREQ_GHZ,
    bandwidth_switching_candidates,
    filter_band as science_filter_band,
    filter_resolution as science_filter_resolution,
    line_names_for_species,
    normalize_target_alias,
    projects_covering_all_lines,
    projects_with_array_combo,
    project_prefix_where,
    redshifted_line_projects,
    select_obscore_query,
    summarize_projects,
)
from services.cross_archive_matcher import (
    PERSEUS_PROTOSTARS,
    alma_bulk_cone_adql,
    attach_nearest_source,
    normalize_source_catalog,
    summarize_cross_archive_matches,
)
from core.prompts.lit_to_code import LIT_TO_CODE_PROMPT
from services.astro_calculators import (
    calculate_redshift, convert_coordinates, calculate_beam,
    calculate_alma_sensitivity,
)
from integrations.datalab_client import DatalabClient
from services import datalab_query_builders, datalab_registry, datalab_sql_policy
from services import datalab_orchestration
from services.datalab_result_store import default_result_store
from services.mmu_hats import (
    MMU_HATS_CATALOGS,
    MMUHatsError,
    MMUHatsUnavailableError,
    compact_preview_frame,
    default_mmu_hats_service,
)
from services.datalab_job_service import default_job_service

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



def _run_result_is_new(before_result: Any, current_result: Any) -> bool:
    """Return True only when this tool assigned a fresh UI run result."""
    return current_result is not None and current_result is not before_result


# Some models (notably deepseek-v4-pro) emit HTML entities inside tool-call
# string args — e.g. a plot title "g&lt;18" that otherwise renders the literal
# "&lt;" in the PNG. Unescape the common entities in every string arg before
# dispatch. Kept to an explicit set (NOT html.unescape) so URL query strings
# like "?a=1&copy=2" aren't mangled by greedy legacy-named-entity matching.
_HTML_ENTITY_REPLACEMENTS = (
    ("&lt;", "<"),
    ("&gt;", ">"),
    ("&quot;", '"'),
    ("&#34;", '"'),
    ("&#39;", "'"),
    ("&#x27;", "'"),
    ("&apos;", "'"),
    ("&amp;", "&"),  # last: a lone "&amp;" must not re-trigger the entities above
)


def _unescape_html_entities(value: str) -> str:
    for entity, char in _HTML_ENTITY_REPLACEMENTS:
        if entity in value:
            value = value.replace(entity, char)
    return value


def _unescape_tool_args(args: Any) -> Any:
    """Recursively unescape common HTML entities in string tool-call args.

    Values only — dict keys (parameter names) are left untouched so a decoded
    key can never fail to match a tool signature."""
    if isinstance(args, str):
        return _unescape_html_entities(args)
    if isinstance(args, dict):
        return {k: _unescape_tool_args(v) for k, v in args.items()}
    if isinstance(args, list):
        return [_unescape_tool_args(v) for v in args]
    return args


@dataclass
class AgentConfig:
    """Configuration for QuasarAgent"""
    api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    ads_api_key: str = field(default_factory=lambda: os.getenv("NASA_ADS_API_KEY", ""))
    model: str = field(default_factory=lambda: os.getenv("DEFAULT_LLM_MODEL", "gpt-oss-120b"))
    temperature: float = 0.7
    # Per-round output-token budget for the MAIN agent loop. The old hardcoded
    # 2000 was the real cause of the deepseek mid-turn truncations (live P6/P8/
    # P9): thinking-mode reasoning counts against max_tokens, so rounds died at
    # finish_reason=length mid-sentence — sometimes mid tool-call-arguments.
    # Providers additionally clamp (DEEPSEEK_MAX_OUTPUT_TOKENS / TACC_MAX_OUTPUT_TOKENS).
    max_tokens: int = field(default_factory=lambda: int(os.getenv("LLM_MAX_OUTPUT_TOKENS", "16384")))
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
        self._conv_run_tokens: Dict[str, str] = {}
        self._conv_ids_lock = threading.Lock()
        self._alma_project_picker_by_conversation: Dict[str, Dict[str, Any]] = {}
        self._alma_project_picker_lock = threading.Lock()

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
        fast_model = os.getenv("QUASAR_FAST_MODEL", "deepseek-v4-flash")
        reasoning_model = os.getenv("QUASAR_REASONING_MODEL", "deepseek-v4-pro")
        conductor_model = os.getenv("QUASAR_CONDUCTOR_MODEL", reasoning_model)
        synthesis_model = os.getenv("QUASAR_SYNTHESIS_MODEL", conductor_model)
        # Initialize complexity detector (gates Conductor activation)
        print("DEBUG: Init ComplexityDetector")
        self.complexity_detector = ComplexityDetector(
            client=self.client,
            model=os.getenv("QUASAR_COMPLEXITY_MODEL", fast_model),
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
            conductor_model=conductor_model,
            synthesis_model=synthesis_model,
            tool_executor=self._conductor_tool_executor,
            recovery_engine=self.recovery_engine,
            model_router=self.model_router,
            agent_pool=self.agent_pool,
            sandbox_executor=self.sandbox_executor,
            ads_client=self.ads_client,
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
    def _accumulated_tool_trace(self):
        # Request-scoped like _accumulated_run_results: concurrent chats must
        # never mix tool traces. Conductor subtask threads get their own
        # (discarded) list — their calls are not traced into the parent
        # request until a collector can be passed through explicitly.
        if not hasattr(self._tls, 'accumulated_tool_trace'):
            self._tls.accumulated_tool_trace = []
        return self._tls.accumulated_tool_trace

    @_accumulated_tool_trace.setter
    def _accumulated_tool_trace(self, value):
        self._tls.accumulated_tool_trace = value

    @property
    def last_search_results(self):
        return getattr(self._tls, 'last_search_results', None)

    @last_search_results.setter
    def last_search_results(self, value):
        self._tls.last_search_results = value

    # ── Per-conversation response ID helpers ────────────────────────

    def _response_state_key(self, conversation_id: str, model: Optional[str] = None) -> str:
        selected_model = model or self.config.model
        return f"{conversation_id}|{detect_provider(selected_model)}|{selected_model}"

    def _get_response_id(
        self,
        conversation_id: str,
        model: Optional[str] = None,
    ) -> Optional[str]:
        """Get provider response state for one conversation/model pair."""
        with self._conv_ids_lock:
            return self._conv_response_ids.get(self._response_state_key(conversation_id, model))

    def _begin_response_run(self, conversation_id: str, model: str, run_token: Optional[str]) -> None:
        if not run_token:
            return
        with self._conv_ids_lock:
            self._conv_run_tokens[self._response_state_key(conversation_id, model)] = run_token

    def _response_run_active(self, conversation_id: str, model: str, run_token: Optional[str]) -> bool:
        if not run_token:
            return True
        with self._conv_ids_lock:
            return self._conv_run_tokens.get(
                self._response_state_key(conversation_id, model)
            ) == run_token

    def _set_response_id(
        self,
        conversation_id: str,
        response_id: Optional[str],
        model: Optional[str] = None,
        run_token: Optional[str] = None,
    ):
        """Set or clear provider response state for one conversation/model pair."""
        key = self._response_state_key(conversation_id, model)
        with self._conv_ids_lock:
            if run_token and self._conv_run_tokens.get(key) != run_token:
                return
            if response_id is None:
                self._conv_response_ids.pop(key, None)
            else:
                self._conv_response_ids[key] = response_id

    def clear_response_state(
        self,
        conversation_id: str,
        model: Optional[str] = None,
        run_token: Optional[str] = None,
    ) -> None:
        """Clear agent and compatibility-adapter history after interruption."""
        if run_token and not self._response_run_active(
            conversation_id, model or self.config.model, run_token
        ):
            return
        response_id = self._get_response_id(conversation_id, model)
        self._set_response_id(conversation_id, None, model, run_token)
        if response_id:
            try:
                self.client.responses.clear_history(response_id)
            except Exception:
                pass

    def cancel_response_run(self, conversation_id: str, model: str, run_token: str) -> None:
        """Invalidate a timed-out run so its worker cannot restore stale state."""
        key = self._response_state_key(conversation_id, model)
        with self._conv_ids_lock:
            if self._conv_run_tokens.get(key) != run_token:
                return
        self.clear_response_state(conversation_id, model, run_token)
        with self._conv_ids_lock:
            if self._conv_run_tokens.get(key) == run_token:
                self._conv_run_tokens.pop(key, None)

    def _cleanup_conv_states(self, max_entries: int = 500):
        """Prevent memory leak — evict oldest conversation entries."""
        with self._conv_ids_lock:
            keys = list(dict.fromkeys([
                *self._conv_response_ids.keys(),
                *self._conv_run_tokens.keys(),
            ]))
            if len(keys) > max_entries:
                for k in keys[:len(keys) // 2]:
                    self._conv_response_ids.pop(k, None)
                    self._conv_run_tokens.pop(k, None)

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
        return f"""You are Quasar, an expert AI research assistant for astronomy — all wavelengths, all archives.

You give science users natural-language access to major astronomical data services:
- **NOIRLab Astro Data Lab survey catalogs** (Gaia DR3, DES DR1, DESI DR1, NSC DR2, SMASH DR1/2,
  DELVE DR3, Legacy Surveys DR9, SDSS DR17, VHS DR5) via TAP/SQL builders, one-shot analysis+plot
  tools, SIA image cutouts, and SPARCL spectra — catalog science is a FIRST-CLASS capability, not a side feature.
- **Radio archives**: the ALMA Science Archive (search, TAP, data-product triage) and VLASS imagery.
- **Multi-wavelength archives**: MAST (JWST/HST/TESS/Kepler) via `search_mast`/`search_mast_by_criteria`,
  ESO (VLT/MUSE, KMOS, X-Shooter, FORS2) via `search_eso_archive`, CADC (Gemini/JCMT/CFHT and more),
  IRSA infrared catalogs (WISE, 2MASS, Spitzer) via `search_irsa`, plus generic VO discovery tools.
- **Literature and people**: NASA ADS papers, researcher profiles, research trends.
Your goal is to help users find, visualize, and analyze astronomical data across ALL of these — pick
the archive/tool that fits the science question, never default to one observatory out of habit.

DOMAIN ROUTING (read first):
- Survey-catalog science (photometry, astrometry, redshift catalogs, CMDs/color cuts, stellar
  populations, crossmatches, density maps, variable-star light curves from survey epochs) →
  datalab_* tools. NEVER route these to ALMA tools or observatory documentation.
- Radio interferometry data, ALMA/VLA observations, proposal/policy/instrument questions → ALMA
  archive tools and the documentation context.
- Named-target imagery across wavelengths → hips_cutout / hips_multiband_panel / datalab imaging.
- The documentation (RAG) context, when present, covers observatory/instrument manuals (mostly
  ALMA/radio). It is IRRELEVANT to survey-catalog data requests — if the user wants catalog data,
  call the data tools and ignore weak documentation snippets.

ARTIFACT HONESTY (hard rule):
- Only claim a plot/image/data card "is shown above" when a tool in THIS turn actually returned
  success with an attached visual (image_attached/path in its result). The UI renders visuals from
  tool events only — your words cannot create a figure.
- If a plotting/query tool failed or was never called, say plainly that no figure was produced and
  what you would run next. NEVER describe the appearance/features of a figure that does not exist,
  and NEVER invent counts, coordinates, or table contents you did not retrieve this turn.

GUIDELINES:
- **ACTION OVER CHATTER**: If the user asks for data/search/plots, **IMMEDIATELY** call the appropriate tool. Answering a data, catalog, imagery, or plotting request with generic how-to instructions, an SQL sketch, or a description of what one COULD do — instead of actually calling the tools — is UNACCEPTABLE.
- **RECOVER FROM TOOL ERRORS**: When a tool returns an error with a hint (wrong column name, unsupported expression, bad table), fix the call using the hint (e.g. consult datalab_describe_table for valid columns) and retry. Never end the turn on a tool error without at least one corrected retry, and always finish with a plain-text answer for the user summarizing what worked.
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
- **PAPER SEARCH (MANDATORY TOOL)**: When the user asks for papers, publications, articles, literature, or studies — you MUST call the `search_papers` tool. Pass the user's request as NATURAL LANGUAGE (e.g. "recent papers on protoplanetary disks", "best ALMA papers on disk gaps", "foundational papers on planet formation"). If the user gives a proposal ID, project code, MOUS UID, ASDM UID, or archive dataset identifier and asks for papers connected to it, call `search_papers_by_observation_id` instead so QUASAR searches ADS for the exact identifier. Do NOT try to construct ADS field syntax yourself. NEVER use `web_search` for paper requests. After the tool runs, do NOT write any text listing the papers — output NOTHING. The UI renders the papers as interactive cards automatically.
- **AUTO-LINKING LITERATURE**: The backend automatically exact-links top ALMA project/proposal codes from `search_by_target` or `search_by_position` to NASA ADS papers for the Observation-Paper Graph. Do NOT call `search_papers_by_observation_id` merely to auto-link normal archive search results. Only call it when the user explicitly asks for papers connected to a specific identifier.
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
- **ALMA DATA PRODUCT TRIAGE**: When the user asks to fetch, inspect, list, or triage ALMA FITS/data products, call `triage_alma_data_products`. If they provide a project/proposal code, MOUS UID, ASDM UID, or dataset ID, triage that exact identifier directly. If they provide only a target name such as "M87", first use `triage_alma_data_products` to show available project codes and ask the user which project to triage; do NOT guess. If the previous turn showed a project-code picker and the user replies with a row number like "#4", "number 4", or "use the fourth one", call `triage_alma_data_products` with that reply exactly. If they provide a band preference, pass it so the picker is filtered first. Never auto-download huge products; remote header inspection and product listing are safe.
- **RESEARCH TRENDS**: When the user asks about publication volume, field growth, or funding landscape — "How much research on FRBs?", "Is interest in X growing?", "Who funds research on Y?" — call `get_research_trends`. Returns papers-per-year breakdown and top funders.
- **NEVER MENTION DATA SOURCES**: NEVER mention "OpenAlex", "OpenAlex profile", "OpenAlex database", or any internal data source by name in your response. Present all researcher/trend/enrichment data as if it is native QUASAR knowledge. Do NOT include links to OpenAlex pages or API URLs.
- **WEB TOOLS**: Use `web_search` ONLY for non-paper, non-archive real-time queries: current telescope schedules, observatory news, instrument specs, call-for-proposals, or operational status. Use `web_extract_url` ONLY when the user provides full http(s) URLs to read or when a prior map/search result gives a specific URL whose full page text is truly needed. Use `web_map_site` to discover URLs on a known site before extraction. Use `web_crawl_site` for bounded documentation/site-section extraction. Use `web_research` for comprehensive web reports and comparisons. NEVER use web tools when the user asks for papers/publications — use `search_papers` instead.
- **WEB TOOL ROUTING**: Keyword query → `web_search`. Full URL(s) to read/summarize/quote → `web_extract_url`. Site root URL plus "find pages" → `web_map_site`. Site section plus "crawl/docs" → `web_crawl_site`. Do NOT call `navigate_to_url` after `web_search` unless the user explicitly asks you to open a specific result URL. Do NOT pass keyword queries to `web_extract_url`.
- **IMAGERY ROUTING**: when the user asks to SEE something (show me X / what does X look like / image of X), call an imaging tool (hips_cutout / hips_multiband_panel / vlass_cutout / stamps) in THIS turn - even if a similar image was produced earlier in the conversation. Prior images are not re-displayed with a new answer; an answer about appearance without a fresh tool-produced image is incomplete.
- **DATA LAB / LEGACY SURVEYS IMAGERY**: For a Legacy Surveys / DECam / "coadd" color image (e.g. "color image of M31 from the DECam Legacy Surveys"), call `datalab_color_image` with just ra/dec/fov — it auto-selects an available 3-band triplet and renders the Lupton RGB. Do NOT hand-pick bands or pre-judge coverage. Key facts: (1) **LS DR9 imaging bands are g, r, z — there is NO i band**; never conclude "no color image" because i is missing. (2) Pick the FOV from the target's apparent size and the "center" intent (M31's D25 ≈ 3°, so "center" ≈ 0.1–0.2°), not a fixed constant. (3) The `coadd_all` cutout service has genuinely BROKEN/partial coverage at some bright nearby galaxies (e.g. the exact center of M31 has only usable z-band; the g/r/i tiles there are broken Local Group Survey refs). When `datalab_color_image` returns coverage_gap because fewer than 3 bands are usable, DO NOT just report failure and stop — the user asked for a color image, so **deliver it from a survey that does cover the target**: call `hips_cutout` (DSS2/color) or `hips_multiband_panel` for an optical color view. Report honestly that the Legacy Surveys coadd lacked full multi-band coverage at this position and that the color image shown is from the fallback survey. Only claim an image "shown" when a tool actually rendered one this turn.
- **STRICT WEB SAFETY**: Never provide, summarize, cite, or link to pornographic, sexually explicit, nude, erotic, escort, or adult-entertainment content. Never emit general-web image URLs. If the web safety filter withholds results, state only that results were withheld by the safety filter and do not reconstruct the blocked content from memory.
- After a tool runs (except search_papers), summarize the output concisely.
- If a search returns many results, offer to plot them (but execute the search first).
- If the user says "yes/proceed" to a previous suggestion, ACT on it immediately.
- **DO NOT** output raw tool usage strings like `[TOOL: ...]` or JSON. Just use the Native Tool Calling feature.
- **NAME RESOLUTION**: If search_by_target returns empty for a valid target, use the resolve_target tool to get RA/Dec, then use search_by_position.
- **MINIMAL PARAMETERS**: When calling search_by_target, ONLY include optional parameters (band, max_resolution, min_freq_ghz, scan_intent, etc.) if the user EXPLICITLY requested them. For example, if the user says "Find ALMA data of M87", call search_by_target(target_name="M87") with NO other parameters. Do NOT pass band=0, min_freq_ghz=0, max_resolution=100 etc. Leaving them out returns ALL data.
- **MULTI-TARGET (SAME CONSTRAINTS)**: If the user mentions multiple targets with the SAME constraints (e.g. "M87 and Sz65", or "M87, Sz65, NGC23 and M83"), pass them as a single comma-separated string: search_by_target(target_name="M87, Sz65"). The tool handles splitting and searching each target.
- **MULTI-BAND**: If the user mentions multiple bands (e.g. "Band 6 and Band 7"), pass them as comma-separated: search_by_target(target_name="M87", band="6,7"). The tool handles searching each band separately and shows a data card for each. NEVER make separate tool calls for each band — use comma-separated bands in ONE call.
- **PER-TARGET CONSTRAINTS**: If different targets have DIFFERENT band/constraint requirements (e.g. "M87 in Band 6 and Sz65 in Band 7"), make SEPARATE tool calls for each target-constraint pair: first search_by_target(target_name="M87", band="6"), then search_by_target(target_name="Sz65", band="7"). Each call produces its own data card. You MAY also pass them in one call as search_by_target(target_name="M87 in band 6, Sz65 in band 7") — the tool can parse per-target bands.
- **MULTI-WAVELENGTH / MIXED SOURCES**: For JWST/HST data with rich filtering (instrument, program, filter), prefer `search_mast` or `search_mast_by_criteria` — they provide deeper queries than search_cadc_archive. For ESO/VLT data (MUSE, KMOS, X-Shooter, FORS2), use `search_eso_archive`. For infrared catalog data (WISE, 2MASS, Spitzer), use `search_irsa`. Use `search_cadc_archive` for general multi-wavelength cone searches or telescopes like Gemini, JCMT, and CFHT. When the user asks for data from DIFFERENT archives (e.g. "ALMA data of M87 and JWST data of NGC23"), make SEPARATE tool calls for each: search_by_target(target_name="M87") for ALMA, then search_mast(target_name="NGC23", mission="JWST") for JWST. Each produces its own data card in the UI.
- **ALMA SCIENCE ARCHIVE COUNTS/DIAGNOSTICS**: For Cycle/project counts, solar/Sun projects, array-combination questions (12m, 7m, total power), high-resolution Band N target summaries, required molecular line sets in the same project, or bandwidth-switching likelihood, call `query_alma_science_archive`. Do NOT answer these from memory and do NOT hand-write ADQL unless that tool cannot express the query.
- **MMU/HATS CATALOG RULE**: Use `search_mmu_hats_catalog` when the user asks for source/catalog properties from large surveys -- Gaia astrometry/proper motions/parallaxes, DESI/SDSS redshifts and classifications, TESS source metadata, Chandra spectra metadata, what sources are near this position, source tables for ML, or cross-survey enrichment. Use the archive tools (search_by_target, search_by_position, search_mast, search_cadc_archive, search_eso_archive, triage_alma_data_products) when the user asks for observation availability, project/proposal IDs, FITS/data products, or telescope archive records. For combined requests (find ALMA data for M87 and Gaia sources in the field), call the archive tool FIRST to get observations/positions, THEN search_mmu_hats_catalog to enrich the field. For catalog-to-catalog matching use crossmatch_mmu_hats_catalogs within a bounded cone; if it fails, run two bounded cone searches and say so. Examples: Find ALMA data for M87 -> search_by_target. What Gaia sources are near M87? -> search_mmu_hats_catalog(catalog_key='gaia', target_name='M87'). Download ALMA FITS files -> ALMA/DataLink tools, never MMU/HATS.
- **LIVE IMAGERY RULE**: Use `hips_cutout` or `hips_multiband_panel` for "show me", appearance, and multiwavelength postage-stamp questions; they are deeper and broader than `get_sky_image`. Use `vlass_cutout` for 3 GHz radio continuum imagery (Dec > -40 only). Use `search_ztf_alerts`, `ztf_light_curve`, and `ztf_stamps` for transients and variability. Use `ned_sed_plot` for literature SEDs. Use `sparcl_find_spectra` and `sparcl_plot_spectrum` for real DESI/SDSS optical spectra. MMU/Data Lab remain authoritative for catalog tables.
- **RADIO SED RULE**: Use `radio_sed` for compact-source radio continuum SED or radio spectral-index questions; always repeat its flags and state that v1 uses TGSS/GLEAM/SUMSS/NVSS/FIRST catalog fluxes without resolution matching, flux-scale corrections, or image-plane photometry.
- **SKY MONITOR RULE**: When the user wants ongoing watching ("keep an eye on", "alert me", "monitor"), use `monitor_add_target` then `monitor_check_now`; report only NEW alerts, and use `monitor_list_targets` / `monitor_remove_target` to manage the watchlist.
- **VO DISCOVERY RULE**: When no built-in tool covers an archive/dataset, use the VO chain: `vo_find_services` -> `vo_list_tables` -> `vo_describe_table` -> `vo_adql_query` (SELECT-only). Always inspect the schema before writing ADQL, quote table names containing '/' or '+' in double quotes, and pass a keyword to `vo_list_tables` on big services like VizieR.
- **VARIABILITY RULE**: For variability, use `search_space_lightcurves` / `plot_space_lightcurve` for TESS/Kepler availability and plots, and `period_search` for TESS/Kepler targets or ZTF oids; always report the FAP with any period.
- **PULSAR CATALOG RULE**: Use `search_pulsars` / `pulsar_lookup` for pulsar catalogue or parameter questions (periods, DMs, S1400 fluxes, associations, name lookups, cone searches) instead of web search.
- **SOLAR SYSTEM RULE**: Use `moving_object_check` when a transient could be a known asteroid/comet, and `solar_system_ephemeris` for planet/asteroid/comet positions, distances, magnitude, and visibility over a date range.
- **DISTANCE RULE**: For distances: `gaia_distance` (stars, parallax), `ned_distance` (galaxies, redshift-independent), and `velocity_frame_distance` (flow-corrected Hubble distances) -- do not compute 1/parallax by hand.
- **MOC COVERAGE RULE**: Use `survey_coverage` / `survey_covers_position` before broad archive availability searches to check which surveys actually cover a position, especially for "is there data" or "which surveys observed X" questions; for exact archive IDs, product downloads, or if MOCServer fails, continue with the requested archive tool and report the preflight issue.
- **GALACTIC EXTINCTION RULE**: Use `galactic_extinction` for E(B-V)/A_lambda whenever photometry, colors, or distance moduli need dereddening.
- **RESPECT EXCLUSIONS**: If the user explicitly excludes a source (e.g. "non-ALMA", "not from ALMA", "only CADC"), do NOT call the excluded tool. Only call the tools the user actually wants.
- **FILTERING**: If the user asks for constraints like "resolution < 0.05", use the filter_results tool AFTER a search.
- **LINE COVERAGE**: For one named transition and target (for example, "Check CO(2-1) line coverage for M87"), call `find_alma_line_coverage` once. It resolves the target/redshift, selects the exact Splatalogue transition, and locally verifies ALMA spectral-window coverage. Do NOT use broad `check_co_lines` for a named transition and do NOT report other CO ladder transitions as matches. Keep `check_co_lines` only for explicit requests to inspect the whole CO/13CO/C18O ladder in prior search results.
- **TAP QUERIES**: When generating SQL/ADQL queries, use the column names in the schema below.
- **DATA LAB COLUMNS**: If you are not certain of a Data Lab table's column names, call `datalab_describe_table` BEFORE writing SQL — never guess column names (e.g. NSC DR2 uses gmag/rmag, not gmagmag). If a query fails with 'column ... does not exist', correct the name from the HINT or describe-table output and re-run.
- **DATA LAB ONE-SHOT TOOLS FIRST**: For a requested end product, call the matching one-shot tool — it queries, filters, plots, AND renders the figure in the UI in a single call:
  `datalab_color_magnitude_diagram` (CMD / "g vs g−r"; point_sources=true for stars),
  `datalab_color_color_diagram` (color-color / star-galaxy split),
  `datalab_sed_plot` + `svo_filter_wavelength` (SEDs — wavelengths come from the SVO service, never from memory),
  `datalab_lss_wedge` (cone/wedge large-scale-structure plots),
  `datalab_sky_density_map` / `datalab_density_aggregate` (density maps — use the COARSE HEALPix column, e.g. ring256, for regions wider than a few degrees),
  `datalab_density_vetting` (densest-clump search + cutout grid),
  `datalab_period_fold` (variable-star phase folding),
  `datalab_q3c_crossmatch` (two-catalog positional crossmatch — never hand-write q3c_join SQL),
  `datalab_image_cutout` / `datalab_color_image` / `datalab_cutout_grid` (survey imagery),
  `datalab_tiled_search` + `datalab_confirm_sky_area` (wide-area candidate searches — ALWAYS confirm the sky area with the user before scanning more than ~100 deg²).
  Chain datalab_select_catalog_rows→plotting only when no one-shot tool fits.
- **DATA LAB EXPERT SQL**: datalab_sql_query requires a bound: a q3c cone (q3c_radial_query), an indexed equality (e.g. SMASH `fieldid = 169`, `id = '169.429960'`, DESI `targetid = N`), a registry-approved BETWEEN box, or a GROUP BY aggregate on an aggregate-safe table. All-sky ROW-level pulls are rejected — use aggregates for footprints/histograms. Wide `datalab_density_aggregate` cones that exceed the sync window auto-tile into sub-cones and merge — call it ONCE with the full cone rather than hand-tiling. If a query returns a jobid, poll datalab_job_status a FEW times only; when the result says stop_polling, end the turn and tell the user the job is still running.
- **SURVEY COVERAGE CLAIMS**: Before claiming a catalog contains (or lacks) a target/region, check the `footprint` field returned by datalab_list_catalogs / datalab_describe_table, or call survey_covers_position for the exact position. NEVER list every catalog as covering a target — curate by footprint (e.g. the LMC is NOT covered by SDSS, DESI, LS DR9, or DES).
- **CROSSMATCH → MEMBER SELECTION**: For stream/cluster membership science (e.g. Pal 5 tidal tails), a raw positional crossmatch is only step one. Apply the science cuts server-side (value_cuts for proper-motion windows, color_cut for the population/CMD locus) and make the FINAL sky/CMD plots from the SELECTED member sample — never present the raw crossmatch as the result. State the exact cuts in your answer. For the ON-SKY plot, use the PM+CMD-SELECTED single-catalog rows over the FULL cone (e.g. the Gaia datalab_select_catalog_rows result), NOT a row-capped q3c_crossmatch result — the crossmatch LIMIT slices the sample to a spatial corner and the map then misses the cluster/stream entirely. Only claim the map shows the cluster/tails if the cluster center is actually within the plotted RA/Dec range.

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
- **SCIENTIFIC TONE**: Avoid decorative emoji in scientific headings. Do not use lab-themed emoji; prefer plain Markdown headings or astronomy terms such as ALMA, JWST, HST, telescope, archive, source, and observation.
- **GROUNDED SUMMARY MODE**: If the user prompt contains `[GROUNDED_SUMMARY_MODE]`, your final answer must be constrained to data retrieved by tools in the current run. Only summarize returned rows, counts, identifiers, coordinates, links, and explicit tool errors. Do not add outside background knowledge, inferred archive coverage, likely targets, or unstated counts. If no rows were retrieved, say that the current run returned no rows and do not fill gaps from memory.
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
                    "scan_intent": {"type": "string", "description": "ALMA scan intent to filter, such as TARGET, BANDPASS, PHASE, FLUX, or WVR. ONLY pass if user explicitly asks for a scan intent."},
                    "max_results": {"type": "integer", "description": "Maximum results to return"}
                },
                "required": ["ra", "dec"]
            }
        ))

        self.tool_registry.register(Tool(
            name="search_by_target",
            description=(
                "Search the ALMA archive (default) by target name. Supports multiple targets "
                "separated by 'and' or comma (e.g. 'M87 and Sz65' or 'M87, NGC 1068').\n"
                "Pass facility='VLA', 'VLBA', or 'GBT' to search the NRAO archive instead — "
                "ONLY when the user explicitly asks for those telescopes.\n"
                "CRITICAL: ONLY pass optional filter parameters (band, resolution, frequency, scan_intent) "
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
                    "scan_intent":    {"type": "string",  "description": "ALMA scan intent to filter, such as TARGET, BANDPASS, PHASE, FLUX, or WVR. ONLY pass if user explicitly asks for a scan intent."},
                    "public_only":    {"type": "boolean", "description": "Only return publicly available data."},
                },
                "required": ["target_name"]
            }
        ))

        self.tool_registry.register(Tool(
            name="search_by_frequency",
            description="Search archives by frequency range. Defaults to ALMA; pass facility='VLA'/'VLBA'/'GBT' for the NRAO archive.",
            function=self._search_by_frequency,
            parameters={
                "type": "object",
                "properties": {
                    "min_freq_ghz": {"type": "number", "description": "Minimum frequency in GHz"},
                    "max_freq_ghz": {"type": "number", "description": "Maximum frequency in GHz"},
                    "facility": {"type": "string", "enum": ["ALMA", "VLA", "VLBA", "GBT"], "description": "Observatory facility. Default ALMA."},
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
                "ALMA ONLY — NOT for NOIRLab Data Lab catalogs (gaia_dr3/des_dr1/desi_dr1/nsc_dr2/smash/...): "
                "use datalab_sql_query for those.\n"
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

        self.tool_registry.register(Tool(
            name="query_alma_science_archive",
            description=(
                "Run deterministic ALMA Science Archive query templates for hard archive-science questions. "
                "Use this instead of raw ADQL for: Cycle N project counts, Sun/solar projects, projects using "
                "12m+7m+total-power arrays, high-resolution Band N continuum candidates for a target, projects "
                "covering a required molecular line set such as 12CO/13CO/C18O in the same project, and "
                "bandwidth-switching calibration diagnostics."
            ),
            function=self._query_alma_science_archive,
            parameters={
                "type": "object",
                "properties": {
                    "query_type": {
                        "type": "string",
                        "enum": [
                            "cycle_solar_projects",
                            "cycle_array_combo_projects",
                            "high_resolution_band_data",
                            "line_set_projects",
                            "redshifted_line_projects",
                            "bandwidth_switching_candidates",
                        ],
                        "description": "Specific ALMA science/archive query template to run."
                    },
                    "cycle": {"type": "integer", "description": "ALMA cycle number, e.g. 10."},
                    "target": {"type": "string", "description": "Target/source name, e.g. HH212."},
                    "band": {"type": "integer", "description": "ALMA band number, e.g. 6 or 7."},
                    "max_resolution_arcsec": {"type": "number", "description": "Maximum angular resolution in arcsec for high-resolution data."},
                    "arrays": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Required array types, e.g. ['12m','7m','TP']."
                    },
                    "lines": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Required lines, e.g. ['12CO','13CO','C18O']."
                    },
                    "topic_filter": {"type": "string", "description": "Optional science keyword/category filter such as protostellar disks."},
                    "redshift_min": {"type": "number", "description": "Minimum redshift for redshifted rest-line searches."},
                    "redshift_max": {"type": "number", "description": "Maximum redshift for redshifted rest-line searches."},
                    "rest_species": {"type": "string", "description": "Rest species for line searches, e.g. CO, 12CO, 13CO, C18O."},
                    "science_category": {"type": "string", "description": "Optional ALMA science category filter, e.g. Galaxy evolution."},
                    "require_same_project": {"type": "boolean", "description": "Require requested line matches in the same proposal_id. Default true."},
                    "include_adql": {"type": "boolean", "description": "Include executed ADQL in provenance. Default true."},
                    "max_results": {"type": "integer", "description": "Maximum TAP rows to fetch before grouping. Default 5000."},
                },
                "required": ["query_type"]
            },
            category="archive"
        ))

        self.tool_registry.register(Tool(
            name="match_cross_archive_sources",
            description=(
                "Cross-match a source catalog against ALMA and MAST/JWST observations. "
                "Use this for source-list location questions such as Perseus protostars observed with ALMA and JWST, "
                "or pass explicit source coordinates for any catalog."
            ),
            function=self._match_cross_archive_sources,
            parameters={
                "type": "object",
                "properties": {
                    "catalog_name": {"type": "string", "description": "Catalog key. Built-in: 'perseus_protostars'. For arbitrary catalogs, pass sources."},
                    "sources": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "source_name": {"type": "string"},
                                "ra": {"type": "number", "description": "ICRS right ascension in degrees."},
                                "dec": {"type": "number", "description": "ICRS declination in degrees."},
                            },
                            "required": ["ra", "dec"],
                        },
                        "description": "Optional inline coordinate catalog. Each item needs source_name/name plus ra and dec in degrees."
                    },
                    "archives": {"type": "array", "items": {"type": "string"}, "description": "Archives to match, e.g. ['ALMA','JWST']."},
                    "radius_arcsec": {"type": "number", "description": "Match radius in arcsec. Default 5."},
                    "max_sources": {"type": "integer", "description": "Max catalog sources to test. Default 12."},
                    "max_alma_rows": {"type": "integer", "description": "Max ALMA TAP rows to fetch. Default 5000."},
                    "max_mast_results_per_source": {"type": "integer", "description": "Max MAST/JWST rows per source. Default 80."},
                    "require_all_archives": {"type": "boolean", "description": "If true, only return sources matched in every requested archive. Default false for diagnostic cross-match tables."},
                },
                "required": []
            },
            category="archive"
        ))

        self.tool_registry.register(Tool(
            name="match_perseus_protostars_alma_jwst",
            description=(
                "Cross-match a built-in Perseus protostar source list against ALMA and MAST/JWST observations. "
                "Use this for questions like 'Show locations of protostars in Perseus observed with ALMA and JWST'. "
                "Returns sources with both ALMA and JWST matches, counts, project/program IDs, and sky coordinates."
            ),
            function=self._match_perseus_protostars_alma_jwst,
            parameters={
                "type": "object",
                "properties": {
                    "radius_arcsec": {"type": "number", "description": "Match radius in arcsec. Default 5."},
                    "max_sources": {"type": "integer", "description": "Max built-in Perseus sources to test. Default 12."},
                    "max_alma_rows": {"type": "integer", "description": "Max ALMA TAP rows to fetch. Default 5000."},
                    "max_mast_results_per_source": {"type": "integer", "description": "Max MAST/JWST rows per source. Default 80."},
                },
                "required": []
            },
            category="archive"
        ))

        # plot_alma_results is registered once below under "Publication Plotting Tools"
        
        self.tool_registry.register(Tool(
            name="download_alma_data",
            description="Download ALMA data (FITS) for current results",
            function=self._download_alma_data,
            parameters={
                "type": "object",
                "properties": {
                    "dry_run": {"type": "boolean", "description": "If true, only simulates download. Default False.", "default": False}
                },
                "required": []
            }
        ))


        # NEW: Advanced ALminer Tools
        self.tool_registry.register(Tool(
            name="find_alma_line_coverage",
            description=(
                "Resolve one named spectral transition with Splatalogue and return only "
                "ALMA projects whose exact spectral windows cover its observed frequency."
            ),
            function=self._find_alma_line_coverage,
            parameters={
                "type": "object",
                "properties": {
                    "target_name": {
                        "type": "string",
                        "description": "Astronomical target name, e.g. M87",
                    },
                    "species": {
                        "type": "string",
                        "description": "Molecular species/formula, e.g. CO",
                    },
                    "transition": {
                        "type": "string",
                        "description": "Exact transition text, e.g. 2-1",
                    },
                    "redshift": {
                        "type": "number",
                        "description": "Optional explicit redshift; overrides SIMBAD/NED",
                    },
                    "tolerance_mhz": {
                        "type": "number",
                        "description": "Optional additional frequency tolerance in MHz",
                    },
                    "velocity_width_kms": {
                        "type": "number",
                        "description": "Optional full velocity width in km/s",
                    },
                },
                "required": ["target_name", "species", "transition"],
            },
            category="archive",
        ))

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
            description="Apply numeric filters to the LAST ALMA/archive search results table (e.g. 'resolution < 0.05 arcsec', 'sensitivity > 10 mJy'). This does NOT see Data Lab catalog results — for those, put the cut in the query itself (datalab_select_catalog_rows value_cuts, datalab_sql_query WHERE) or use the one-shot diagram tools' point_sources/morphology options.",
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
                "Uses Brave, Tavily, and Exa through Quasar's web search router for grounded, source-cited results. "
                "Use the user's query as-is — do NOT add years or dates unless the user explicitly mentioned them. "
                "For keyword queries, use this tool directly and answer from its returned sources. "
                "Do NOT call navigate_to_url after web_search unless the user explicitly asks to open a specific result URL "
                "or the search result is insufficient and you need one full page from a known http(s) URL. "
                "Examples: 'ALMA proprietary period policy', 'JWST cycle 4 call for proposals', "
                "'VLA sensitivity at 1.4 GHz'."
            ),
            function=self._tavily_web_search,
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Web search query string"},
                    "max_results": {"type": "integer", "description": "Number of results to return (default 10, max 10)"},
                    "search_depth": {"type": "string", "enum": ["basic", "advanced"], "description": "'basic' for quick real-time search, 'advanced' for Exa deep search; very hard comparisons/research route to Exa deep-reasoning (default: basic)"},
                },
                "required": ["query"]
            }
        ))

        self.tool_registry.register(Tool(
            name="web_extract_url",
            description=(
                "Extract clean markdown/text from one or more specific URLs using Tavily Extract. "
                "Use when the user gives URLs and asks to read, summarize, quote, or pull page content. "
                "Only pass full http(s) URLs. Never pass search terms or keyword queries here; use web_search for those. "
                "Do not call this immediately after web_search unless the final answer truly needs full-page text "
                "from a specific result URL. "
                "For JavaScript-heavy pages, set extract_depth='advanced'."
            ),
            function=self._tavily_extract_url,
            parameters={
                "type": "object",
                "properties": {
                    "urls": {
                        "type": "string",
                        "description": "Single full http(s) URL, comma-separated URLs, or list of URLs. Max 20. Not a keyword query.",
                    },
                    "query": {"type": "string", "description": "Optional focus query to return only relevant chunks."},
                    "chunks_per_source": {"type": "integer", "description": "Relevant chunks per URL when query is provided. 1-5, default 3."},
                    "extract_depth": {"type": "string", "enum": ["basic", "advanced"], "description": "Use advanced for JS-heavy pages."},
                    "include_images": {"type": "boolean", "description": "Whether to include image URLs from the pages."},
                    "content_format": {"type": "string", "enum": ["markdown", "text"], "description": "Extracted content format."},
                },
                "required": ["urls"],
            }
        ))

        self.tool_registry.register(Tool(
            name="web_map_site",
            description=(
                "Discover URLs on a website using Tavily Map without extracting page content. "
                "Use before crawling a large site, or when the user asks for site structure or to find the right page."
            ),
            function=self._tavily_map_site,
            parameters={
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "Root URL to map."},
                    "instructions": {"type": "string", "description": "Optional natural-language filter, e.g. 'find API authentication docs'."},
                    "max_depth": {"type": "integer", "description": "Crawl depth for URL discovery. 1-5, default 1."},
                    "max_breadth": {"type": "integer", "description": "Links explored per page, default 20."},
                    "limit": {"type": "integer", "description": "Maximum URLs to return. Default 100, hard-capped at 500."},
                    "select_paths": {"type": "array", "items": {"type": "string"}, "description": "Regex path patterns to include."},
                    "exclude_paths": {"type": "array", "items": {"type": "string"}, "description": "Regex path patterns to exclude."},
                    "select_domains": {"type": "array", "items": {"type": "string"}, "description": "Regex domain patterns to include."},
                    "exclude_domains": {"type": "array", "items": {"type": "string"}, "description": "Regex domain patterns to exclude."},
                    "allow_external": {"type": "boolean", "description": "Whether to include external links. Default false."},
                },
                "required": ["url"],
            }
        ))

        self.tool_registry.register(Tool(
            name="web_crawl_site",
            description=(
                "Crawl a bounded website section with Tavily Crawl and extract content from discovered pages. "
                "Use for documentation sections or site areas where multiple pages are needed. Keep limit small unless the user asks for broad coverage."
            ),
            function=self._tavily_crawl_site,
            parameters={
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "Root URL to crawl."},
                    "instructions": {"type": "string", "description": "Semantic focus for relevant pages/chunks."},
                    "chunks_per_source": {"type": "integer", "description": "Chunks per page when instructions are provided. 1-5, default 3."},
                    "max_depth": {"type": "integer", "description": "Crawl depth. 1-5, default 1."},
                    "max_breadth": {"type": "integer", "description": "Links explored per page, default 20."},
                    "limit": {"type": "integer", "description": "Maximum pages to extract. Default 20, hard-capped at 50."},
                    "extract_depth": {"type": "string", "enum": ["basic", "advanced"], "description": "Use advanced for JS-heavy pages."},
                    "content_format": {"type": "string", "enum": ["markdown", "text"], "description": "Extracted content format."},
                    "include_images": {"type": "boolean", "description": "Whether to include image URLs from crawled pages."},
                    "select_paths": {"type": "array", "items": {"type": "string"}, "description": "Regex path patterns to include."},
                    "exclude_paths": {"type": "array", "items": {"type": "string"}, "description": "Regex path patterns to exclude."},
                    "select_domains": {"type": "array", "items": {"type": "string"}, "description": "Regex domain patterns to include."},
                    "exclude_domains": {"type": "array", "items": {"type": "string"}, "description": "Regex domain patterns to exclude."},
                    "allow_external": {"type": "boolean", "description": "Whether to include external links. Default false."},
                },
                "required": ["url"],
            }
        ))

        self.tool_registry.register(Tool(
            name="web_research",
            description=(
                "Run Tavily Research for comprehensive multi-source web research with citations. "
                "Use for deep web reports, market/landscape comparisons, or current-topic investigations. "
                "Do not use for astronomy paper searches; use search_papers for publications."
            ),
            function=self._tavily_research,
            parameters={
                "type": "object",
                "properties": {
                    "research_input": {"type": "string", "description": "Research task or question."},
                    "model": {"type": "string", "enum": ["mini", "pro", "auto"], "description": "mini for narrow tasks, pro for broad comparisons, auto by default."},
                    "citation_format": {"type": "string", "enum": ["numbered", "mla", "apa", "chicago"], "description": "Citation style."},
                    "wait_for_completion": {"type": "boolean", "description": "Poll for completion before returning. Default true."},
                    "timeout_seconds": {"type": "integer", "description": "Maximum polling time. Default 120 seconds."},
                },
                "required": ["research_input"],
            }
        ))

        self.tool_registry.register(Tool(
            name="web_research_status",
            description="Get the status or completed content for a Tavily Research request_id.",
            function=self._tavily_research_status,
            parameters={
                "type": "object",
                "properties": {
                    "request_id": {"type": "string", "description": "Tavily Research request ID."}
                },
                "required": ["request_id"],
            }
        ))

        self.tool_registry.register(Tool(
            name="navigate_to_url",
            description=(
                "Navigate the browser to a specific full http(s) URL and return its page content. "
                "Use for interactive browser-only tasks on ALMA archive, NASA ADS, ESO portal, VizieR, arXiv pages, etc. "
                "Do NOT use this for keyword searches and do NOT call it after web_search unless the user explicitly "
                "asked to open/navigate to a specific URL."
            ),
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
            description="Search Splatalogue for spectral-line transitions of a molecule, with optional frequency, energy, intensity, transition, and catalog filters.",
            function=lambda **kw: self.splatalogue_tool.search_lines_by_molecule(**kw),
            parameters={
                "type": "object",
                "properties": {
                    "molecule_name": {"type": "string", "description": "Molecule name (e.g., 'CO', 'HCN', 'CH3OH', 'SiO')"},
                    "freq_min_ghz": {"type": "number", "description": "Minimum frequency filter (GHz)"},
                    "freq_max_ghz": {"type": "number", "description": "Maximum frequency filter (GHz)"},
                    "top_n": {"type": "integer", "minimum": 1, "description": "Maximum number of normalized transitions to return"},
                    "transition": {"type": "string", "description": "Optional quantum-number or transition filter, e.g. '2-1'"},
                    "energy_min": {"type": "number", "description": "Optional lower energy bound"},
                    "energy_max": {"type": "number", "description": "Optional upper energy bound"},
                    "energy_type": {
                        "type": "string",
                        "enum": ["el_cm1", "eu_cm1", "el_k", "eu_k"],
                        "description": "Energy field used for energy_min/energy_max",
                    },
                    "intensity_lower_limit": {"type": "number", "description": "Optional lower line-intensity threshold"},
                    "intensity_type": {
                        "type": "string",
                        "enum": ["CDMS/JPL (log)", "Sij-mu2", "Aij (log)"],
                        "description": "Intensity field used by intensity_lower_limit",
                    },
                    "line_lists": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": ["LovasNIST", "SLAIM", "JPL", "CDMS", "ToyaMA", "OSU", "TopModel", "Recombination", "RFI"],
                        },
                        "description": "Splatalogue source catalogs to include",
                    },
                    "only_astronomically_observed": {"type": "boolean", "description": "Only return transitions observed in space"},
                    "only_nrao_recommended": {"type": "boolean", "description": "Only return NRAO-recommended frequencies"},
                },
                "required": ["molecule_name"]
            }
        ))

        self.tool_registry.register(Tool(
            name="search_spectral_lines",
            description="Run an advanced Splatalogue frequency-range query. Use this for filtered line surveys, line-confusion checks, and catalog comparisons.",
            function=lambda **kw: self.splatalogue_tool.search_spectral_lines(**kw),
            parameters={
                "type": "object",
                "properties": {
                    "freq_min_ghz": {"type": "number", "description": "Minimum rest frequency in GHz"},
                    "freq_max_ghz": {"type": "number", "description": "Maximum rest frequency in GHz"},
                    "molecule_name": {"type": "string", "description": "Optional molecule/species name or formula"},
                    "transition": {"type": "string", "description": "Optional quantum-number or transition filter"},
                    "energy_min": {"type": "number", "description": "Optional lower energy bound"},
                    "energy_max": {"type": "number", "description": "Optional upper energy bound"},
                    "energy_type": {
                        "type": "string",
                        "enum": ["el_cm1", "eu_cm1", "el_k", "eu_k"],
                        "description": "Energy field used for energy_min/energy_max",
                    },
                    "intensity_lower_limit": {"type": "number", "description": "Optional lower line-intensity threshold"},
                    "intensity_type": {
                        "type": "string",
                        "enum": ["CDMS/JPL (log)", "Sij-mu2", "Aij (log)"],
                        "description": "Intensity field used by intensity_lower_limit",
                    },
                    "version": {
                        "type": "string",
                        "enum": ["v1.0", "v2.0", "v3.0", "vall"],
                        "description": "Splatalogue data version",
                    },
                    "exclude": {
                        "type": "array",
                        "items": {"type": "string", "enum": ["atmospheric", "potential", "probable", "known", "none"]},
                        "description": "Species classifications to exclude",
                    },
                    "line_lists": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": ["LovasNIST", "SLAIM", "JPL", "CDMS", "ToyaMA", "OSU", "TopModel", "Recombination", "RFI"],
                        },
                        "description": "Splatalogue source catalogs to include",
                    },
                    "only_astronomically_observed": {"type": "boolean", "description": "Only return transitions observed in space"},
                    "only_nrao_recommended": {"type": "boolean", "description": "Only return NRAO-recommended frequencies"},
                    "top_n": {"type": "integer", "minimum": 1, "description": "Maximum number of normalized transitions to return"},
                },
                "required": ["freq_min_ghz", "freq_max_ghz"]
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

        # Data Lab P0 catalog-TAP tools
        self.tool_registry.register(Tool(
            name="datalab_list_catalogs",
            description="List supported NOIRLab Astro Data Lab P0 catalogs and registered tables.",
            function=self._datalab_list_catalogs,
            parameters={"type": "object", "properties": {}, "required": []},
            category="datalab",
        ))

        self.tool_registry.register(Tool(
            name="datalab_describe_table",
            description="Describe a registered Data Lab catalog table, columns, region strategy, morphology hints, and citation.",
            function=self._datalab_describe_table,
            parameters={
                "type": "object",
                "properties": {
                    "catalog": {"type": "string", "description": "Registered catalog, e.g. gaia_dr3 or nsc_dr2."},
                    "table": {"type": "string", "description": "Registered table within the catalog, e.g. gaia_source or object."},
                },
                "required": ["catalog", "table"],
            },
            category="datalab",
        ))

        self.tool_registry.register(Tool(
            name="datalab_cone_count",
            description="Count rows in a Data Lab catalog cone using a governed q3c_radial_query builder.",
            function=self._datalab_cone_count,
            parameters={
                "type": "object",
                "properties": {
                    "catalog": {"type": "string"},
                    "table": {"type": "string"},
                    "ra": {"type": "number", "description": "ICRS right ascension in degrees."},
                    "dec": {"type": "number", "description": "ICRS declination in degrees."},
                    "radius_deg": {"type": "number", "description": "Cone radius in degrees."},
                },
                "required": ["catalog", "table", "ra", "dec", "radius_deg"],
            },
            category="datalab",
        ))

        self.tool_registry.register(Tool(
            name="datalab_select_catalog_rows",
            description="Select capped rows from a Data Lab catalog cone using governed structured SQL; returns result_id, not the full table. Apply selection cuts server-side via value_cuts/color_cut/morphology so the row budget is spent on rows you want.",
            function=self._datalab_select_catalog_rows,
            parameters={
                "type": "object",
                "properties": {
                    "catalog": {"type": "string"},
                    "table": {"type": "string"},
                    "ra": {"type": "number"},
                    "dec": {"type": "number"},
                    "radius_deg": {"type": "number"},
                    "columns": {"type": "array", "items": {"type": "string"}},
                    "limit": {"type": "integer", "default": 500},
                    "value_cuts": {"type": "array", "items": {"type": "object"}, "description": "e.g. [{'column':'parallax_over_error','op':'>','value':5}]"},
                    "color_cut": {"type": "object", "description": "{'bands':['gmag','rmag'],'min':-0.5,'max':0.5}"},
                    "morphology": {"type": "object", "description": "{'column':'class_star','op':'>','value':0.5}"},
                },
                "required": ["catalog", "table", "ra", "dec", "radius_deg"],
            },
            category="datalab",
        ))

        self.tool_registry.register(Tool(
            name="datalab_density_aggregate",
            description="Aggregate Data Lab source density by RA/Dec grid or registered HEALPix column over a cone region; returns a stable result_id. Requires a cone (ra/dec/radius_deg) unless all_sky=true is set explicitly. Wide cones that exceed the 60s sync window are automatically tiled into sub-cones and merged — do NOT hand-tile the region yourself; call once with the full cone.",
            function=self._datalab_density_aggregate,
            parameters={
                "type": "object",
                "properties": {
                    "catalog": {"type": "string"},
                    "table": {"type": "string"},
                    "mode": {"type": "string", "enum": ["grid", "healpix"], "default": "grid"},
                    "step_deg": {"type": "number", "default": 0.1},
                    "healpix_column": {"type": "string"},
                    "ra": {"type": "number", "description": "Cone center RA (deg); with dec+radius_deg bounds the aggregate."},
                    "dec": {"type": "number", "description": "Cone center Dec (deg)."},
                    "radius_deg": {"type": "number", "description": "Cone radius (deg) bounding the aggregate."},
                    "all_sky": {"type": "boolean", "default": False, "description": "Explicitly run an unbounded whole-catalog aggregate (slow/expensive)."},
                    "color_cut": {"type": "object", "description": "e.g. {'bands':['gmag','rmag'],'min':-0.5,'max':0.5}"},
                    "value_cuts": {"type": "array", "items": {"type": "object"}, "description": "e.g. [{'column':'gmag','op':'>','value':19.5}]"},
                    "morphology": {"type": "object", "description": "e.g. {'column':'class_star','op':'>','value':0.5}"},
                    "limit": {"type": "integer", "default": 5000},
                },
                "required": ["catalog", "table"],
            },
            category="datalab",
        ))

        self.tool_registry.register(Tool(
            name="datalab_q3c_crossmatch",
            description="Planner-safe Data Lab q3c crossmatch: materializes the small Gaia-like side first and joins the large indexed catalog second.",
            function=self._datalab_q3c_crossmatch,
            parameters={
                "type": "object",
                "properties": {
                    "small_catalog": {"type": "string", "default": "gaia_dr3"},
                    "small_table": {"type": "string", "default": "gaia_source"},
                    "big_catalog": {"type": "string", "default": "nsc_dr2"},
                    "big_table": {"type": "string", "default": "object"},
                    "ra": {"type": "number"},
                    "dec": {"type": "number"},
                    "radius_deg": {"type": "number"},
                    "match_radius_arcsec": {"type": "number", "default": 1.0},
                    "small_columns": {"type": "array", "items": {"type": "string"}},
                    "big_columns": {"type": "array", "items": {"type": "string"}},
                    "small_limit": {"type": "integer", "default": 10000},
                    "limit": {"type": "integer", "default": 500},
                },
                "required": ["ra", "dec", "radius_deg"],
            },
            category="datalab",
        ))

        self.tool_registry.register(Tool(
            name="datalab_sql_query",
            description=(
                "EXPERT/DEBUG ONLY: run governed raw native SQL against Data Lab. "
                "Requires expert_ack=true and a reason; q3c_join remains blocked outside the structured crossmatch builder. "
                "Returns result_id only, not the full table."
            ),
            function=self._datalab_sql_query,
            parameters={
                "type": "object",
                "properties": {
                    "sql": {"type": "string", "description": "Read-only native SQL SELECT/WITH query."},
                    "expert_ack": {"type": "boolean", "description": "Must be true to acknowledge expert/debug raw SQL mode."},
                    "reason": {"type": "string", "description": "Brief justification for not using structured builders."},
                },
                "required": ["sql", "expert_ack", "reason"],
            },
            category="datalab",
        ))

        self.tool_registry.register(Tool(
            name="datalab_get_result",
            description="Fetch the rows of a stored Data Lab result_id (from a prior datalab_* tool), capped at max_rows (<=5000).",
            function=self._datalab_get_result,
            parameters={
                "type": "object",
                "properties": {
                    "result_id": {"type": "string", "description": "result_id returned by a datalab_* tool."},
                    "max_rows": {"type": "integer", "default": 200, "description": "Maximum rows to return (capped at 5000)."},
                },
                "required": ["result_id"],
            },
            category="datalab",
        ))

        # Data Lab P1 SIA image, SVO, and catalog-analysis tools
        self.tool_registry.register(Tool(
            name="datalab_image_cutout",
            description="Render a single-band NOIRLab Astro Data Lab SIA cutout at RA/Dec or a resolvable target name.",
            function=self._datalab_image_cutout,
            parameters={
                "type": "object",
                "properties": {
                    "ra": {"type": "number", "description": "ICRS right ascension in degrees."},
                    "dec": {"type": "number", "description": "ICRS declination in degrees."},
                    "target_name": {"type": "string", "description": "Optional target name to resolve if ra/dec are not supplied."},
                    "fov_deg": {"type": "number", "default": 0.05, "description": "Cutout field of view in degrees."},
                    "band": {"type": "string", "default": "g", "description": "Band prefix, e.g. g, r, or i."},
                    "catalog": {"type": "string", "default": "ls_dr9", "description": "Registered Data Lab catalog used to choose SIA endpoint."},
                    "endpoint": {"type": "string", "description": "Optional explicit SIA endpoint override."},
                    "title": {"type": "string"},
                },
                "required": ["fov_deg"],
            },
            category="datalab",
        ))

        self.tool_registry.register(Tool(
            name="datalab_color_image",
            description="Render a Data Lab color image from deepest SIA stack images, reprojected to a common WCS before Lupton RGB composition. Auto-selects RGB bands (red=i or z, green=r, blue=g) unless 'bands' is given.",
            function=self._datalab_color_image,
            parameters={
                "type": "object",
                "properties": {
                    "ra": {"type": "number"},
                    "dec": {"type": "number"},
                    "target_name": {"type": "string"},
                    "fov_deg": {"type": "number", "default": 0.05},
                    "catalog": {"type": "string", "default": "ls_dr9"},
                    "endpoint": {"type": "string"},
                    "bands": {"type": "array", "items": {"type": "string"}, "description": "Optional (red, green, blue) band override, e.g. ['z','r','g']. Defaults to auto-selection."},
                    "q": {"type": "number", "default": 8.0},
                    "stretch": {"type": "number", "default": 0.5},
                    "title": {"type": "string"},
                },
                "required": ["fov_deg"],
            },
            category="datalab",
        ))

        self.tool_registry.register(Tool(
            name="datalab_cutout_grid",
            description="Render a multi-panel Data Lab SIA cutout grid for peak coordinates; panels without coverage are labeled instead of failing the grid.",
            function=self._datalab_cutout_grid,
            parameters={
                "type": "object",
                "properties": {
                    "peaks": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "ra": {"type": "number"},
                                "dec": {"type": "number"},
                                "label": {"type": "string"},
                            },
                            "required": ["ra", "dec"],
                        },
                    },
                    "fov_deg": {"type": "number", "default": 0.05},
                    "band": {"type": "string", "default": "g"},
                    "catalog": {"type": "string", "default": "ls_dr9"},
                    "endpoint": {"type": "string"},
                    "title": {"type": "string"},
                },
                "required": ["peaks", "fov_deg"],
            },
            category="datalab",
        ))

        self.tool_registry.register(Tool(
            name="svo_filter_wavelength",
            description="Look up effective and pivot wavelengths for SVO FPS filter IDs, including LS DR9 shorthand g/r/z/w1/w2.",
            function=self._svo_filter_wavelength,
            parameters={
                "type": "object",
                "properties": {
                    "filter_id": {"type": "string", "description": "Single filter ID or shorthand."},
                    "filter_ids": {"type": "array", "items": {"type": "string"}, "description": "Multiple filter IDs or shorthands."},
                },
                "required": [],
            },
            category="datalab",
        ))

        self.tool_registry.register(Tool(
            name="datalab_catalog_scatter",
            description=(
                "Render a Data Lab result_id as a CMD/CCD/HR-style scatter plot using safe column expressions. "
                "x_expr/y_expr combine columns of the stored result with + - * / ** %, parentheses, numeric literals, "
                "and functions log10/log/sqrt/abs/exp/power (e.g. absolute magnitude: "
                "'phot_g_mean_mag + 5*log10(parallax/100)' with parallax in mas). Only columns present in the stored "
                "result can be referenced — derived columns like M_G do NOT pre-exist; compute them inline here or "
                "alias them in the SQL SELECT first."
            ),
            function=self._datalab_catalog_scatter,
            parameters={
                "type": "object",
                "properties": {
                    "result_id": {"type": "string"},
                    "x_expr": {
                        "type": "string",
                        "description": "Expression over result columns for the x axis, e.g. 'bp_rp' or 'g - r'.",
                    },
                    "y_expr": {
                        "type": "string",
                        "description": "Expression over result columns for the y axis, e.g. 'phot_g_mean_mag + 5*log10(parallax/100)'.",
                    },
                    "color_by": {"type": "string"},
                    "invert_y": {"type": "boolean", "default": False},
                    "invert_x": {"type": "boolean", "default": False},
                    "title": {"type": "string"},
                    "x_label": {"type": "string"},
                    "y_label": {"type": "string"},
                    "overlay_locus": {"type": "string"},
                },
                "required": ["result_id", "x_expr", "y_expr"],
            },
            category="datalab",
        ))

        self.tool_registry.register(Tool(
            name="datalab_sky_density_map",
            description=(
                "Render a Data Lab result_id as a RA/Dec density map or sparse HEALPix density map, "
                "with optional matched-filter peak detection. The colorbar is LOG-scaled by default "
                "(log_scale=true) — leave it on for 'log counts'/'log source count' requests; set "
                "log_scale=false only when the user explicitly wants a linear count scale."
            ),
            function=self._datalab_sky_density_map,
            parameters={
                "type": "object",
                "properties": {
                    "result_id": {"type": "string"},
                    "mode": {"type": "string", "enum": ["hist2d", "healpix"], "default": "hist2d"},
                    "ra_col": {"type": "string"},
                    "dec_col": {"type": "string"},
                    "count_col": {"type": "string", "default": "source_count"},
                    "bins": {"type": "integer", "default": 80},
                    "healpix_col": {"type": "string", "default": "healpix"},
                    "nside": {"type": "integer"},
                    "order": {"type": "string", "default": "nested"},
                    "matched_filter": {"type": "boolean", "default": False},
                    "sigma_small": {"type": "number", "default": 1.0},
                    "sigma_large": {"type": "number", "default": 3.0},
                    "peak_threshold": {"type": "number", "default": 3.0},
                    "max_peaks": {"type": "integer", "default": 10},
                    "log_scale": {"type": "boolean", "default": True, "description": "Log-scale the count colorbar (matplotlib LogNorm). True honors 'log counts' requests; set false for a linear scale."},
                    "title": {"type": "string"},
                },
                "required": ["result_id"],
            },
            category="datalab",
        ))

        self.tool_registry.register(Tool(
            name="datalab_period_fold",
            description=(
                "Run Lomb-Scargle period search on a stored Data Lab light curve and render the folded light curve. "
                "Multi-band light curves (NSC/DES/SMASH interleave g/r/i/z epochs in one table) should be folded ONE band "
                "at a time — pass band (e.g. 'g') to restrict to a single filter; mixing bands smears the phased curve."
            ),
            function=self._datalab_period_fold,
            parameters={
                "type": "object",
                "properties": {
                    "result_id": {"type": "string"},
                    "time_col": {"type": "string", "default": "mjd"},
                    "mag_col": {"type": "string", "default": "cmag"},
                    "error_col": {"type": "string", "default": "cerr"},
                    "band": {"type": "string", "description": "Single filter to fold (e.g. 'g', 'r', 'i', 'z'). Restricts to rows where band_col equals this value. Leave unset to fold all rows."},
                    "band_col": {"type": "string", "default": "filter", "description": "Column holding the filter/band label (NSC/SMASH use 'filter')."},
                    "min_frequency": {"type": "number", "default": 1.0},
                    "max_frequency": {"type": "number", "default": 10.0},
                    "title": {"type": "string"},
                },
                "required": ["result_id"],
            },
            category="datalab",
        ))

        self.tool_registry.register(Tool(
            name="datalab_sed_plot",
            description="Render an LS DR9-style SED from a stored Data Lab result_id using SVO FPS wavelengths.",
            function=self._datalab_sed_plot,
            parameters={
                "type": "object",
                "properties": {
                    "result_id": {"type": "string"},
                    "row_index": {"type": "integer", "default": 0},
                    "filter_columns": {"type": "object", "additionalProperties": {"type": "string"}},
                    "title": {"type": "string"},
                },
                "required": ["result_id"],
            },
            category="datalab",
        ))

        self.tool_registry.register(Tool(
            name="datalab_lss_wedge",
            description="Render a stored spectroscopic Data Lab result_id as a comoving large-scale-structure wedge or 3D scatter plot.",
            function=self._datalab_lss_wedge,
            parameters={
                "type": "object",
                "properties": {
                    "result_id": {"type": "string"},
                    "ra_col": {"type": "string"},
                    "dec_col": {"type": "string"},
                    "z_col": {"type": "string", "default": "z"},
                    "class_col": {"type": "string"},
                    "pie_slice": {"type": "boolean", "default": False},
                    "title": {"type": "string"},
                },
                "required": ["result_id"],
            },
            category="datalab",
        ))
        self.tool_registry.register(Tool(
            name="datalab_density_vetting",
            description="P12: find the densest catalog cells within a cone (with optional color/magnitude/morphology cuts) and pull a SIA cutout grid of the top-N densest locations to eyeball.",
            function=self._datalab_density_vetting,
            parameters={
                "type": "object",
                "properties": {
                    "catalog": {"type": "string"},
                    "table": {"type": "string"},
                    "radius_deg": {"type": "number"},
                    "ra": {"type": "number"},
                    "dec": {"type": "number"},
                    "target_name": {"type": "string"},
                    "step_deg": {"type": "number", "default": 0.05},
                    "color_cut": {"type": "object", "description": "e.g. {'bands':['gmag','rmag'],'min':-0.5,'max':0.5}"},
                    "value_cuts": {"type": "array", "items": {"type": "object"}, "description": "e.g. [{'column':'gmag','op':'<','value':25}]"},
                    "morphology": {"type": "object", "description": "e.g. {'column':'class_star','op':'>','value':0.5} or {'column':'ext_coadd','between':[0,1]}"},
                    "top_n": {"type": "integer", "default": 5},
                    "fov_deg": {"type": "number", "default": 0.05},
                    "band": {"type": "string", "default": "g"},
                },
                "required": ["catalog", "table", "radius_deg"],
            },
            category="datalab",
        ))
        self.tool_registry.register(Tool(
            name="datalab_color_color_diagram",
            description="ONE-SHOT color-color diagram (e.g. g-r vs r-i) for a catalog cone. Queries + plots in a single call; auto-splits into stars vs galaxies (2 panels) using the catalog's morphology column (e.g. DES spread_model_r) unless split_col is given. Sentinel magnitudes (99.99) are excluded automatically. Use this for 'show me a color-color diagram'/'separate stars from galaxies' requests — do NOT chain separate query+plot tools.",
            function=self._datalab_color_color_diagram,
            parameters={
                "type": "object",
                "properties": {
                    "catalog": {"type": "string"},
                    "table": {"type": "string", "description": "Optional; defaults to the catalog's primary table."},
                    "radius_deg": {"type": "number", "default": 0.5},
                    "ra": {"type": "number"},
                    "dec": {"type": "number"},
                    "target_name": {"type": "string"},
                    "x_bands": {"type": "array", "items": {"type": "string"}, "description": "Two bands for the x color, e.g. ['g','r']."},
                    "y_bands": {"type": "array", "items": {"type": "string"}, "description": "Two bands for the y color, e.g. ['r','i']."},
                    "split_col": {"type": "string", "description": "Morphology column to split stars/galaxies (auto from registry if omitted, e.g. spread_model_r)."},
                    "split_threshold": {"type": "number", "default": 0.005},
                    "limit": {"type": "integer", "default": 3000},
                    "point_sources": {"type": "boolean", "default": False, "description": "True = keep only point sources via the catalog's registered star cut (single panel, no star/galaxy split)."},
                    "value_cuts": {"type": "array", "items": {"type": "object"}, "description": "Extra server-side cuts, e.g. [{'column':'flags_g','op':'=','value':0}]."},
                },
                "required": ["catalog"],
            },
            category="datalab",
        ))
        self.tool_registry.register(Tool(
            name="datalab_color_magnitude_diagram",
            description="ONE-SHOT color-magnitude diagram (CMD): mag_band vs (blue-red) color for a catalog cone. Queries + plots in a single call (magnitude axis inverted). Sentinel magnitudes (99.99) are excluded automatically; set point_sources=true when the user asks for stars/point sources. Use this for 'plot a CMD'/'g vs g-r' requests instead of chaining query+plot tools.",
            function=self._datalab_color_magnitude_diagram,
            parameters={
                "type": "object",
                "properties": {
                    "catalog": {"type": "string"},
                    "table": {"type": "string", "description": "Optional; defaults to the catalog's primary table."},
                    "radius_deg": {"type": "number", "default": 0.4},
                    "ra": {"type": "number"},
                    "dec": {"type": "number"},
                    "target_name": {"type": "string"},
                    "blue_band": {"type": "string", "default": "g"},
                    "red_band": {"type": "string", "default": "r"},
                    "mag_band": {"type": "string", "description": "Magnitude (y) band; defaults to blue_band."},
                    "limit": {"type": "integer", "default": 5000},
                    "point_sources": {"type": "boolean", "default": False, "description": "True = keep only point sources via the catalog's registered star cut (e.g. NSC class_star>0.5)."},
                    "value_cuts": {"type": "array", "items": {"type": "object"}, "description": "Extra server-side cuts, e.g. [{'column':'parallax_over_error','op':'>','value':5}]."},
                },
                "required": ["catalog"],
            },
            category="datalab",
        ))
        self.tool_registry.register(Tool(
            name="datalab_tiled_search",
            description="P15: tiled region-bounded overdensity search over a footprint. Runs a server-side density aggregate per q3c cone tile, finds matched-filter peaks, and ranks candidates. Executes as a background job; for a large area it returns needs_confirmation first — re-call with confirm=true after confirming the sky area with the user.",
            function=self._datalab_tiled_search,
            parameters={
                "type": "object",
                "properties": {
                    "catalog": {"type": "string"},
                    "table": {"type": "string"},
                    "ra_min": {"type": "number"},
                    "ra_max": {"type": "number"},
                    "dec_min": {"type": "number"},
                    "dec_max": {"type": "number"},
                    "tile_radius_deg": {"type": "number", "default": 2.0},
                    "step_deg": {"type": "number", "default": 0.05},
                    "color_cut": {"type": "object"},
                    "value_cuts": {"type": "array", "items": {"type": "object"}},
                    "morphology": {"type": "object"},
                    "peak_threshold": {"type": "number", "default": 3.0},
                    "max_tiles": {"type": "integer", "default": 64},
                    "candidate_budget": {"type": "integer", "default": 50},
                    "confirm": {"type": "boolean", "default": False},
                },
                "required": ["catalog", "table", "ra_min", "ra_max", "dec_min", "dec_max"],
            },
            category="datalab",
        ))
        self.tool_registry.register(Tool(
            name="datalab_confirm_sky_area",
            description="Estimate the sky area and tile count for a tiled search before fanning out (guardrail: confirm wide scans with the user first).",
            function=self._datalab_confirm_sky_area,
            parameters={
                "type": "object",
                "properties": {
                    "ra_min": {"type": "number"}, "ra_max": {"type": "number"},
                    "dec_min": {"type": "number"}, "dec_max": {"type": "number"},
                    "tile_radius_deg": {"type": "number", "default": 2.0},
                },
                "required": ["ra_min", "ra_max", "dec_min", "dec_max"],
            },
            category="datalab",
        ))
        self.tool_registry.register(Tool(
            name="datalab_job_status",
            description="Poll the status of a Data Lab background job (e.g. a tiled search).",
            function=self._datalab_job_status,
            parameters={"type": "object", "properties": {"job_id": {"type": "string"}}, "required": ["job_id"]},
            category="datalab",
        ))
        self.tool_registry.register(Tool(
            name="datalab_job_results",
            description="Fetch the result of a Data Lab background job (ranked candidates for a tiled search).",
            function=self._datalab_job_results,
            parameters={"type": "object", "properties": {"job_id": {"type": "string"}}, "required": ["job_id"]},
            category="datalab",
        ))
        self.tool_registry.register(Tool(
            name="datalab_job_cancel",
            description="Cancel a running Data Lab background job.",
            function=self._datalab_job_cancel,
            parameters={"type": "object", "properties": {"job_id": {"type": "string"}}, "required": ["job_id"]},
            category="datalab",
        ))
        self.tool_registry.register(Tool(
            name="datalab_export_notebook",
            description="Export a reproducible Jupyter notebook for a Data Lab analysis: the governed TAP SQL (qc.query(sql=...)), an optional SIA cutout recipe, an optional SVO filter-wavelength lookup, and a data-citation cell.",
            function=self._datalab_export_notebook,
            parameters={
                "type": "object",
                "properties": {
                    "title": {"type": "string", "default": "NOIRLab Data Lab analysis"},
                    "sql": {"type": "string", "description": "The governed native SQL to reproduce (e.g. a datalab tool's query_summary)."},
                    "catalog": {"type": "string"},
                    "table": {"type": "string"},
                    "sia_ra": {"type": "number"},
                    "sia_dec": {"type": "number"},
                    "sia_fov_deg": {"type": "number", "default": 0.1},
                    "sia_endpoint": {"type": "string"},
                    "svo_filters": {"type": "array", "items": {"type": "string"}},
                },
                "required": [],
            },
            category="datalab",
        ))
        # -- Multimodal Universe HATS catalogs (LSDB / Hugging Face) --
        self.tool_registry.register(Tool(
            name="list_mmu_hats_catalogs",
            description="List available Multimodal Universe HATS catalogs (Gaia, DESI, SDSS, TESS, Chandra) that Quasar can cone-search from Hugging Face via LSDB. These provide catalog/source properties, not archive observations.",
            function=self._list_mmu_hats_catalogs,
            parameters={"type": "object", "properties": {}, "required": []},
            category="mmu_hats",
        ))
        self.tool_registry.register(Tool(
            name="search_mmu_hats_catalog",
            description=(
                "Cone-search a Multimodal Universe HATS catalog using LSDB/Hugging Face. "
                "Use this for catalog/source properties such as Gaia astrometry, DESI/SDSS redshifts, "
                "TESS source metadata, Chandra spectra metadata, or cross-survey source enrichment "
                "around a sky position. Do NOT use this for finding archive observations, project/proposal IDs, "
                "or FITS/data products -- use the archive tools for those."
            ),
            function=self._search_mmu_hats_catalog,
            parameters={
                "type": "object",
                "properties": {
                    "catalog_key": {"type": "string", "enum": sorted(MMU_HATS_CATALOGS)},
                    "ra": {"type": "number", "description": "ICRS right ascension in degrees."},
                    "dec": {"type": "number", "description": "ICRS declination in degrees."},
                    "radius_arcsec": {"type": "number", "default": 120, "description": "Cone radius in arcseconds; capped by MMU_HATS_MAX_RADIUS_ARCSEC."},
                    "columns": {"type": "array", "items": {"type": "string"}},
                    "max_rows": {"type": "integer", "default": 500, "description": "Maximum rows returned to the UI; capped by MMU_HATS_MAX_ROWS."},
                    "target_name": {"type": "string", "description": "Resolve this name to RA/Dec instead of passing ra/dec."},
                },
                "required": [],
            },
            category="mmu_hats",
        ))
        self.tool_registry.register(Tool(
            name="crossmatch_mmu_hats_catalogs",
            description=(
                "Crossmatch two Multimodal Universe HATS catalogs (e.g. gaia x desi_edr_sv3) "
                "within a bounded cone region using LSDB margin caches. Requires ra/dec/radius -- "
                "all-sky crossmatches are not allowed. If this fails, run two bounded "
                "search_mmu_hats_catalog cone searches instead."
            ),
            function=self._crossmatch_mmu_hats_catalogs,
            parameters={
                "type": "object",
                "properties": {
                    "left_catalog_key": {"type": "string", "enum": sorted(MMU_HATS_CATALOGS)},
                    "right_catalog_key": {"type": "string", "enum": sorted(MMU_HATS_CATALOGS)},
                    "ra": {"type": "number"},
                    "dec": {"type": "number"},
                    "radius_arcsec": {"type": "number", "default": 120},
                    "match_radius_arcsec": {"type": "number", "default": 1.0},
                    "columns_left": {"type": "array", "items": {"type": "string"}},
                    "columns_right": {"type": "array", "items": {"type": "string"}},
                    "max_rows": {"type": "integer", "default": 500},
                },
                "required": ["left_catalog_key", "right_catalog_key", "ra", "dec"],
            },
            category="mmu_hats",
        ))

        # -- Live imagery and external spectra (hips2fits, ZTF, NED, SparCL) --
        self.tool_registry.register(Tool(
            name="hips_cutout",
            description=(
                "Fetch a live CDS hips2fits PNG cutout for 'show me X', 'what does X look like', "
                "or multiwavelength postage-stamp requests. Supports aliases optical/dss/dss2, sdss, "
                "2mass/nir, wise/mir, galex/uv, rosat/xray, fermi/gamma, vlass/radio, or raw HiPS IDs. "
                "Use this for broad survey imagery; hips2fits exposes roughly 1000 HiPS surveys."
            ),
            function=self._hips_cutout,
            parameters={
                "type": "object",
                "properties": {
                    "target_name": {"type": "string", "description": "Resolve this target name to RA/Dec if ra/dec are not supplied."},
                    "ra": {"type": "number", "description": "ICRS right ascension in degrees."},
                    "dec": {"type": "number", "description": "ICRS declination in degrees."},
                    "survey": {"type": "string", "default": "optical", "description": "Survey alias or raw HiPS ID. Aliases: optical/dss/dss2, dss2_red, sdss, 2mass/nir, wise/mir, galex/uv, rosat/xray, fermi/gamma, vlass/radio."},
                    "fov_deg": {"type": "number", "default": 0.25},
                    "width": {"type": "integer", "default": 512},
                },
                "required": [],
            },
            category="archive",
        ))
        self.tool_registry.register(Tool(
            name="hips_multiband_panel",
            description=(
                "Render a multi-panel CDS hips2fits survey view for appearance or multiwavelength "
                "postage-stamp requests. Defaults to optical, 2MASS, and WISE; failed panels are labeled."
            ),
            function=self._hips_multiband_panel,
            parameters={
                "type": "object",
                "properties": {
                    "target_name": {"type": "string"},
                    "ra": {"type": "number"},
                    "dec": {"type": "number"},
                    "surveys": {"type": "array", "items": {"type": "string"}, "default": ["optical", "2mass", "wise"]},
                    "fov_deg": {"type": "number", "default": 0.25},
                },
                "required": [],
            },
            category="archive",
        ))
        self.tool_registry.register(Tool(
            name="vlass_cutout",
            description="Fetch a VLASS 3 GHz radio-continuum cutout via hips2fits. Use for radio appearance; VLASS covers Dec > -40 deg only.",
            function=self._vlass_cutout,
            parameters={
                "type": "object",
                "properties": {
                    "target_name": {"type": "string"},
                    "ra": {"type": "number"},
                    "dec": {"type": "number"},
                    "fov_deg": {"type": "number", "default": 0.1},
                },
                "required": [],
            },
            category="archive",
        ))
        self.tool_registry.register(Tool(
            name="search_ztf_alerts",
            description="Search ALeRCE/ZTF alert objects near a target or sky position for transients and variability; returns a UI table of object IDs and detection metadata.",
            function=self._search_ztf_alerts,
            parameters={
                "type": "object",
                "properties": {
                    "target_name": {"type": "string"},
                    "ra": {"type": "number"},
                    "dec": {"type": "number"},
                    "radius_arcsec": {"type": "number", "default": 120},
                    "max_rows": {"type": "integer", "default": 25},
                },
                "required": [],
            },
            category="archive",
        ))
        self.tool_registry.register(Tool(
            name="ztf_light_curve",
            description="Plot an ALeRCE/ZTF light curve for an alert object oid, including detections and non-detection limits.",
            function=self._ztf_light_curve,
            parameters={"type": "object", "properties": {"oid": {"type": "string"}}, "required": ["oid"]},
            category="analysis",
        ))
        self.tool_registry.register(Tool(
            name="ztf_stamps",
            description="Render ALeRCE/ZTF science, template, and difference stamp PNG panels for an alert object oid and optional candid.",
            function=self._ztf_stamps,
            parameters={
                "type": "object",
                "properties": {"oid": {"type": "string"}, "candid": {"type": "string"}},
                "required": ["oid"],
            },
            category="analysis",
        ))
        self.tool_registry.register(Tool(
            name="search_space_lightcurves",
            description="List available TESS/Kepler/K2 light curves for a target; returns a table whose index can be used with plot_space_lightcurve or period_search.",
            function=self._search_space_lightcurves,
            parameters={
                "type": "object",
                "properties": {
                    "target": {"type": "string"},
                    "mission": {"type": "string", "enum": ["TESS", "Kepler", "K2"]},
                    "max_rows": {"type": "integer", "default": 20},
                },
                "required": ["target"],
            },
            category="archive",
        ))
        self.tool_registry.register(Tool(
            name="plot_space_lightcurve",
            description="Download and plot a TESS/Kepler/K2 light curve for a target.",
            function=self._plot_space_lightcurve,
            parameters={
                "type": "object",
                "properties": {
                    "target": {"type": "string"},
                    "mission": {"type": "string", "enum": ["TESS", "Kepler", "K2"]},
                    "index": {"type": "integer", "default": 0},
                },
                "required": ["target"],
            },
            category="analysis",
        ))
        self.tool_registry.register(Tool(
            name="period_search",
            description="Lomb-Scargle period search plus phase-folded plot on a TESS/Kepler light curve or a ZTF object by ALeRCE oid; returns best period, FAP, and top alternatives.",
            function=self._period_search,
            parameters={
                "type": "object",
                "properties": {
                    "source": {"type": "string", "enum": ["tess", "kepler", "ztf"]},
                    "identifier": {"type": "string"},
                    "mission": {"type": "string", "enum": ["TESS", "Kepler", "K2"]},
                    "min_period_d": {"type": "number", "default": 0.05},
                    "max_period_d": {"type": "number", "default": 30.0},
                    "fid": {"type": "integer"},
                    "index": {"type": "integer", "default": 0},
                },
                "required": ["source", "identifier"],
            },
            category="analysis",
        ))
        self.tool_registry.register(Tool(
            name="ned_sed_plot",
            description="Plot a literature SED from NED photometry for a named target.",
            function=self._ned_sed_plot,
            parameters={"type": "object", "properties": {"target_name": {"type": "string"}}, "required": ["target_name"]},
            category="analysis",
        ))
        self.tool_registry.register(Tool(
            name="sparcl_find_spectra",
            description="Search NOIRLab SparCL for actual DESI/SDSS optical spectra near a target or position, not just redshift catalog values.",
            function=self._sparcl_find_spectra,
            parameters={
                "type": "object",
                "properties": {
                    "target_name": {"type": "string"},
                    "ra": {"type": "number"},
                    "dec": {"type": "number"},
                    "radius_arcsec": {"type": "number", "default": 60},
                    "data_release": {"type": "array", "items": {"type": "string"}},
                    "limit": {"type": "integer", "default": 20},
                },
                "required": [],
            },
            category="archive",
        ))
        self.tool_registry.register(Tool(
            name="sparcl_plot_spectrum",
            description="Plot an actual SparCL optical spectrum by sparcl_id with optional model overlay and redshifted line markers.",
            function=self._sparcl_plot_spectrum,
            parameters={
                "type": "object",
                "properties": {
                    "sparcl_id": {"type": "string"},
                    "mark_lines": {"type": "boolean", "default": True},
                    "smooth": {"type": "integer", "default": 0},
                },
                "required": ["sparcl_id"],
            },
            category="analysis",
        ))
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
            name="search_papers_by_observation_id",
            description=(
                "Find NASA ADS papers explicitly connected to a specific archive identifier. "
                "Use this instead of generic search_papers when the user provides an ALMA project/proposal code "
                "(e.g. 2019.1.00123.S), MOUS/member_ous_uid (uid://...), ASDM UID, or archive dataset ID. "
                "The lookup uses exact identifier searches and returns provenance metadata for the graph."
            ),
            function=lambda identifier, max_results=20, facility="ALMA", **kw: (
                self._search_papers_by_observation_identifier(
                    identifier=identifier,
                    max_results=max_results,
                    facility=facility,
                )
                if self.ads_client else {"error": "ADS client not configured"}
            ),
            parameters={
                "type": "object",
                "properties": {
                    "identifier": {"type": "string", "description": "Project/proposal code, MOUS UID, ASDM UID, or archive dataset identifier."},
                    "facility": {"type": "string", "description": "Facility/bibgroup hint, default ALMA."},
                    "max_results": {"type": "integer", "description": "Number of results to return (default 20, max 50)"},
                },
                "required": ["identifier"]
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
            name="overlay_archive_images",
            description=(
                "End-to-end archive image overlay workflow. Queries MAST/JWST for a "
                "background FITS image and ALMA/DataLink for contour FITS near a named "
                "region, WCS-aligns them, and renders a PNG. Use for requests like "
                "'Overlay ALMA contours on JWST image for HUDF'."
            ),
            function=self._overlay_archive_images,
            parameters={
                "type": "object",
                "properties": {
                    "region": {"type": "string", "description": "Named region or source, e.g. HUDF, M87, HH 212. Can also contain decimal RA/Dec."},
                    "ra_deg": {"type": "number", "description": "Optional ICRS right ascension in degrees. Use with dec_deg for arbitrary regions."},
                    "dec_deg": {"type": "number", "description": "Optional ICRS declination in degrees. Use with ra_deg for arbitrary regions."},
                    "base_archive": {"type": "string", "description": "Base image archive, default MAST."},
                    "base_collection": {"type": "string", "description": "Base collection/mission, default JWST."},
                    "contour_archive": {"type": "string", "description": "Contour archive, default ALMA."},
                    "radius_arcmin": {"type": "number", "description": "Search radius around the region center. Default 1."},
                    "max_product_mb": {"type": "number", "description": "Maximum FITS product size to select. Default 150."},
                },
                "required": ["region"]
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
            name="triage_alma_data_products",
            description=(
                "Discover and triage ALMA deliverable data products. Use this when the user asks "
                "to fetch, list, inspect, or triage ALMA FITS/data products. Exact project/proposal "
                "codes, MOUS UIDs, ASDM UIDs, and dataset IDs are routed directly. Generic target "
                "names are treated as ambiguous: the tool returns a project-code picker table and "
                "asks the user to choose before product triage. If a project picker was just shown, "
                "row-number replies such as '#4' or 'use number 4' are resolved to that project. "
                "It lists DataLink products and reads "
                "remote FITS headers only; it does not download large science files."
            ),
            function=self._triage_alma_data_products,
            parameters={
                "type": "object",
                "properties": {
                    "identifier_or_target": {
                        "type": "string",
                        "description": "ALMA project code, MOUS UID, dataset ID, or target name from the user request."
                    },
                    "band": {
                        "type": "string",
                        "description": "Optional ALMA band preference, e.g. '6' or 'Band 7'."
                    },
                    "max_projects": {
                        "type": "integer",
                        "description": "Optional max project-code options to show for ambiguous target searches. By default all matched projects are shown."
                    },
                    "max_mous": {
                        "type": "integer",
                        "description": "Max MOUS datasets to inspect for exact IDs. Default 5."
                    },
                    "max_products": {
                        "type": "integer",
                        "description": "Max product rows to list. Default 40."
                    },
                    "max_header_checks": {
                        "type": "integer",
                        "description": "Max FITS products to inspect with remote header reads. Default 6."
                    },
                },
                "required": ["identifier_or_target"]
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

        self.tool_registry.register(Tool(
            name="survey_coverage",
            description="List surveys/datasets whose sky coverage (MOC) includes a position or region; use first for data-existence/coverage questions before broad archive searches, with optional dataproduct_type, regime, or keyword filters.",
            function=self._survey_coverage,
            parameters={
                "type": "object",
                "properties": {
                    "target_name": {"type": "string", "description": "Astronomical target name to resolve, e.g. NGC 253 or 3C 273."},
                    "ra": {"type": "number", "description": "Right ascension in decimal degrees (ICRS). Used with dec if target_name is omitted."},
                    "dec": {"type": "number", "description": "Declination in decimal degrees (ICRS). Used with ra if target_name is omitted."},
                    "radius_deg": {"type": "number", "description": "Cone radius in degrees. Default 0 for a point query.", "default": 0.0},
                    "dataproduct_type": {"type": "string", "enum": ["image", "catalog", "cube"], "description": "Optional product type filter."},
                    "keyword": {"type": "string", "description": "Optional case-insensitive substring filter on dataset ID and title."},
                    "regime": {"type": "string", "enum": ["radio", "mm/sub-mm", "infrared", "optical", "UV", "X-ray", "gamma"], "description": "Optional wavelength-regime filter derived from em_min/em_max."},
                    "max_rows": {"type": "integer", "description": "Maximum rows to return, capped at 200.", "default": 50},
                },
                "required": []
            },
            category="archive"
        ))

        self.tool_registry.register(Tool(
            name="survey_covers_position",
            description="Check whether a named survey (e.g. VLASS, SDSS, GLEAM) covers a given position; returns covered true/false and matching dataset IDs.",
            function=self._survey_covers_position,
            parameters={
                "type": "object",
                "properties": {
                    "survey_keyword": {"type": "string", "description": "Survey name or keyword to match in MOCServer dataset IDs/titles."},
                    "target_name": {"type": "string", "description": "Astronomical target name to resolve."},
                    "ra": {"type": "number", "description": "Right ascension in decimal degrees (ICRS). Used with dec if target_name is omitted."},
                    "dec": {"type": "number", "description": "Declination in decimal degrees (ICRS). Used with ra if target_name is omitted."},
                },
                "required": ["survey_keyword"]
            },
            category="archive"
        ))

        self.tool_registry.register(Tool(
            name="galactic_extinction",
            description=(
                "Galactic dust reddening E(B-V) (SFD98 + Schlafly-Finkbeiner 2011) "
                "and per-band extinction A_lambda at a sky position -- use before any "
                "photometric correction, color, or distance-modulus work."
            ),
            function=self._galactic_extinction,
            parameters={
                "type": "object",
                "properties": {
                    "target_name": {"type": "string", "description": "Astronomical target name to resolve, e.g. M87 or 3C 273."},
                    "ra": {"type": "number", "description": "Right ascension in decimal degrees (ICRS). Used with dec if target_name is omitted."},
                    "dec": {"type": "number", "description": "Declination in decimal degrees (ICRS). Used with ra if target_name is omitted."},
                    "bands": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional extinction bands. Exact keys include V, sdss_r, ps1_g, J, Ks, W1. Omit for all supported bands."
                    },
                },
                "required": []
            },
            category="archive"
        ))

        self.tool_registry.register(Tool(
            name="gaia_distance",
            description="Bailer-Jones (2021) geometric/photogeometric distances for Gaia DR3 sources near a position -- the correct way to turn parallax into distance for stars.",
            function=self._gaia_distance,
            parameters={
                "type": "object",
                "properties": {
                    "target_name": {"type": "string", "description": "Astronomical target name to resolve, e.g. Barnard's Star."},
                    "ra": {"type": "number", "description": "Right ascension in decimal degrees (ICRS). Used with dec if target_name is omitted."},
                    "dec": {"type": "number", "description": "Declination in decimal degrees (ICRS). Used with ra if target_name is omitted."},
                    "radius_arcsec": {"type": "number", "description": "Cone radius in arcseconds, capped at 300. Default 10; named-target default calls expand to 30 for high-proper-motion tolerance.", "default": 10},
                    "max_rows": {"type": "integer", "description": "Maximum rows to return, capped at 50.", "default": 10},
                },
                "required": []
            },
            category="archive"
        ))

        self.tool_registry.register(Tool(
            name="ned_distance",
            description="NED redshift-independent distance measurements (Cepheids, TRGB, SNIa, ...) for a named galaxy, with median summary.",
            function=self._ned_distance,
            parameters={
                "type": "object",
                "properties": {
                    "target_name": {"type": "string", "description": "Named galaxy, e.g. M83 or NGC 253."},
                },
                "required": ["target_name"]
            },
            category="archive"
        ))

        self.tool_registry.register(Tool(
            name="velocity_frame_distance",
            description="Convert a heliocentric velocity or redshift to GSR, Local Group, and CMB frames and give Hubble-flow distances (Planck18 H0) -- use for nearby-galaxy distances and flow corrections.",
            function=self._velocity_frame_distance,
            parameters={
                "type": "object",
                "properties": {
                    "target_name": {"type": "string", "description": "Astronomical target name to resolve."},
                    "ra": {"type": "number", "description": "Right ascension in decimal degrees (ICRS). Used with dec if target_name is omitted."},
                    "dec": {"type": "number", "description": "Declination in decimal degrees (ICRS). Used with ra if target_name is omitted."},
                    "v_helio_kms": {"type": "number", "description": "Heliocentric velocity in km/s. Provide exactly one of v_helio_kms or z."},
                    "z": {"type": "number", "description": "Redshift converted relativistically to heliocentric velocity. Provide exactly one of v_helio_kms or z."},
                },
                "required": []
            },
            category="analysis"
        ))

        self.tool_registry.register(Tool(
            name="search_pulsars",
            description="Search the ATNF pulsar catalogue around a sky position; returns period, DM, 1400 MHz flux, distance, binarity, associations.",
            function=self._search_pulsars,
            parameters={
                "type": "object",
                "properties": {
                    "target_name": {"type": "string", "description": "Astronomical target name to resolve, e.g. Crab Nebula."},
                    "ra": {"type": "number", "description": "Right ascension in decimal degrees (ICRS). Used with dec if target_name is omitted."},
                    "dec": {"type": "number", "description": "Declination in decimal degrees (ICRS). Used with ra if target_name is omitted."},
                    "radius_deg": {"type": "number", "description": "Cone radius in degrees, capped at 30. Default 1.", "default": 1.0},
                    "max_rows": {"type": "integer", "description": "Maximum rows to return, capped at 200. Default 25.", "default": 25},
                },
                "required": []
            },
            category="archive"
        ))

        self.tool_registry.register(Tool(
            name="pulsar_lookup",
            description="Look up a pulsar by J/B name in the ATNF catalogue and return its full timing/derived parameters.",
            function=self._pulsar_lookup,
            parameters={
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Pulsar J-name or B-name, with or without the PSR prefix."},
                },
                "required": ["name"]
            },
            category="archive"
        ))
        self.tool_registry.register(Tool(
            name="solar_system_ephemeris",
            description="JPL Horizons ephemeris for a planet, asteroid, or comet over a date range: RA/Dec, observer/heliocentric distances, magnitude, and elongation.",
            function=self._solar_system_ephemeris,
            parameters={
                "type": "object",
                "properties": {
                    "target": {"type": "string", "description": "Solar-system body name or designation, e.g. 'Ceres', 'Mars', '2020 SO'."},
                    "start": {"type": "string", "description": "Start date/time, ISO, e.g. '2026-07-03'."},
                    "stop": {"type": "string", "description": "Stop date/time, ISO, e.g. '2026-07-08'."},
                    "step": {"type": "string", "description": "Step size, e.g. '1d', '6h', '30m'.", "default": "1d"},
                },
                "required": ["target", "start", "stop"]
            },
            category="archive"
        ))
        self.tool_registry.register(Tool(
            name="moving_object_check",
            description="List known asteroids/comets inside a field at a given epoch (IMCCE SkyBoT) — use to check whether a transient or odd detection is a known moving object.",
            function=self._moving_object_check,
            parameters={
                "type": "object",
                "properties": {
                    "target_name": {"type": "string", "description": "Target name to resolve for the field centre."},
                    "ra": {"type": "number", "description": "RA in decimal degrees (ICRS); used with dec if target_name is omitted."},
                    "dec": {"type": "number", "description": "Dec in decimal degrees (ICRS); used with ra if target_name is omitted."},
                    "radius_deg": {"type": "number", "description": "Cone radius in degrees (<=10).", "default": 0.2},
                    "epoch": {"type": "string", "description": "Epoch (ISO UTC or JD); defaults to now."},
                },
                "required": []
            },
            category="archive"
        ))
        self.tool_registry.register(Tool(
            name="radio_sed",
            description=(
                "Compile a compact-source radio continuum SED from TGSS, GLEAM, SUMSS, NVSS, and FIRST catalog fluxes; "
                "fit the spectral index alpha (S~nu^alpha), plot it, and return flags for resolution/epoch caveats."
            ),
            function=self._radio_sed,
            parameters={
                "type": "object",
                "properties": {
                    "target_name": {"type": "string", "description": "Astronomical target name to resolve, e.g. 3C 273 or M87."},
                    "ra": {"type": "number", "description": "Right ascension in decimal degrees (ICRS). Used with dec if target_name is omitted."},
                    "dec": {"type": "number", "description": "Declination in decimal degrees (ICRS). Used with ra if target_name is omitted."},
                    "radius_arcsec": {"type": "number", "description": "Cone-search radius in arcsec, capped at 120. Default 30.", "default": 30.0},
                },
                "required": []
            },
            category="analysis"
        ))
        self.tool_registry.register(Tool(
            name="monitor_add_target",
            description="Add a sky position to the standing ZTF-alert watchlist; Quasar remembers it across sessions and reports only NEW alerts on each check.",
            function=self._monitor_add_target,
            parameters={
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Label for the watchlist entry, e.g. 'SN 2026abc field'."},
                    "target_name": {"type": "string", "description": "Astronomical name to resolve for the position (defaults to `name`)."},
                    "ra": {"type": "number", "description": "RA in decimal degrees (ICRS); used with dec if no name resolves."},
                    "dec": {"type": "number", "description": "Dec in decimal degrees (ICRS)."},
                    "radius_arcsec": {"type": "number", "description": "Match radius in arcsec, capped at 600.", "default": 120},
                    "note": {"type": "string", "description": "Optional free-text note."},
                },
                "required": ["name"]
            },
            category="analysis"
        ))
        self.tool_registry.register(Tool(
            name="monitor_list_targets",
            description="List the standing sky-monitor watchlist with hit counts and last-checked times.",
            function=self._monitor_list_targets,
            parameters={"type": "object", "properties": {}, "required": []},
            category="analysis"
        ))
        self.tool_registry.register(Tool(
            name="monitor_remove_target",
            description="Remove a sky-monitor watchlist entry (and its recorded alerts) by id.",
            function=self._monitor_remove_target,
            parameters={
                "type": "object",
                "properties": {"target_id": {"type": "integer", "description": "Watchlist entry id from monitor_list_targets."}},
                "required": ["target_id"]
            },
            category="analysis"
        ))
        self.tool_registry.register(Tool(
            name="monitor_check_now",
            description="Check the watchlist (or one target) against ALeRCE/ZTF NOW and report only alerts that are new since the previous check.",
            function=self._monitor_check_now,
            parameters={
                "type": "object",
                "properties": {"target_id": {"type": "integer", "description": "Optional: check only this watchlist entry."}},
                "required": []
            },
            category="analysis"
        ))
        self.tool_registry.register(Tool(
            name="vo_find_services",
            description="Discover Virtual Observatory services (TAP/SIA/SSA/cone) by keyword and waveband — finds archives Quasar has no built-in client for. Follow with vo_list_tables / vo_adql_query on the access_url.",
            function=self._vo_find_services,
            parameters={
                "type": "object",
                "properties": {
                    "keywords": {"type": "string", "description": "Search keywords, e.g. 'HI 21cm survey' or 'GLEAM'."},
                    "service_type": {"type": "string", "enum": ["tap", "sia", "ssa", "scs"], "description": "Optional service type filter."},
                    "waveband": {"type": "string", "description": "Optional waveband filter, e.g. 'radio', 'x-ray'."},
                    "max_rows": {"type": "integer", "description": "Max services to return, capped at 100.", "default": 30},
                },
                "required": ["keywords"]
            },
            category="archive"
        ))
        self.tool_registry.register(Tool(
            name="vo_list_tables",
            description="List (and keyword-filter) the tables of any TAP service found via vo_find_services.",
            function=self._vo_list_tables,
            parameters={
                "type": "object",
                "properties": {
                    "access_url": {"type": "string", "description": "TAP service base URL."},
                    "keyword": {"type": "string", "description": "Substring filter on table name/description — strongly recommended for big services like VizieR."},
                    "max_tables": {"type": "integer", "description": "Max tables to list, capped at 200.", "default": 50},
                },
                "required": ["access_url"]
            },
            category="archive"
        ))
        self.tool_registry.register(Tool(
            name="vo_describe_table",
            description="Column schema (names, datatypes, units, UCDs) of a table on any TAP service — call before writing ADQL.",
            function=self._vo_describe_table,
            parameters={
                "type": "object",
                "properties": {
                    "access_url": {"type": "string", "description": "TAP service base URL."},
                    "table_name": {"type": "string", "description": "Exact table name from vo_list_tables."},
                },
                "required": ["access_url", "table_name"]
            },
            category="archive"
        ))
        self.tool_registry.register(Tool(
            name="vo_adql_query",
            description="Run a guarded SELECT-only ADQL query against any TAP service URL. On ADQL errors the server's message is returned - read it and fix the query.",
            function=self._vo_adql_query,
            parameters={
                "type": "object",
                "properties": {
                    "access_url": {"type": "string", "description": "TAP service base URL."},
                    "adql": {"type": "string", "description": "SELECT-only ADQL. Quote table names containing '/' or '+' in double quotes."},
                    "max_rows": {"type": "integer", "description": "Row cap (service cap also applies).", "default": 200},
                },
                "required": ["access_url", "adql"]
            },
            category="archive"
        ))
        self.tool_registry.register(Tool(
            name="vo_cone_search",
            description="Cone search any VO simple-cone-search service by position.",
            function=self._vo_cone_search,
            parameters={
                "type": "object",
                "properties": {
                    "access_url": {"type": "string", "description": "SCS service base URL."},
                    "target_name": {"type": "string", "description": "Target name to resolve."},
                    "ra": {"type": "number", "description": "RA in decimal degrees (ICRS)."},
                    "dec": {"type": "number", "description": "Dec in decimal degrees (ICRS)."},
                    "radius_deg": {"type": "number", "description": "Cone radius in degrees, capped at 5.", "default": 0.1},
                    "max_rows": {"type": "integer", "description": "Row cap.", "default": 100},
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

    def _route_alma_science_archive_query(self, query: str) -> Optional[Dict[str, Any]]:
        """Map known hard ALMA science prompts to deterministic tool args."""
        raw = str(query or "")
        q = raw.lower()
        cycle_match = re.search(r"\bcycle\s+(\d{1,2})\b", q)
        cycle = int(cycle_match.group(1)) if cycle_match else None

        if cycle is not None and re.search(r"\b(?:sun|solar)\b", q):
            return {"query_type": "cycle_solar_projects", "cycle": cycle}

        if cycle is not None and all(term in q for term in ("12m", "7m")) and re.search(r"total\s+power|\btp\b", q):
            return {"query_type": "cycle_array_combo_projects", "cycle": cycle, "arrays": ["12m", "7m", "TP"]}

        if re.search(r"\bhh\s*212\b", q) and re.search(r"band\s*7|\bb7\b", q):
            return {
                "query_type": "high_resolution_band_data",
                "target": "HH 212",
                "band": 7,
                "max_resolution_arcsec": 0.1,
            }

        if all(term in q for term in ("12co", "13co", "c18o")):
            args: Dict[str, Any] = {
                "query_type": "line_set_projects",
                "band": 6,
                "lines": ["12CO", "13CO", "C18O"],
                "require_same_project": True,
            }
            if re.search(r"protostellar|proto-stellar|disk", q):
                args["topic_filter"] = "protostellar disks"
            return args

        z_match = re.search(r"\bz\s*[=~]?\s*(\d+(?:\.\d+)?)\s*(?:-|to|\u2013)\s*(\d+(?:\.\d+)?)", q)
        if z_match and re.search(r"\bco\b|carbon monoxide|rest frequenc", q):
            return {
                "query_type": "redshifted_line_projects",
                "redshift_min": float(z_match.group(1)),
                "redshift_max": float(z_match.group(2)),
                "rest_species": "CO",
                "science_category": "Galaxy",
                "require_same_project": True,
            }

        if re.search(r"bandwidth\s+switching|spectral\s+setup", q):
            args: Dict[str, Any] = {"query_type": "bandwidth_switching_candidates"}
            if cycle is not None:
                args["cycle"] = cycle
            return args

        return None

    # ── Astronomy acronym dictionary for web search disambiguation ─────
    def _route_cross_archive_source_match_query(self, query: str) -> Optional[Dict[str, Any]]:
        """Map custom coordinate cross-match prompts to deterministic tool args."""
        raw = str(query or "")
        q = raw.lower()
        archive_aliases = {
            "alma": "ALMA",
            "jwst": "JWST",
            "hst": "HST",
            "mast": "MAST",
            "tess": "TESS",
            "kepler": "KEPLER",
            "k2": "K2",
            "galex": "GALEX",
            "swift": "SWIFT",
        }
        archives: List[str] = []
        for needle, label in archive_aliases.items():
            if re.search(rf"\b{re.escape(needle)}\b", q) and label not in archives:
                archives.append(label)
        if not archives:
            return None

        has_match_intent = bool(re.search(r"\bcross[- ]?match|crossmatch|match\b", q))
        has_coverage_intent = bool(re.search(r"\bboth\b|all requested|every archive|in all\b|which\b|coverage", q))
        has_catalog_signal = bool(re.search(r"\bperseus\b|\bprotostar|\bsource\b|\bRA\s*\d", raw, flags=re.IGNORECASE))
        if not (has_match_intent or (has_coverage_intent and has_catalog_signal)):
            return None

        radius_arcsec = 5.0
        radius_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:arcsec|arcsecond|arcseconds|as|['\"]{2})\b", q)
        if radius_match:
            radius_arcsec = float(radius_match.group(1))

        sources: List[Dict[str, Any]] = []
        source_re = re.compile(
            r"(?P<name>[A-Za-z0-9_.+/-][A-Za-z0-9_.+/-]*(?:\s+[A-Za-z0-9_.+/-]+){0,4})"
            r"\s+(?:at\s+)?RA\s*[=:]?\s*(?P<ra>\d+(?:\.\d+)?)"
            r"\s*(?:,|\s)+Dec\s*[=:]?\s*(?P<dec>[+-]?\d+(?:\.\d+)?)",
            flags=re.IGNORECASE,
        )
        for match in source_re.finditer(raw):
            name = re.sub(
                r"^(?:and|with|sources|catalog|target)\s+",
                "",
                match.group("name").strip(" ,.;:"),
                flags=re.IGNORECASE,
            ).strip()
            name = re.sub(r"\s+at$", "", name, flags=re.IGNORECASE).strip() or f"source_{len(sources) + 1}"
            try:
                ra = float(match.group("ra"))
                dec = float(match.group("dec"))
            except (TypeError, ValueError):
                continue
            sources.append({"source_name": name, "ra": ra, "dec": dec})

        args: Dict[str, Any] = {
            "archives": archives,
            "radius_arcsec": radius_arcsec,
            "require_all_archives": bool(re.search(r"\bboth\b|all requested|every archive|in all\b", q)),
        }
        if sources:
            args["catalog_name"] = "inline"
            args["sources"] = sources
            args["max_sources"] = len(sources)
        elif "perseus" in q and "protostar" in q:
            args["catalog_name"] = "perseus_protostars"
            args["require_all_archives"] = True
        else:
            return None
        return args

    def _ensure_alma_project_picker_state(self) -> None:
        if not hasattr(self, "_alma_project_picker_by_conversation"):
            self._alma_project_picker_by_conversation = {}
        if not hasattr(self, "_alma_project_picker_lock"):
            self._alma_project_picker_lock = threading.Lock()

    def _current_conversation_id(self) -> str:
        return str(getattr(getattr(self, "_tls", None), "current_conversation_id", "") or "")

    def _store_alma_project_picker(
        self,
        picker: pd.DataFrame,
        *,
        target: str = "",
        band: str = "",
    ) -> None:
        conversation_id = self._current_conversation_id()
        if not conversation_id or picker is None or not hasattr(picker, "empty") or picker.empty:
            return
        self._ensure_alma_project_picker_state()
        with self._alma_project_picker_lock:
            self._alma_project_picker_by_conversation[conversation_id] = {
                "target": target,
                "band": band,
                "projects": picker.to_dict("records"),
            }

    def _clear_alma_project_picker(self) -> None:
        conversation_id = self._current_conversation_id()
        if not conversation_id:
            return
        self._ensure_alma_project_picker_state()
        with self._alma_project_picker_lock:
            self._alma_project_picker_by_conversation.pop(conversation_id, None)

    def _has_pending_alma_project_picker(self, conversation_id: str) -> bool:
        if not conversation_id:
            return False
        self._ensure_alma_project_picker_state()
        with self._alma_project_picker_lock:
            pending = self._alma_project_picker_by_conversation.get(conversation_id)
        return bool(pending and pending.get("projects"))

    @staticmethod
    def _parse_project_picker_selection_index(text: str) -> Optional[int]:
        clean = str(text or "").strip().lower()
        if not clean:
            return None

        ordinal_map = {
            "first": 1,
            "second": 2,
            "third": 3,
            "fourth": 4,
            "fifth": 5,
            "sixth": 6,
            "seventh": 7,
            "eighth": 8,
            "ninth": 9,
            "tenth": 10,
        }
        ordinal_match = re.search(
            r"\b(?:use|choose|select|pick|triage|fetch|inspect)?\s*"
            r"(first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth)\b",
            clean,
        )
        if ordinal_match:
            return ordinal_map[ordinal_match.group(1)]

        bare_number = re.fullmatch(r"#?\s*(\d{1,3})", clean)
        if bare_number:
            return int(bare_number.group(1))

        numbered = re.search(
            r"\b(?:#|row|option|number|no\.?|use|choose|select|pick|triage|fetch|inspect)\s*#?\s*(\d{1,3})\b",
            clean,
        )
        if numbered:
            return int(numbered.group(1))

        return None

    def _resolve_alma_project_picker_selection(self, text: str) -> str:
        conversation_id = self._current_conversation_id()
        if not conversation_id:
            return ""
        index = self._parse_project_picker_selection_index(text)
        if index is None:
            return ""

        self._ensure_alma_project_picker_state()
        with self._alma_project_picker_lock:
            pending = self._alma_project_picker_by_conversation.get(conversation_id) or {}
            projects = list(pending.get("projects") or [])

        if index < 1 or index > len(projects):
            return ""
        return str(projects[index - 1].get("proposal_id") or "").strip()

    @log_tool
    def _triage_alma_data_products(
        self,
        identifier_or_target: str,
        band: str = "",
        max_projects: int = None,
        max_mous: int = 5,
        max_products: int = 40,
        max_header_checks: int = 6,
    ) -> Dict[str, Any]:
        """Discover ALMA products and perform safe remote FITS-header triage."""
        request = classify_alma_product_request(identifier_or_target)
        if request.get("kind") == "target":
            selected_project = self._resolve_alma_project_picker_selection(identifier_or_target)
            if selected_project:
                request = {
                    "kind": "project_code",
                    "identifier": selected_project,
                    "target": "",
                    "band": request.get("band", ""),
                }
        band_value = str(band or request.get("band") or "").replace("Band", "").replace("band", "").strip()
        max_projects = max(1, min(int(max_projects), 500)) if max_projects else None
        max_mous = max(1, min(int(max_mous or 5), 20))
        max_products = max(1, min(int(max_products or 40), 200))
        max_header_checks = max(0, min(int(max_header_checks or 6), 20))

        kind = request.get("kind", "target")
        identifier = request.get("identifier", "")
        target = request.get("target", "") or str(identifier_or_target or "").strip()

        try:
            if kind == "mous_uid":
                self._clear_alma_project_picker()
                return self._triage_alma_mous_products(
                    mous_uids=[identifier],
                    label=identifier,
                    band=band_value,
                    max_products=max_products,
                    max_header_checks=max_header_checks,
                )

            if kind in {"project_code", "dataset_id"}:
                observations = self._search_alma_observations_for_identifier(kind, identifier)
                lookup_label = identifier
            else:
                observations = self.search_service.search_by_target(target, facility="ALMA", max_results=200)
                lookup_label = target

            if observations is None or not hasattr(observations, "empty") or observations.empty:
                return {
                    "success": False,
                    "mode": "no_observations",
                    "query": identifier_or_target,
                    "message": f"No public ALMA observations were found for {lookup_label}.",
                }

            if band_value:
                observations = filter_observations_by_band(observations, band_value)
                if observations.empty:
                    return {
                        "success": False,
                        "mode": "no_band_match",
                        "query": identifier_or_target,
                        "band": band_value,
                        "message": f"No ALMA observations for {lookup_label} matched Band {band_value}.",
                    }

            if kind == "target":
                picker = summarize_project_options(observations, max_projects=max_projects)
                project_count = len(picker) if picker is not None else 0
                if project_count != 1:
                    self._store_alma_project_picker(picker, target=target, band=band_value)
                    self.last_search_results = observations
                    self.last_run_result = {
                        "type": "data",
                        "data": picker,
                        "source": "ALMA Project Picker",
                        "filter_label": f"ALMA project options for {target}" + (f" Band {band_value}" if band_value else ""),
                        "tool_name": "triage_alma_data_products",
                        "table_kind": "alma_project_picker",
                    }
                    return {
                        "success": True,
                        "mode": "needs_project_selection",
                        "target": target,
                        "band": band_value or None,
                        "project_count": project_count,
                        "message": (
                            f"I found {project_count} possible ALMA project codes for {target}"
                            + (f" in Band {band_value}" if band_value else "")
                            + ". Ask the user to choose one project code before fetching products."
                        ),
                        "project_options": picker.to_dict("records") if picker is not None else [],
                    }

                identifier = str(picker.iloc[0]["proposal_id"])
                lookup_label = identifier

            self._clear_alma_project_picker()
            proposal_id = self._best_observation_value(observations, ["proposal_id", "project_code"]) or identifier
            target_name = self._best_observation_value(observations, ["target_name"]) or target
            mous_uids = unique_values(observations, ["member_ous_uid"], limit=max_mous)
            observation_metadata_by_mous: Dict[str, Dict[str, Any]] = {}
            if "member_ous_uid" in observations.columns:
                for _, row in observations.iterrows():
                    mous_uid = str(row.get("member_ous_uid") or "").strip()
                    if not mous_uid or mous_uid in observation_metadata_by_mous:
                        continue
                    observation_metadata_by_mous[mous_uid] = {
                        "scan_intent": row.get("scan_intent", ""),
                        "qa2_passed": row.get("qa2_passed", ""),
                    }
            if not mous_uids:
                self.last_search_results = observations
                self.last_run_result = {
                    "type": "data",
                    "data": observations,
                    "source": "ALMA observations",
                    "filter_label": f"ALMA observations for {lookup_label}",
                    "tool_name": "triage_alma_data_products",
                }
                return {
                    "success": False,
                    "mode": "no_mous_uid",
                    "project_code": proposal_id,
                    "observation_count": len(observations),
                    "message": "Observation rows were found, but no MOUS UID was available for DataLink product discovery.",
                }

            return self._triage_alma_mous_products(
                mous_uids=mous_uids,
                label=lookup_label,
                band=band_value,
                proposal_id=proposal_id,
                target_name=target_name,
                observation_count=len(observations),
                observation_metadata_by_mous=observation_metadata_by_mous,
                max_products=max_products,
                max_header_checks=max_header_checks,
            )
        except Exception as e:
            return {"success": False, "mode": "error", "error": str(e), "query": identifier_or_target}

    def _search_alma_observations_for_identifier(self, kind: str, identifier: str) -> pd.DataFrame:
        clean = str(identifier or "").strip()
        if not clean:
            return pd.DataFrame()

        safe = clean.replace("'", "''")
        if kind == "project_code":
            direct_query = (
                "SELECT TOP 500 * FROM ivoa.obscore "
                f"WHERE proposal_id = '{safe}' OR obs_publisher_did LIKE '%{safe}%'"
            )
        else:
            direct_query = (
                "SELECT TOP 500 * FROM ivoa.obscore "
                f"WHERE obs_publisher_did = '{safe}' OR obs_publisher_did LIKE '%{safe}%'"
            )
        try:
            alma_client = getattr(self.search_service, "alminer_client", None)
            if alma_client is not None and hasattr(alma_client, "_get_tap_service"):
                service = alma_client._get_tap_service()
                result = service.search(direct_query)
                df = result.to_table().to_pandas()
                if df is not None and not df.empty:
                    if hasattr(alma_client, "_standardize_columns"):
                        return alma_client._standardize_columns(df)
                    return df
        except Exception as tap_err:
            print(f"[ALMA product triage] Direct TAP identifier query failed: {tap_err}")

        keyword_attempts: List[Dict[str, Any]] = []
        if kind == "project_code":
            keyword_attempts.extend([
                {"project_code": clean},
                {"proposal_id": clean},
            ])
        elif kind == "dataset_id":
            keyword_attempts.append({"obs_publisher_did": clean})

        for keywords in keyword_attempts:
            df = self.search_service.search_alma_with_keywords(keywords)
            if df is not None and hasattr(df, "empty") and not df.empty:
                return df

        return self.search_service.advanced_search(direct_query)

    def _triage_alma_mous_products(
        self,
        mous_uids: List[str],
        label: str,
        band: str = "",
        proposal_id: str = "",
        target_name: str = "",
        observation_count: int = 0,
        observation_metadata_by_mous: Optional[Dict[str, Dict[str, Any]]] = None,
        max_products: int = 40,
        max_header_checks: int = 6,
    ) -> Dict[str, Any]:
        product_rows: List[Dict[str, Any]] = []
        all_files: List[Dict[str, Any]] = []
        errors: List[str] = []

        for mous_uid in mous_uids:
            result = self.datalink_client.list_files(mous_uid=mous_uid)
            if not result.get("success"):
                errors.append(f"{mous_uid}: {result.get('error', 'DataLink query failed')}")
                continue
            for file_info in result.get("files", []):
                enriched = dict(file_info)
                enriched["_mous_uid"] = mous_uid
                all_files.append(enriched)

        ranked_files = sorted(all_files, key=product_rank)
        selected_files = ranked_files[:max_products]

        header_checks = 0
        header_summaries: List[Dict[str, Any]] = []
        for file_info in selected_files:
            metadata = None
            if header_checks < max_header_checks and is_fits_product(file_info) and file_info.get("access_url"):
                metadata = self.fits_service.extract_metadata_from_url(str(file_info["access_url"]))
                header_checks += 1
                header_summaries.append({
                    "filename": file_info.get("filename"),
                    "success": bool(metadata.get("success")) if isinstance(metadata, dict) else False,
                    "object_name": metadata.get("object_name") if isinstance(metadata, dict) else None,
                    "beam_major_arcsec": metadata.get("beam_major_arcsec") if isinstance(metadata, dict) else None,
                    "beam_minor_arcsec": metadata.get("beam_minor_arcsec") if isinstance(metadata, dict) else None,
                    "rest_freq_ghz": metadata.get("rest_freq_ghz") if isinstance(metadata, dict) else None,
                    "image_size": metadata.get("image_size") if isinstance(metadata, dict) else None,
                    "bunit": metadata.get("bunit") if isinstance(metadata, dict) else None,
                })
            observation_metadata = (observation_metadata_by_mous or {}).get(str(file_info.get("_mous_uid") or ""), {})
            product_rows.append(build_product_row(
                file_info,
                member_ous_uid=str(file_info.get("_mous_uid") or ""),
                proposal_id=proposal_id,
                target_name=target_name,
                scan_intent=str(observation_metadata.get("scan_intent") or ""),
                qa2_passed=observation_metadata.get("qa2_passed", ""),
                metadata=metadata,
            ))

        product_df = pd.DataFrame(product_rows)
        if not product_df.empty:
            self.last_run_result = {
                "type": "data",
                "data": product_df,
                "source": "ALMA Data Products",
                "filter_label": f"ALMA products for {label}" + (f" Band {band}" if band else ""),
                "tool_name": "triage_alma_data_products",
                "table_kind": "alma_products",
            }

        fits_count = sum(1 for item in all_files if is_fits_product(item))
        large_count = sum(1 for item in all_files if float(item.get("size_mb") or 0) > 500)
        return {
            "success": True,
            "mode": "triage",
            "label": label,
            "project_code": proposal_id or None,
            "target_name": target_name or None,
            "band": band or None,
            "observation_count": observation_count,
            "mous_checked": len(mous_uids),
            "total_products_found": len(all_files),
            "products_listed": len(product_rows),
            "fits_products_found": fits_count,
            "header_checks": header_checks,
            "large_products_over_500mb": large_count,
            "header_summaries": header_summaries,
            "errors": errors,
            "message": (
                f"Found {len(all_files)} ALMA DataLink product(s) across {len(mous_uids)} MOUS dataset(s); "
                f"listed {len(product_rows)} and inspected {header_checks} FITS header(s) without downloading full files."
            ),
            "safety": "No large science files were downloaded. Header checks used remote FITS header reads only.",
        }

    @staticmethod
    def _best_observation_value(df: pd.DataFrame, columns: List[str]) -> str:
        values = unique_values(df, columns, limit=1)
        return values[0] if values else ""

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
        web_data = sanitize_web_payload(web_data)
        filter_meta = web_data.get("content_filter") if isinstance(web_data, dict) else {}
        if (
            isinstance(filter_meta, dict)
            and filter_meta.get("filtered")
            and not web_data.get("results")
        ):
            return str(filter_meta.get("notice") or FILTER_NOTICE)

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
            "If the snippets do not contain information that helps answer the "
            "question, reply with exactly NO_RELEVANT_INFO and nothing else. "
            "Do NOT repeat background context the user already knows. "
            "Do NOT include generic descriptions of organizations or telescopes."
        )

        user_prompt = (
            f"User question: {query}\n\n"
            f"Web search snippets:\n{context_block}\n\n"
            f"Write a concise, directly relevant summary:"
        )

        try:
            from core.llm_client import LLMClient
            synthesis_model = os.getenv("QUASAR_WEB_SYNTHESIS_MODEL") or os.getenv("QUASAR_FAST_MODEL", "gpt-4.1-mini")
            client = LLMClient(model=synthesis_model)
            resp = client.responses.create(
                model=synthesis_model,
                instructions=system_prompt,
                input=user_prompt,
                max_output_tokens=200,
                temperature=0.2,
            )
            summary = resp.output_text.strip()
            if "NO_RELEVANT_INFO" in summary.upper():
                # Web results don't answer the question — suppress the
                # "From the Web" section entirely rather than appending a
                # "the snippets do not provide..." non-answer.
                print("[WEB SEARCH] Synthesis judged web snippets irrelevant; omitting web section")
                return ""
            if summary:
                return safe_assistant_text(summary)
        except Exception as e:
            print(f"[WEB SEARCH] Summary synthesis failed: {e}")

        # Fallback to raw Tavily answer
        return safe_assistant_text(web_data.get("answer", "").strip())

    def _synthesize_web_tool_answer(self, query: str, web_results: List[Dict[str, Any]]) -> str:
        """Create a final answer when web tools returned sources but the model emitted no text."""
        web_results = [
            sanitize_web_payload(result)
            for result in web_results
            if isinstance(result, dict)
        ]
        if web_results and all(
            isinstance(result.get("content_filter"), dict)
            and result["content_filter"].get("filtered")
            and not result.get("results")
            and not result.get("sources")
            for result in web_results
        ):
            return FILTER_NOTICE

        snippets = []
        for web_data in web_results[:4]:
            provider = web_data.get("provider", "Web")
            answer = str(web_data.get("answer") or web_data.get("content") or "").strip()
            if answer:
                snippets.append(f"{provider} answer: {answer[:900]}")

            result_items = web_data.get("results", [])
            if not isinstance(result_items, list):
                response = web_data.get("response")
                result_items = response.get("results", []) if isinstance(response, dict) else []

            for result in result_items[:6]:
                if not isinstance(result, dict):
                    continue
                title = str(result.get("title") or result.get("url") or "Source").strip()
                url = str(result.get("url") or result.get("link") or "").strip()
                snippet = str(
                    result.get("snippet")
                    or result.get("content")
                    or result.get("raw_content")
                    or result.get("text")
                    or ""
                ).strip()
                if snippet:
                    snippets.append(f"[{title}]({url}): {snippet[:700]}")

        if not snippets:
            return ""

        context_block = "\n\n".join(snippets)[:9000]
        system_prompt = (
            "You are a careful astronomy research assistant. Use only the provided web "
            "and documentation snippets. Answer the user's question directly. If the "
            "evidence is incomplete, say what is incomplete. Include source links inline "
            "where URLs are provided."
        )
        user_prompt = (
            f"User question: {query}\n\n"
            f"Retrieved source snippets:\n{context_block}\n\n"
            "Write a concise but useful final answer."
        )

        try:
            from core.llm_client import LLMClient
            synthesis_model = os.getenv("QUASAR_WEB_SYNTHESIS_MODEL") or os.getenv("QUASAR_FAST_MODEL", "gpt-4.1-mini")
            client = LLMClient(model=synthesis_model)
            resp = client.responses.create(
                model=synthesis_model,
                instructions=system_prompt,
                input=user_prompt,
                max_output_tokens=700,
                temperature=0.2,
            )
            answer = resp.output_text.strip()
            if answer:
                return safe_assistant_text(answer)
        except Exception as e:
            print(f"[WEB SEARCH] Tool-answer synthesis failed: {e}")

        return (
            "I found relevant web sources, but the model did not synthesize a final answer. "
            "Open the source cards below for the retrieved material."
        )

    @log_tool
    def _tavily_web_search(self, query: str, max_results: int = 10, search_depth: str = "basic") -> Dict[str, Any]:
        """
        Real-time web search routed dynamically between Brave, Tavily, and Exa.
        Returns source URLs + related images for ChatGPT-style inline display.
        Falls back to BrowserService if keys are unavailable.
        """
        # ── Langfuse: create a child span for this tool call ──
        from core.llm_client import get_langfuse_parent
        parent = get_langfuse_parent()
        lf_span = None
        if parent:
            try:
                lf_span = parent.span(
                    name="tool: web_search",
                    input=query,
                    metadata={"max_results": max_results, "search_depth": search_depth}
                )
            except Exception:
                pass

        ret_val = None
        try:
            from services.web_search_service import WebSearchService
            search_service = WebSearchService(browser_service=self.browser_service)
            ret_val = search_service.route_and_search(
                query=query,
                max_results=max_results,
                search_depth=search_depth
            )
        except Exception as e:
            print(f"[SEARCH ROUTER] Router execution failed: {e}. Falling back to basic BrowserService.")

        # -- Ultimate Fallback: BrowserService --
        if not ret_val or not ret_val.get("success"):
            try:
                search_query = self._expand_astro_query(query)
                fallback = self.browser_service.web_search(query=search_query)
                fallback_results = (
                    fallback.get("results", [])
                    if isinstance(fallback, dict)
                    else fallback
                )
                if not isinstance(fallback_results, list):
                    fallback_results = []
                ret_val = {
                    "success": True,
                    "provider": "BrowserService (fallback)",
                    "query": query,
                    "results": fallback_results,
                    "raw_text": fallback.get("raw_text", "") if isinstance(fallback, dict) else "",
                    "images": [],
                }
            except Exception:
                ret_val = {"success": False, "error": "Search failed"}

        # ── Langfuse: end the span with the results ──
        if lf_span:
            try:
                lf_span.end(output=ret_val)
            except Exception:
                pass

        return sanitize_web_payload(ret_val)

    def _web_search_service(self):
        """Create the shared web provider service."""
        from services.web_search_service import WebSearchService

        return WebSearchService(browser_service=self.browser_service)

    def _has_web_provider_key(self) -> bool:
        """Return True when Brave, Tavily, or Exa is configured."""
        from services.web_search_service import WebSearchService

        return WebSearchService.has_any_provider_key()

    def _web_tool_status_label(self, tool_name: str, args: Dict[str, Any]) -> str:
        """Return user-facing tool labels for the Thought panel."""
        query_hint = str(
            args.get("query")
            or args.get("url")
            or args.get("urls")
            or ""
        ).strip()
        query_lower = query_hint.lower()

        if tool_name == "web_search":
            exa_keywords = {
                "compare", "versus", "vs", "formula", "equations", "papers",
                "documentation", "handbook", "innovations", "architecture",
                "literature", "review", "research", "academic",
            }
            is_advanced = (
                args.get("search_depth") == "advanced"
                or any(keyword in query_lower for keyword in exa_keywords)
            )
            label = (
                "Calling advanced web search agent"
                if is_advanced
                else "Calling web search agent"
            )
        elif tool_name == "web_extract_url":
            label = "Extracting web source"
        elif tool_name == "web_map_site":
            label = "Mapping website"
        elif tool_name == "web_crawl_site":
            label = "Crawling website"
        elif tool_name == "web_research":
            label = "Calling deep web research agent"
        elif tool_name == "web_research_status":
            label = "Checking web research status"
        else:
            label = f"Calling tool: {tool_name}"

        if query_hint:
            label += f' ("{query_hint[:60]}")'
        return label

    def _tool_status_label(self, tool_name: str, args: Dict[str, Any]) -> str:
        """Return archive-aware status labels for the live run phase UI."""
        if tool_name.startswith("web_") or tool_name == "web_search":
            return self._web_tool_status_label(tool_name, args)

        target = str(args.get("target_name") or args.get("target") or "").strip()
        query = str(args.get("query") or "").strip()
        mission = str(args.get("mission") or "").strip().upper()
        instrument = str(args.get("instrument") or "").strip().upper()
        archives = args.get("archives") or []
        if isinstance(archives, str):
            archive_label = archives
        else:
            archive_label = " + ".join(str(a).upper() for a in archives if str(a).strip())

        tool_labels = {
            "search_by_target": "Querying ALMA by target",
            "search_by_position": "Querying ALMA cone search",
            "search_by_frequency": "Querying ALMA frequency range",
            "advanced_search": "Running ALMA TAP query",
            "search_alma_with_keywords": "Searching ALMA project metadata",
            "search_alma_co_in_redshift_range": "Searching ALMA CO redshift coverage",
            "find_alma_line_coverage": "Resolving line and checking exact ALMA coverage",
            "query_alma_science_archive": "Querying ALMA Science Archive",
            "triage_alma_data_products": "Inspecting ALMA data products",
            "list_alma_files": "Listing ALMA files",
            "download_alma_data": "Downloading ALMA data",
            "match_cross_archive_sources": f"Cross-matching {archive_label or 'archives'}",
            "match_perseus_protostars_alma_jwst": "Cross-matching Perseus protostars in ALMA + JWST",
            "search_mast": f"Searching {mission or 'MAST'} archive",
            "search_mast_by_criteria": f"Searching {mission or 'MAST'} archive by criteria",
            "get_mast_products": "Listing MAST products",
            "download_mast_data": "Downloading MAST products",
            "search_cadc_archive": "Searching CADC archive",
            "search_eso_archive": "Searching ESO Science Archive",
            "search_irsa": "Searching IRSA catalogs",
            "galactic_extinction": "Querying IRSA dust maps",
            "get_sky_image": "Fetching sky image",
            "overlay_archive_images": "Overlaying archive images",
            "overlay_fits_images": "Overlaying FITS images",
            "render_fits_image": "Rendering FITS image",
            "inspect_fits_header": "Inspecting FITS header",
            "extract_spectrum": "Extracting spectrum",
            "compute_moment_map": "Computing moment map",
            "fit_spectral_line": "Fitting spectral line",
            "search_papers": "Searching astronomy literature",
            "search_papers_by_observation_id": "Searching papers linked to observation",
            "lookup_researcher": "Looking up researcher profile",
            "list_mmu_hats_catalogs": "Listing Multimodal Universe catalogs",
            "search_mmu_hats_catalog": f"Searching Multimodal Universe {args.get('catalog_key', 'catalog')}",
            "crossmatch_mmu_hats_catalogs": "Crossmatching Multimodal Universe catalogs",
            "hips_cutout": f"Fetching {args.get('survey', 'optical')} cutout",
            "hips_multiband_panel": "Fetching multiband HiPS panel",
            "vlass_cutout": "Fetching VLASS 3 GHz cutout",
            "search_ztf_alerts": "Searching ALeRCE/ZTF alerts",
            "search_space_lightcurves": "Searching TESS/Kepler light curves",
            "plot_space_lightcurve": "Plotting space light curve",
            "period_search": "Running Lomb-Scargle period search",
            "search_pulsars": "Searching ATNF pulsar catalogue",
            "pulsar_lookup": "Looking up pulsar parameters",
            "solar_system_ephemeris": "Querying JPL Horizons",
            "moving_object_check": "Checking for moving objects (SkyBoT)",
            "gaia_distance": "Querying Gaia/Bailer-Jones distances",
            "ned_distance": "Fetching NED-D distances",
            "velocity_frame_distance": "Computing velocity-frame corrections",
            "survey_coverage": "Checking sky coverage (MOCServer)",
            "survey_covers_position": "Checking survey footprint",
            "ztf_light_curve": f"Plotting ZTF light curve {args.get('oid', '')}",
            "ztf_stamps": f"Fetching ZTF stamps {args.get('oid', '')}",
            "ned_sed_plot": f"Plotting NED SED {args.get('target_name', '')}",
            "radio_sed": "Compiling radio SED + spectral index",
            "monitor_add_target": "Adding sky-monitor target",
            "monitor_list_targets": "Listing sky-monitor watchlist",
            "monitor_remove_target": "Removing sky-monitor target",
            "monitor_check_now": "Checking watchlist for new ZTF alerts",
            "vo_find_services": "Searching the IVOA registry",
            "vo_list_tables": "Listing TAP service tables",
            "vo_describe_table": "Inspecting table schema",
            "vo_adql_query": "Running ADQL on remote TAP service",
            "vo_cone_search": "Running VO cone search",
            "sparcl_find_spectra": "Searching SparCL spectra",
            "sparcl_plot_spectrum": f"Plotting SparCL spectrum {args.get('sparcl_id', '')}",
        }
        label = tool_labels.get(tool_name, f"Running {tool_name.replace('_', ' ')}")

        hint_parts = []
        if target:
            hint_parts.append(target)
        if query:
            hint_parts.append(query)
        if instrument:
            hint_parts.append(instrument)
        if args.get("ra") is not None and args.get("dec") is not None:
            hint_parts.append(f"RA {args.get('ra')}, Dec {args.get('dec')}")
        if hint_parts:
            label += f' ("{", ".join(hint_parts)[:80]}")'
        return label

    def _execute_tool_with_progress(
        self,
        tool,
        args: Dict[str, Any],
        *,
        tool_name: str,
        step_label: str,
        on_status=None,
        heartbeat_seconds: Optional[float] = None,
    ):
        """Execute a synchronous tool while emitting hidden liveness events."""
        if on_status is None:
            return tool.execute(**args)

        interval = (
            float(heartbeat_seconds)
            if heartbeat_seconds is not None
            else float(os.getenv("TOOL_PROGRESS_HEARTBEAT_SECONDS", "15"))
        )
        interval = max(1.0 if heartbeat_seconds is None else 0.001, interval)
        stopped = threading.Event()

        def emit_heartbeats():
            while not stopped.wait(interval):
                try:
                    on_status(
                        f"__tool_heartbeat__{tool_name}::{step_label}",
                        "meta",
                    )
                except Exception:
                    pass

        heartbeat = threading.Thread(
            target=emit_heartbeats,
            name=f"quasar-tool-heartbeat-{tool_name[:32]}",
            daemon=True,
        )
        heartbeat.start()
        try:
            return tool.execute(**args)
        finally:
            stopped.set()
            heartbeat.join(timeout=min(1.0, interval))

    def _build_web_sources_event(self, web_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Normalize web-tool outputs into the frontend source-card event."""
        web_data = sanitize_web_payload(web_data)
        if not isinstance(web_data, dict) or not web_data.get("success", True):
            return None

        def _normalize_web_url(value: Any) -> str:
            url = str(value or "").strip().strip("<>")
            url = url.rstrip(".,;:)]}'\"")
            if not url:
                return ""
            if url.startswith(("http://", "https://")):
                return url
            if url.startswith("www."):
                return f"https://{url}"
            # Text summaries sometimes include a schemeless URL with a path.
            if re.match(r"^[A-Za-z0-9.-]+\.[A-Za-z]{2,}/\S+$", url):
                return f"https://{url}"
            return ""

        def _title_from_url(url: str) -> str:
            clean = re.sub(r"^https?://", "", url).split("/", 1)[0]
            return clean.replace("www.", "") or url

        def _source_from_item(item: Any) -> Optional[Dict[str, Any]]:
            if isinstance(item, str):
                url = _normalize_web_url(item)
                candidate = {"title": _title_from_url(url), "url": url, "snippet": ""}
                return annotate_web_source_evidence(candidate) if url and is_safe_web_source(candidate) else None
            if not isinstance(item, dict):
                return None
            url = _normalize_web_url(
                item.get("url")
                or item.get("link")
                or item.get("href")
                or item.get("source_url")
                or ""
            )
            if not url:
                return None
            candidate = {
                "title": str(item.get("title") or item.get("name") or _title_from_url(url)).strip(),
                "url": url,
                "snippet": str(
                    item.get("snippet")
                    or item.get("content")
                    or item.get("text")
                    or item.get("description")
                    or ""
                ).strip()[:500],
                "evidenceQuality": item.get("evidenceQuality") or item.get("evidence_quality") or {},
            }
            return annotate_web_source_evidence(candidate) if is_safe_web_source(candidate) else None

        def _image_from_item(item: Any) -> Optional[Dict[str, str]]:
            if isinstance(item, str):
                url = item.strip()
                candidate = {"url": url, "description": ""}
                return candidate if url and is_safe_web_image(candidate) else None
            if not isinstance(item, dict):
                return None
            url = str(item.get("url") or item.get("src") or item.get("image_url") or "").strip()
            if not url:
                return None
            image = {
                "url": url,
                "description": str(
                    item.get("description") or item.get("alt") or item.get("title") or ""
                ).strip(),
            }
            source_url = _normalize_web_url(
                item.get("sourceUrl")
                or item.get("source_url")
                or item.get("sourcePageUrl")
                or item.get("source_page_url")
                or item.get("pageUrl")
                or item.get("page_url")
                or item.get("source")
                or ""
            )
            if source_url:
                image["sourceUrl"] = source_url
            source_title = str(
                item.get("sourceTitle")
                or item.get("source_title")
                or item.get("sourcePageTitle")
                or item.get("source_page_title")
                or item.get("pageTitle")
                or item.get("page_title")
                or ""
            ).strip()
            if source_title:
                image["sourceTitle"] = source_title
            return image if is_safe_web_image(image) else None

        def _source_items_from_text(text: Any) -> List[Dict[str, str]]:
            if not isinstance(text, str) or "." not in text:
                return []
            matches = re.findall(
                r"https?://[^\s<>\]\)\"']+|www\.[^\s<>\]\)\"']+|[A-Za-z0-9.-]+\.[A-Za-z]{2,}/[^\s<>\]\)\"']+",
                text,
            )
            sources_from_text = []
            for match in matches:
                url = _normalize_web_url(match)
                if url:
                    sources_from_text.append({
                        "title": _title_from_url(url),
                        "url": url,
                        "snippet": "",
                    })
            return sources_from_text

        source_items: List[Any] = []
        for key in ("results", "sources"):
            value = web_data.get(key)
            if isinstance(value, list):
                source_items.extend(value)
            elif isinstance(value, dict):
                nested = value.get("results") or value.get("sources")
                if isinstance(nested, list):
                    source_items.extend(nested)

        response = web_data.get("response")
        if isinstance(response, dict):
            for key in ("results", "sources"):
                value = response.get(key)
                if isinstance(value, list):
                    source_items.extend(value)
                elif isinstance(value, dict):
                    nested = value.get("results") or value.get("sources")
                    if isinstance(nested, list):
                        source_items.extend(nested)

        for container in (web_data, response if isinstance(response, dict) else {}):
            for key in ("answer", "content", "raw_text", "text", "summary"):
                source_items.extend(_source_items_from_text(container.get(key)))

        seen_urls = set()
        sources = []
        for item in source_items:
            source = _source_from_item(item)
            if not source or source["url"] in seen_urls:
                continue
            seen_urls.add(source["url"])
            sources.append(source)
        sources = rank_web_sources(sources)

        image_items = web_data.get("images", [])
        if isinstance(response, dict) and not image_items:
            image_items = response.get("images", [])
        images = []
        seen_image_urls = set()
        if isinstance(image_items, list):
            for item in image_items:
                image = _image_from_item(item)
                if image and image["url"] not in seen_image_urls:
                    seen_image_urls.add(image["url"])
                    images.append(image)

        if not sources and not images:
            return None

        provider = (
            web_data.get("provider")
            or (response.get("provider") if isinstance(response, dict) else None)
            or ""
        )
        image_provider = web_data.get("image_provider") or ""
        search_type = web_data.get("search_type") or ""
        query_value = (
            web_data.get("query")
            or web_data.get("url")
            or ", ".join(web_data.get("urls", []) if isinstance(web_data.get("urls"), list) else [])
            or ""
        )

        return {
            "type": "web_sources",
            "sources": sources,
            "images": images,
            "query": query_value,
            "provider": str(provider),
            "image_provider": str(image_provider),
            "search_type": str(search_type),
        }

    @log_tool
    def _tavily_extract_url(
        self,
        urls,
        query: Optional[str] = None,
        chunks_per_source: int = 3,
        extract_depth: str = "basic",
        include_images: bool = False,
        content_format: str = "markdown",
    ) -> Dict[str, Any]:
        """Extract clean content from specific URLs via Tavily."""
        return self._web_search_service().extract_tavily(
            urls=urls,
            query=query,
            chunks_per_source=chunks_per_source,
            extract_depth=extract_depth,
            include_images=include_images,
            content_format=content_format,
        )

    @log_tool
    def _tavily_map_site(
        self,
        url: str,
        instructions: Optional[str] = None,
        max_depth: int = 1,
        max_breadth: int = 20,
        limit: int = 100,
        select_paths: Optional[List[str]] = None,
        exclude_paths: Optional[List[str]] = None,
        select_domains: Optional[List[str]] = None,
        exclude_domains: Optional[List[str]] = None,
        allow_external: bool = False,
    ) -> Dict[str, Any]:
        """Discover URLs on a website via Tavily Map."""
        return self._web_search_service().map_tavily(
            url=url,
            instructions=instructions,
            max_depth=max_depth,
            max_breadth=max_breadth,
            limit=limit,
            select_paths=select_paths,
            exclude_paths=exclude_paths,
            select_domains=select_domains,
            exclude_domains=exclude_domains,
            allow_external=allow_external,
        )

    @log_tool
    def _tavily_crawl_site(
        self,
        url: str,
        instructions: Optional[str] = None,
        chunks_per_source: int = 3,
        max_depth: int = 1,
        max_breadth: int = 20,
        limit: int = 20,
        extract_depth: str = "basic",
        content_format: str = "markdown",
        include_images: bool = False,
        select_paths: Optional[List[str]] = None,
        exclude_paths: Optional[List[str]] = None,
        select_domains: Optional[List[str]] = None,
        exclude_domains: Optional[List[str]] = None,
        allow_external: bool = False,
    ) -> Dict[str, Any]:
        """Crawl and extract a bounded website section via Tavily Crawl."""
        return self._web_search_service().crawl_tavily(
            url=url,
            instructions=instructions,
            chunks_per_source=chunks_per_source,
            max_depth=max_depth,
            max_breadth=max_breadth,
            limit=limit,
            extract_depth=extract_depth,
            content_format=content_format,
            include_images=include_images,
            select_paths=select_paths,
            exclude_paths=exclude_paths,
            select_domains=select_domains,
            exclude_domains=exclude_domains,
            allow_external=allow_external,
        )

    @log_tool
    def _tavily_research(
        self,
        research_input: str,
        model: str = "auto",
        citation_format: str = "numbered",
        wait_for_completion: bool = True,
        timeout_seconds: int = 120,
    ) -> Dict[str, Any]:
        """Run Tavily Research and return a cited report when available."""
        return self._web_search_service().research_tavily(
            research_input=research_input,
            model=model,
            citation_format=citation_format,
            wait_for_completion=wait_for_completion,
            timeout_seconds=timeout_seconds,
        )

    @log_tool
    def _tavily_research_status(self, request_id: str) -> Dict[str, Any]:
        """Fetch a Tavily Research task by request_id."""
        return self._web_search_service().get_tavily_research_status(request_id=request_id)

    @staticmethod
    def _filter_by_scan_intent(results: pd.DataFrame, scan_intent: Optional[Any]) -> Tuple[pd.DataFrame, str]:
        """Filter ALMA rows by the archive scan_intent column when requested."""
        raw = str(scan_intent or "").strip()
        if not raw or results is None or results.empty:
            return results, ""

        intent_col = next((c for c in ["scan_intent", "Scan Intent", "intent"] if c in results.columns), None)
        if not intent_col:
            return results, ""

        known_intents = re.findall(
            r"\b(TARGET|BANDPASS|PHASE|FLUX|WVR|CHECK|POINTING|FOCUS|AMPLITUDE|ATMOSPHERE)\b",
            raw.upper(),
        )
        requested = known_intents or [part.strip().upper() for part in re.split(r"[,;/]+|\s+and\s+", raw) if part.strip()]
        requested = [part for part in requested if part and part not in {"ONLY", "EXCLUDE", "EXCLUDING", "CALIBRATORS"}]
        if not requested:
            return results, ""

        mask = results[intent_col].astype(str).str.upper().apply(
            lambda value: any(intent in value for intent in requested)
        )
        label = "Scan Intent " + ",".join(requested)
        return results[mask].copy(), label

    @log_tool
    def _search_by_position(self, ra: float, dec: float, radius: float = 0.5,
                           facility: Optional[str] = None,
                           band: Optional[str] = None,
                           scan_intent: Optional[str] = None,
                           max_results: int = 100, **kwargs) -> Dict[str, Any]:
        """Search archives by sky position.
        
        Accepts `band` and extra kwargs so the LLM can pass them without
        crashing, even though the underlying cone_search doesn't use them.
        Band filtering is applied as a post-filter on the results.
        """
        facility_label = (facility or "ALMA").strip().upper()
        if facility_label in ("EVLA", "JVLA"):
            facility_label = "VLA"
        if facility_label not in ("VLA", "VLBA", "GBT"):
            facility_label = "ALMA"

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

            scan_filter_label = ""
            if facility_label == "ALMA" and scan_intent:
                before = len(results)
                results, scan_filter_label = self._filter_by_scan_intent(results, scan_intent)
                if scan_filter_label:
                    print(f"[FILTER] {scan_filter_label}: {before} → {len(results)} rows")

            self.last_search_results = results
            self.last_run_result = {
                "type": "data",
                "data": results,
                "source": facility_label,
                "filter_label": f"{facility_label} › position" + (f" [{scan_filter_label}]" if scan_filter_label else ""),
                "tool_name": "search_by_position",
            }

            # Include top MOUS UIDs + access URLs so Conductor subtasks
            # can use them for list_alma_files / render_fits_image.
            top_mous = []
            if hasattr(results, 'columns') and 'member_ous_uid' in results.columns:
                top_mous = results['member_ous_uid'].dropna().unique()[:5].tolist()
            top_urls = []
            if hasattr(results, 'columns') and 'access_url' in results.columns:
                top_urls = results['access_url'].dropna().head(5).tolist()

            # Extract unique project codes
            top_projects = []
            if hasattr(results, 'columns'):
                project_col = next((col for col in ["proposal_id", "project_code", "Project", "obs_publisher_did"] if col in results.columns), None)
                if project_col:
                    raw_projects = results[project_col].dropna().astype(str).str.strip().unique()
                    import re as _re_proj
                    proj_regex = _re_proj.compile(r"\b\d{4}\.\d\.\d{5}\.[A-Za-z]\b")
                    for p in raw_projects:
                        match = proj_regex.search(p)
                        if match:
                            code = match.group(0).upper()
                            if code not in top_projects:
                                top_projects.append(code)
                        elif len(p) >= 10 and '.' in p:
                            p_upper = p.upper()
                            if p_upper not in top_projects:
                                top_projects.append(p_upper)
                    top_projects = top_projects[:3]

            return {
                "success": True,
                "total_results": len(results),
                "ra": ra, "dec": dec, "radius_deg": radius,
                "top_mous_uids": top_mous,
                "top_access_urls": top_urls,
                "top_project_codes": top_projects,
                "filters_applied": [scan_filter_label] if scan_filter_label else [],
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
                          scan_intent: Optional[str] = None,
                          public_only: bool = False) -> Dict[str, Any]:
        """Search ALMA (default) or NRAO VLA/VLBA/GBT archives by target name,
        with optional native post-filters."""
        # ── Normalize facility for routing + result labeling ──
        facility_label = (facility or "ALMA").strip().upper()
        if facility_label in ("EVLA", "JVLA"):
            facility_label = "VLA"
        if facility_label not in ("VLA", "VLBA", "GBT"):
            facility_label = "ALMA"

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
                self.last_run_result = {"type": "data", "data": results, "source": facility_label, "tool_name": "search_by_target"}
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
            if facility_label == "ALMA" and scan_intent:
                before = len(results)
                results, scan_filter_label = self._filter_by_scan_intent(results, scan_intent)
                if scan_filter_label:
                    filter_parts.append(scan_filter_label)
                    print(f"[FILTER] {scan_filter_label}: {before} → {len(results)} rows")

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
                    filter_label = f"{facility_label} › {target_name} [{band_label}" + (", ".join([""] + filter_parts) if filter_parts else "") + "]"
                else:
                    # None of the bands matched — show unfiltered
                    filter_label = f"{facility_label} › {target_name}"
                    if filter_parts:
                        filter_label += " [" + ", ".join(filter_parts) + "]"

                self.last_search_results = results
                self.last_run_result = {
                    "type": "data", "data": results,
                    "source": facility_label, "filter_label": filter_label,
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

                filter_label = f"{facility_label} › {target_name}"
                if filter_parts:
                    filter_label += " [" + ", ".join(filter_parts) + "]"
                self.last_search_results = results
                self.last_run_result = {
                    "type": "data", "data": results,
                    "source": facility_label, "filter_label": filter_label,
                    "tool_name": "search_by_target"
                }
            else:
                # No band filter
                filter_label = f"{facility_label} › {target_name}"
                if filter_parts:
                    filter_label += " [" + ", ".join(filter_parts) + "]"
                self.last_search_results = results
                self.last_run_result = {
                    "type": "data", "data": results,
                    "source": facility_label, "filter_label": filter_label,
                    "tool_name": "search_by_target"
                }

            # Compact summary + top MOUS UIDs & URLs for Conductor subtask chaining
            top_mous = []
            if 'member_ous_uid' in results.columns:
                top_mous = results['member_ous_uid'].dropna().unique()[:5].tolist()
            top_urls = []
            if 'access_url' in results.columns:
                top_urls = results['access_url'].dropna().head(5).tolist()

            # Extract unique project codes
            top_projects = []
            if hasattr(results, 'columns'):
                project_col = next((col for col in ["proposal_id", "project_code", "Project", "obs_publisher_did"] if col in results.columns), None)
                if project_col:
                    raw_projects = results[project_col].dropna().astype(str).str.strip().unique()
                    import re as _re_proj
                    proj_regex = _re_proj.compile(r"\b\d{4}\.\d\.\d{5}\.[A-Za-z]\b")
                    for p in raw_projects:
                        match = proj_regex.search(p)
                        if match:
                            code = match.group(0).upper()
                            if code not in top_projects:
                                top_projects.append(code)
                        elif len(p) >= 10 and '.' in p:
                            p_upper = p.upper()
                            if p_upper not in top_projects:
                                top_projects.append(p_upper)
                    top_projects = top_projects[:3]

            return {
                "success": True,
                "total_results": len(results),
                "filters_applied": filter_parts,
                "target": target_name,
                "top_mous_uids": top_mous,
                "top_access_urls": top_urls,
                "top_project_codes": top_projects,
                "note": f"Found {len(results)} observations matching your constraints. Full data with sky previews shown in UI table. Do NOT render a table — the UI already displays one."
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    @log_tool
    def _search_by_frequency(self, min_freq_ghz: float, max_freq_ghz: float,
                            facility: Optional[str] = None,
                            max_results: int = 100) -> Dict[str, Any]:
        """Search archives by frequency range"""
        facility_label = (facility or "ALMA").strip().upper()
        if facility_label in ("EVLA", "JVLA"):
            facility_label = "VLA"
        if facility_label not in ("VLA", "VLBA", "GBT"):
            facility_label = "ALMA"

        try:
            results = self.search_service.search_by_frequency(
                min_freq_ghz, max_freq_ghz, facility, max_results
            )
            self.last_search_results = results
            self.last_run_result = {"type": "data", "data": results,
                                    "source": facility_label,
                                    "filter_label": f"{facility_label} › {min_freq_ghz}–{max_freq_ghz} GHz",
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


    # Data Lab P0 handlers

    def _get_datalab_client(self) -> DatalabClient:
        if not hasattr(self, "_datalab_client_instance"):
            self._datalab_client_instance = DatalabClient()
        return self._datalab_client_instance

    def _get_datalab_result_store(self):
        if not hasattr(self, "_datalab_result_store_instance"):
            self._datalab_result_store_instance = default_result_store()
        return self._datalab_result_store_instance

    def _get_mmu_hats_service(self):
        if not hasattr(self, "_mmu_hats_service_instance"):
            self._mmu_hats_service_instance = default_mmu_hats_service()
        return self._mmu_hats_service_instance

    def _get_hips_image_service(self):
        if not hasattr(self, "_hips_image_service_instance"):
            from services.hips_images import HipsImageService

            self._hips_image_service_instance = HipsImageService()
        return self._hips_image_service_instance

    def _get_alerce_client(self):
        if not hasattr(self, "_alerce_client_instance"):
            from services.alerce_client import AlerceClient

            self._alerce_client_instance = AlerceClient()
        return self._alerce_client_instance

    def _get_radio_sed_service(self):
        if not hasattr(self, "_radio_sed_service_instance"):
            from services.radio_sed import RadioSedService

            self._radio_sed_service_instance = RadioSedService()
        return self._radio_sed_service_instance

    def _get_sky_monitor_service(self):
        if not hasattr(self, "_sky_monitor_service_instance"):
            from services.sky_monitor import SkyMonitorService

            self._sky_monitor_service_instance = SkyMonitorService()
        return self._sky_monitor_service_instance

    def _get_vo_registry_service(self):
        if not hasattr(self, "_vo_registry_service_instance"):
            from services.vo_registry import VoRegistryService

            self._vo_registry_service_instance = VoRegistryService()
        return self._vo_registry_service_instance

    def _get_lightcurve_suite(self):
        if not hasattr(self, "_lightcurve_suite_instance"):
            from services.lightcurve_suite import LightCurveSuite

            self._lightcurve_suite_instance = LightCurveSuite()
        return self._lightcurve_suite_instance
    def _get_pulsar_catalog_service(self):
        if not hasattr(self, "_pulsar_catalog_service_instance"):
            from services.pulsar_catalog import PulsarCatalogService

            self._pulsar_catalog_service_instance = PulsarCatalogService()
        return self._pulsar_catalog_service_instance

    def _get_solar_system_service(self):
        if not hasattr(self, "_solar_system_service_instance"):
            from services.solar_system import SolarSystemService

            self._solar_system_service_instance = SolarSystemService()
        return self._solar_system_service_instance

    def _get_distance_service(self):
        if not hasattr(self, "_distance_service_instance"):
            from services.distance_service import DistanceService

            self._distance_service_instance = DistanceService()
        return self._distance_service_instance

    def _get_dust_extinction_service(self):
        if not hasattr(self, "_dust_extinction_service_instance"):
            from services.dust_extinction import DustExtinctionService

            self._dust_extinction_service_instance = DustExtinctionService()
        return self._dust_extinction_service_instance

    def _get_moc_coverage_service(self):
        if not hasattr(self, "_moc_coverage_service_instance"):
            from services.moc_coverage import MocCoverageService

            self._moc_coverage_service_instance = MocCoverageService()
        return self._moc_coverage_service_instance

    def _get_ned_photometry_service(self):
        if not hasattr(self, "_ned_photometry_service_instance"):
            from services.ned_photometry import NedPhotometryService

            self._ned_photometry_service_instance = NedPhotometryService()
        return self._ned_photometry_service_instance

    def _get_sparcl_spectra_service(self):
        if not hasattr(self, "_sparcl_spectra_service_instance"):
            from services.sparcl_spectra import SparclSpectraService

            self._sparcl_spectra_service_instance = SparclSpectraService()
        return self._sparcl_spectra_service_instance

    def _datalab_list_catalogs(self) -> Dict[str, Any]:
        self.last_run_result = None  # Data Lab tools emit summaries, not stale data cards
        try:
            catalogs = datalab_registry.list_catalogs()
            return {"success": True, "catalogs": catalogs, "count": len(catalogs)}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _datalab_describe_table(self, catalog: str, table: str) -> Dict[str, Any]:
        self.last_run_result = None  # Data Lab tools emit summaries, not stale data cards
        try:
            return {"success": True, "table": datalab_registry.describe_table(catalog, table)}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _datalab_cone_count(self, catalog: str, table: str, ra: float, dec: float, radius_deg: float) -> Dict[str, Any]:
        self.last_run_result = None  # builder may raise before _execute_datalab_sql clears it
        try:
            sql, meta = datalab_query_builders.build_cone_count(catalog, table, ra=ra, dec=dec, radius_deg=radius_deg)
            return self._execute_datalab_sql(sql, meta, tool_name="datalab_cone_count")
        except Exception as e:
            return self._datalab_error(e)

    def _datalab_select_catalog_rows(
        self,
        catalog: str,
        table: str,
        ra: float,
        dec: float,
        radius_deg: float,
        columns: Optional[List[str]] = None,
        limit: int = 500,
        value_cuts: Optional[List[Dict[str, Any]]] = None,
        color_cut: Optional[Dict[str, Any]] = None,
        morphology: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        self.last_run_result = None  # builder may raise before _execute_datalab_sql clears it
        try:
            # Models consistently expect the density-tool cut schema here too
            # (live test RV-P6 guessed value_cuts twice); accept it so selective
            # cuts run server-side instead of failing on an unknown kwarg.
            predicates = None
            if value_cuts or color_cut or morphology:
                predicates = datalab_query_builders.build_catalog_predicates(
                    catalog, table, color_cut=color_cut, value_cuts=value_cuts, morphology=morphology
                )
            sql, meta = datalab_query_builders.build_cone_select(
                catalog, table, ra=ra, dec=dec, radius_deg=radius_deg, columns=columns, limit=limit,
                predicates=predicates,
            )
            return self._execute_datalab_sql(sql, meta, tool_name="datalab_select_catalog_rows")
        except Exception as e:
            return self._datalab_error(e)

    def _datalab_density_aggregate(
        self,
        catalog: str,
        table: str,
        mode: str = "grid",
        step_deg: float = 0.1,
        healpix_column: Optional[str] = None,
        ra: Optional[float] = None,
        dec: Optional[float] = None,
        radius_deg: Optional[float] = None,
        all_sky: bool = False,
        color_cut: Optional[Dict[str, Any]] = None,
        value_cuts: Optional[List[Dict[str, Any]]] = None,
        morphology: Optional[Dict[str, Any]] = None,
        limit: int = 5000,
    ) -> Dict[str, Any]:
        self.last_run_result = None  # builder may raise before _execute_datalab_sql clears it
        try:
            predicates = datalab_query_builders.build_catalog_predicates(
                catalog, table, color_cut=color_cut, value_cuts=value_cuts, morphology=morphology,
            )
            sql, meta = datalab_query_builders.build_density_aggregate(
                catalog, table, mode=mode, step_deg=step_deg, healpix_column=healpix_column,
                ra=ra, dec=dec, radius_deg=radius_deg, all_sky=all_sky, predicates=predicates, limit=limit,
            )
            has_cone = ra is not None and dec is not None and radius_deg is not None
            # Once one aggregate on this table has sync-timed-out this turn,
            # go straight to tiling for further wide cones — the doomed 60s
            # sync attempt per call burned ~3 minutes of live DS-P8's clock.
            timeout_tables = getattr(self, "_datalab_agg_timeout_tables", None)
            if timeout_tables is None:
                timeout_tables = self._datalab_agg_timeout_tables = set()
            table_key = f"{catalog}.{table}".lower()
            skip_sync = has_cone and float(radius_deg) >= 2.0 and table_key in timeout_tables
            if skip_sync:
                out = {"success": False, "error": "sync skipped: earlier aggregate on this table timed out"}
            else:
                out = self._execute_datalab_sql(sql, meta, tool_name="datalab_density_aggregate")
            # Wide-cone sync timeout (anonymous tokens cannot use the async-job
            # path — live P8 both models): auto-tile the cone into sub-cones
            # sized for the 60s window and merge, instead of failing the tool.
            if (
                not out.get("success")
                and ("timed out" in str(out.get("error", "")).lower() or skip_sync)
                and has_cone
                and float(radius_deg) >= 2.0
            ):
                print(
                    f"[DATALAB] density aggregate timed out at radius {radius_deg}° — "
                    f"auto-tiling ({catalog}.{table}, mode={mode}, sync_skipped={skip_sync})"
                )
                # Repeat attempts on a table that already proved slow get a
                # smaller tiling budget, so the model keeps enough turn clock
                # for more probes and the final render (live DS-P8 attempt 3:
                # four 210s probes of the Galactic centre ate the whole 900s).
                _budget_override = 120.0 if table_key in timeout_tables else None
                timeout_tables.add(table_key)
                return datalab_orchestration.tiled_density_aggregate(
                    catalog, table, mode=mode, step_deg=step_deg, healpix_column=healpix_column,
                    ra=float(ra), dec=float(dec), radius_deg=float(radius_deg),
                    predicates=predicates, limit=limit,
                    max_seconds=_budget_override,
                    client=self._get_datalab_client(),
                    result_store=self._get_datalab_result_store(),
                )
            return out
        except Exception as e:
            return self._datalab_error(e)

    def _datalab_q3c_crossmatch(
        self,
        ra: float,
        dec: float,
        radius_deg: float,
        small_catalog: str = "gaia_dr3",
        small_table: str = "gaia_source",
        big_catalog: str = "nsc_dr2",
        big_table: str = "object",
        match_radius_arcsec: float = 1.0,
        small_columns: Optional[List[str]] = None,
        big_columns: Optional[List[str]] = None,
        small_limit: int = 10000,
        limit: int = 500,
    ) -> Dict[str, Any]:
        self.last_run_result = None  # builder may raise before _execute_datalab_sql clears it
        try:
            sql, meta = datalab_query_builders.build_q3c_crossmatch(
                small_catalog=small_catalog,
                small_table=small_table,
                big_catalog=big_catalog,
                big_table=big_table,
                ra=ra,
                dec=dec,
                radius_deg=radius_deg,
                match_radius_arcsec=match_radius_arcsec,
                small_columns=small_columns,
                big_columns=big_columns,
                small_limit=small_limit,
                limit=limit,
            )
            return self._execute_datalab_sql(sql, meta, tool_name="datalab_q3c_crossmatch")
        except Exception as e:
            return self._datalab_error(e)

    def _datalab_sql_query(self, sql: str, expert_ack: bool = False, reason: str = "") -> Dict[str, Any]:
        self.last_run_result = None  # clear before the early expert-ack rejection path too
        try:
            if expert_ack is not True or not str(reason or "").strip():
                return {
                    "success": False,
                    "error": "datalab_sql_query is restricted expert/debug mode and requires expert_ack=true plus a reason.",
                }
            meta = {"source": "expert", "builder": "raw_sql", "expert_reason": str(reason).strip()}
            return self._execute_datalab_sql(sql, meta, tool_name="datalab_sql_query", source="expert")
        except Exception as e:
            return self._datalab_error(e)

    def _execute_datalab_sql(
        self,
        sql: str,
        meta: Dict[str, Any],
        *,
        tool_name: str,
        source: str = "builder",
    ) -> Dict[str, Any]:
        # Data Lab tools return summaries/result_ids, not data cards; clear any prior
        # tool's last_run_result so the streaming loop can't re-emit a stale card.
        self.last_run_result = None
        try:
            validated = datalab_sql_policy.validate(sql, source=source, meta=meta)
            result = self._get_datalab_client().query(sql=validated.sql, fmt="pandas")
            store_meta = {
                **validated.meta,
                "tool_name": tool_name,
                "validated_sql": validated.sql,
                "warnings": validated.warnings,
                "provenance": {
                    **result.provenance,
                    "query": validated.sql,
                    "tool_name": tool_name,
                    "policy_source": source,
                    # HEALPix pixelization travels with the result so renderers
                    # decode with the true scheme/nside, not their defaults.
                    **(
                        {"healpix": validated.meta["healpix"]}
                        if isinstance(validated.meta, dict) and validated.meta.get("healpix")
                        else {}
                    ),
                },
            }
            result_id = self._get_datalab_result_store().put(result.dataframe, store_meta)
            # Preview the most COMPLETE rows first: Data Lab often returns NaN-heavy
            # rows at the top, and a NaN-leading preview misled the model into
            # believing the whole result was NaN (live test DS-P6 burned 3 debug
            # rounds on it). The stored result keeps the original order.
            preview_df = result.dataframe
            preview_reordered = False
            if len(preview_df) > 10:
                _nan_counts = preview_df.isna().sum(axis=1)
                if int(_nan_counts.head(10).sum()) > 0:
                    preview_df = preview_df.loc[_nan_counts.sort_values(kind="stable").index]
                    preview_reordered = True
            preview_rows, preview_more = self._datalab_fit_rows(preview_df, 10, char_budget=4000)
            summary: Dict[str, Any] = {
                "success": True,
                "tool_name": tool_name,
                "result_id": result_id,
                "rowcount": int(len(result.dataframe)),
                "columns": result.columns[:30],
                "warnings": validated.warnings,
                "catalog": validated.meta.get("catalog") or result.provenance.get("catalog"),
                "table": validated.meta.get("table") or result.provenance.get("table"),
                "query_summary": self._datalab_query_summary(validated.sql),
                "preview": preview_rows,
                "preview_truncated": preview_more,
                "note": (
                    "Preview shows the most complete rows (some rows contain NaNs; the full "
                    "result keeps its original order); fetch up to 5000 rows with "
                    "datalab_get_result(result_id)."
                    if preview_reordered
                    else "Preview shows the first rows; fetch up to 5000 rows with datalab_get_result(result_id)."
                ),
            }
            if "row_count" in result.dataframe.columns and not result.dataframe.empty:
                summary["reported_count"] = int(result.dataframe.iloc[0]["row_count"])
            return summary
        except Exception as e:
            return self._datalab_error(e)

    def _datalab_error(self, error: Exception) -> Dict[str, Any]:
        payload = {"success": False, "error": str(error)}
        if isinstance(error, datalab_sql_policy.DatalabPolicyError):
            payload["fix_hint"] = error.fix_hint
            return payload
        text = str(error).lower()
        if "column" in text and "does not exist" in text:
            payload["fix_hint"] = (
                "A column name in the query is wrong. Use the HINT in the error if present, or call "
                "datalab_describe_table(catalog, table) to list the valid columns, then re-run the "
                "corrected query. Do not give up after this error."
            )
        elif "relation" in text and "does not exist" in text:
            payload["fix_hint"] = (
                "The table name is wrong. Call datalab_list_catalogs and datalab_describe_table to "
                "find the correct schema-qualified table, then re-run the query."
            )
        elif "not found for expression" in str(error) or "unsupported expression" in text:
            payload["fix_hint"] = (
                "Fix the expression using the available columns listed in the error, then call the "
                "plot tool again. Derived quantities can be computed inline, e.g. "
                "'phot_g_mean_mag + 5*log10(parallax/100)'."
            )
        return payload

    @staticmethod
    def _datalab_query_summary(sql: str) -> str:
        compact = " ".join(str(sql or "").split())
        return compact[:700] + ("..." if len(compact) > 700 else "")

    @staticmethod
    def _summarize_tool_outcomes(tool_results) -> str:
        """When the model called tools but emitted no final text, summarize what the tools
        did (and surface any errors) so the user gets a useful reply — and so a silent failure
        becomes self-explaining — instead of the dead-end 'didn't generate a text response'."""
        produced, errors, queried = [], [], []
        for tr in (tool_results or []):
            try:
                out = json.loads(tr.get("output") or "{}")
            except Exception:
                continue
            if not isinstance(out, dict):
                continue
            if out.get("error"):
                errors.append(str(out["error"]))
            elif out.get("success"):
                if out.get("image_attached") or out.get("path") or out.get("image_base64"):
                    produced.append("a plot/image")
                elif out.get("coverage_gap"):
                    produced.append("a coverage-gap result (no image at this position)")
                elif out.get("result_id") is not None:
                    cat, tab, rc = out.get("catalog") or "", out.get("table") or "", out.get("rowcount")
                    queried.append((f"{cat}.{tab}".strip(".") or "a Data Lab table") + (f" ({rc} rows)" if rc is not None else ""))
                elif str(out.get("source") or "").startswith("Multimodal Universe"):
                    rc = out.get("rowcount")
                    queried.append(str(out.get("source")) + (f" ({rc} rows)" if rc is not None else ""))
        parts = []
        if queried:
            query_label = "Queried catalogs" if any(str(q).startswith("Multimodal Universe") for q in queried) else "Queried Data Lab"
            parts.append(query_label + ": " + "; ".join(dict.fromkeys(queried)) + ".")
        if produced:
            parts.append("Produced " + "; ".join(dict.fromkeys(produced)) + " (shown above).")
        if errors:
            parts.append("Some steps failed: " + "; ".join(dict.fromkeys(errors))[:500])
        return " ".join(parts)

    @staticmethod
    def _user_facing_provider_error(error: Exception) -> str:
        """Convert a raw provider exception into a message fit for the chat UI.

        Raw litellm/openai error payloads (harmony channel markup, model-group
        dumps) must never appear as the assistant's answer."""
        text = str(error)
        if "Failed to parse tool call" in text or "Invalid function calling output" in text:
            return (
                "The language model produced a malformed tool call that the provider rejected, "
                "so this request could not be completed. This occasionally happens with open "
                "models such as gpt-oss-120b — please resend the question (a retry usually "
                "succeeds), or switch to another model if it persists."
            )
        status = getattr(error, "status_code", None)
        detail = f" (HTTP {status})" if status else ""
        return (
            f"The language-model provider returned an error{detail} and the request could not "
            "be completed. Please try again; if the problem persists, try a different model."
        )

    def _compose_final_answer_from_tools(
        self,
        user_query: str,
        tool_results,
        selected_model: str,
        user_id=None,
        conversation_id=None,
        run_token=None,
    ) -> str:
        """One extra no-tools round that turns raw tool results into a real answer.

        Weaker models sometimes exhaust the tool loop without emitting any final
        text; without this round the user would only see the mechanical step
        summary from _summarize_tool_outcomes."""
        compact = [str(tr.get("output") or "")[:1500] for tr in (tool_results or [])[-12:]]
        compact = [c for c in compact if c]
        if not compact:
            return ""
        prompt = (
            "The user asked:\n" + str(user_query or "") + "\n\n"
            "Tools were already executed for this request. Their JSON results (possibly truncated):\n"
            + "\n".join(compact)
            + "\n\nWrite the final answer to the user's question based ONLY on these results. "
            "Plots and data cards produced by the tools are already displayed above your reply, "
            "so refer to them naturally. If some steps failed, briefly say what failed and answer "
            "with what succeeded. Do not call tools; reply in plain text now."
        )
        request_kwargs = {
            "model": selected_model,
            "input": prompt,
            "instructions": self.system_prompt,
            "temperature": self.config.temperature,
            "max_output_tokens": self.config.max_tokens,
            "stream": False,
            "user_id": user_id,
            "session_id": conversation_id,
        }
        _no_temp = {"o1", "o1-mini", "o1-pro", "o3", "o3-mini", "o3-pro", "o4-mini", "gpt-5-nano", "gpt-5-mini", "gpt-5.4-mini", "deepseek-v4-pro", "deepseek-v4-flash"}
        if selected_model in _no_temp:
            request_kwargs.pop("temperature", None)
        try:
            resp = self.client.responses.create(**request_kwargs)
            text = safe_assistant_text(getattr(resp, "output_text", "") or "").strip()
            # Deliberately do NOT register this response as the conversation
            # head: the prompt above is a standalone summary built WITHOUT
            # previous_response_id, so making it the head would drop the real
            # conversation/tool chain and break follow-ups ("use the second
            # source"). This round only produces display text; the main loop
            # keeps ownership of conversation state.
            return text
        except Exception as final_err:
            print(f"[WARNING] Final-answer composition round failed: {final_err}")
            return ""

    @staticmethod
    def _datalab_fit_rows(frame, max_rows: int, *, char_budget: int = 6000):
        """Return (rows, truncated) trimmed so the JSON stays under char_budget.

        Tool outputs are sliced at 8000 chars downstream; an unbounded row dump
        would be cut mid-structure into invalid JSON. Trimming the row count here
        keeps the payload valid and self-describing via the truncated flag.
        """
        total = int(len(frame))
        if total == 0:
            return [], False
        n = max(1, min(int(max_rows or 1), total))
        rows = frame.head(n).to_dict(orient="records")
        truncated = total > len(rows)
        blob = json.dumps(rows, default=str)
        if len(blob) > char_budget:
            # Estimate how many rows fit (one dump), then trim in small steps —
            # avoids an O(n^2) pop-one-at-a-time loop on large results.
            avg = max(1, len(blob) // max(1, len(rows)))
            rows = rows[: max(1, char_budget // avg)]
            truncated = True
            while len(rows) > 1 and len(json.dumps(rows, default=str)) > char_budget:
                rows = rows[: max(1, len(rows) - 5)]
        # A single very wide row can still exceed the budget (e.g. a long text/description
        # column); clip long string cells so the payload stays valid after the 8000-char slice.
        if rows and len(json.dumps(rows, default=str)) > char_budget:
            ncols = max(1, len(rows[0]))
            per_cell = max(40, char_budget // (len(rows) * ncols))
            rows = [
                {k: (v[:per_cell] + "…" if isinstance(v, str) and len(v) > per_cell else v) for k, v in r.items()}
                for r in rows
            ]
            truncated = True
        return rows, truncated

    def _datalab_get_result(self, result_id: str, max_rows: int = 200) -> Dict[str, Any]:
        """Fetch stored Data Lab rows (size-bounded) for a result_id from a prior tool."""
        self.last_run_result = None
        try:
            res = self._get_datalab_result_store().get(result_id)
            cap = max(1, min(int(max_rows or 200), 5000))
            rows, truncated = self._datalab_fit_rows(res.dataframe, cap)
            return {
                "success": True,
                "result_id": result_id,
                "rowcount": int(len(res.dataframe)),
                "returned_rows": len(rows),
                "columns": res.columns,
                "rows": rows,
                "provenance": res.provenance,
                "truncated": truncated,
            }
        except Exception as e:
            return self._datalab_error(e)

    # MMU/HATS catalog handlers

    def _list_mmu_hats_catalogs(self) -> Dict[str, Any]:
        self.last_run_result = None  # MMU catalog listing emits summaries, not stale data cards
        service = self._get_mmu_hats_service()
        catalogs = service.list_catalogs()
        available, reason = service.is_available()
        return {
            "success": True,
            "enabled": bool(getattr(service, "enabled", True)),
            "available": available,
            "reason": reason or None,
            "catalogs": catalogs,
            "count": len(catalogs),
        }

    def _search_mmu_hats_catalog(
        self,
        catalog_key: Optional[str] = None,
        ra: Optional[float] = None,
        dec: Optional[float] = None,
        radius_arcsec: Optional[float] = None,
        columns: Optional[List[str]] = None,
        max_rows: Optional[int] = None,
        target_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        self.last_run_result = None
        service = self._get_mmu_hats_service()
        if not catalog_key:
            return {"success": False, "error": f"catalog_key is required. Valid catalogs: {sorted(MMU_HATS_CATALOGS)}"}
        if (ra is None or dec is None) and target_name:
            resolved = self._resolve_target(str(target_name))
            if not resolved.get("success"):
                return {"success": False, "error": resolved.get("error") or f"Could not resolve target {target_name!r}"}
            ra = resolved.get("ra_deg")
            dec = resolved.get("dec_deg")
        try:
            result = service.cone_search(
                catalog_key,
                ra=ra,
                dec=dec,
                radius_arcsec=radius_arcsec,
                columns=columns,
                max_rows=max_rows,
            )
        except MMUHatsUnavailableError as e:
            return {"success": False, "unavailable": True, "error": str(e)}
        except MMUHatsError as e:
            return {"success": False, "error": str(e)}

        df = result["dataframe"]
        provenance = result.get("provenance", {})
        cone = provenance.get("cone", {})
        eff_ra = float(cone.get("ra_deg", ra))
        eff_dec = float(cone.get("dec_deg", dec))
        eff_radius = float(cone.get("radius_arcsec", radius_arcsec or 0))
        catalog_label = provenance.get("catalog_label") or str(catalog_key)
        source = f"Multimodal Universe / {catalog_label}"
        self.last_run_result = {
            "type": "data",
            "data": df,
            "source": source,
            "filter_label": f"{catalog_label} › cone RA={eff_ra:.5f}, Dec={eff_dec:.5f}, r={eff_radius:g} arcsec",
            "tool_name": "search_mmu_hats_catalog",
            "table_kind": "mmu_hats",
            "warnings": result.get("warnings", []),
            "partial": bool(result.get("warnings")),
        }
        # Compact nested/oversized cells first: a single embedded spectrum could
        # otherwise push the tool JSON past the 8000-char dispatch slice.
        preview_rows, _ = self._datalab_fit_rows(compact_preview_frame(df), 10, char_budget=4000)
        preview_more = int(len(df)) > len(preview_rows)
        return {
            "success": True,
            "catalog_key": catalog_key,
            "source": source,
            "rowcount": result.get("rowcount", int(len(df))),
            "returned_rows": result.get("returned_rows", int(len(df))),
            "columns": result.get("columns", list(df.columns)),
            "results_preview": preview_rows,
            "preview_truncated": preview_more,
            "warnings": result.get("warnings", []),
            "provenance": provenance,
            "note": "Full table is rendered as a data card in the UI; do not repeat the rows in text.",
        }

    def _crossmatch_mmu_hats_catalogs(
        self,
        left_catalog_key: str,
        right_catalog_key: str,
        ra: float,
        dec: float,
        radius_arcsec: Optional[float] = None,
        match_radius_arcsec: float = 1.0,
        columns_left: Optional[List[str]] = None,
        columns_right: Optional[List[str]] = None,
        max_rows: Optional[int] = None,
    ) -> Dict[str, Any]:
        self.last_run_result = None
        service = self._get_mmu_hats_service()
        try:
            result = service.crossmatch_catalogs(
                left_catalog_key,
                right_catalog_key,
                ra=ra,
                dec=dec,
                radius_arcsec=radius_arcsec,
                match_radius_arcsec=match_radius_arcsec,
                columns_left=columns_left,
                columns_right=columns_right,
                max_rows=max_rows,
            )
        except MMUHatsUnavailableError as e:
            return {"success": False, "unavailable": True, "error": str(e)}
        except MMUHatsError as e:
            return {"success": False, "error": str(e)}

        df = result["dataframe"]
        provenance = result.get("provenance", {})
        cone = provenance.get("cone", {})
        eff_ra = float(cone.get("ra_deg", ra))
        eff_dec = float(cone.get("dec_deg", dec))
        eff_radius = float(cone.get("radius_arcsec", radius_arcsec or 0))
        left_label = provenance.get("left_catalog_label") or str(left_catalog_key)
        right_label = provenance.get("right_catalog_label") or str(right_catalog_key)
        source = f"Multimodal Universe / {left_label} x {right_label}"
        self.last_run_result = {
            "type": "data",
            "data": df,
            "source": source,
            "filter_label": f"{left_label} × {right_label} › crossmatch RA={eff_ra:.5f}, Dec={eff_dec:.5f}, r={eff_radius:g} arcsec",
            "tool_name": "crossmatch_mmu_hats_catalogs",
            "table_kind": "mmu_hats",
            "warnings": result.get("warnings", []),
            "partial": bool(result.get("warnings")),
        }
        preview_rows, _ = self._datalab_fit_rows(compact_preview_frame(df), 10, char_budget=4000)
        preview_more = int(len(df)) > len(preview_rows)
        return {
            "success": True,
            "left_catalog_key": left_catalog_key,
            "right_catalog_key": right_catalog_key,
            "source": source,
            "rowcount": result.get("rowcount", int(len(df))),
            "returned_rows": result.get("returned_rows", int(len(df))),
            "columns": result.get("columns", list(df.columns)),
            "results_preview": preview_rows,
            "preview_truncated": preview_more,
            "warnings": result.get("warnings", []),
            "provenance": provenance,
            "note": "Full table is rendered as a data card in the UI; do not repeat the rows in text.",
        }

    # Data Lab P1 handlers

    def _get_datalab_image_service(self):
        if not hasattr(self, "_datalab_image_service_instance"):
            from services.datalab_image_service import DatalabImageService

            self._datalab_image_service_instance = DatalabImageService()
        return self._datalab_image_service_instance

    def _get_svo_fps_client(self):
        if not hasattr(self, "_svo_fps_client_instance"):
            from integrations.svo_fps_client import SvoFpsClient

            self._svo_fps_client_instance = SvoFpsClient()
        return self._svo_fps_client_instance

    def _datalab_coordinates(
        self,
        target_name: Optional[str] = None,
        ra: Optional[float] = None,
        dec: Optional[float] = None,
    ) -> Tuple[float, float, str]:
        if ra is not None and dec is not None:
            ra_f, dec_f = float(ra), float(dec)
            if not 0.0 <= ra_f < 360.0 or not -90.0 <= dec_f <= 90.0:
                raise ValueError("ra/dec must be valid ICRS degrees")
            return ra_f, dec_f, f"RA={ra_f:.5f}, Dec={dec_f:.5f}"
        if target_name:
            resolved = self._resolve_target(str(target_name))
            if not resolved.get("success"):
                raise ValueError(resolved.get("error") or f"Could not resolve target {target_name!r}")
            return float(resolved["ra_deg"]), float(resolved["dec_deg"]), str(target_name)
        raise ValueError("Provide either ra+dec or target_name")

    # ── P2 orchestration handlers (Tier 6-7) ───────────────────────────────
    # Live imagery / external catalog handlers
    def _live_imagery_coordinates(
        self,
        target_name: Optional[str] = None,
        ra: Optional[float] = None,
        dec: Optional[float] = None,
    ) -> Tuple[float, float, str]:
        import math

        if (ra is None or dec is None) and target_name:
            resolved = self._resolve_target(str(target_name))
            if not resolved.get("success"):
                raise ValueError(resolved.get("error") or f"Could not resolve target {target_name!r}")
            ra = resolved.get("ra_deg")
            dec = resolved.get("dec_deg")
            label = str(target_name)
        else:
            label = f"RA={float(ra):.5f}, Dec={float(dec):.5f}" if ra is not None and dec is not None else "sky position"
        if ra is None or dec is None:
            raise ValueError("Provide either target_name or both ra and dec.")
        ra_f = float(ra)
        dec_f = float(dec)
        if not math.isfinite(ra_f) or not math.isfinite(dec_f) or not 0.0 <= ra_f < 360.0 or not -90.0 <= dec_f <= 90.0:
            raise ValueError("ra/dec must be finite ICRS degrees with 0 <= ra < 360 and -90 <= dec <= 90.")
        return ra_f, dec_f, label

    def _external_catalog_table_result(
        self,
        rows: List[Dict[str, Any]],
        *,
        columns: List[str],
        source: str,
        filter_label: str,
        tool_name: str,
        warnings: Optional[List[str]] = None,
        provenance: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        df = pd.DataFrame(rows, columns=columns)
        warnings_list = list(warnings or [])
        self.last_run_result = {
            "type": "data",
            "data": df,
            "source": source,
            "filter_label": filter_label,
            "tool_name": tool_name,
            "table_kind": "external_catalog",
            "warnings": warnings_list,
            "partial": bool(warnings_list),
        }
        preview_rows, _ = self._datalab_fit_rows(compact_preview_frame(df), 10, char_budget=4000)
        return {
            "success": True,
            "source": source,
            "rowcount": int(len(df)),
            "returned_rows": int(len(df)),
            "columns": list(df.columns),
            "results_preview": preview_rows,
            "preview_truncated": int(len(df)) > len(preview_rows),
            "warnings": warnings_list,
            "provenance": provenance or {},
            "note": "Full table is rendered as a data card in the UI; do not repeat the rows in text.",
        }

    def _hips_cutout(
        self,
        target_name: Optional[str] = None,
        ra: Optional[float] = None,
        dec: Optional[float] = None,
        survey: str = "optical",
        fov_deg: float = 0.25,
        width: int = 512,
    ) -> Dict[str, Any]:
        self.last_run_result = None
        try:
            ra_f, dec_f, label = self._live_imagery_coordinates(target_name=target_name, ra=ra, dec=dec)
            result = self._get_hips_image_service().cutout(ra_f, dec_f, fov_deg=fov_deg, survey=survey, width=width)
            caption = f"HiPS {survey} cutout: {label}"
            meta = {
                "kind": "hips",
                "ra": ra_f,
                "dec": dec_f,
                "fov_deg": result.get("fov_deg", fov_deg),
                "survey": result.get("survey_id") or result.get("survey") or survey,
            }
            return self._datalab_attach_image_result(result, caption, meta=meta)
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _hips_multiband_panel(
        self,
        target_name: Optional[str] = None,
        ra: Optional[float] = None,
        dec: Optional[float] = None,
        surveys: Optional[List[str]] = None,
        fov_deg: float = 0.25,
    ) -> Dict[str, Any]:
        self.last_run_result = None
        try:
            ra_f, dec_f, label = self._live_imagery_coordinates(target_name=target_name, ra=ra, dec=dec)
            survey_list = surveys or ["optical", "2mass", "wise"]
            result = self._get_hips_image_service().multiband_panel(
                ra_f,
                dec_f,
                fov_deg=fov_deg,
                surveys=survey_list,
                title=f"HiPS multiband panel: {label}",
            )
            meta = {"kind": "hips_panel", "ra": ra_f, "dec": dec_f, "fov_deg": result.get("fov_deg", fov_deg)}
            return self._datalab_attach_image_result(result, f"HiPS multiband panel: {label}", meta=meta)
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _vlass_cutout(
        self,
        target_name: Optional[str] = None,
        ra: Optional[float] = None,
        dec: Optional[float] = None,
        fov_deg: float = 0.1,
    ) -> Dict[str, Any]:
        self.last_run_result = None
        try:
            ra_f, dec_f, label = self._live_imagery_coordinates(target_name=target_name, ra=ra, dec=dec)
            result = self._get_hips_image_service().vlass_cutout(ra_f, dec_f, fov_deg=fov_deg)
            meta = {
                "kind": "hips",
                "ra": ra_f,
                "dec": dec_f,
                "fov_deg": result.get("fov_deg", fov_deg),
                "survey": result.get("survey_id") or result.get("survey") or "NRAO/P/VLASS-Quicklook-MedianStack",
            }
            return self._datalab_attach_image_result(result, f"VLASS 3 GHz cutout: {label}", meta=meta)
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _survey_coverage(
        self,
        target_name: Optional[str] = None,
        ra: Optional[float] = None,
        dec: Optional[float] = None,
        radius_deg: float = 0.0,
        dataproduct_type: Optional[str] = None,
        keyword: Optional[str] = None,
        max_rows: int = 50,
        regime: Optional[str] = None,
    ) -> Dict[str, Any]:
        self.last_run_result = None
        try:
            ra_f, dec_f, label = self._live_imagery_coordinates(target_name=target_name, ra=ra, dec=dec)
            result = self._get_moc_coverage_service().coverage_at(
                ra_f,
                dec_f,
                radius_deg=radius_deg,
                dataproduct_type=dataproduct_type,
                keyword=keyword,
                max_rows=max_rows,
                regime=regime,
            )
            if not result.get("success"):
                return result
            columns = ["id", "title", "dataproduct_type", "regime", "moc_sky_fraction"]
            rows = result.get("rows") or []
            present_columns = [col for col in columns if any(col in row for row in rows)] or columns
            radius_label = float(result.get("provenance", {}).get("radius_deg", radius_deg))
            return self._external_catalog_table_result(
                rows,
                columns=present_columns,
                source="CDS MOCServer",
                filter_label=f"coverage at {label}, r={radius_label:g} deg",
                tool_name="survey_coverage",
                warnings=result.get("warnings", []),
                provenance=result.get("provenance", {}),
            )
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _survey_covers_position(
        self,
        survey_keyword: str,
        target_name: Optional[str] = None,
        ra: Optional[float] = None,
        dec: Optional[float] = None,
    ) -> Dict[str, Any]:
        self.last_run_result = None
        try:
            ra_f, dec_f, label = self._live_imagery_coordinates(target_name=target_name, ra=ra, dec=dec)
            result = self._get_moc_coverage_service().survey_covers(survey_keyword, ra_f, dec_f)
            out = dict(result)
            out["target"] = label
            return out
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _galactic_extinction(
        self,
        target_name: Optional[str] = None,
        ra: Optional[float] = None,
        dec: Optional[float] = None,
        bands: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        self.last_run_result = None
        try:
            ra_f, dec_f, label = self._live_imagery_coordinates(target_name=target_name, ra=ra, dec=dec)
            result = self._get_dust_extinction_service().extinction_table(ra_f, dec_f, bands=bands)
            if not result.get("success"):
                return result
            columns = ["band", "A_lambda", "coeff"]
            rows = result.get("rows") or []
            present_columns = [col for col in columns if any(col in row for row in rows)] or columns
            ebv_sfd = float(result.get("ebv_sfd"))
            ebv_sf11 = float(result.get("ebv_sf11"))
            table_result = self._external_catalog_table_result(
                rows,
                columns=present_columns,
                source="IRSA DUST (SFD98/SF11)",
                filter_label=f"E(B-V) at {label}: SFD={ebv_sfd:.4f}, SF11={ebv_sf11:.4f}",
                tool_name="galactic_extinction",
                warnings=result.get("warnings", []),
                provenance=result.get("provenance", {}),
            )
            for key in ("ebv_sfd", "ebv_sf11", "ebv_sfd_mean", "ebv_sf11_mean"):
                table_result[key] = result.get(key)
                if self.last_run_result is not None:
                    self.last_run_result[key] = result.get(key)
            table_result["target"] = label
            return table_result
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _gaia_distance(
        self,
        target_name: Optional[str] = None,
        ra: Optional[float] = None,
        dec: Optional[float] = None,
        radius_arcsec: float = 10,
        max_rows: int = 10,
    ) -> Dict[str, Any]:
        self.last_run_result = None
        try:
            ra_f, dec_f, label = self._live_imagery_coordinates(target_name=target_name, ra=ra, dec=dec)
            wrapper_warnings: List[str] = []
            effective_radius_arcsec = radius_arcsec
            try:
                radius_value = float(radius_arcsec)
            except (TypeError, ValueError):
                radius_value = None
            if target_name and (ra is None or dec is None) and radius_value == 10.0:
                effective_radius_arcsec = 30
                wrapper_warnings.append(
                    "named-target Gaia query radius expanded from 10 to 30 arcsec to tolerate proper-motion/epoch offsets"
                )
            result = self._get_distance_service().gaia_distances(
                ra_f,
                dec_f,
                radius_arcsec=effective_radius_arcsec,
                max_rows=max_rows,
            )
            if not result.get("success"):
                return result
            columns = [
                "source_id",
                "g_mag",
                "parallax_mas",
                "r_geo_pc",
                "r_geo_lo_pc",
                "r_geo_hi_pc",
                "r_photogeo_pc",
                "r_photogeo_lo_pc",
                "r_photogeo_hi_pc",
                "sep_arcsec",
            ]
            rows = result.get("rows") or []
            present_columns = [col for col in columns if any(col in row for row in rows)] or columns[:6]
            radius_label = float(result.get("provenance", {}).get("radius_arcsec", effective_radius_arcsec))
            return self._external_catalog_table_result(
                rows,
                columns=present_columns,
                source="ESA Gaia DR3 x Bailer-Jones EDR3 distances",
                filter_label=f"Gaia/Bailer-Jones cone {label}, r={radius_label:g} arcsec",
                tool_name="gaia_distance",
                warnings=wrapper_warnings + list(result.get("warnings", [])),
                provenance=result.get("provenance", {}),
            )
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _ned_distance(self, target_name: str) -> Dict[str, Any]:
        self.last_run_result = None
        try:
            result = self._get_distance_service().ned_distances(target_name)
            if not result.get("success"):
                return result
            columns = ["dist_mpc", "dist_modulus", "dist_modulus_err", "method", "refcode"]
            rows = result.get("rows") or []
            present_columns = [col for col in columns if any(col in row for row in rows)] or columns
            summary = result.get("summary") or {}
            median = summary.get("median_mpc")
            median_label = f"median={float(median):.4g} Mpc" if median is not None else "no valid moduli"
            table_result = self._external_catalog_table_result(
                rows,
                columns=present_columns,
                source="NED-D redshift-independent distances",
                filter_label=f"NED-D {target_name}: {median_label}",
                tool_name="ned_distance",
                warnings=result.get("warnings", []),
                provenance=result.get("provenance", {}),
            )
            table_result["summary"] = summary
            if self.last_run_result is not None:
                self.last_run_result["summary"] = summary
            return table_result
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _velocity_frame_distance(
        self,
        target_name: Optional[str] = None,
        ra: Optional[float] = None,
        dec: Optional[float] = None,
        v_helio_kms: Optional[float] = None,
        z: Optional[float] = None,
    ) -> Dict[str, Any]:
        self.last_run_result = None
        try:
            ra_f, dec_f, label = self._live_imagery_coordinates(target_name=target_name, ra=ra, dec=dec)
            result = self._get_distance_service().velocity_frames(ra_f, dec_f, v_helio_kms=v_helio_kms, z=z)
            if not result.get("success"):
                return result
            columns = ["frame", "v_kms", "d_hubble_mpc"]
            rows = result.get("rows") or []
            present_columns = [col for col in columns if any(col in row for row in rows)] or columns
            input_label = f"z={float(z):g}" if z is not None else f"v_helio={float(v_helio_kms):g} km/s"
            table_result = self._external_catalog_table_result(
                rows,
                columns=present_columns,
                source="Velocity-frame corrections (NED/Planck conventions)",
                filter_label=f"Velocity frames for {label}, {input_label}",
                tool_name="velocity_frame_distance",
                warnings=result.get("warnings", []),
                provenance=result.get("provenance", {}),
            )
            table_result["target"] = label
            if self.last_run_result is not None:
                self.last_run_result["target"] = label
            return table_result
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _search_ztf_alerts(
        self,
        target_name: Optional[str] = None,
        ra: Optional[float] = None,
        dec: Optional[float] = None,
        radius_arcsec: float = 120,
        max_rows: int = 25,
    ) -> Dict[str, Any]:
        self.last_run_result = None
        try:
            ra_f, dec_f, label = self._live_imagery_coordinates(target_name=target_name, ra=ra, dec=dec)
            result = self._get_alerce_client().cone_objects(ra_f, dec_f, radius_arcsec=radius_arcsec, max_rows=max_rows)
            if not result.get("success"):
                return result
            columns = ["oid", "ndet", "meanra", "meandec", "firstmjd", "lastmjd", "classalerce", "probability", "classifier", "class", "classification", "prob"]
            rows = result.get("rows") or []
            present_columns = [col for col in columns if any(col in row for row in rows)] or columns[:6]
            return self._external_catalog_table_result(
                rows,
                columns=present_columns,
                source="ALeRCE ZTF alerts",
                filter_label=f"ALeRCE/ZTF cone {label}, r={float(result.get('provenance', {}).get('radius_arcsec', radius_arcsec)):g} arcsec",
                tool_name="search_ztf_alerts",
                warnings=result.get("warnings", []),
                provenance=result.get("provenance", {}),
            )
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _radio_sed(
        self,
        target_name: Optional[str] = None,
        ra: Optional[float] = None,
        dec: Optional[float] = None,
        radius_arcsec: float = 30.0,
    ) -> Dict[str, Any]:
        self.last_run_result = None
        try:
            ra_f, dec_f, label = self._live_imagery_coordinates(target_name=target_name, ra=ra, dec=dec)
            service = self._get_radio_sed_service()
            compile_out = service.compile_sed(ra_f, dec_f, radius_arcsec=radius_arcsec)
            if not compile_out.get("success"):
                return compile_out

            points = list(compile_out.get("points") or [])
            warnings = list(compile_out.get("warnings") or [])
            provenance = {"compile": compile_out.get("provenance", {})}
            if not points:
                statuses = compile_out.get("provenance", {}).get("survey_status", [])
                all_failed = bool(statuses) and all(row.get("status") == "failed" for row in statuses)
                note = "No usable radio catalog responses; this is not evidence of a radio nondetection." if all_failed else "No radio catalog detections within radius."
                return {
                    "success": True,
                    "note": note,
                    "target": label,
                    "points": [],
                    "count": 0,
                    "warnings": warnings,
                    "provenance": compile_out.get("provenance", {}),
                }

            fit = None
            fit_flags: List[str] = []
            if len(points) >= 2:
                fit_out = service.fit_spectral_index(points)
                if fit_out.get("success"):
                    fit = fit_out
                    fit_flags = list(fit_out.get("flags") or [])
                    warnings.extend(fit_out.get("warnings") or [])
                    provenance["fit"] = fit_out.get("provenance", {})
                else:
                    warnings.extend(fit_out.get("warnings") or [])
                    warnings.append(f"Spectral-index fit skipped: {fit_out.get('error', 'fit failed')}.")
                    provenance["fit"] = fit_out.get("provenance", {})
            else:
                warnings.append("Only one radio catalog detection; spectral index was not fitted.")

            plot_result = service.plot_sed(points, fit=fit, title=f"Radio SED: {label}")
            warnings.extend(plot_result.get("warnings") or [])
            provenance["plot"] = plot_result.get("provenance", {})
            if not plot_result.get("success"):
                plot_result["points"] = points
                plot_result["warnings"] = warnings
                plot_result["flags"] = fit_flags
                plot_result["provenance"] = provenance
                return plot_result

            n_points = len(points)
            if fit:
                alpha = float(fit["alpha"])
                alpha_err = None
                try:
                    alpha_err_candidate = float(fit.get("alpha_err"))
                    if alpha_err_candidate == alpha_err_candidate and alpha_err_candidate not in (float("inf"), float("-inf")) and alpha_err_candidate >= 0:
                        alpha_err = alpha_err_candidate
                except (TypeError, ValueError):
                    alpha_err = None
                if alpha_err is None:
                    caption = f"Radio SED: {label} - alpha = {alpha:.2f} ({n_points} surveys)"
                else:
                    caption = f"Radio SED: {label} - alpha = {alpha:.2f} +/- {alpha_err:.2f} ({n_points} surveys)"
            else:
                caption = f"Radio SED: {label} ({n_points} survey point{'s' if n_points != 1 else ''}; spectral index not fitted)"

            meta = {
                "kind": "radio_sed",
                "ra": ra_f,
                "dec": dec_f,
                "radius_arcsec": compile_out.get("provenance", {}).get("radius_arcsec", radius_arcsec),
                "flags": fit_flags,
            }
            attached = self._datalab_attach_image_result(plot_result, caption, meta=meta)
            attached.update(
                {
                    "points": points,
                    "count": n_points,
                    "flags": fit_flags,
                    "warnings": warnings,
                    "provenance": provenance,
                    "target": label,
                }
            )
            if fit:
                attached.update(
                    {
                        "alpha": fit.get("alpha"),
                        "alpha_err": fit.get("alpha_err"),
                        "chi2_red": fit.get("chi2_red"),
                        "n_points": fit.get("n_points"),
                        "s_1400_mjy_predicted": fit.get("s_1400_mjy_predicted"),
                    }
                )
            else:
                attached["note"] = "Radio SED plotted, but spectral index was not fitted."
            return attached
        except Exception as e:
            return {"success": False, "error": str(e)}


    def _search_space_lightcurves(
        self,
        target: str,
        mission: Optional[str] = None,
        max_rows: int = 20,
    ) -> Dict[str, Any]:
        self.last_run_result = None
        try:
            result = self._get_lightcurve_suite().search_space_lightcurves(target, mission=mission, max_rows=max_rows)
            if not result.get("success"):
                return result
            columns = ["index", "mission", "year", "author", "exptime_s", "target_name", "distance_arcsec"]
            rows = result.get("rows") or []
            present_columns = [col for col in columns if any(col in row for row in rows)] or columns
            return self._external_catalog_table_result(
                rows,
                columns=present_columns,
                source="MAST via lightkurve",
                filter_label=f"TESS/Kepler/K2 light curves for {target}",
                tool_name="search_space_lightcurves",
                warnings=result.get("warnings", []),
                provenance=result.get("provenance", {}),
            )
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _plot_space_lightcurve(
        self,
        target: str,
        mission: Optional[str] = None,
        index: int = 0,
    ) -> Dict[str, Any]:
        self.last_run_result = None
        try:
            result = self._get_lightcurve_suite().plot_space_lightcurve(target, mission=mission, index=index)
            return self._datalab_attach_image_result(result, f"{mission or 'TESS/Kepler'} light curve: {target}")
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _period_search(
        self,
        source: str,
        identifier: str,
        mission: Optional[str] = None,
        min_period_d: float = 0.05,
        max_period_d: float = 30.0,
        fid: Optional[int] = None,
        index: int = 0,
    ) -> Dict[str, Any]:
        self.last_run_result = None
        try:
            result = self._get_lightcurve_suite().period_search(
                source,
                identifier,
                mission=mission,
                min_period_d=min_period_d,
                max_period_d=max_period_d,
                fid=fid,
                index=index,
            )
            if not result.get("success"):
                return result
            period = result.get("best_period_d")
            fap = result.get("fap")
            period_text = f"{float(period):.6g}" if period is not None else "n/a"
            fap_text = f"{float(fap):.2g}" if fap is not None and math.isfinite(float(fap)) else "n/a"
            caption = f"Period search: {identifier} - P={period_text} d (FAP={fap_text})"
            return self._datalab_attach_image_result(result, caption)
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _search_pulsars(
        self,
        target_name: Optional[str] = None,
        ra: Optional[float] = None,
        dec: Optional[float] = None,
        radius_deg: float = 1.0,
        max_rows: int = 25,
    ) -> Dict[str, Any]:
        self.last_run_result = None
        try:
            ra_f, dec_f, label = self._live_imagery_coordinates(target_name=target_name, ra=ra, dec=dec)
            result = self._get_pulsar_catalog_service().search_pulsars(ra_f, dec_f, radius_deg=radius_deg, max_rows=max_rows)
            if not result.get("success"):
                return result
            columns = ["jname", "p0_s", "dm_pc_cm3", "s1400_mjy", "dist_kpc", "binary", "assoc", "sep_arcmin"]
            rows = result.get("rows") or []
            present_columns = [col for col in columns if any(col in row for row in rows)] or columns
            radius_label = float(result.get("provenance", {}).get("radius_deg", radius_deg))
            table_result = self._external_catalog_table_result(
                rows,
                columns=present_columns,
                source="ATNF Pulsar Catalogue",
                filter_label=f"ATNF pulsar cone {label}, r={radius_label:g} deg",
                tool_name="search_pulsars",
                warnings=result.get("warnings", []),
                provenance=result.get("provenance", {}),
            )
            table_result["target"] = label
            if self.last_run_result is not None:
                self.last_run_result["target"] = label
            return table_result
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _pulsar_lookup(self, name: str) -> Dict[str, Any]:
        self.last_run_result = None
        try:
            result = self._get_pulsar_catalog_service().pulsar_lookup(name)
            if not result.get("success"):
                return result
            columns = [
                "jname",
                "name",
                "bname",
                "ra_deg",
                "dec_deg",
                "p0_s",
                "p1",
                "dm_pc_cm3",
                "dist_kpc",
                "age_yr",
                "bsurf_g",
                "edot_erg_s",
                "s1400_mjy",
                "binary",
                "assoc",
                "type",
            ]
            rows = result.get("rows") or []
            present_columns = [col for col in columns if any(col in row for row in rows)] or columns
            table_result = self._external_catalog_table_result(
                rows,
                columns=present_columns,
                source="ATNF Pulsar Catalogue",
                filter_label=f"ATNF pulsar lookup {name}",
                tool_name="pulsar_lookup",
                warnings=result.get("warnings", []),
                provenance=result.get("provenance", {}),
            )
            table_result["name"] = name
            if self.last_run_result is not None:
                self.last_run_result["name"] = name
            return table_result
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _solar_system_ephemeris(
        self,
        target: str,
        start: str,
        stop: str,
        step: str = "1d",
    ) -> Dict[str, Any]:
        self.last_run_result = None
        try:
            result = self._get_solar_system_service().horizons_ephemeris(target, start, stop, step=step)
            if not result.get("success"):
                return result
            columns = ["datetime", "ra_deg", "dec_deg", "delta_au", "r_au", "v_mag", "elong_deg"]
            rows = result.get("rows") or []
            present_columns = [col for col in columns if any(col in row for row in rows)] or columns
            return self._external_catalog_table_result(
                rows,
                columns=present_columns,
                source="JPL Horizons",
                filter_label=f"{target}: {start} → {stop} (step {step})",
                tool_name="solar_system_ephemeris",
                warnings=result.get("warnings", []),
                provenance=result.get("provenance", {}),
            )
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _moving_object_check(
        self,
        target_name: Optional[str] = None,
        ra: Optional[float] = None,
        dec: Optional[float] = None,
        radius_deg: float = 0.2,
        epoch: Optional[str] = None,
    ) -> Dict[str, Any]:
        self.last_run_result = None
        try:
            ra_f, dec_f, label = self._live_imagery_coordinates(target_name=target_name, ra=ra, dec=dec)
            result = self._get_solar_system_service().skybot_cone(ra_f, dec_f, radius_deg=radius_deg, epoch=epoch)
            if not result.get("success"):
                return result
            columns = ["name", "class", "v_mag", "ra_deg", "dec_deg", "sep_arcsec", "pos_err_arcsec"]
            rows = result.get("rows") or []
            present_columns = [col for col in columns if any(col in row for row in rows)] or columns
            epoch_iso = result.get("provenance", {}).get("epoch_iso", "now")
            radius_label = float(result.get("provenance", {}).get("radius_deg", radius_deg))
            return self._external_catalog_table_result(
                rows,
                columns=present_columns,
                source="IMCCE SkyBoT",
                filter_label=f"moving objects at {label}, r={radius_label:g} deg, epoch {epoch_iso}",
                tool_name="moving_object_check",
                warnings=result.get("warnings", []),
                provenance=result.get("provenance", {}),
            )
        except Exception as e:
            return {"success": False, "error": str(e)}

    # ── F08: standing sky monitors ─────────────────────────────────────────
    def _monitor_add_target(
        self,
        name: str,
        target_name: Optional[str] = None,
        ra: Optional[float] = None,
        dec: Optional[float] = None,
        radius_arcsec: float = 120,
        note: Optional[str] = None,
    ) -> Dict[str, Any]:
        self.last_run_result = None
        try:
            ra_f, dec_f, label = self._live_imagery_coordinates(
                target_name=target_name or name, ra=ra, dec=dec
            )
            result = self._get_sky_monitor_service().add_target(
                name, ra_f, dec_f, radius_arcsec=radius_arcsec, note=note
            )
            if result.get("success"):
                result["resolved_position"] = label
            return result
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _monitor_list_targets(self) -> Dict[str, Any]:
        self.last_run_result = None
        try:
            result = self._get_sky_monitor_service().list_targets()
            if not result.get("success"):
                return result
            columns = ["id", "name", "ra", "dec", "radius_arcsec", "enabled",
                       "last_checked_at", "hits"]
            rows = result.get("rows") or []
            present_columns = [col for col in columns if any(col in row for row in rows)] or columns
            return self._external_catalog_table_result(
                rows,
                columns=present_columns,
                source="Quasar sky monitor",
                filter_label=f"watchlist ({len(rows)} target(s))",
                tool_name="monitor_list_targets",
                warnings=result.get("warnings", []),
                provenance=result.get("provenance", {}),
            )
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _monitor_remove_target(self, target_id: int) -> Dict[str, Any]:
        self.last_run_result = None
        try:
            return self._get_sky_monitor_service().remove_target(target_id)
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _monitor_check_now(self, target_id: Optional[int] = None) -> Dict[str, Any]:
        self.last_run_result = None
        try:
            result = self._get_sky_monitor_service().check_now(target_id=target_id)
            if not result.get("success"):
                return result
            new_alerts = result.get("new_alerts") or []
            if not new_alerts:
                out = dict(result)
                out["note"] = "No new alerts since last check."
                return out
            columns = ["target_name", "oid", "ndet", "lastmjd", "class_name"]
            present_columns = [col for col in columns if any(col in row for row in new_alerts)] or columns
            return self._external_catalog_table_result(
                new_alerts,
                columns=present_columns,
                source="ALeRCE ZTF via sky monitor",
                filter_label=f"{len(new_alerts)} NEW alert(s) across "
                             f"{result.get('n_targets_checked')} target(s)",
                tool_name="monitor_check_now",
                warnings=result.get("warnings", []),
                provenance=result.get("provenance", {}),
            )
        except Exception as e:
            return {"success": False, "error": str(e)}

    # ── F01: VO registry discovery chain ───────────────────────────────────
    def _vo_find_services(
        self,
        keywords: str,
        service_type: Optional[str] = None,
        waveband: Optional[str] = None,
        max_rows: int = 30,
    ) -> Dict[str, Any]:
        self.last_run_result = None
        try:
            result = self._get_vo_registry_service().registry_search(
                keywords, service_type=service_type, waveband=waveband, max_rows=max_rows
            )
            if not result.get("success"):
                return result
            columns = ["short_name", "title", "service_type", "waveband", "access_url"]
            rows = result.get("rows") or []
            present_columns = [col for col in columns if any(col in row for row in rows)] or columns
            return self._external_catalog_table_result(
                rows,
                columns=present_columns,
                source="IVOA Registry",
                filter_label=f"VO services for {keywords!r}"
                             + (f" [{service_type}]" if service_type else ""),
                tool_name="vo_find_services",
                warnings=result.get("warnings", []),
                provenance=result.get("provenance", {}),
            )
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _vo_list_tables(
        self, access_url: str, keyword: Optional[str] = None, max_tables: int = 50
    ) -> Dict[str, Any]:
        self.last_run_result = None
        try:
            result = self._get_vo_registry_service().list_tables(
                access_url, keyword=keyword, max_tables=max_tables
            )
            if not result.get("success"):
                return result
            columns = ["table_name", "n_columns", "description"]
            rows = result.get("rows") or []
            present_columns = [col for col in columns if any(col in row for row in rows)] or columns
            return self._external_catalog_table_result(
                rows,
                columns=present_columns,
                source=access_url,
                filter_label=f"TAP tables" + (f" matching {keyword!r}" if keyword else ""),
                tool_name="vo_list_tables",
                warnings=result.get("warnings", []),
                provenance=result.get("provenance", {}),
            )
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _vo_describe_table(self, access_url: str, table_name: str) -> Dict[str, Any]:
        self.last_run_result = None
        try:
            result = self._get_vo_registry_service().describe_table(access_url, table_name)
            if not result.get("success"):
                return result
            columns = ["name", "datatype", "unit", "ucd", "description"]
            rows = result.get("rows") or []
            present_columns = [col for col in columns if any(col in row for row in rows)] or columns
            return self._external_catalog_table_result(
                rows,
                columns=present_columns,
                source=f"Schema: {table_name}",
                filter_label=f"columns of {table_name}",
                tool_name="vo_describe_table",
                warnings=result.get("warnings", []),
                provenance=result.get("provenance", {}),
            )
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _vo_adql_query(self, access_url: str, adql: str, max_rows: int = 200) -> Dict[str, Any]:
        self.last_run_result = None
        try:
            result = self._get_vo_registry_service().run_adql(access_url, adql, max_rows=max_rows)
            if not result.get("success"):
                return result
            rows = result.get("rows") or []
            columns = list(result.get("columns") or [])
            warnings = list(result.get("warnings") or [])
            if len(columns) > 12:
                columns = columns[:12]
                warnings.append("Displaying the first 12 of the result's columns.")
            return self._external_catalog_table_result(
                rows,
                columns=columns or ["result"],
                source=f"TAP: {access_url}",
                filter_label=(adql or "")[:120],
                tool_name="vo_adql_query",
                warnings=warnings,
                provenance=result.get("provenance", {}),
            )
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _vo_cone_search(
        self,
        access_url: str,
        target_name: Optional[str] = None,
        ra: Optional[float] = None,
        dec: Optional[float] = None,
        radius_deg: float = 0.1,
        max_rows: int = 100,
    ) -> Dict[str, Any]:
        self.last_run_result = None
        try:
            ra_f, dec_f, label = self._live_imagery_coordinates(target_name=target_name, ra=ra, dec=dec)
            result = self._get_vo_registry_service().cone_search(
                access_url, ra_f, dec_f, radius_deg=radius_deg, max_rows=max_rows
            )
            if not result.get("success"):
                return result
            rows = result.get("rows") or []
            columns = list(result.get("columns") or [])
            warnings = list(result.get("warnings") or [])
            if len(columns) > 12:
                columns = columns[:12]
                warnings.append("Displaying the first 12 of the result's columns.")
            radius_label = float(result.get("provenance", {}).get("radius_deg", radius_deg))
            return self._external_catalog_table_result(
                rows,
                columns=columns or ["result"],
                source=f"SCS: {access_url}",
                filter_label=f"cone at {label}, r={radius_label:g} deg",
                tool_name="vo_cone_search",
                warnings=warnings,
                provenance=result.get("provenance", {}),
            )
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _ztf_light_curve(self, oid: str) -> Dict[str, Any]:
        self.last_run_result = None
        try:
            result = self._get_alerce_client().plot_light_curve(oid)
            return self._datalab_attach_image_result(result, f"ZTF light curve: {oid}")
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _ztf_stamps(self, oid: str, candid: Optional[str] = None) -> Dict[str, Any]:
        self.last_run_result = None
        try:
            result = self._get_alerce_client().stamp_triplet(oid, candid=candid)
            caption = f"ZTF stamps: {oid}" + (f" / {candid}" if candid else "")
            return self._datalab_attach_image_result(result, caption)
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _ned_sed_plot(self, target_name: str) -> Dict[str, Any]:
        self.last_run_result = None
        try:
            result = self._get_ned_photometry_service().sed_plot(target_name)
            return self._datalab_attach_image_result(result, f"NED SED: {target_name}")
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _sparcl_find_spectra(
        self,
        target_name: Optional[str] = None,
        ra: Optional[float] = None,
        dec: Optional[float] = None,
        radius_arcsec: float = 60,
        data_release: Optional[List[str]] = None,
        limit: int = 20,
    ) -> Dict[str, Any]:
        self.last_run_result = None
        try:
            ra_f, dec_f, label = self._live_imagery_coordinates(target_name=target_name, ra=ra, dec=dec)
            result = self._get_sparcl_spectra_service().find_spectra(
                ra_f,
                dec_f,
                radius_arcsec=radius_arcsec,
                data_release=data_release,
                limit=limit,
            )
            if not result.get("success"):
                return result
            columns = ["sparcl_id", "ra", "dec", "distance_arcsec", "redshift", "spectype", "data_release"]
            return self._external_catalog_table_result(
                result.get("rows") or [],
                columns=columns,
                source="NOIRLab SparCL spectra",
                filter_label=f"SparCL cone {label}, r={float(result.get('provenance', {}).get('radius_arcsec', radius_arcsec)):g} arcsec",
                tool_name="sparcl_find_spectra",
                warnings=result.get("warnings", []),
                provenance=result.get("provenance", {}),
            )
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _sparcl_plot_spectrum(self, sparcl_id: str, mark_lines: bool = True, smooth: int = 0) -> Dict[str, Any]:
        self.last_run_result = None
        try:
            result = self._get_sparcl_spectra_service().plot_spectrum(sparcl_id, mark_lines=mark_lines, smooth=smooth)
            return self._datalab_attach_image_result(result, f"SparCL spectrum: {sparcl_id}")
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _datalab_confirm_sky_area(self, ra_min: float, ra_max: float, dec_min: float, dec_max: float, tile_radius_deg: float = 2.0) -> Dict[str, Any]:
        self.last_run_result = None
        try:
            fp = {"ra_min": ra_min, "ra_max": ra_max, "dec_min": dec_min, "dec_max": dec_max}
            return {"success": True, **datalab_orchestration.confirm_sky_area(fp, float(tile_radius_deg))}
        except Exception as e:
            return self._datalab_error(e)

    def _datalab_density_vetting(
        self,
        catalog: str,
        table: str,
        radius_deg: float,
        ra: Optional[float] = None,
        dec: Optional[float] = None,
        target_name: Optional[str] = None,
        step_deg: float = 0.05,
        color_cut: Optional[Dict[str, Any]] = None,
        value_cuts: Optional[List[Dict[str, Any]]] = None,
        morphology: Optional[Dict[str, Any]] = None,
        top_n: int = 5,
        fov_deg: float = 0.05,
        band: str = "g",
    ) -> Dict[str, Any]:
        self.last_run_result = None
        try:
            # Null-coerce optional numerics: models sometimes send explicit
            # "step_deg": null / "fov_deg": null / "top_n": null, which would
            # crash on float(None)/int(None). Fall back to the documented defaults
            # (mirrors the color-magnitude/color-color handlers).
            radius_deg = 0.5 if radius_deg is None else float(radius_deg)
            step_deg = 0.05 if step_deg is None else float(step_deg)
            top_n = 5 if top_n is None else int(top_n)
            fov_deg = 0.05 if fov_deg is None else float(fov_deg)
            band = band or "g"
            ra_f, dec_f, label = self._datalab_coordinates(target_name=target_name, ra=ra, dec=dec)
            out = datalab_orchestration.density_then_cutouts(
                catalog, table, ra_f, dec_f, radius_deg, step_deg=step_deg,
                color_cut=color_cut, value_cuts=value_cuts, morphology=morphology,
                top_n=top_n, fov_deg=fov_deg, band=band,
            )
            out["target"] = label
            # The cutout grid must reach the UI as an image card (2026-07-04 live test:
            # the base64 grid stayed buried in the tool output, the chat showed nothing,
            # and the answer still told the user to "eyeball the cutouts above").
            grid = out.get("cutout_grid")
            if isinstance(grid, dict) and (grid.get("image_base64") or grid.get("path")):
                grid = dict(grid)
                grid.setdefault("success", True)
                out["cutout_grid"] = self._datalab_attach_image_result(
                    grid, f"Density-peak cutout grid: {label} (top {int(top_n)})"
                )
            return out
        except Exception as e:
            return self._datalab_error(e)

    def _datalab_color_color_diagram(
        self,
        catalog: str,
        table: Optional[str] = None,
        radius_deg: float = 0.5,
        ra: Optional[float] = None,
        dec: Optional[float] = None,
        target_name: Optional[str] = None,
        x_bands: Optional[List[str]] = None,
        y_bands: Optional[List[str]] = None,
        split_col: Optional[str] = None,
        split_threshold: float = 0.005,
        limit: int = 3000,
        title: Optional[str] = None,
        point_sources: bool = False,
        value_cuts: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        self.last_run_result = None
        try:
            from services import datalab_registry as _dl_reg
            # Coerce null/omitted optional args (models often pass null) to sane defaults.
            table = table or _dl_reg.default_table(catalog)
            radius_deg = float(radius_deg) if radius_deg is not None else 0.5
            split_threshold = float(split_threshold) if split_threshold is not None else 0.005
            limit = int(limit) if limit is not None else 3000
            ra_f, dec_f, label = self._datalab_coordinates(target_name=target_name, ra=ra, dec=dec)
            out = datalab_orchestration.color_color_diagram(
                catalog, table, ra_f, dec_f, radius_deg,
                x_bands=tuple(x_bands) if x_bands else ("g", "r"),
                y_bands=tuple(y_bands) if y_bands else ("r", "i"),
                split_col=split_col, split_threshold=split_threshold, limit=limit,
                title=title or f"{catalog} color-color: {label}",
                point_sources=bool(point_sources), value_cuts=value_cuts,
            )
            return self._datalab_attach_image_result(out, title or f"Color-color diagram: {label}")
        except Exception as e:
            return self._datalab_error(e)

    def _datalab_color_magnitude_diagram(
        self,
        catalog: str,
        table: Optional[str] = None,
        radius_deg: float = 0.4,
        ra: Optional[float] = None,
        dec: Optional[float] = None,
        target_name: Optional[str] = None,
        blue_band: str = "g",
        red_band: str = "r",
        mag_band: Optional[str] = None,
        limit: int = 5000,
        title: Optional[str] = None,
        point_sources: bool = False,
        value_cuts: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        self.last_run_result = None
        try:
            from services import datalab_registry as _dl_reg
            table = table or _dl_reg.default_table(catalog)
            radius_deg = float(radius_deg) if radius_deg is not None else 0.4
            limit = int(limit) if limit is not None else 5000
            blue_band = blue_band or "g"
            red_band = red_band or "r"
            ra_f, dec_f, label = self._datalab_coordinates(target_name=target_name, ra=ra, dec=dec)
            out = datalab_orchestration.color_magnitude_diagram(
                catalog, table, ra_f, dec_f, radius_deg,
                blue_band=blue_band, red_band=red_band, mag_band=mag_band, limit=limit,
                title=title or f"{catalog} CMD: {label}",
                point_sources=bool(point_sources), value_cuts=value_cuts,
            )
            return self._datalab_attach_image_result(out, title or f"Color-magnitude diagram: {label}")
        except Exception as e:
            return self._datalab_error(e)

    def _datalab_tiled_search(
        self,
        catalog: str,
        table: str,
        ra_min: float,
        ra_max: float,
        dec_min: float,
        dec_max: float,
        tile_radius_deg: float = 2.0,
        step_deg: float = 0.05,
        color_cut: Optional[Dict[str, Any]] = None,
        value_cuts: Optional[List[Dict[str, Any]]] = None,
        morphology: Optional[Dict[str, Any]] = None,
        peak_threshold: float = 3.0,
        max_tiles: int = 64,
        candidate_budget: int = 50,
        confirm: bool = False,
    ) -> Dict[str, Any]:
        self.last_run_result = None
        try:
            fp = {"ra_min": ra_min, "ra_max": ra_max, "dec_min": dec_min, "dec_max": dec_max}
            decision = datalab_orchestration.confirm_sky_area(fp, float(tile_radius_deg), max_tiles=int(max_tiles))
            # HITL gate: don't start a wide scan until the user confirms the area.
            if decision["needs_confirmation"] and not confirm:
                return {"success": False, "needs_confirmation": True, **decision,
                        "hint": "Re-call datalab_tiled_search with confirm=true to run the scan over this area."}

            def _job(cancel_check):
                return datalab_orchestration.tiled_sky_scan(
                    catalog, table, fp, tile_radius_deg=float(tile_radius_deg), step_deg=float(step_deg),
                    color_cut=color_cut, value_cuts=value_cuts, morphology=morphology,
                    peak_threshold=float(peak_threshold), max_tiles=int(max_tiles),
                    candidate_budget=int(candidate_budget), confirm=True, cancel_check=cancel_check,
                )

            job_id = default_job_service().start("tiled_sky_scan", _job, params={"catalog": catalog, "table": table, **fp})
            return {"success": True, "job_id": job_id, "status": "queued", **decision,
                    "note": "Tiled scan started; poll with datalab_job_status / datalab_job_results."}
        except Exception as e:
            return self._datalab_error(e)

    def _datalab_job_status(self, job_id: str) -> Dict[str, Any]:
        self.last_run_result = None
        try:
            out = {"success": True, **default_job_service().status(job_id)}
            # Job-aware turn ending (live DS-P15: the model polled a slow tiled
            # scan 18x until it silently hit HARD_MAX_ITERATIONS with no closing
            # message). After a few polls of a still-running job, tell the model
            # to stop polling and end the turn gracefully.
            if str(out.get("status", "")).lower() in {"queued", "running"}:
                counts = getattr(self, "_job_poll_counts", None)
                if counts is None:
                    counts = self._job_poll_counts = {}
                counts[str(job_id)] = counts.get(str(job_id), 0) + 1
                if counts[str(job_id)] >= 3:
                    out["stop_polling"] = True
                    out["instruction"] = (
                        f"This job is still {out.get('status')} server-side after "
                        f"{counts[str(job_id)]} polls. STOP polling now. End your answer: "
                        "summarize any results you already have, state that job "
                        f"{job_id} is still running, and tell the user to ask you to "
                        "check it again in a few minutes (datalab_job_status / "
                        "datalab_job_results). Do NOT call datalab_job_status again this turn."
                    )
            return out
        except Exception as e:
            return self._datalab_error(e)

    def _datalab_job_results(self, job_id: str) -> Dict[str, Any]:
        self.last_run_result = None
        try:
            return {"success": True, **default_job_service().results(job_id)}
        except Exception as e:
            return self._datalab_error(e)

    def _datalab_job_cancel(self, job_id: str) -> Dict[str, Any]:
        self.last_run_result = None
        try:
            return {"success": True, **default_job_service().cancel(job_id)}
        except Exception as e:
            return self._datalab_error(e)

    def _datalab_export_notebook(
        self,
        title: str = "NOIRLab Data Lab analysis",
        sql: Optional[str] = None,
        catalog: Optional[str] = None,
        table: Optional[str] = None,
        sia_ra: Optional[float] = None,
        sia_dec: Optional[float] = None,
        sia_fov_deg: float = 0.1,
        sia_endpoint: Optional[str] = None,
        svo_filters: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        # NB: _generate_notebook sets last_run_result to the notebook card (the deliverable),
        # so we intentionally do NOT clear last_run_result here.
        try:
            from services.notebook_gen import datalab_notebook_steps
            citation = None
            if catalog:
                try:
                    citation = datalab_registry.citation(catalog)
                except Exception:
                    citation = None
            sia = None
            if sia_ra is not None and sia_dec is not None:
                sia = {"ra": sia_ra, "dec": sia_dec, "fov_deg": sia_fov_deg, "endpoint": sia_endpoint}
            steps = datalab_notebook_steps(
                sql=sql, catalog=catalog, table=table, sia=sia, svo_filters=svo_filters, citation=citation,
            )
            return self._generate_notebook(title, steps)
        except Exception as e:
            return self._datalab_error(e)

    def _datalab_attach_image_result(self, result: Dict[str, Any], caption: str, meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        # Surface a displayable image card. Prefer the served path ("/plots/...", now mounted by
        # the API); fall back to a base64 data URI so the image still renders if no path is set.
        if result.get("success") and not result.get("coverage_gap"):
            b64 = result.get("image_base64")
            image_url = result.get("path") or (f"data:image/png;base64,{b64}" if b64 else None)
            if image_url:
                image_result = {
                    "type": "image",
                    "image_url": image_url,
                    "caption": caption,
                }
                if result.get("plotly_spec"):
                    # Interactive figure spec rides along; the SSE layer emits a
                    # "plotly" card with the PNG as fallback.
                    image_result["plotly_spec"] = result["plotly_spec"]
                if meta is not None:
                    image_result["meta"] = dict(meta)
                self.last_run_result = image_result
        # Keep the heavy base64/figure spec OUT of the LLM-facing tool result: they bloat context
        # and get truncated mid-string by the 8000-char tool-output slice → malformed JSON → the
        # model emits an empty response. The visual goes to the UI card above.
        if result.get("image_base64") or result.get("plotly_spec"):
            result = {k: v for k, v in result.items() if k not in ("image_base64", "plotly_spec")}
            result["image_attached"] = True
        return result

    def _datalab_image_cutout(
        self,
        fov_deg: float,
        ra: Optional[float] = None,
        dec: Optional[float] = None,
        target_name: Optional[str] = None,
        band: str = "g",
        catalog: str = "ls_dr9",
        endpoint: Optional[str] = None,
        title: Optional[str] = None,
    ) -> Dict[str, Any]:
        self.last_run_result = None
        try:
            ra_f, dec_f, label = self._datalab_coordinates(target_name=target_name, ra=ra, dec=dec)
            service = self._get_datalab_image_service()
            caption = title or f"Data Lab {band}-band cutout: {label}"
            result = service.cutout(
                ra_f,
                dec_f,
                float(fov_deg),
                band=band,
                catalog=catalog,
                endpoint=endpoint,
                title=caption,
            )
            # Auto-substitute a working band: models often stop instead of retrying, so if the
            # requested band has no usable tiles but another does (e.g. M31: no g/r/i, working
            # MzLS z), render that band deterministically and label the substitution clearly.
            no_image = not (result.get("image_base64") or result.get("path"))
            suggested = list(result.get("suggested_bands") or [])
            if no_image and suggested:
                sub_band = suggested[0]
                sub_caption = title or f"Data Lab {sub_band}-band cutout: {label} (requested {band}, not available here)"
                retry = service.cutout(
                    ra_f, dec_f, float(fov_deg), band=sub_band,
                    catalog=catalog, endpoint=endpoint, title=sub_caption,
                )
                if retry.get("image_base64") or retry.get("path"):
                    retry["band_substituted"] = {"requested": str(band), "used": sub_band}
                    retry["note"] = (
                        f"No usable {band}-band tiles at this position; rendered the {sub_band}-band "
                        f"cutout instead. State the substitution to the user."
                    )
                    return self._datalab_attach_image_result(retry, sub_caption)
            return self._datalab_attach_image_result(result, caption)
        except Exception as e:
            return self._datalab_error(e)

    def _datalab_color_image(
        self,
        fov_deg: float,
        ra: Optional[float] = None,
        dec: Optional[float] = None,
        target_name: Optional[str] = None,
        catalog: str = "ls_dr9",
        endpoint: Optional[str] = None,
        bands: Optional[List[str]] = None,
        q: float = 8.0,
        stretch: float = 0.5,
        title: Optional[str] = None,
    ) -> Dict[str, Any]:
        self.last_run_result = None
        try:
            ra_f, dec_f, label = self._datalab_coordinates(target_name=target_name, ra=ra, dec=dec)
            caption = title or f"Data Lab color image: {label}"
            result = self._get_datalab_image_service().color_image(
                ra_f,
                dec_f,
                float(fov_deg),
                catalog=catalog,
                endpoint=endpoint,
                bands=bands,
                q=float(q),
                stretch=float(stretch),
                title=caption,
            )
            return self._datalab_attach_image_result(result, caption)
        except Exception as e:
            return self._datalab_error(e)

    def _datalab_cutout_grid(
        self,
        peaks: List[Dict[str, Any]],
        fov_deg: float,
        band: str = "g",
        catalog: str = "ls_dr9",
        endpoint: Optional[str] = None,
        title: str = "Data Lab cutout grid",
    ) -> Dict[str, Any]:
        self.last_run_result = None
        try:
            result = self._get_datalab_image_service().cutout_grid(
                peaks,
                float(fov_deg),
                band=band,
                catalog=catalog,
                endpoint=endpoint,
                title=title,
            )
            return self._datalab_attach_image_result(result, title)
        except Exception as e:
            return self._datalab_error(e)

    def _svo_filter_wavelength(
        self,
        filter_id: Optional[str] = None,
        filter_ids: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        self.last_run_result = None
        try:
            client = self._get_svo_fps_client()
            if filter_ids:
                return {"success": True, "wavelengths": client.wavelengths(filter_ids)}
            if filter_id:
                return client.wavelength(filter_id)
            return {"success": False, "error": "Provide filter_id or filter_ids"}
        except Exception as e:
            return self._datalab_error(e)

    def _datalab_catalog_scatter(
        self,
        result_id: str,
        x_expr: str,
        y_expr: str,
        color_by: Optional[str] = None,
        invert_y: bool = False,
        invert_x: bool = False,
        title: str = "Data Lab catalog scatter",
        x_label: Optional[str] = None,
        y_label: Optional[str] = None,
        overlay_locus: Optional[str] = None,
    ) -> Dict[str, Any]:
        self.last_run_result = None
        try:
            from services import datalab_analysis

            result = datalab_analysis.catalog_scatter(
                result_id,
                x_expr,
                y_expr,
                color_by=color_by,
                invert_y=invert_y,
                invert_x=invert_x,
                title=title,
                x_label=x_label,
                y_label=y_label,
                overlay_locus=overlay_locus,
                result_store=self._get_datalab_result_store(),
            )
            return self._datalab_attach_image_result(result, title)
        except Exception as e:
            return self._datalab_error(e)

    def _datalab_sky_density_map(
        self,
        result_id: str,
        mode: str = "hist2d",
        ra_col: Optional[str] = None,
        dec_col: Optional[str] = None,
        count_col: str = "source_count",
        bins: int = 80,
        healpix_col: str = "healpix",
        nside: Optional[int] = None,
        order: str = "nested",
        matched_filter: bool = False,
        sigma_small: float = 1.0,
        sigma_large: float = 3.0,
        peak_threshold: float = 3.0,
        max_peaks: int = 10,
        log_scale: bool = True,
        title: str = "Data Lab sky density map",
    ) -> Dict[str, Any]:
        self.last_run_result = None
        try:
            from services import datalab_analysis

            result = datalab_analysis.sky_density_map(
                result_id,
                mode=mode,
                ra_col=ra_col,
                dec_col=dec_col,
                count_col=count_col,
                bins=bins,
                healpix_col=healpix_col,
                nside=nside,
                order=order,
                matched_filter=matched_filter,
                sigma_small=sigma_small,
                sigma_large=sigma_large,
                peak_threshold=peak_threshold,
                max_peaks=max_peaks,
                log_scale=log_scale,
                title=title,
                result_store=self._get_datalab_result_store(),
            )
            return self._datalab_attach_image_result(result, title)
        except Exception as e:
            return self._datalab_error(e)

    def _datalab_period_fold(
        self,
        result_id: str,
        time_col: str = "mjd",
        mag_col: str = "cmag",
        error_col: Optional[str] = "cerr",
        band: Optional[str] = None,
        band_col: str = "filter",
        min_frequency: float = 1.0,
        max_frequency: float = 10.0,
        title: str = "Data Lab period-folded light curve",
    ) -> Dict[str, Any]:
        self.last_run_result = None
        try:
            from services import datalab_analysis

            result = datalab_analysis.period_fold(
                result_id,
                time_col=time_col,
                mag_col=mag_col,
                error_col=error_col,
                band=band,
                band_col=band_col,
                min_frequency=min_frequency,
                max_frequency=max_frequency,
                title=title,
                result_store=self._get_datalab_result_store(),
            )
            return self._datalab_attach_image_result(result, title)
        except Exception as e:
            return self._datalab_error(e)

    def _datalab_sed_plot(
        self,
        result_id: str,
        row_index: int = 0,
        filter_columns: Optional[Dict[str, str]] = None,
        title: str = "Data Lab SED",
    ) -> Dict[str, Any]:
        self.last_run_result = None
        try:
            from services import datalab_analysis

            result = datalab_analysis.sed_plot(
                result_id,
                row_index=row_index,
                filter_columns=filter_columns,
                title=title,
                result_store=self._get_datalab_result_store(),
                svo_client=self._get_svo_fps_client(),
            )
            return self._datalab_attach_image_result(result, title)
        except Exception as e:
            return self._datalab_error(e)

    def _datalab_lss_wedge(
        self,
        result_id: str,
        ra_col: Optional[str] = None,
        dec_col: Optional[str] = None,
        z_col: str = "z",
        class_col: Optional[str] = None,
        pie_slice: bool = False,
        title: str = "Data Lab large-scale structure wedge",
    ) -> Dict[str, Any]:
        self.last_run_result = None
        try:
            from services import datalab_analysis

            result = datalab_analysis.lss_wedge(
                result_id,
                ra_col=ra_col,
                dec_col=dec_col,
                z_col=z_col,
                class_col=class_col,
                pie_slice=pie_slice,
                title=title,
                result_store=self._get_datalab_result_store(),
            )
            return self._datalab_attach_image_result(result, title)
        except Exception as e:
            return self._datalab_error(e)

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

            # If we have a preview PNG, set it as the run result for UI display.
            # NB: the SSE layer emits image cards from the `image_url` key ONLY, and the
            # preview lives in ~/quasar_data (not web-served) — so copy it into the served
            # /plots dir and reference that URL (base64 data-URI as a fallback). The old
            # `image_path` key was silently ignored and SkyView images never displayed.
            if result.get("preview_path"):
                image_url = None
                try:
                    import shutil
                    from services.plotting import PLOT_OUTPUT_DIR
                    os.makedirs(PLOT_OUTPUT_DIR, exist_ok=True)
                    dest_name = f"skyview_{uuid.uuid4().hex[:10]}.png"
                    shutil.copyfile(result["preview_path"], os.path.join(PLOT_OUTPUT_DIR, dest_name))
                    image_url = f"/plots/{dest_name}"
                except Exception:
                    try:
                        import base64 as _b64
                        with open(result["preview_path"], "rb") as _f:
                            image_url = "data:image/png;base64," + _b64.b64encode(_f.read()).decode()
                    except Exception:
                        image_url = None
                if image_url:
                    self.last_run_result = {
                        "type": "image",
                        "image_url": image_url,
                        "caption": f"SkyView {result.get('survey', 'DSS2')}: {target_name or f'RA={ra}, Dec={dec}'}",
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
                return {
                    "success": False,
                    "error": "No ALMA/archive search results to filter. This tool only sees archive search tables.",
                    "hint": (
                        "For Data Lab catalog data, apply the cut inside the query instead: "
                        "datalab_select_catalog_rows(value_cuts=[{'column': ..., 'op': ..., 'value': ...}]), "
                        "a WHERE clause in datalab_sql_query, or point_sources=true on the one-shot diagram tools."
                    ),
                }

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
            # This tool queries the ALMA Science Archive TAP service ONLY. Queries
            # against Data Lab schemas used to be forwarded and came back as a
            # silent success with 0 rows, which the model then reported as "no
            # data exists" (2026-07-04 live test P11: a DESI LRG selection).
            query_lower = str(query or "").lower()
            try:
                from services import datalab_registry as _dl_reg
                _dl_schemas = sorted(_dl_reg.DATALAB_CATALOGS.keys())
            except Exception:
                _dl_schemas = []
            _foreign = [s for s in _dl_schemas if re.search(rf"\b{re.escape(s)}\s*\.", query_lower)]
            if _foreign:
                return {
                    "success": False,
                    "error": (
                        f"advanced_search only queries the ALMA Science Archive (ivoa.obscore). "
                        f"The query references NOIRLab Data Lab schema(s): {', '.join(_foreign)}."
                    ),
                    "hint": "Run this SQL with datalab_sql_query (or a datalab_* builder tool) instead.",
                }
            if "obscore" not in query_lower:
                return {
                    "success": False,
                    "error": "advanced_search queries the ALMA ivoa.obscore table; the query does not reference it.",
                    "hint": (
                        "Use FROM ivoa.obscore for ALMA archive searches. For survey-catalog SQL "
                        "(Gaia/DES/DESI/NSC/SMASH/...), use datalab_sql_query instead."
                    ),
                }
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
        correct indirect approach (Î½_obs = Î½_rest / (1+z)).
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
        # Î½_obs_min (at z_max) to Î½_obs_max (at z_min)
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

    def _tap_obscore_dataframe(self, where_clause: str, *, max_results: int = 5000, order_by: str = "proposal_id") -> pd.DataFrame:
        query = select_obscore_query(where_clause, top=max_results, order_by=order_by)
        self._last_alma_tap_query = query
        self._last_alma_tap_url = "https://almascience.nrao.edu/tap"
        service = self.search_service.alminer_client._get_tap_service()
        result = service.search(query)
        df = result.to_table().to_pandas()
        if hasattr(self.search_service.alminer_client, "_standardize_columns"):
            return self.search_service.alminer_client._standardize_columns(df)
        return df

    def _redshifted_line_where(
        self,
        rest_species: str,
        redshift_min: float,
        redshift_max: float,
        science_category: str = "",
    ) -> Tuple[str, List[str]]:
        z_min = float(redshift_min)
        z_max = float(redshift_max)
        line_names = line_names_for_species(rest_species or "CO")
        conditions = []
        for line_name in line_names:
            rest_freq = LINE_REST_FREQ_GHZ[line_name]
            nu_low = rest_freq / (1.0 + max(z_min, z_max))
            nu_high = rest_freq / (1.0 + min(z_min, z_max))
            conditions.append(
                f"((frequency - 0.5*bandwidth/1e9) < {nu_high:.6f} "
                f"AND (frequency + 0.5*bandwidth/1e9) > {nu_low:.6f})"
            )
        where = "(" + " OR ".join(conditions) + ")"
        if science_category:
            safe_category = str(science_category).replace("'", "''")
            where += f" AND LOWER(scientific_category) LIKE '%{safe_category.lower()}%'"
        else:
            where += (
                " AND (LOWER(scientific_category) LIKE '%galaxy%' "
                "OR LOWER(scientific_category) LIKE '%cosmology%' "
                "OR LOWER(scientific_category) LIKE '%active%')"
            )
        return where, line_names

    def _query_alma_science_archive(
        self,
        query_type: str,
        cycle: int = None,
        target: str = "",
        band: int = None,
        max_resolution_arcsec: float = None,
        arrays: List[str] = None,
        lines: List[str] = None,
        topic_filter: str = "",
        redshift_min: float = None,
        redshift_max: float = None,
        rest_species: str = "CO",
        science_category: str = "",
        require_same_project: bool = True,
        include_adql: bool = True,
        max_results: int = 5000,
    ) -> Dict[str, Any]:
        """Run deterministic ALMA archive query templates for science questions."""
        started = time.perf_counter()
        query_type = str(query_type or "").strip()
        max_results = max(1, min(int(max_results or 5000), 20000))
        self._last_alma_tap_query = None
        self._last_alma_tap_url = None
        warnings: List[str] = []
        query_summary = ""
        try:
            if query_type == "cycle_solar_projects":
                if cycle is None:
                    return {"success": False, "error": "cycle is required"}
                where = (
                    f"{project_prefix_where(int(cycle))} AND ("
                    "LOWER(target_name) LIKE '%sun%' "
                    "OR LOWER(science_keyword) LIKE '%sun%' "
                    "OR LOWER(scientific_category) LIKE '%sun%' "
                    "OR LOWER(obs_title) LIKE '%sun%'"
                    ")"
                )
                df = self._tap_obscore_dataframe(where, max_results=max_results)
                result_df = summarize_projects(df)
                source = f"ALMA Cycle {cycle} solar projects"
                mode = "cycle_solar_projects"
                query_summary = f"Cycle {cycle} projects with solar/Sun terms in target, keyword, category, or title."

            elif query_type == "cycle_array_combo_projects":
                if cycle is None:
                    return {"success": False, "error": "cycle is required"}
                required_arrays = arrays or ["12m", "7m", "TP"]
                df = self._tap_obscore_dataframe(project_prefix_where(int(cycle)), max_results=max_results)
                result_df = projects_with_array_combo(df, required_arrays)
                source = f"ALMA Cycle {cycle} array combo projects"
                mode = "cycle_array_combo_projects"
                query_summary = f"Cycle {cycle} projects grouped by proposal_id requiring arrays {', '.join(required_arrays)}."

            elif query_type == "high_resolution_band_data":
                if not target:
                    return {"success": False, "error": "target is required"}
                normalized_target = normalize_target_alias(target)
                if max_resolution_arcsec is None:
                    max_resolution_arcsec = 0.1
                    warnings.append("Defaulted high-resolution threshold to <0.1 arcsec.")
                df = self.search_service.search_by_target(normalized_target, facility="ALMA", max_results=max_results)
                df = science_filter_band(df, band)
                df = science_filter_resolution(df, max_resolution_arcsec)
                if "dataproduct_type" in df.columns:
                    image_mask = df["dataproduct_type"].astype(str).str.contains("image|cube", case=False, regex=True, na=False)
                    df = df[image_mask].copy()
                result_df = summarize_projects(df)
                source = f"ALMA {normalized_target} Band {band or 'any'} high-resolution candidates"
                mode = "high_resolution_band_data"
                query_summary = (
                    f"Target search for {normalized_target}, Band {band or 'any'}, "
                    f"resolution < {max_resolution_arcsec} arcsec, image/cube products when available."
                )

            elif query_type == "line_set_projects":
                required_lines = lines or ["12CO", "13CO", "C18O"]
                requested_band = band or 6
                topic = str(topic_filter or "").strip()
                where_parts = [f"(band_list LIKE '%{requested_band}%')"]
                if topic:
                    safe_topic = topic.replace("'", "''").lower()
                    where_parts.append(
                        "("
                        f"LOWER(science_keyword) LIKE '%{safe_topic}%' "
                        f"OR LOWER(scientific_category) LIKE '%{safe_topic}%' "
                        f"OR LOWER(obs_title) LIKE '%{safe_topic}%'"
                        ")"
                    )
                where = " AND ".join(where_parts)
                df = self._tap_obscore_dataframe(where, max_results=max_results)
                result_df = projects_covering_all_lines(df, required_lines, z=0.0)
                source = f"ALMA Band {requested_band} projects covering {', '.join(required_lines)}"
                mode = "line_set_projects"
                query_summary = (
                    f"Band {requested_band} rows grouped by proposal_id; retained projects covering all requested "
                    f"rest-frame lines: {', '.join(required_lines)}."
                )

            elif query_type == "redshifted_line_projects":
                z_min = 1.0 if redshift_min is None else float(redshift_min)
                z_max = 2.0 if redshift_max is None else float(redshift_max)
                where, line_names = self._redshifted_line_where(rest_species or "CO", z_min, z_max, science_category)
                df = self._tap_obscore_dataframe(where, max_results=max_results)
                result_df = redshifted_line_projects(df, rest_species=rest_species or "CO", z_min=z_min, z_max=z_max)
                source = f"ALMA {rest_species or 'CO'} redshifted line projects z={z_min:g}-{z_max:g}"
                mode = "redshifted_line_projects"
                if require_same_project is False:
                    warnings.append("require_same_project=False is accepted for API compatibility; this summary is still grouped by proposal_id.")
                query_summary = (
                    f"Frequency-containment query for {', '.join(line_names)} shifted to z={z_min:g}-{z_max:g}, "
                    "restricted to extragalactic science categories unless science_category is supplied."
                )

            elif query_type == "bandwidth_switching_candidates":
                where = project_prefix_where(int(cycle)) if cycle is not None else "proposal_id IS NOT NULL"
                df = self._tap_obscore_dataframe(where, max_results=max_results)
                result_df = bandwidth_switching_candidates(df)
                source = "ALMA bandwidth-switching calibration candidates" + (f" Cycle {cycle}" if cycle is not None else "")
                mode = "bandwidth_switching_candidates"
                warnings.append("Bandwidth Switching likelihood is inferred from public spectral setup metadata; it is not proof of calibration intent.")
                query_summary = "Projects scored by spectral-window count, bandwidth diversity, tuning diversity, and calibration-like metadata."

            else:
                return {"success": False, "error": f"Unknown query_type: {query_type}"}

            elapsed_ms = round((time.perf_counter() - started) * 1000, 1)
            provenance = {
                "archive": "ALMA Science Archive",
                "tap_url": getattr(self, "_last_alma_tap_url", None),
                "adql": getattr(self, "_last_alma_tap_query", None) if include_adql else None,
                "elapsed_ms": elapsed_ms,
                "fresh_query": True,
            }
            self.last_search_results = result_df
            self.last_run_result = {
                "type": "data",
                "data": result_df,
                "source": source,
                "filter_label": source,
                "tool_name": "query_alma_science_archive",
            }
            unique_projects = int(result_df["proposal_id"].nunique()) if "proposal_id" in result_df.columns else len(result_df)
            return {
                "success": True,
                "mode": mode,
                "count": len(result_df),
                "unique_projects": unique_projects,
                "source": source,
                "query_summary": query_summary,
                "results": result_df.head(100).to_dict("records") if not result_df.empty else [],
                "warnings": warnings,
                "provenance": provenance,
                "note": "Full result table is shown in the UI data card.",
            }
        except Exception as e:
            import traceback
            logger.error("ALMA science query failed: %s\n%s", e, traceback.format_exc())
            return {"success": False, "error": str(e), "query_type": query_type}

    def _match_cross_archive_sources(
        self,
        catalog_name: str = "perseus_protostars",
        sources: List[Dict[str, Any]] = None,
        archives: List[str] = None,
        radius_arcsec: float = 5.0,
        max_sources: int = 12,
        max_alma_rows: int = 5000,
        max_mast_results_per_source: int = 80,
        require_all_archives: bool = False,
    ) -> Dict[str, Any]:
        """Cross-match a built-in or inline source catalog against archives."""
        requested_archives = {str(a).upper() for a in (archives or ["ALMA", "JWST"])}
        radius_arcsec = max(0.5, min(float(radius_arcsec or 5.0), 60.0))
        requested_mast_missions = sorted(requested_archives & {"JWST", "HST", "TESS", "KEPLER", "K2", "GALEX", "SWIFT"})
        try:
            catalog_label, source_catalog = normalize_source_catalog(
                catalog_name=catalog_name,
                sources=sources,
                max_sources=max_sources,
            )
        except ValueError as e:
            return {"success": False, "error": str(e)}

        archive_errors: List[str] = []
        alma_df = pd.DataFrame()
        alma_matches = pd.DataFrame()
        mast_by_source: Dict[str, pd.DataFrame] = {}
        if "ALMA" in requested_archives:
            try:
                service = self.search_service.alminer_client._get_tap_service()
                query = alma_bulk_cone_adql(source_catalog, radius_arcsec=radius_arcsec, top=max_alma_rows)
                self._last_alma_tap_query = query
                self._last_alma_tap_url = "https://almascience.nrao.edu/tap"
                alma_result = service.search(query)
                alma_df = alma_result.to_table().to_pandas()
                if hasattr(self.search_service.alminer_client, "_standardize_columns"):
                    alma_df = self.search_service.alminer_client._standardize_columns(alma_df)
                alma_matches = attach_nearest_source(alma_df, source_catalog, radius_arcsec=radius_arcsec)
            except Exception as e:
                import traceback
                logger.error("ALMA cross-match query failed: %s\n%s", e, traceback.format_exc())
                archive_errors.append(f"ALMA TAP failed: {e}")

        if {"MAST", "JWST", "HST", "TESS", "KEPLER", "K2", "GALEX", "SWIFT"} & requested_archives:
            mission = requested_mast_missions[0] if len(requested_mast_missions) == 1 else None
            for source in source_catalog:
                try:
                    mast_by_source[source["source_name"]] = self.mast_client.search_by_position(
                        float(source["ra"]),
                        float(source["dec"]),
                        radius_arcmin=radius_arcsec / 60.0,
                        mission=mission,
                        max_results=max_mast_results_per_source,
                    )
                except Exception as e:
                    logger.error("MAST cross-match query failed for %s: %s", source["source_name"], e)
                    archive_errors.append(f"MAST query failed for {source['source_name']}: {e}")
                    mast_by_source[source["source_name"]] = pd.DataFrame()

        summary = summarize_cross_archive_matches(
            source_catalog,
            alma_matches,
            mast_by_source,
            sorted(requested_archives),
            require_all_archives=bool(require_all_archives),
        )
        self.last_search_results = summary
        self.last_run_result = {
            "type": "data",
            "data": summary,
            "source": f"{' + '.join(sorted(requested_archives))} {catalog_label} Cross-match",
            "filter_label": f"{catalog_label} within {radius_arcsec:g} arcsec",
            "tool_name": "match_cross_archive_sources",
            "warnings": archive_errors,
            "partial": bool(archive_errors),
        }

        return {
            "success": True,
            "mode": "cross_archive_source_match",
            "catalog_name": catalog_label,
            "archives": sorted(requested_archives),
            "sources_tested": len(source_catalog),
            "matched_sources": len(summary),
            "radius_arcsec": radius_arcsec,
            "alma_rows": len(alma_df) if alma_df is not None else 0,
            "archive_errors": archive_errors,
            "results": summary.head(100).to_dict("records") if not summary.empty else [],
            "note": "Full cross-match table is shown in the UI data card with sky coordinates.",
        }

    def _match_perseus_protostars_alma_jwst(
        self,
        radius_arcsec: float = 5.0,
        max_sources: int = 12,
        max_alma_rows: int = 5000,
        max_mast_results_per_source: int = 80,
    ) -> Dict[str, Any]:
        """Backward-compatible wrapper for the generic cross-archive matcher."""
        result = self._match_cross_archive_sources(
            catalog_name="perseus_protostars",
            archives=["ALMA", "JWST"],
            radius_arcsec=radius_arcsec,
            max_sources=max_sources,
            max_alma_rows=max_alma_rows,
            max_mast_results_per_source=max_mast_results_per_source,
            require_all_archives=True,
        )
        if result.get("success") and self.last_run_result:
            self.last_run_result["tool_name"] = "match_perseus_protostars_alma_jwst"
        if result.get("success"):
            result["mode"] = "perseus_alma_jwst_cross_match"
        return result

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

    def _overlay_region_coordinates(
        self,
        region: str,
        ra_deg: float = None,
        dec_deg: float = None,
    ) -> Optional[Tuple[float, float, str]]:
        if ra_deg is not None and dec_deg is not None:
            try:
                ra = float(ra_deg)
                dec = float(dec_deg)
                if 0.0 <= ra < 360.0 and -90.0 <= dec <= 90.0:
                    return ra, dec, str(region or f"RA {ra:.5f} Dec {dec:.5f}").strip()
            except (TypeError, ValueError):
                return None

        text = str(region or "").strip()
        coord_match = re.search(
            r"(?:ra\s*[=:]?\s*)?(\d+(?:\.\d+)?)\s*[, ]+\s*(?:dec\s*[=:]?\s*)?([+-]?\d+(?:\.\d+)?)",
            text,
            flags=re.IGNORECASE,
        )
        if coord_match:
            try:
                ra = float(coord_match.group(1))
                dec = float(coord_match.group(2))
                if 0.0 <= ra < 360.0 and -90.0 <= dec <= 90.0:
                    return ra, dec, text
            except (TypeError, ValueError):
                pass

        key = str(region or "").strip().lower().replace(" ", "")
        if key in {"hudf", "hubbleultradeepfield", "ultradeepfield"}:
            return 53.1625, -27.7914, "HUDF"
        if not text:
            return None

        try:
            from integrations.alminer_client import _resolve_simbad_cached
            resolved_ra, resolved_dec = _resolve_simbad_cached(text)
            if resolved_ra is not None and resolved_dec is not None:
                return float(resolved_ra), float(resolved_dec), text
        except Exception:
            pass

        try:
            from astropy.coordinates import SkyCoord
            coord = SkyCoord.from_name(text)
            return float(coord.ra.deg), float(coord.dec.deg), text
        except Exception:
            return None
        return None

    def _mast_product_access_url(self, row: Dict[str, Any]) -> str:
        import urllib.parse

        for key in ("access_url", "dataURL", "data_url", "url"):
            value = str(row.get(key) or "").strip()
            if value.startswith("http"):
                return value
        data_uri = str(row.get("dataURI") or row.get("data_uri") or "").strip()
        if data_uri:
            return "https://mast.stsci.edu/api/v0.1/Download/file?uri=" + urllib.parse.quote(data_uri, safe="")
        return ""

    def _pick_mast_fits_product(self, products: pd.DataFrame, max_product_mb: float) -> Optional[Dict[str, Any]]:
        if products is None or products.empty:
            return None
        candidates = products.copy()
        filename_col = next((c for c in ("productFilename", "filename", "File") if c in candidates.columns), None)
        if filename_col:
            candidates = candidates[candidates[filename_col].astype(str).str.contains(r"\.fits?(\.gz)?$", case=False, regex=True, na=False)]
        type_col = next((c for c in ("productType", "product_type") if c in candidates.columns), None)
        if type_col:
            science = candidates[candidates[type_col].astype(str).str.upper().str.contains("SCIENCE", na=False)]
            if not science.empty:
                candidates = science
        size_col = next((c for c in ("size_mb", "Size (MB)", "productSize", "size") if c in candidates.columns), None)
        if size_col:
            sizes = pd.to_numeric(candidates[size_col], errors="coerce")
            if size_col not in {"size_mb", "Size (MB)"}:
                sizes = sizes / (1024 * 1024)
            under = candidates[(sizes.isna()) | (sizes <= float(max_product_mb))]
            if not under.empty:
                candidates = under
        for _, product in candidates.iterrows():
            row = product.to_dict()
            url = self._mast_product_access_url(row)
            if url:
                row["access_url"] = url
                return row
        return None

    def _find_alma_overlay_product(self, ra_deg: float, dec_deg: float, radius_arcmin: float, max_product_mb: float) -> Optional[Dict[str, Any]]:
        radius_deg = max(float(radius_arcmin or 1.0), 0.1) / 60.0
        query = f"""
SELECT TOP 100
       target_name, proposal_id, member_ous_uid, s_ra, s_dec, band_list,
       dataproduct_type, s_resolution, frequency, bandwidth
FROM ivoa.obscore
WHERE CONTAINS(POINT('ICRS', s_ra, s_dec), CIRCLE('ICRS', {ra_deg:.8f}, {dec_deg:.8f}, {radius_deg:.8f})) = 1
ORDER BY s_resolution
"""
        service = self.search_service.alminer_client._get_tap_service()
        result = service.search(query)
        alma_df = result.to_table().to_pandas()
        if hasattr(self.search_service.alminer_client, "_standardize_columns"):
            alma_df = self.search_service.alminer_client._standardize_columns(alma_df)
        for mous_uid in unique_values(alma_df, ["member_ous_uid"], limit=8):
            listing = self.datalink_client.list_files(mous_uid=mous_uid)
            if not listing.get("success"):
                continue
            files = sorted(listing.get("files", []), key=product_rank)
            for file_info in files:
                size_mb = float(file_info.get("size_mb") or 0)
                if size_mb and size_mb > float(max_product_mb):
                    continue
                if is_fits_product(file_info) and file_info.get("access_url"):
                    selected = dict(file_info)
                    selected["member_ous_uid"] = mous_uid
                    return selected
        return None

    def _overlay_archive_images(
        self,
        region: str,
        ra_deg: float = None,
        dec_deg: float = None,
        base_archive: str = "MAST",
        base_collection: str = "JWST",
        contour_archive: str = "ALMA",
        radius_arcmin: float = 1.0,
        max_product_mb: float = 150.0,
    ) -> Dict[str, Any]:
        """Find archive FITS products around a region and overlay ALMA contours."""
        coords = self._overlay_region_coordinates(region, ra_deg=ra_deg, dec_deg=dec_deg)
        if not coords:
            return {"success": False, "error": f"Could not resolve overlay region: {region}. Provide ra_deg and dec_deg for arbitrary regions."}
        if str(base_archive or "MAST").upper() != "MAST" or str(contour_archive or "ALMA").upper() != "ALMA":
            return {"success": False, "error": "overlay_archive_images currently supports MAST/JWST base images with ALMA contours."}

        ra_deg, dec_deg, label = coords
        try:
            mast_obs = self.mast_client.search_by_position(
                ra_deg,
                dec_deg,
                radius_arcmin=float(radius_arcmin or 1.0),
                mission=base_collection or "JWST",
                max_results=80,
            )
            if mast_obs is None or mast_obs.empty:
                return {"success": False, "error": f"No {base_collection or 'JWST'} MAST observations found near {label}."}
            mast_products = self.mast_client.get_product_list(mast_obs, productType="SCIENCE", extension="fits")
            base_product = self._pick_mast_fits_product(mast_products, max_product_mb=max_product_mb)
            if not base_product:
                return {"success": False, "error": f"No small science FITS product found in MAST near {label}."}

            contour_product = self._find_alma_overlay_product(ra_deg, dec_deg, float(radius_arcmin or 1.0), max_product_mb)
            if not contour_product:
                return {"success": False, "error": f"No small public ALMA FITS product found near {label}."}

            from services.fits_service import overlay_fits_images
            result = overlay_fits_images(
                base_product["access_url"],
                contour_product["access_url"],
                base_label=f"{base_collection or 'JWST'} {label}",
                contour_label=f"ALMA {label}",
                base_cmap="inferno",
                contour_levels=8,
            )
            result["region"] = label
            result["selected_products"] = {
                "base": {
                    "filename": base_product.get("productFilename") or base_product.get("filename"),
                    "access_url": base_product.get("access_url"),
                },
                "contour": {
                    "filename": contour_product.get("filename"),
                    "access_url": contour_product.get("access_url"),
                    "member_ous_uid": contour_product.get("member_ous_uid"),
                },
            }
            if result.get("success"):
                self.last_run_result = {
                    "type": "image",
                    "image_url": result["image_path"],
                    "caption": result.get("caption", ""),
                }
            return result
        except Exception as e:
            return {"success": False, "error": str(e), "region": region}

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

    @log_tool
    def _search_papers_by_observation_identifier(
        self,
        identifier: str,
        max_results: int = 20,
        facility: str = "ALMA",
    ) -> Dict[str, Any]:
        """Search ADS for papers explicitly tied to an archive/project identifier."""
        try:
            if not self.ads_client:
                return {"success": False, "error": "NASA ADS Client not initialized (check API Key)"}

            identifiers_to_search = [str(identifier or "").strip()]
            derived_identifiers = []
            try:
                derived_identifiers = self._derive_archive_identifiers_for_paper_search(identifier)
                for derived in derived_identifiers:
                    if derived and derived not in identifiers_to_search:
                        identifiers_to_search.append(derived)
            except Exception as _derive_err:
                print(f"[ADS identifier] Archive identifier derivation failed (non-fatal): {_derive_err}")

            merged_papers: Dict[str, Dict[str, Any]] = {}
            ads_queries = []
            identifier_types = {}
            for search_identifier in identifiers_to_search:
                result = self.ads_client.search_by_observation_identifier(
                    identifier=search_identifier,
                    max_results=max_results,
                    facility=facility,
                )
                ads_queries.append(result.get("query", ""))
                identifier_types[search_identifier] = result.get("identifier_type", "identifier")
                for paper in result.get("papers", []):
                    key = paper.get("bibcode") or paper.get("doi") or paper.get("title") or str(id(paper))
                    if key not in merged_papers:
                        merged_papers[key] = paper
                    else:
                        current_links = merged_papers[key].setdefault("observation_links", [])
                        for link in paper.get("observation_links", []):
                            if link not in current_links:
                                current_links.append(link)

            papers_list = list(merged_papers.values())[:max_results]
            ads_query = " OR ".join(q for q in ads_queries if q)

            try:
                oalex = self.openalex_client
                dois = [p.get("doi", "") for p in papers_list if p.get("doi")]
                if dois and oalex:
                    enrichments = oalex.enrich_batch_dois(dois)
                    if enrichments:
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
            except Exception as _enrich_err:
                print(f"[OpenAlex] Enrichment failed (non-fatal): {_enrich_err}")

            self.last_run_result = {
                "type": "papers",
                "papers": papers_list,
                "source": f"ADS identifier: {identifier}",
                "paper_provenance": {
                    "identifier": identifier,
                    "identifier_type": identifier_types.get(str(identifier or "").strip(), "identifier"),
                    "derived_identifiers": derived_identifiers,
                    "ads_query": ads_query,
                    "facility": facility,
                },
            }
            return {
                "success": True,
                "count": len(papers_list),
                "identifier": identifier,
                "identifier_type": identifier_types.get(str(identifier or "").strip(), "identifier"),
                "derived_identifiers": derived_identifiers,
                "ads_query": ads_query,
                "papers": papers_list,
                "top_title": papers_list[0]["title"] if papers_list else "No results",
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _auto_link_project_papers_from_result(
        self,
        tool_name: str,
        tool_result: Any,
        max_project_codes: int = 3,
        max_results: int = 20,
    ) -> Optional[Dict[str, Any]]:
        """Build an exact project-code paper result for ALMA archive searches."""
        return build_exact_project_paper_links(
            self.ads_client,
            tool_name,
            tool_result,
            max_project_codes=max_project_codes,
            max_results=max_results,
        )

    def _derive_archive_identifiers_for_paper_search(self, identifier: str) -> List[str]:
        """Resolve a MOUS/dataset identifier to proposal/project IDs when possible."""
        raw = str(identifier or "").strip()
        if not raw or not self.search_service:
            return []

        id_type = self.ads_client.classify_observation_identifier(raw) if self.ads_client else "identifier"
        if id_type == "project_code":
            return []

        keyword_by_type = {
            "mous_uid": "member_ous_uid",
            "asdm_uid": "asdm_uid",
            "dataset_id": "obs_publisher_did",
            "uid": "member_ous_uid",
        }
        keyword = keyword_by_type.get(id_type)
        if not keyword:
            return []

        df = self.search_service.search_alma_with_keywords({keyword: raw})
        if df is None or not hasattr(df, "columns") or df.empty:
            return []

        derived = []
        for col in ("proposal_id", "project_code"):
            if col in df.columns:
                for value in df[col].dropna().astype(str).unique().tolist()[:5]:
                    clean = value.strip()
                    if clean and clean not in derived:
                        derived.append(clean)
        return derived

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

            response = self.client.responses.create(
                model=self.config.model,
                instructions="You are a meticulous scientific literature analyst who produces rigorous, well-cited consensus evaluations.",
                input=consensus_prompt,
                temperature=0.1,
                max_output_tokens=4000,
            )
            
            analysis = response.output_text.strip()
            
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

            response = self.client.responses.create(
                model=self.config.model,
                instructions="You are an expert radio astronomy data reduction specialist.",
                input=prompt,
                temperature=0.2,
                max_output_tokens=4000
            )

            script = response.output_text.strip()

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

    def _download_alma_data(self, dry_run: bool = False) -> Dict[str, Any]:
        """Download ALMA data for observations in current context"""
        try:
            if not hasattr(self, 'last_search_results') or self.last_search_results is None or self.last_search_results.empty:
                 return {"success": False, "error": "No results available to download."}
            
            msg = self.search_service.download_alma_data(self.last_search_results, dry_run=dry_run)
            return {"success": True, "message": msg}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _find_alma_line_coverage(
        self,
        target_name: str,
        species: str,
        transition: str,
        redshift: Optional[float] = None,
        tolerance_mhz: float = 0.0,
        velocity_width_kms: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Resolve one transition and verify exact ALMA SPW coverage."""
        try:
            from services.spectral_line_explorer import (
                find_alma_line_coverage as run_alma_line_coverage,
            )

            result = run_alma_line_coverage(
                target_name=target_name,
                species=species,
                transition=transition,
                redshift=redshift,
                tolerance_mhz=tolerance_mhz,
                velocity_width_kms=velocity_width_kms,
            )
            if not result.get("success"):
                return result
            project_rows = []
            for project in result.get("projects") or []:
                project_rows.append(
                    {
                        "proposal_id": project.get("proposal_id"),
                        "target_name": project.get("target_name"),
                        "covered_line_count": project.get("covered_line_count"),
                        "all_lines_full": project.get("all_lines_full"),
                        "minimum_edge_margin_mhz": project.get(
                            "minimum_edge_margin_mhz"
                        ),
                        "angular_separation_arcsec": project.get(
                            "angular_separation_arcsec"
                        ),
                        "best_angular_resolution_arcsec": project.get(
                            "best_angular_resolution_arcsec"
                        ),
                        "total_exposure_seconds": project.get(
                            "total_exposure_seconds"
                        ),
                        "archive_url": project.get("archive_url"),
                    }
                )
            frame = pd.DataFrame(project_rows)
            self.last_search_results = frame
            self.last_run_result = {
                "type": "data",
                "data": frame,
                "source": (
                    f"Exact ALMA coverage: {target_name} "
                    f"{species}({transition})"
                ),
            }
            selected = result.get("selected_line") or {}
            target = result.get("target") or {}
            return {
                "success": True,
                "target_name": target_name,
                "species": species,
                "transition": transition,
                "rest_frequency_ghz": selected.get("frequency_ghz"),
                "observed_frequency_ghz": selected.get(
                    "observed_frequency_ghz"
                ),
                "redshift": target.get("redshift"),
                "redshift_source": target.get("redshift_source"),
                "coordinates": {
                    "ra_deg": target.get("ra_deg"),
                    "dec_deg": target.get("dec_deg"),
                    "source": target.get("coordinate_source"),
                },
                "project_count": result.get("project_count", 0),
                "projects": project_rows[:25],
                "backend": result.get("backend"),
                "degraded": result.get("degraded", False),
                "warnings": result.get("warnings") or [],
                "line_explorer_url": result.get("deep_link"),
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "target_name": target_name,
                "species": species,
                "transition": transition,
            }

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
            
            # Get coordinates from first match. astroquery <0.4.8 returns
            # 'RA'/'DEC' sexagesimal strings; newer versions return lowercase
            # 'ra'/'dec' already in degrees.
            cols = {c.lower(): c for c in result.colnames}
            ra_val = result[cols['ra']][0]
            dec_val = result[cols['dec']][0]
            try:
                coord = SkyCoord(ra=float(ra_val) * u.deg, dec=float(dec_val) * u.deg)
            except (TypeError, ValueError):
                coord = SkyCoord(str(ra_val), str(dec_val), unit=(u.hourangle, u.deg))
            
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
            return {
                "success": False,
                "error": "No ALMA/archive search results to filter. This tool only sees archive search tables.",
                "hint": (
                    "For Data Lab catalog data, apply the cut inside the query instead: "
                    "datalab_select_catalog_rows(value_cuts=[{'column': ..., 'op': ..., 'value': ...}]), "
                    "a WHERE clause in datalab_sql_query, or point_sources=true on the one-shot diagram tools."
                ),
            }
        
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

    # ==================================================================
    # CONDUCTOR TOOL EXECUTOR — Bridges sub-agents to the full tool set
    # ==================================================================

    def _conductor_tool_executor(self, task_description: str, dep_context: str = "", subtask_model: str = "") -> str:
        """
        Execute a sub-task using a mini Responses API call with full tool access.

        This is the callback passed to Conductor so each DAG node can use
        all 28+ registered tools (ALMA search, ADS, Splatalogue, etc.).

        Parameters
        ----------
        subtask_model : str, optional
            The model to use for this subtask, as determined by ModelRouter.
            If empty, falls back to conductor_model (deepseek-v4-pro).
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
            # otherwise fall back to conductor_model (deepseek-v4-pro)
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
                    try:
                        _status_args = json.loads(fn_args) if fn_args else {}
                    except Exception:
                        _status_args = {}
                    _status_label = self._tool_status_label(fn_name, _status_args)
                    if on_status := getattr(self, "_last_on_status", None):
                        on_status(_status_label, "running")
                    _rr_before = self.last_run_result
                    result = self._dispatch_tool_call(fn_name, fn_args)
                    if on_status:
                        on_status(_status_label, "completed")

                    # ── Capture image results IMMEDIATELY after each tool call ──
                    _conductor_image_result = (
                        self.last_run_result
                        if _run_result_is_new(_rr_before, self.last_run_result)
                        else None
                    )
                    if isinstance(_conductor_image_result, dict) and _conductor_image_result.get("type") == "image":
                        if hasattr(self, '_conductor_images'):
                            lock = getattr(self, '_conductor_images_lock', None)
                            if lock:
                                with lock:
                                    self._conductor_images.append(_conductor_image_result.copy())
                            else:
                                self._conductor_images.append(_conductor_image_result.copy())
                        img_url = _conductor_image_result.get("image_url", "")
                        caption = _conductor_image_result.get("caption", "")
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
                    model=model_to_use,
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
            # CRITICAL: Preserve self.last_search_results so that subsequent sequential subtasks
            # (e.g. check_co_lines following search_by_target) can access the cached DataFrame.
            self.last_run_result = None
            self._accumulated_run_results = []
            import gc; gc.collect()

    def _record_tool_trace(self, tool_name: str, args, result_str: str,
                           result_obj=None) -> None:
        """Append a compact record of one executed tool call to the per-request
        trace (surfaced as the SSE ``tool_trace`` event; consumed by the UI's
        debug view and by Benchmark/datalabbench). Never raises.

        ``result_obj`` is the untruncated result dict when the caller has it —
        structured fields are read from it so an 8000-char ``result_str``
        slice can never cost the trace its SQL/rowcount."""
        try:
            trace = self._accumulated_tool_trace
            if len(trace) >= 200:
                return
            ok = True
            sql = ""
            rowcount = None
            try:
                parsed = result_obj
                if not isinstance(parsed, dict):
                    parsed = json.loads(result_str) if result_str else {}
                if isinstance(parsed, dict):
                    if parsed.get("error") or parsed.get("success") is False:
                        ok = False
                    # Structured fields survive even when the raw output below
                    # is truncated mid-JSON (the trace consumers rely on them).
                    sql = str(parsed.get("query_summary")
                              or parsed.get("validated_sql") or "")
                    rowcount = parsed.get("reported_count", parsed.get("rowcount"))
            except Exception:
                pass
            if not sql and isinstance(args, dict) and isinstance(args.get("sql"), str):
                sql = args["sql"]
            record = {
                "name": str(tool_name or ""),
                "arguments": args if isinstance(args, dict) else {},
                "output": (result_str or "")[:2000],
                "ok": ok,
            }
            if sql:
                record["sql"] = sql[:1500]
            if isinstance(rowcount, (int, float)):
                record["rowcount"] = int(rowcount)
            trace.append(record)
        except Exception:
            pass

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
        args = _unescape_tool_args(args)

        tool = self.tool_registry.get_tool(tool_name)
        result_obj = None
        if tool:
            try:
                result = tool.execute(**args)
                result_obj = result if isinstance(result, dict) else None
                result_str = _json.dumps(result, default=str)[:8000]
            except Exception as e:
                result_str = _json.dumps({"error": str(e)})
        else:
            result_str = _json.dumps({"error": f"Unknown tool: {tool_name}"})
        self._record_tool_trace(tool_name, args, result_str, result_obj=result_obj)
        return result_str

    # ==================================================================
    # RESPONSES API METHOD (New Architecture)
    # ==================================================================
    
    # ── Knowledge cutoff detection ──────────────────────────────────────────

    # LLM training knowledge cutoff — GPT-4o data ends ~Oct 2024
    _LLM_CUTOFF_YEAR = 2024
    _LLM_CUTOFF_MONTH = 10   # October 2024

    # ── Live-data query detection ───────────────────────────────────────
    # Queries served by QUASAR's built-in live-data tools — observation
    # archives (ALMA/CADC/...), ADS/arXiv papers, ALeRCE/ZTF alerts,
    # Data Lab catalogs, SparCL spectra, NED photometry, SIMBAD/VizieR
    # lookups, HiPS imagery/cutouts, cone searches, crossmatches.
    # These hit live databases directly; web search adds nothing and just
    # clutters the response with an irrelevant "From the Web" section.
    _LIVE_DATA_KEYWORDS_RE = re.compile(
        r'\b(?:observations?|data|archives?|band\s*\d|'
        r'search_by|search_cadc|member_ous|mous|project_code|fits|'
        r'alerts?|light\s*curves?|lightcurves?|cutouts?|cone\s*search(?:es)?|'
        r'cross-?match(?:es|ing|ed)?|photometry|spectra|spectrum|'
        r'catalogs?|catalogues?|sky\s*map|skymap|'
        r'ztf|alerce|sparcl|data\s*lab|vlass|hips|aladin|'
        r'simbad|vizier|gaia|sdss|desi|pan-?starrs|2mass|'
        r'decam|legacy\s*surveys?|smash|delve|skymapper|(?:un|all|cat)wise|'
        r'nsc\s*dr\d|des\s*dr\d|(?:color|colour)[-\s]magnitude|cmds?|'
        r'hr\s*diagrams?|overdensit(?:y|ies)|proper\s*motions?|parallax(?:es)?)\b'
    )
    _LIVE_DATA_VERBS_RE = re.compile(
        r'\b(?:find|search|show|get|list|query|look\s*up|plot|display|select|identify)\b.*'
        r'\b(?:observations?|data|archives?|images?|spectra|spectrum|alerts?|'
        r'stars?|sources?|magnitudes?|cutouts?|diagrams?|candidates?)\b'
    )
    _LIVE_DATA_FACILITY_RE = re.compile(
        r'\b(?:alma|vla|vlba|gbt|ngvla|jwst|hst|gemini|jcmt|cfht|chandra|xmm|'
        r'ztf|desi|sdss|gaia|vlass|euclid|rubin|lsst)\b.*'
        r'\b(?:observations?|data|of)\b'
    )
    # Cone-search-shaped queries: "within 2 arcminutes of M87", "sources
    # around NGC 1275 within a 30 arcsec radius", ...
    _LIVE_DATA_CONE_RE = re.compile(
        r'\b(?:within|around|near)\b.*?\b\d+(?:\.\d+)?\s*'
        r'(?:arc\s*sec(?:onds?)?|arc\s*min(?:utes?)?|deg(?:rees?)?)\b'
    )
    # Papers come from NASA ADS (search_papers tool), not web search.
    # "recent papers on X" should NOT trigger web search just because
    # of the word "recent".
    _PAPER_QUERY_RE = re.compile(
        r'\b(?:papers?|publications?|articles?|literature|studies)\b'
    )

    def _is_live_data_query(self, query: str) -> bool:
        """True when the query is answered by QUASAR's built-in live-data
        tools (archives, papers, alerts, catalogs, imagery, spectra,
        photometry, cone searches) — web search adds nothing for these."""
        _q = query.lower()
        return bool(
            self._LIVE_DATA_KEYWORDS_RE.search(_q)
            or self._LIVE_DATA_VERBS_RE.search(_q)
            or self._LIVE_DATA_FACILITY_RE.search(_q)
            or self._LIVE_DATA_CONE_RE.search(_q)
            or self._PAPER_QUERY_RE.search(_q)
        )

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

        # ── Skip web search for live-data / paper queries ───────────────
        # These are served by dedicated live databases (ALMA archive, ADS,
        # ALeRCE alerts, Data Lab, SparCL, NED, HiPS imagery, ...) — web
        # search adds nothing and just wastes time / clutters the response.
        if self._is_live_data_query(_q):
            return None

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

    def _detect_web_search_needed_via_llm(self, query: str) -> bool:
        """Use the configured fast model to classify if a query requires web search."""
        try:
            # 1. Direct keyword override for policy/time-sensitive queries
            query_lower = query.lower()
            policy_keywords = [
                "proprietary period", "proprietary time", "deadline", "policy", "policies",
                "guideline", "guidelines", "regulation", "regulations", "cycle 12", "cycle 13", "cycle 14",
                "call for proposals", "call-for-proposals", "proposers guide", "proposer's guide"
            ]
            if any(kw in query_lower for kw in policy_keywords):
                print(f"[WEB SEARCH DETECTION] Forcing web search due to policy keywords in query: '{query}'")
                return True

            from core.llm_client import LLMClient
            intent_model = os.getenv("QUASAR_WEB_INTENT_MODEL") or os.getenv("QUASAR_FAST_MODEL", "deepseek-v4-flash")
            client = LLMClient(model=intent_model)
            
            # Get recent conversation history (e.g. last 2 turns / 4 messages) to provide context
            recent_turns = self.memory.get_last_n_turns(2)
            history_str = ""
            if recent_turns:
                history_str = "Recent Conversation History:\n"
                for msg in recent_turns:
                    role_label = "User" if msg["role"] == "user" else "Assistant"
                    content_preview = msg["content"]
                    if len(content_preview) > 500:
                        content_preview = content_preview[:500] + "... [truncated]"
                    history_str += f"{role_label}: {content_preview}\n"
                history_str += "\n"

            prompt = (
                "Classify if the new user query requires searching the web for real-time, current, or highly fresh information.\n\n"
                f"{history_str}"
                f"New Query: \"{query}\"\n\n"
                "Reply with YES if the query:\n"
                "1. Asks about recent astronomical events, discoveries, or news (e.g., 'latest news from JWST', 'recent coordinate changes of X', 'who won the Nobel prize in physics recently?').\n"
                "2. Asks about current telescope operational status, schedules, or call-for-proposals deadlines (e.g., 'ALMA Cycle 14 deadlines', 'current status of GBT').\n"
                "3. References dates, years, or events after 2024.\n"
                "4. Requires highly specific or real-time web facts to answer accurately.\n"
                "5. Asks about telescope rules, guidelines, policies, regulations, or proprietary periods that may change or be updated in real-time (e.g., 'What is the ALMA proprietary period?', 'HST public data policies').\n\n"
                "Reply with NO if the query:\n"
                "1. Asks for general physics/astronomy textbook knowledge, mathematical derivations, or static concepts (e.g., 'what is a black hole?', 'derive the Jeans mass', 'explain redshift').\n"
                "2. Is purely conversational or a follow-up (e.g., 'hello', 'thank you', 'can you explain more?', 'now show me the band 7 of the same' when preceding messages refer to telescope observations).\n"
                "3. Asks you to write code, scripts, or format something (e.g., 'write a python script to plot a fits file').\n"
                "4. Asks for scientific papers or publications (these are searched via NASA ADS/arXiv tool, not general web search).\n"
                "5. Is a follow-up query related to astronomical data, observations, or archives discussed in the recent conversation (e.g. asking for another band, project code, or target details of an observation already found).\n"
                "6. Asks for astronomical data served by the assistant's built-in live tools: observation archives, transient/ZTF alerts, catalog cone-searches or crossmatches, photometry, spectra, light curves, or sky images/cutouts (e.g., 'any ZTF alerts near M87?', 'DESI spectra of this target', 'SDSS photometry of NGC 1275'). These query live astronomical databases directly — even though the data is real-time, general web search adds nothing.\n\n"
                "Reply with ONLY one word: YES or NO"
            )
            
            resp = client.responses.create(
                model=intent_model,
                input=prompt,
                temperature=0,
                max_output_tokens=1024,
            )
            ans = resp.output_text.strip().upper()
            is_needed = "YES" in ans
            print(f"[WEB SEARCH DETECTION] LLM classified query: '{query[:60]}...' -> {ans} (needed={is_needed})")
            return is_needed
        except Exception as e:
            # Fallback to False on failure to be conservative and prevent unnecessary web searches
            print(f"[WEB SEARCH DETECTION] LLM classification failed: {e}")
            return False

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
        on_thought=None,
        web_search: bool = False,
        model: Optional[str] = None,
        run_token: Optional[str] = None,
        _history_recovery_attempted: bool = False,
    ) -> str:
        # Assign a unique conversation_id if none provided (isolates anonymous
        # concurrent requests so they never share OpenAI response state).
        if not conversation_id:
            conversation_id = f"anon_{uuid.uuid4().hex[:12]}"
        selected_model = model or self.config.model
        self._begin_response_run(conversation_id, selected_model, run_token)

        # Periodic cleanup to prevent unbounded memory growth
        self._cleanup_conv_states()

        # Reset per-request thread-local state
        self._accumulated_run_results = []
        self._accumulated_tool_trace = []
        self.last_run_result = None
        self._job_poll_counts = {}  # per-turn datalab_job_status poll counter
        self._datalab_agg_timeout_tables = set()  # tables whose aggregates sync-timed-out this turn
        self.last_search_results = None
        self._tls.current_conversation_id = conversation_id
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
        # Derive the bare user question for routing / classification
        _user_query = raw_query or query

        if is_explicit_query(_user_query):
            if on_token:
                on_token(FILTER_NOTICE)
            return FILTER_NOTICE

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

        # 0a. Refined Conditional Web Search orchestration
        _web_result_holder = {}   # will be filled by background thread
        _web_search_reason = None
        _email_result_holder = {}  # dedicated email search for researcher queries
        _email_thread = None
        _web_thread = None

        _uq = _user_query.lower()
        # Detect explicit request to search the web
        _explicit_web_search = bool(re.search(
            r'\b(?:use web\s*search|search the web|web\s*search|internet search|google it|tavily|online search)\b',
            _uq
        ))
        # Detect explicit request NOT to search the web
        _explicit_no_web = bool(re.search(
            r'\b(?:no web search|dont search the web|dont use web search|without web search|no internet search)\b',
            _uq
        )) or not web_search or "[GROUNDED_SUMMARY_MODE]" in query

        # Skip web search for live-data and paper queries — archives, alerts,
        # catalogs, imagery, spectra, photometry all hit dedicated live
        # databases (ALMA/CADC, ALeRCE, Data Lab, SparCL, NED, ...), not the web.
        _is_archive_or_paper = self._is_live_data_query(_uq)

        # Detect OpenAlex-targeted researcher query (copied from below for early execution)
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
            r'\b(?:correlator|band\s?\d|pipeline|calibrat|antenna|baseline)\b',
            _bare_lower,
        )) and not self._LIVE_DATA_KEYWORDS_RE.search(_bare_lower)

        _web_search_query = None

        if not _explicit_no_web:
            if _explicit_web_search:
                _web_search_query = _user_query
                _web_search_reason = "explicit"
            elif _is_researcher_query:
                _web_search_query = _user_query
                _web_search_reason = "researcher_supplement"
            elif not _is_archive_or_paper:
                # Check standard year/cutoff/freshness matches
                _cutoff_match = self._detect_beyond_cutoff(_user_query)
                if _cutoff_match:
                    _web_search_query = _cutoff_match
                    _web_search_reason = "cutoff"
                else:
                    # Run deepseek-v4-flash intent classification fallback
                    if self._has_web_provider_key() and self._detect_web_search_needed_via_llm(_user_query):
                        _web_search_query = _user_query
                        _web_search_reason = "intent_detection"

        if _web_search_query:
            if _web_search_reason == "researcher_supplement":
                if on_status:
                    on_status("Searching the web for researcher profile", "running")
                # Extract the person's name for targeted search
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

                # Thread 2: Targeted email search
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
            else:
                if on_status:
                    msg = "Searching the web in parallel"
                    if _web_search_reason == "cutoff":
                        msg = "⚡ Time period beyond training knowledge cutoff detected — searching the web in parallel"
                    elif _web_search_reason == "intent_detection":
                        msg = "🌐 Query requires real-time information — searching the web in parallel"
                    on_status(msg, "running")

                def _bg_web_search():
                    try:
                        _web_result_holder["data"] = self._tavily_web_search(
                            query=_web_search_query,
                            max_results=10,
                            search_depth="basic",
                        )
                    except Exception as _e:
                        _web_result_holder["error"] = str(_e)

                _web_thread = threading.Thread(target=_bg_web_search, daemon=True)
                _web_thread.start()

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
        _query_lower = _user_query.lower()
        from services.rag_service import is_domain_relevant
        # Word-boundary keyword matching only: bare substring matching routed
        # "phase-fold the light curve" (a SMASH variable-star request) into the
        # ALMA documentation path via "phase" (2026-07-04 live test, P14).
        _rag_hits = [
            kw for kw in _rag_keywords
            if re.search(rf"(?<![\w-]){re.escape(kw)}(?![\w-])", _query_lower)
        ]
        # Documentation RAG is for observatory/instrument questions. Require either
        # an explicit radio-facility context or 2+ independent keyword hits.
        _strong_rag_context = bool(re.search(
            r"\b(alma|vla|vlba|gbt|ngvla|casa|tclean|correlator|interferomet\w+|"
            r"observing\s+tool|technical\s+handbook|proposer'?s?\s+guide|cycle\s+\d{1,2}|"
            r"data\s+reduction|calibrat\w+)\b",
            _query_lower,
        ))
        # Survey-catalog science (Data Lab catalogs, coordinates, photometry
        # workflows) is a DATA request — never answer it from ALMA manuals.
        _catalog_science_context = bool(re.search(
            r"\b(gaia|des\s+dr\d|desi|nsc|smash|delve|legacy\s+surveys?|ls_dr\d|sdss|boss|"
            r"vhs|pan-?starrs|unwise|2mass|data\s?lab|datalab|cone\s+search|cross-?match|"
            r"light\s?curves?|proper\s+motions?|parallax|photometr\w+|magnitudes?|"
            r"color-magnitude|cmd|hr\s+diagram|redshift\s+catalog|dwarf\s+galax\w+|globular)\b",
            _query_lower,
        )) or bool(re.search(r"\bra\s*[=~]?\s*[\d.]+\s*,?\s*dec\s*[=~]?\s*[+\-]?[\d.]+", _query_lower))
        _should_rag = (
            is_domain_relevant(_user_query)
            and (_strong_rag_context or len(_rag_hits) >= 2)
            and not (_catalog_science_context and not _strong_rag_context)
        )

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

        _is_alma_project_picker_selection_followup = (
            self._has_pending_alma_project_picker(conversation_id)
            and self._parse_project_picker_selection_index(_user_query) is not None
        )

        _is_data_product_triage_query = _is_alma_project_picker_selection_followup or bool(
            re.search(r'\b(?:fetch|get|list|show|inspect|triage|analy[sz]e)\b', _query_lower)
            and re.search(r'\b(?:alma|project\s+code|proposal\s+id|mous|member_ous|asdm|fits|data\s+products?|products?|files?)\b', _query_lower)
            and re.search(r'\b(?:fits|data\s+products?|products?|files?|mous|member_ous|asdm)\b', _query_lower)
        )
        if _is_data_product_triage_query:
            _should_rag = False

        _alma_science_route = self._route_alma_science_archive_query(_user_query)
        _is_alma_science_archive_query = bool(_alma_science_route) or bool(
            re.search(
                r"\b(?:cycle\s+\d{1,2}|observed\s+the\s+sun|solar\s+projects?|"
                r"12m|7m|total\s+power|hh\s*212|high[-\s]?resolution|"
                r"12co|13co|c18o|bandwidth\s+switching|spectral\s+setup|z\s*[=~]?\s*\d+(?:\.\d+)?\s*(?:-|to|\u2013)\s*\d+(?:\.\d+)?)\b",
                _query_lower,
            )
            and re.search(r"\b(?:alma|archive|projects?|observations?|band\s*\d|data|co|continuum)\b", _query_lower)
        )
        if _is_alma_science_archive_query:
            _should_rag = False

        _cross_archive_route = self._route_cross_archive_source_match_query(_user_query)
        _is_cross_archive_source_match_query = bool(_cross_archive_route) or bool(
            re.search(r"\bperseus\b", _query_lower)
            and re.search(r"\bprotostar", _query_lower)
            and re.search(r"\balma\b", _query_lower)
            and re.search(r"\b(?:jwst|mast)\b", _query_lower)
        )
        _is_archive_overlay_query = bool(
            re.search(r"\b(?:overlay|contours?)\b", _query_lower)
            and re.search(r"\balma\b", _query_lower)
            and re.search(r"\b(?:jwst|mast)\b", _query_lower)
        )
        if _is_cross_archive_source_match_query or _is_archive_overlay_query:
            _should_rag = False

        # Imagery requests ("show me a color image of M31 from DECam") must end
        # in a fresh tool-produced image (hips_cutout & co.), never a text-only
        # answer or web snippets — force a tool call on round 0.
        # Only FORCE a tool when an explicit imagery noun is present. The bare
        # "what does X look like" phrasing is intentionally NOT forced here: it
        # matches knowledge questions ("what does the ALMA pipeline look
        # like?") and would push an arbitrary tool call. The model can still
        # choose an imagery tool for those on its own.
        _is_imagery_request = bool(re.search(
            r'\b(?:show|display|make|create|generate|render|get|give)\b.*'
            r'\b(?:image|images|imagery|cutouts?|postage\s*stamps?|picture|pictures)\b',
            _query_lower,
        ))

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
            # Fast LLM intent verification.
            try:
                from core.llm_client import LLMClient
                _intent_model = os.getenv("QUASAR_PAPER_INTENT_MODEL") or os.getenv("QUASAR_FAST_MODEL", "gpt-4o-mini")
                _mini = LLMClient(model=_intent_model)
                _intent_resp = _mini.responses.create(
                    model=_intent_model,
                    input=(
                        f"Classify this astronomy query into exactly one category.\n\n"
                        f"Query: \"{_user_query}\"\n\n"
                        f"PAPERS = The user wants to FIND scientific papers, publications, or literature from NASA ADS or arXiv. "
                        f"Example: 'Find papers about protoplanetary disks', 'Recent publications on galaxy mergers'.\n"
                        f"KNOWLEDGE = The user wants general information, how-to guides, ALMA policies, procedures, or technical details. "
                        f"Words like 'proposal', 'archival', 'access', 'deadline', 'review process' in context of ALMA operations are KNOWLEDGE, not PAPERS.\n"
                        f"Example: 'What are the proposal submission deadlines?', 'How do I access archival data?'\n\n"
                        f"Reply with ONLY one word: PAPERS or KNOWLEDGE"
                    ),
                    temperature=0,
                    max_output_tokens=5,
                )
                _intent = _intent_resp.output_text.strip().upper()
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
        )) and not self._LIVE_DATA_KEYWORDS_RE.search(_bare_lower)
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
            if _is_researcher_query and _web_thread is None and self._has_web_provider_key():
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

                docs, _rag_diag = self.rag_service.search_with_diagnostics(
                    _user_query,
                    min_year=_rag_min_year,
                    min_score=0.35,
                )

                # Defense-in-depth relevance gate: min_score above applies to the vector
                # similarity, but the reranked _score shown in citations can still be
                # near-zero for off-domain queries (live test P14 cited chunks at 0.02).
                # Junk context is worse than none — it invites answering from the docs.
                def _rag_doc_score(d):
                    raw = d.metadata.get("_score", d.metadata.get("_semantic_score"))
                    try:
                        return float(raw)
                    except (TypeError, ValueError):
                        return None

                docs = [d for d in docs if (_rag_doc_score(d) is None or _rag_doc_score(d) >= 0.15)]
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

                    # Surface the freshness diagnostic computed during retrieval so
                    # the model is told when retrieved ALMA docs span multiple cycles
                    # (newer specs may supersede older ones).
                    _year_conflict = (_rag_diag or {}).get("year_conflict")
                    if _year_conflict and _year_conflict.get("message"):
                        rag_context += (
                            "\n\n⚠️ FRESHNESS NOTICE: "
                            + _year_conflict["message"]
                            + " Attribute each value to its specific cycle/year. If the "
                            "user asked about a particular cycle or year, use that "
                            "version's values; otherwise prefer the most recent and "
                            "explicitly flag any version differences."
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
                    if _web_thread is None and self._has_web_provider_key():
                        _uq = _user_query.lower()

                        # Check if the user explicitly requested NOT to use web search
                        _explicit_no_web = bool(re.search(
                            r'\b(?:no web search|dont search the web|dont use web search|without web search|no internet search)\b',
                            _uq
                        )) or not web_search or "[GROUNDED_SUMMARY_MODE]" in query

                        if _explicit_no_web:
                            _needs_web_supplement = False
                        else:
                            # Trigger web search only if the query asks for fresh/current info
                            # Note: policies, rules, regulations, and proprietary periods can be updated in real-time,
                            # so it is always safer to add a web search supplement to retrieve the latest version.
                            _FRESHNESS_KEYWORDS = re.compile(
                                r'\b(?:latest|current|recent|today|now|deadline|schedule|'
                                r'status|update|20(?:2[5-9]|[3-9]\d)|cycle\s*\d{1,2}|'
                                r'policy|policies|rules?|regulations?|proprietary|period)\b',
                                re.IGNORECASE,
                            )
                            _needs_web_supplement = bool(_FRESHNESS_KEYWORDS.search(_user_query))

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
        disabled_web_note = ""
        if not web_search:
            tools = [t for t in tools if not (t.get("name", "").startswith("web_") or t.get("name", "") == "web_search")]
            disabled_web_note = (
                "\n\nNOTE: Web search is DISABLED for this request by the user. You have no web tools. "
                "Do not include an 'Updated Information from the Web' section or any web-sourced "
                "claims/links, and do not imply web verification. Answer from internal tools, "
                "documentation context, and prior knowledge only, and if freshness matters, say web "
                "search was disabled."
            )

        # Emit model step
        if on_status:
            on_status(f"Calling {selected_model}", "running")
            on_status(f"Calling {selected_model}", "completed")
        
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
        full_input = f"{memory_context}{rag_context}{citation_note}{disabled_web_note}\n\nUser: {query}"

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
        elif _is_data_product_triage_query:
            if _is_alma_project_picker_selection_followup:
                product_directive = (
                    "\n\nMANDATORY INSTRUCTION: The user is selecting a row from the previously displayed ALMA project-code picker. "
                    "You MUST call `triage_alma_data_products` now with the user's reply exactly as `identifier_or_target`. "
                    "Do NOT ask them to copy the project code. The tool will resolve the row number to the stored project code. "
                    "If the tool returns mode `triage`, summarize the product counts, header checks, warnings, and safety note."
                )
            else:
                product_directive = (
                    "\n\nMANDATORY INSTRUCTION: The user is asking for ALMA FITS/data-product discovery or triage. "
                    "You MUST call `triage_alma_data_products` now. Pass the user-provided project code, MOUS UID, "
                    "dataset ID, or target name as `identifier_or_target`, and pass a band only if the user specified one. "
                    "If the tool returns mode `needs_project_selection`, ask the user to choose from the displayed project-code table. "
                    "If the tool returns mode `triage`, summarize the product counts, header checks, warnings, and safety note. "
                    "Do NOT call generic `search_by_target` first and do NOT claim files were downloaded."
                )
            full_input += product_directive
        elif _is_archive_overlay_query:
            full_input += (
                "\n\nMANDATORY INSTRUCTION: The user is asking for a real archive image overlay. "
                "You MUST call `overlay_archive_images` now. Use the user's named region/source as `region`; "
                "if they provided explicit coordinates, pass `ra_deg` and `dec_deg`. Use base_archive='MAST', "
                "base_collection='JWST', and contour_archive='ALMA' unless the user specified another MAST collection. "
                "Do NOT describe the workflow without calling the tool."
            )
        elif _is_cross_archive_source_match_query:
            route_text = json.dumps(_cross_archive_route) if _cross_archive_route else '{"catalog_name":"perseus_protostars","archives":["ALMA","JWST"],"radius_arcsec":5,"require_all_archives":true}'
            full_input += (
                "\n\nMANDATORY INSTRUCTION: The user is asking for cross-archive source locations. "
                "You MUST call `match_cross_archive_sources` now using these exact arguments: "
                f"{route_text}. Do NOT answer from memory. Do NOT retry by describing another plan; "
                "if one archive fails, summarize the tool's partial table and archive_errors."
            )
        elif _is_alma_science_archive_query:
            route_text = json.dumps(_alma_science_route) if _alma_science_route else "{}"
            science_directive = (
                "\n\nMANDATORY INSTRUCTION: The user is asking a live ALMA Science Archive count/filter/diagnostic question. "
                "You MUST call `query_alma_science_archive` now. Map the request as follows: "
                "Cycle Sun/solar projects -> query_type='cycle_solar_projects'; "
                "Cycle array combo with 12m/7m/total power -> query_type='cycle_array_combo_projects' and arrays=['12m','7m','TP']; "
                "HH212 or target Band high-resolution continuum -> query_type='high_resolution_band_data'; "
                "12CO/13CO/C18O in Band 6 same project -> query_type='line_set_projects' with lines=['12CO','13CO','C18O']; "
                "Galaxies at z=1-2 with CO -> query_type='redshifted_line_projects'; "
                "Bandwidth Switching -> query_type='bandwidth_switching_candidates'. "
                f"If an exact argument mapping is provided here, use it exactly: {route_text}. "
                "Do NOT answer from memory or documentation context."
            )
            full_input += science_directive
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
                if on_status:
                    on_status("__run_mode__:conductor", "meta")

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
                                    user_id=user_id,
                                    session_id=conversation_id,
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
                                web_event_payload = dict(web_data)
                                if tavily_answer:
                                    web_event_payload["answer"] = "\n".join(
                                        part for part in [
                                            str(web_data.get("answer", "") or ""),
                                            tavily_answer,
                                        ]
                                        if part.strip()
                                    )
                                web_event = self._build_web_sources_event(web_event_payload)
                                if web_event:
                                    on_event(web_event)
                    # Re-emit accumulated images via last_run_result so the SSE
                    # loop in main.py can emit them as inline image events.
                    if hasattr(self, '_conductor_images') and self._conductor_images:
                        self.last_run_result = {
                            "type": "conductor_result",
                            "images": self._conductor_images,
                        }

                    # Companion notebook attachment has been disabled for Conductor tasks as per requirements.

                    return safe_assistant_text(conductor_answer)
                # Conductor returned None → not complex enough, fall through to standard path
        except Exception as e:
            print(f"[WARNING] Complexity detection failed: {e}. Using standard path.")
        
        try:
            
            # Smart token budget replaces hard MAX_TOOL_ROUNDS = 12
            _token_budget = TokenBudget(max_budget=100_000)
            last_id = self._get_response_id(conversation_id, selected_model)
            output_text = ""
            _had_tool_calls = False
            _web_tool_results: List[Dict[str, Any]] = []
            _all_tool_results: List[Dict[str, Any]] = []  # every round's tool outputs (for the no-text safety net)

            # Provider-truncation recovery (live P6/P8/P9: DeepSeek hits its
            # output-token cap mid-round; the stream ends cleanly and the
            # dangling text used to be treated as the final answer). When a
            # no-tool-call round is truncated, we inject up to 2 continuation
            # rounds instead of breaking.
            _MAX_CONTINUATIONS = 2
            _continuation_rounds = 0
            _promise_tail_re = re.compile(
                r"(?i)\b(let me|now (?:i|let(?:'s)?|we)|i(?:'ll| will)|we(?:'ll| will)|next,? (?:i|we))\b"
                r"[^.!?]{0,150}\b(render|plot|generat|creat|build|draw|run|execut|quer|fetch|"
                r"retriev|visuali[sz]|comput|calculat|mak|produc|call|select|look)\w*"
                r"[^.!?]{0,200}[.…]{0,3}\s*$"
            )

            # 5. Call Responses API with manual streaming loop
            for _round in range(_token_budget.HARD_MAX_ITERATIONS if hasattr(_token_budget, 'HARD_MAX_ITERATIONS') else 25):
                # A deadline-killed run invalidates the token and clears the
                # provider history mid-flight (live DS-P8: the next round then
                # 400'd against a broken chain). Stop instead of working into
                # the void.
                if _round > 0 and not self._response_run_active(conversation_id, selected_model, run_token):
                    print("[STREAM] Run no longer active (cancelled/timed out) — ending the tool loop")
                    break
                _buffer_round_text = _round == 0 and (
                    _is_archive_fetch or _is_paper_query or _is_openalex_query
                    or _is_data_product_triage_query or _is_alma_science_archive_query
                    or _is_cross_archive_source_match_query or _is_archive_overlay_query
                )
                _round_text_buffer = ""
                request_kwargs = {
                    "model": selected_model,
                    "input": full_input if _round == 0 else tool_results,
                    "instructions": self.system_prompt,
                    "previous_response_id": last_id,
                    "tools": tools,
                    "temperature": self.config.temperature,
                    "max_output_tokens": self.config.max_tokens,
                    "stream": True,
                    "user_id": user_id,
                    "session_id": conversation_id,
                }
                if _round == 0 and attachments:
                    request_kwargs["attachments"] = attachments

                # Force tool call on first round for data-fetch queries.
                # This prevents the LLM from answering from conversation
                # memory and ensures a fresh data card is always shown.
                if _round == 0 and (
                    _is_archive_fetch or _is_data_product_triage_query or _is_alma_science_archive_query
                    or _is_cross_archive_source_match_query or _is_archive_overlay_query
                    or _is_imagery_request
                ):
                    request_kwargs["tool_choice"] = "required"

                # Strip unsupported params (e.g. temperature for o-series/gpt-5-mini/deepseek-v4-pro)
                _no_temp = {"o1", "o1-mini", "o1-pro", "o3", "o3-mini", "o3-pro", "o4-mini", "gpt-5-nano", "gpt-5-mini", "gpt-5.4-mini", "deepseek-v4-pro", "deepseek-v4-flash"}
                if request_kwargs.get("model", "") in _no_temp:
                    request_kwargs.pop("temperature", None)

                # Enable reasoning summary streaming for thinking models
                # These models support the `reasoning` parameter which returns
                # a model-provided reasoning summary that we stream to the UI.
                _thinking_models = {
                    "o1", "o1-mini", "o1-pro",
                    "o3", "o3-mini", "o3-pro",
                    "o4-mini",
                    "deepseek-v4-pro", "deepseek-v4-flash",
                }
                _current_model = request_kwargs.get("model", "")
                _is_thinking_model = (
                    _current_model in _thinking_models
                    or _current_model.startswith("gpt-5")  # GPT-5.x adaptive thinking
                )
                if _is_thinking_model:
                    request_kwargs["reasoning"] = {"summary": "auto"}

                try:
                    response_stream = self.client.responses.create(**request_kwargs)
                except Exception as e:
                    _is_hanging_tool_err = any(
                        msg in str(e)
                        for msg in [
                            "No tool output found for function call",
                            "must be followed by tool messages",
                            "insufficient tool messages",
                            "tool_calls",
                        ]
                    )
                    if _is_hanging_tool_err and "previous_response_id" in request_kwargs:
                        # Recover from hanging tool call in a previous interrupted turn
                        print(f"[WARNING] Recovering from hanging tool call state for conv={conversation_id}. Dropping previous_response_id.")
                        del request_kwargs["previous_response_id"]
                        last_id = None
                        self.clear_response_state(conversation_id, selected_model, run_token)
                        response_stream = self.client.responses.create(**request_kwargs)
                    else:
                        raise e
                
                function_calls = {} # call_id -> dict
                item_id_to_call_id = {}  # item.id -> call_id mapping
                _round_finish_reason = None  # provider finish_reason for THIS round
                _round_text_len_before = len(output_text)

                _reasoning_summary_text = ""  # Accumulate reasoning summary for this round
                _reasoning_emitted = False     # Track if we emitted the reasoning header
                _emit_reasoning_details = True  # Stream the model-provided reasoning summary.

                for event in response_stream:
                    if event.type == "response.created":
                        last_id = event.response.id
                        self._set_response_id(
                            conversation_id,
                            last_id,
                            selected_model,
                            run_token,
                        )
                    elif event.type == "response.reasoning_summary_text.delta":
                        # Stream the model-provided reasoning summary to the Thinking box.
                        _reasoning_summary_text += event.delta
                        if on_thought:
                            on_thought(event.delta)
                        elif not _reasoning_emitted and on_status:
                            on_status("🧠 Reasoning", "running")
                            _reasoning_emitted = True
                    elif event.type == "response.reasoning_summary_text.done":
                        # Reasoning summary complete — emit the full text as a thinking step
                        if _emit_reasoning_details and not on_thought and _reasoning_summary_text and on_status:
                            # Split into individual lines for readable thinking steps
                            for line in _reasoning_summary_text.strip().splitlines():
                                line = line.strip()
                                if line:
                                    on_status(f"💭 {line}", "completed")
                            on_status("🧠 Reasoning", "completed")
                        _reasoning_summary_text = ""
                        _reasoning_emitted = False
                    elif event.type == "response.output_text.delta":
                        # If reasoning was still accumulating when text starts,
                        # finalize it now (edge case: some models skip the .done event)
                        if _emit_reasoning_details and not on_thought and _reasoning_summary_text and on_status:
                            for line in _reasoning_summary_text.strip().splitlines():
                                line = line.strip()
                                if line:
                                    on_status(f"💭 {line}", "completed")
                            if _reasoning_emitted:
                                on_status("🧠 Reasoning", "completed")
                            _reasoning_summary_text = ""
                            _reasoning_emitted = False
                        if _buffer_round_text:
                            _round_text_buffer += event.delta
                        else:
                            output_text += event.delta
                        if on_token and not _buffer_round_text:
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
                        # Try all possible ID fields the API might use.
                        # OpenAI native: call_id / item_id on event top-level.
                        # DeepSeek/shim: call_id lives on event.item (FunctionCallItem).
                        raw_id = (
                            getattr(event, 'call_id', None)
                            or getattr(event, 'item_id', None)
                            or (getattr(event.item, 'call_id', None) if getattr(event, 'item', None) else None)
                        )
                        # Resolve to the canonical call_id we stored
                        cid = item_id_to_call_id.get(raw_id, raw_id)
                        if cid and cid in function_calls:
                            function_calls[cid]["arguments"] += event.delta
                    elif event.type == "response.completed":
                        # Final sweep: reconcile call_ids AND arguments from the
                        # completed response.  The streaming deltas may have
                        # failed to accumulate arguments (e.g. if call_id was
                        # not resolvable during delta events).
                        completed_resp = getattr(event, 'response', None)
                        if completed_resp is not None:
                            _round_finish_reason = getattr(completed_resp, 'finish_reason', None)
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
                                            # Backfill arguments if streaming failed to accumulate them
                                            if fn_args and not function_calls[old_key]["arguments"]:
                                                function_calls[old_key]["arguments"] = fn_args
                                            if fn_name and not function_calls[old_key]["name"]:
                                                function_calls[old_key]["name"] = fn_name
                                        elif item_id in function_calls:
                                            function_calls[item_id]["call_id"] = final_cid
                                            if fn_args and not function_calls[item_id]["arguments"]:
                                                function_calls[item_id]["arguments"] = fn_args
                                            if fn_name and not function_calls[item_id]["name"]:
                                                function_calls[item_id]["name"] = fn_name
                                        elif final_cid not in function_calls:
                                            # Entirely new — create the entry
                                            function_calls[final_cid] = {
                                                "name": fn_name,
                                                "arguments": fn_args,
                                                "call_id": final_cid,
                                                "_item_id": item_id,
                                            }
                
                if not function_calls:
                    if _buffer_round_text and _round_text_buffer:
                        output_text += _round_text_buffer
                        if on_token:
                            on_token(_round_text_buffer)
                    # A no-tool-call round normally means the final answer — but a
                    # provider-truncated round looks identical (live DS-P6/P8/P9:
                    # stream ended mid-sentence after "Now I'll render...").
                    # finish_reason=length is authoritative; for DeepSeek we also
                    # accept textual evidence (mid-sentence ending or a dangling
                    # action promise) because the provider has been observed to
                    # end truncated streams without reporting length.
                    _is_deepseek_model = "deepseek" in str(selected_model or "").lower()
                    _tail = output_text.rstrip()[-300:]
                    _mid_sentence = bool(_tail) and _tail[-1] not in ".!?…\"'`)]}|"
                    _dangling_promise = bool(_tail) and bool(_promise_tail_re.search(_tail))
                    _round_grew = len(output_text) > _round_text_len_before
                    _truncated = _round_finish_reason == "length" or (
                        _is_deepseek_model
                        and _had_tool_calls
                        and _round_grew
                        and _round_finish_reason in (None, "stop")
                        and (_mid_sentence or _dangling_promise)
                    )
                    if _truncated and _continuation_rounds < _MAX_CONTINUATIONS:
                        _continuation_rounds += 1
                        print(
                            f"[PROVIDER] Round {_round} looks truncated "
                            f"(finish_reason={_round_finish_reason!r}, mid_sentence={_mid_sentence}, "
                            f"dangling_promise={_dangling_promise}, textLen={len(output_text)}) — "
                            f"auto-continuation {_continuation_rounds}/{_MAX_CONTINUATIONS}"
                        )
                        if on_status:
                            on_status("Resuming after provider cutoff", "completed")
                        tool_results = (
                            "[SYSTEM CONTINUATION] Your previous message was cut off before it "
                            f"finished (finish_reason={_round_finish_reason or 'unknown'}). Continue "
                            "EXACTLY from where you stopped — do not repeat text you already sent. "
                            "If you announced a tool call (rendering a plot, running a query), MAKE "
                            "that tool call now. Then finish your answer for the user."
                        )
                        continue
                    break  # No tool calls — we have the final text

                _had_tool_calls = True
                if _buffer_round_text and _round_text_buffer:
                    print(
                        f"[STREAM] Suppressed pre-tool assistant text "
                        f"({len(_round_text_buffer)} chars)"
                    )

                # Track output growth for smart budget. Tool-productive rounds count
                # as progress even when the interleaved narration is short.
                _token_budget.record_output(len(output_text), tool_calls=len(function_calls))
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
                    args = _unescape_tool_args(args)

                    print(f"[TOOL CALL] {tool_name}({args})")

                    # Emit archive-aware tool status to the live phase tracker.
                    step_label = self._tool_status_label(tool_name, args)
                    if on_status:
                        on_status(step_label, "running")

                    _trace_result_obj = None
                    tool = self.tool_registry.get_tool(tool_name)
                    if not tool:
                        # gpt-oss habitually typos tool names ("datlab_density_vetting")
                        # and then gives up after the Unknown-tool error (live test P12).
                        # Resolve unambiguous near-misses automatically; keep the original
                        # name in the trace note so the correction is auditable.
                        import difflib as _difflib
                        _registered = [t.name for t in self.tool_registry.list_tools()]
                        _fuzzy = _difflib.get_close_matches(tool_name, _registered, n=2, cutoff=0.75)
                        _unambiguous = len(_fuzzy) == 1 or (
                            len(_fuzzy) >= 2
                            and _difflib.SequenceMatcher(None, tool_name, _fuzzy[0]).ratio()
                            - _difflib.SequenceMatcher(None, tool_name, _fuzzy[1]).ratio() >= 0.08
                        )
                        if _fuzzy and _unambiguous:
                            print(f"[TOOL CALL] Auto-corrected unknown tool '{tool_name}' -> '{_fuzzy[0]}'")
                            tool_name = _fuzzy[0]
                            tool = self.tool_registry.get_tool(tool_name)
                    if tool:
                        try:
                            _acc_len_before = len(self._accumulated_run_results)
                            _rr_before = self.last_run_result
                            result = self._execute_tool_with_progress(
                                tool,
                                args,
                                tool_name=tool_name,
                                step_label=step_label,
                                on_status=on_status,
                            )
                            if _run_result_is_new(_rr_before, self.last_run_result):
                                _primary_run_result = (
                                    self.last_run_result.copy()
                                    if isinstance(self.last_run_result, dict)
                                    else self.last_run_result
                                )
                            else:
                                _primary_run_result = None
                            _auto_paper_result = None
                            if tool_name in {"search_by_target", "search_by_position"}:
                                try:
                                    if on_status:
                                        on_status("Searching papers linked to observation", "running")
                                    _auto_paper_result = self._auto_link_project_papers_from_result(tool_name, result)
                                    if (
                                        _auto_paper_result
                                        and isinstance(_primary_run_result, dict)
                                        and _primary_run_result.get("type") == "data"
                                    ):
                                        existing_warnings = list(_primary_run_result.get("warnings") or [])
                                        for warning in _auto_paper_result.get("warnings") or []:
                                            if warning and warning not in existing_warnings:
                                                existing_warnings.append(warning)
                                        if existing_warnings:
                                            _primary_run_result["warnings"] = existing_warnings
                                except Exception as _auto_link_err:
                                    _auto_paper_result = None
                                    if isinstance(_primary_run_result, dict) and _primary_run_result.get("type") == "data":
                                        existing_warnings = list(_primary_run_result.get("warnings") or [])
                                        warning = f"Exact ADS project-code paper lookup failed: {_auto_link_err}"
                                        if warning not in existing_warnings:
                                            existing_warnings.append(warning)
                                        _primary_run_result["warnings"] = existing_warnings
                                finally:
                                    if on_status:
                                        on_status("Searching papers linked to observation", "completed")
                            _trace_result_obj = result if isinstance(result, dict) else None
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
                                        if _new_rc.get("type") in ("data", "papers", "image"):
                                            # Send the result directly (not just index) so the
                                            # event-loop thread doesn't read thread-local state.
                                            _payload = json.dumps({"_eager_result": True, "_idx": _new_idx, "_inline": True})
                                            on_status(f"__data_ready__{_payload}", "ready")
                                            # Stash inline data for the SSE handler to pick up
                                            on_status(f"__eager_data__{json.dumps(_new_rc, default=str)}", "data")
                            elif _primary_run_result is not None:
                                # Tool didn't accumulate — add last_run_result ourselves
                                _rc = (
                                    _primary_run_result.copy()
                                    if isinstance(_primary_run_result, dict)
                                    else _primary_run_result
                                )
                                if isinstance(_rc, dict):
                                    _rc["_result_id"] = id(_primary_run_result)
                                self._accumulated_run_results.append(_rc)
                                # "image" included so figures render AS TOOLS COMPLETE —
                                # a turn killed later (deadline) no longer loses them
                                # (live DS-P8: a density map rendered server-side but the
                                # end-of-turn emission never ran).
                                if on_status and isinstance(_rc, dict) and _rc.get("type") in ("data", "papers", "image"):
                                    _payload = json.dumps({"_eager_result": True, "_idx": len(self._accumulated_run_results) - 1, "_inline": True})
                                    on_status(f"__data_ready__{_payload}", "ready")
                                    on_status(f"__eager_data__{json.dumps(_rc, default=str)}", "data")
                            if _auto_paper_result and _auto_paper_result.get("papers"):
                                _paper_rc = _auto_paper_result.copy()
                                _paper_rc["_result_id"] = id(_auto_paper_result)
                                self._accumulated_run_results.append(_paper_rc)
                                if on_status:
                                    _payload = json.dumps({"_eager_result": True, "_idx": len(self._accumulated_run_results) - 1, "_inline": True})
                                    on_status(f"__data_ready__{_payload}", "ready")
                                    on_status(f"__eager_data__{json.dumps(_paper_rc, default=str)}", "data")
                            # Record tool calls for session memory
                            self.session_memory.record_tool_calls(1)

                            # Emit web_sources event for LLM-initiated web searches
                            # so source cards + images always appear in the UI.
                            if tool_name in {
                                "web_search",
                                "web_extract_url",
                                "web_map_site",
                                "web_crawl_site",
                                "web_research",
                                "web_research_status",
                            } and isinstance(result, dict) and result.get("success"):
                                _web_tool_results.append(result)
                                web_event = self._build_web_sources_event(result)
                                if web_event:
                                    on_event(web_event)
                        except Exception as te:
                            result_str = json.dumps({"error": str(te)})
                    else:
                        import difflib as _difflib
                        _suggestions = _difflib.get_close_matches(
                            tool_name, [t.name for t in self.tool_registry.list_tools()], n=3, cutoff=0.55
                        )
                        _unknown = {"error": f"Unknown tool: {tool_name}"}
                        if _suggestions:
                            _unknown["did_you_mean"] = _suggestions
                            _unknown["hint"] = f"Retry with the exact tool name, e.g. {_suggestions[0]}."
                        result_str = json.dumps(_unknown)

                    if on_status:
                        on_status(step_label, "completed")

                    self._record_tool_trace(tool_name, args, result_str,
                                            result_obj=_trace_result_obj)
                    tool_results.append({
                        "type": "function_call_output",
                        "call_id": fc["call_id"],
                        "output": result_str,
                    })

                # Apply tool result budget — truncate oversized old results
                tool_results = apply_tool_result_budget(tool_results)
                _all_tool_results.extend(tool_results)

            # Emit final step
            if on_status:
                on_status("Generating response", "running")
                on_status("Generating response", "completed")

            _has_rich_tool_output = bool(
                getattr(self, "_accumulated_run_results", None) or self.last_run_result
            )
            if not output_text and _web_tool_results:
                output_text = self._synthesize_web_tool_answer(_user_query, _web_tool_results)
                if output_text and on_token:
                    on_token(output_text)
            # Safety net: the model called tools but produced no final text. First ask the
            # model once more — without tools — to compose a real answer from the tool
            # results; only if that also yields nothing fall back to the mechanical
            # step summary.
            if (
                not output_text
                and _had_tool_calls
                and self._response_run_active(conversation_id, selected_model, run_token)
            ):
                if on_status:
                    on_status("Composing final answer from tool results", "running")
                output_text = self._compose_final_answer_from_tools(
                    _user_query,
                    _all_tool_results,
                    selected_model,
                    user_id=user_id,
                    conversation_id=conversation_id,
                    run_token=run_token,
                )
                if on_status:
                    on_status("Composing final answer from tool results", "completed")
                if output_text and on_token:
                    on_token(output_text)
            if not output_text and _had_tool_calls:
                _tool_summary = self._summarize_tool_outcomes(_all_tool_results)
                if _tool_summary:
                    output_text = _tool_summary
                    if on_token:
                        on_token(output_text)
            if not output_text and (not _had_tool_calls or not _has_rich_tool_output):
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
                        web_event_payload = dict(web_data)
                        if tavily_answer:
                            web_event_payload["answer"] = "\n".join(
                                part for part in [
                                    str(web_data.get("answer", "") or ""),
                                    tavily_answer,
                                ]
                                if part.strip()
                            )
                        web_event = self._build_web_sources_event(web_event_payload)
                        if web_event:
                            on_event(web_event)
                elif _web_result_holder.get("error"):
                    print(f"[WEB SEARCH] Parallel web search failed: {_web_result_holder['error']}")


            # 7b. Claim-vs-artifact guard — models (esp. gpt-oss-120b) sometimes assert
            # that a plot/data card "is displayed above" when nothing visual was emitted
            # this turn (2026-07-04 live test P3/P6/P7/P9/P15). Append an explicit,
            # user-visible correction instead of letting the fabrication stand.
            _visual_artifact_types = {"image", "plotly", "data", "conductor_result", "notebook"}

            def _is_visual_artifact(rr: Any) -> bool:
                return isinstance(rr, dict) and rr.get("type") in _visual_artifact_types

            _turn_visuals = [
                rr for rr in (getattr(self, "_accumulated_run_results", None) or [])
                if _is_visual_artifact(rr)
            ]
            if _is_visual_artifact(self.last_run_result):
                _turn_visuals.append(self.last_run_result)
            _artifact_claim_re = re.compile(
                r"(?:display|shown|attach|plott|render|generat)\w*\s+(?:above|below|here|in\s+the\s+ui)"
                r"|(?:data\s+cards?|cutouts?|figures?|plots?|images?|diagrams?)\s+"
                r"(?:above|below|shown|displayed|attached|already\s+generated)"
                r"|see\s+the\s+(?:plot|figure|image|cmd|diagram|cutout|data\s+cards?)",
                re.IGNORECASE,
            )
            if output_text and not _turn_visuals and _artifact_claim_re.search(output_text):
                _artifact_correction = (
                    "\n\n> ⚠️ Correction: no plot, image, or data card was actually generated in "
                    "this turn, so references above to a displayed figure are inaccurate. Ask me "
                    "to run the corresponding one-shot plotting tool (e.g. "
                    "datalab_color_magnitude_diagram, datalab_sed_plot, datalab_lss_wedge) to "
                    "produce the real figure."
                )
                output_text += _artifact_correction
                if on_token:
                    on_token(_artifact_correction)
                print("[GUARD] Claim-vs-artifact correction appended (no visual artifact this turn)")

            # 8. Update long-term memory — only for authenticated users
            output_text = safe_assistant_text(output_text)
            output_text = append_citation_warning(
                output_text,
                self.ads_client,
                on_token=on_token,
            )
            output_text = safe_assistant_text(output_text)
            if self.long_term_memory and not _is_anonymous:
                try:
                    messages = [
                        {"role": "user", "content": query},
                        {"role": "assistant", "content": output_text}
                    ]
                    self.long_term_memory.add(messages, user_id=user_id)
                except Exception as e:
                    print(f"[WARNING] mem0 memory add failed: {e}")
            
            return safe_assistant_text(output_text)
            
        except AttributeError as ae:
            # Responses API not available in this OpenAI version
            error_msg = f"Responses API not available: {ae}. Please upgrade the openai package."
            print(f"[ERROR] {error_msg}")
            return error_msg
            
        except Exception as e:
            # If the error is about a hanging tool call, clear the poisoned
            # response ID for THIS conversation so it doesn't keep failing.
            # Malformed tool-call 400s (gpt-oss sometimes emits arithmetic or
            # comments inside tool-call JSON, rejected upstream at parse time)
            # are stochastic, so they get the same clear-state-and-retry.
            _is_hanging_tool_err = any(
                msg in str(e)
                for msg in [
                    "No tool output found for function call",
                    "must be followed by tool messages",
                    "insufficient tool messages",
                    "tool_calls",
                    "Failed to parse tool call",
                    "Invalid function calling output",
                ]
            )
            if _is_hanging_tool_err:
                print(f"[WARNING] Clearing poisoned response_id for conv={conversation_id} to break error loop.")
                self.clear_response_state(conversation_id, selected_model, run_token)
                if (
                    detect_provider(selected_model) in {"tacc", "deepseek"}
                    and not _history_recovery_attempted
                    and not output_text
                    and self._response_run_active(conversation_id, selected_model, run_token)
                ):
                    logger.warning(
                        "Retrying interrupted %s tool history without cached state "
                        "for conversation %s",
                        detect_provider(selected_model),
                        conversation_id,
                    )
                    return self.stream_response_api(
                        query,
                        message_placeholder=message_placeholder,
                        user_id=user_id,
                        on_token=on_token,
                        on_status=on_status,
                        attachments=attachments,
                        raw_query=raw_query,
                        conversation_id=conversation_id,
                        plan_feedback_queue=plan_feedback_queue,
                        on_thought=on_thought,
                        web_search=web_search,
                        model=selected_model,
                        run_token=run_token,
                        _history_recovery_attempted=True,
                    )
            print(f"[ERROR] Error with Responses API: {str(e)}")
            return self._user_facing_provider_error(e)
    
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
            prefix = f"{conversation_id}|"
            with self._conv_ids_lock:
                response_ids = [
                    response_id
                    for key, response_id in self._conv_response_ids.items()
                    if key.startswith(prefix)
                ]
                for key in [key for key in self._conv_response_ids if key.startswith(prefix)]:
                    self._conv_response_ids.pop(key, None)
                for key in [key for key in self._conv_run_tokens if key.startswith(prefix)]:
                    self._conv_run_tokens.pop(key, None)
            for response_id in response_ids:
                try:
                    self.client.responses.clear_history(response_id)
                except Exception:
                    pass
        else:
            # Clear ALL conversation states (full reset)
            with self._conv_ids_lock:
                self._conv_response_ids.clear()
                self._conv_run_tokens.clear()
            try:
                self.client.responses.clear_history()
            except Exception:
                pass
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
                model=os.getenv("QUASAR_PERSONAL_MEMORY_MODEL") or os.getenv("QUASAR_FAST_MODEL", "gpt-4o-mini"),
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






