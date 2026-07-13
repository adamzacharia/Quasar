"""
QuasarAgent — Central orchestrator for Quasar AI.

CALLED BY: ui/app.py (Streamlit), ui-pro/api/main.py (FastAPI SSE),
           core/cli.py (terminal REPL), telegram.py (webhook)
CALLS:     All services/* modules, integrations/*, core/complexity.py,
           core/sandbox.py, OpenAI API (GPT-4o), mem0 (long-term memory)

This is the heart of Quasar. The QuasarAgent class:
  1. Registers 140+ tools as OpenAI function-calling schemas
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
# NOTE: the services.alma_science_queries / services.cross_archive_matcher
# imports moved to capabilities/alma.py with the ALMA family; the
# core.prompts.lit_to_code import moved to capabilities/papers.py (docs/v2 P1).
from services.astro_calculators import (
    calculate_redshift, convert_coordinates, calculate_beam,
    calculate_alma_sensitivity,
)
from integrations.datalab_client import DatalabClient
# V2 shared core: the Data Lab family lives in capabilities/datalab.py (one
# implementation; the inline copies are deleted). Only the shared helpers that
# non-datalab tools still use are imported here. (docs/v2 P1)
from capabilities.datalab import datalab_error as capability_datalab_error
from capabilities.datalab import fit_rows as datalab_fit_rows
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

# Phase 5: OpenClaude-inspired reliability & context management
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


_MCP_TOOL_CALL_TIMEOUT_SECONDS = 120


def _build_mcp_tool_wrapper(session, orig_name, loop):
    """Return a sync Tool callable that dispatches the MCP call_tool coroutine
    onto the bridge's long-lived event loop (C10 fix — no per-call loop)."""
    import asyncio
    from concurrent.futures import TimeoutError as FutureTimeoutError

    def wrapper(**kwargs):
        async def _do_call():
            res = await session.call_tool(orig_name, arguments=kwargs)
            if getattr(res, "content", None):
                return [c.text for c in res.content if getattr(c, "type", "") == "text"]
            elif getattr(res, "isError", False):
                return {"error": "Tool execution failed"}
            return {"status": "success"}

        if loop.is_closed() or not loop.is_running():
            return {"error": "MCP bridge event loop is not running"}

        coro = _do_call()
        try:
            fut = asyncio.run_coroutine_threadsafe(coro, loop)
        except Exception as e:
            coro.close()
            return {"error": str(e)}

        try:
            return fut.result(timeout=_MCP_TOOL_CALL_TIMEOUT_SECONDS)
        except FutureTimeoutError:
            fut.cancel()
            return {
                "error": (
                    "MCP tool call timed out after "
                    f"{_MCP_TOOL_CALL_TIMEOUT_SECONDS} seconds"
                )
            }
        except Exception as e:
            return {"error": str(e)}

    return wrapper


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

DUAL_SOURCE_SCAFFOLD = """DUAL-SOURCE RESPONSE STRUCTURE (RAG + Web):
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

**SECTION 3 — Web Search Updates** (after the disclaimer — ONLY when web-search results were actually provided to you in this turn):
Start with the heading "🌐 Updated Information from the Web:" and summarize what the provided web results contain. This section MUST:
- Use ONLY facts and URLs that appear VERBATIM in the provided web results. NEVER invent, reconstruct, or guess a URL, blog post, forum thread, PDF, version number, or access date — a fabricated link is worse than no link.
- If NO web results were provided this turn, or none are relevant, OMIT Section 3 entirely (no heading, no placeholder). It is always acceptable to present a documentation-only answer.

IMPORTANT: The disclaimer (Section 2) MUST appear BETWEEN the documentation content and the web content. Never place the disclaimer after the web section.
The Section 2 disclaimer applies ONLY when Section 1 used the documentation context — never attach it to answers built purely from live archive/catalog tools.

If ONLY documentation context is available (no web results), still cite sources inline and add the documentation disclaimer.
If ONLY web results are available (no documentation), present them with links and note they are from the web."""

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

        # Phase 5: OpenClaude-inspired modules
        print("DEBUG: Init SessionMemory")
        self.session_memory = SessionMemory(client=self.client)
        print("DEBUG: Init HealthMonitor")
        self.health_monitor = HealthMonitor()
        # Wire health monitor into model router
        self.model_router.health_monitor = self.health_monitor
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
            sandbox_executor=self.sandbox_executor,
            ads_client=self.ads_client,
            verbose=True,
        )

        # Load per-user custom tools (if user_id is set)
        if self.config.user_id:
            print(f"DEBUG: Loading MCP servers for {self.config.user_id}")
            self._load_mcp_servers(self.config.user_id)
        
        print("DEBUG: Agent init done")

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
                loop = asyncio.new_event_loop()
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
                            
                            sync_fn = _build_mcp_tool_wrapper(session, mcp_tool.name, loop)
                            
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
                    finally:
                        try:
                            await stack.aclose()
                        except Exception as e:
                            print(f"[MCPServers] Failed to close bridge {config['name']}: {e}")
                
                # Run the bridge setup in a dedicated background thread per server
                # This ensures the async context manager stays alive and connected.
                def _thread_main():
                    asyncio.set_event_loop(loop)
                    task = loop.create_task(_run_client())

                    def _stop_loop(_):
                        loop.stop()

                    task.add_done_callback(_stop_loop)
                    try:
                        loop.run_forever()
                    finally:
                        task.remove_done_callback(_stop_loop)
                        if not task.done():
                            task.cancel()
                        loop.run_until_complete(asyncio.gather(task, return_exceptions=True))
                        loop.run_until_complete(loop.shutdown_asyncgens())
                        loop.run_until_complete(loop.shutdown_default_executor())
                        loop.close()
                t = threading.Thread(target=_thread_main, daemon=True)
                if not hasattr(self, "_mcp_loops"):
                    self._mcp_loops = []
                self._mcp_loops.append(loop)
                t.start()
                
            from services.mcp_server_service import mcp_stdio_enabled
            for cfg in configs:
                transport = (cfg.get("transport") or "stdio").lower()
                if transport == "stdio" and not mcp_stdio_enabled():
                    # RCE guard: don't spawn local-command MCP servers unless
                    # explicitly enabled for a trusted environment. (S2)
                    print(
                        f"[MCPServers] Skipping stdio server "
                        f"'{cfg.get('name', '?')}' — local command spawning is "
                        "disabled (QUASAR_ENABLE_MCP_STDIO off)."
                    )
                    continue
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
  `datalab_color_magnitude_diagram` (CMD / "g vs g−r"; point_sources=true for stars; for Gaia or ANY absolute-magnitude HR diagram pass x_expr + y_expr, e.g. x_expr='bp_rp', y_expr='phot_g_mean_mag + 5*log10(parallax) - 10' — this renders an INTERACTIVE plot; prefer it over select_rows→catalog_scatter, which is static),
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
- **DATA LAB NaN CONVENTION (CRITICAL for correctness)**: Data Lab tables store missing float values as NaN (not SQL NULL), and Postgres orders NaN ABOVE every real number — so a bare `col > x`, `col >= x`, or `col != x` cut silently ADMITS every missing-value row (e.g. `parallax_over_error > 5` alone returns thousands of rows that have NO astrometry). In datalab_sql_query, every one-sided lower-bound or not-equal cut on a nullable float column (parallax, pm, pmra, pmdec, parallax_over_error, mags, colors, snr_*, chi2, …) MUST carry a finiteness guard: `AND col < 'Infinity'` — e.g. `WHERE parallax_over_error > 5 AND parallax_over_error < 'Infinity' AND pm > 150 AND pm < 'Infinity'`. Cuts with an upper bound (`<`, `<=`, BETWEEN, two-sided ranges) are already NaN-safe. The structured value_cuts on datalab tools add this guard automatically — prefer them when possible.
- **SURVEY COVERAGE CLAIMS**: Before claiming a catalog contains (or lacks) a target/region, check the `footprint` field returned by datalab_list_catalogs / datalab_describe_table, or call survey_covers_position for the exact position. NEVER list every catalog as covering a target — curate by footprint (e.g. the LMC is NOT covered by SDSS, DESI, LS DR9, or DES).
- **DENSITY / SKY-DISTRIBUTION MAPS**: NEVER build a sky-density or overdensity map from a row-LIMITed pull (datalab_select_catalog_rows, datalab_sql_query rows, crossmatch rows) — capped results are storage-order, spatially clustered slices and the map will show one corner of the field. For "where do sources clump / density map / footprint" questions use `datalab_density_aggregate` (server-side GROUP BY counts EVERY row) or `datalab_density_vetting` (finds + ranks peaks). For a WHOLE survey field (e.g. "SMASH field 169"), bound the aggregate with the indexed value_cut `fieldid = N` and NO cone — never guess a cone center for a named field. For overdensity hunts, either use density_vetting or pass `matched_filter=true` to datalab_sky_density_map, and REPORT the detected peak RA/Dec coordinates in the answer — a map alone does not answer "where do they clump". If a result carries a "hit its row cap" warning, do not plot its sky distribution — rerun with an aggregate, and always relay the truncation to the user.
- **DEFAULT QUALITY CUTS**: the Data Lab catalog tools automatically apply registry survey-quality cuts (DESI zpix: zwarn=0 + survey='main' + main_primary; DES: flags_g/r/i=0; SDSS specobj: zwarning=0) unless you pass your own cut on those columns — state the applied cuts when reporting counts. When the user implies an object CLASS on a spectroscopic catalog (galaxies/LRGs → spectype='GALAXY' on DESI zpix, class='GALAXY' on SDSS specobj; quasars → 'QSO'), ADD that class cut yourself.
- **SED SAMPLES**: `datalab_sed_plot` IS multi-object. For "SEDs of a sample / a few hundred objects", call it ONCE with `sample_n` (e.g. `{{"result_id": "dlr_...", "sample_n": 300}}`) — it overlays up to 300 SEDs with the per-band median highlighted. Use `row_index` only for ONE object; never claim the tool is single-object and never loop per row.
- **RELAY TOOL WARNINGS**: if any tool result this turn contains a `warnings` field, a "no significant period", "truncated", "hit its row cap", or "partial coverage" note, you MUST repeat that caveat faithfully in your final answer. NEVER present a result as complete or significant when its own tool output says otherwise — report "no significant period (FAP=0.28)" rather than claiming a period was found, and state coverage gaps rather than describing a partial map as the full region.
- **CROSSMATCH → MEMBER SELECTION**: For stream/cluster membership science (e.g. Pal 5 tidal tails), a raw positional crossmatch is only step one. Apply the science cuts server-side (value_cuts for proper-motion windows, color_cut for the population/CMD locus) and make the FINAL sky/CMD plots from the SELECTED member sample — never present the raw crossmatch as the result. State the exact cuts in your answer. For the ON-SKY plot, use the PM+CMD-SELECTED single-catalog rows over the FULL cone (e.g. the Gaia datalab_select_catalog_rows result), NOT a row-capped q3c_crossmatch result — the crossmatch LIMIT slices the sample to a spatial corner and the map then misses the cluster/stream entirely. Only claim the map shows the cluster/tails if the cluster center is actually within the plotted RA/Dec range.

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
        from core.tool_registrations import register_tools
        register_tools(self)

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
        from core.router import route_alma_science_archive_query
        return route_alma_science_archive_query(self, query)

    # ── Astronomy acronym dictionary for web search disambiguation ─────
    def _route_cross_archive_source_match_query(self, query: str) -> Optional[Dict[str, Any]]:
        from core.router import route_cross_archive_source_match_query
        return route_cross_archive_source_match_query(self, query)

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
            "Link ONLY to URLs that appear verbatim in the snippets — never "
            "invent, reconstruct, or guess a URL. "
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

    _EXTERNAL_URL_RE = re.compile(r"https?://[^\s)\]>\"'`]+", re.IGNORECASE)

    @staticmethod
    def _normalize_url(url: str) -> str:
        return str(url or "").rstrip(".,;:!?'\")]}>").lower()

    def _strip_unverified_urls(self, text: str, *, sources: List[str], on_token=None) -> str:
        """Strip external URLs that no tool, web search, or documentation context
        returned this turn.

        The dual-source scaffold used to INDUCE gpt-oss into inventing links
        (live P3: a fake blog slug, stackexchange /q/123456, a fake schema PDF).
        The citation footnote only flags DOIs/bibcodes — fabricated plain URLs
        need removal, not annotation. Whitelist = every URL present in this
        turn's tool outputs, web-search payloads, and RAG context.
        """
        if not text or "http" not in text.lower():
            return text
        allowed = {
            self._normalize_url(m)
            for blob in sources
            if blob
            for m in self._EXTERNAL_URL_RE.findall(str(blob))
        }
        removed: List[str] = []

        def _known(url: str) -> bool:
            return self._normalize_url(url) in allowed

        def _md_sub(match: "re.Match[str]") -> str:
            label, url = match.group(1), match.group(2)
            if _known(url):
                return match.group(0)
            removed.append(url)
            return label

        cleaned = re.sub(r"\[([^\]]*)\]\((https?://[^)\s]+)\)", _md_sub, text)

        def _bare_sub(match: "re.Match[str]") -> str:
            url = match.group(0)
            if _known(url):
                return url
            # Keep trailing punctuation that the URL regex swallowed.
            tail = url[len(url.rstrip(".,;:!?'\")]}>")):]
            removed.append(url)
            return tail

        cleaned = self._EXTERNAL_URL_RE.sub(_bare_sub, cleaned)
        if not removed:
            return text
        note = (
            f"\n\n> 🔗 Removed {len(removed)} external link(s) that were not returned by "
            "any tool, search, or documentation source in this turn (fabricated-link guard)."
        )
        cleaned += note
        if on_token:
            try:
                on_token(note)
            except Exception:
                pass
        print(f"[GUARD] Stripped {len(removed)} unverified external URL(s): {removed[:5]}")
        return cleaned

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
            "sparcl_search_spectra": "Searching SparCL by constraints",
            "sparcl_get_spectrum": "Retrieving SparCL spectrum arrays",
            "sparcl_stack_spectra": "Stacking SparCL spectra",
            "survey_footprint": "Drawing survey footprints",
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

    # NOTE: _filter_by_scan_intent migrated to capabilities/alma.py as the
    # module function filter_by_scan_intent (pure logic). (docs/v2 P1)

    # NOTE: _search_by_position migrated to capabilities/alma.py (registered
    # via _alma_tool_fn with log_name to keep its @log_tool span). (docs/v2 P1)

    # NOTE: _search_by_target migrated to capabilities/alma.py (registered via
    # _alma_tool_fn with log_name). Its dead positional fallback (resolved.get('ra')
    # vs ra_deg) is preserved verbatim there — see docs/v2 OPEN_ISSUES. (docs/v2 P1)

    # NOTE: _search_by_frequency migrated to capabilities/alma.py. (docs/v2 P1)

    # NOTE: _search_cadc migrated to capabilities/alma.py as search_cadc_archive
    # (SIMBAD fallback resolution via the injected resolve_target). (docs/v2 P1)

    # NOTE: _search_mast / _search_mast_by_criteria / _get_mast_products /
    # _search_eso / _search_irsa migrated to capabilities/archives.py
    # (registered via _archives_tool_fn; none carried @log_tool). (docs/v2 P1)


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

    # ── Per-turn (request-scoped) Data Lab state ───────────────────────────
    # Owned by the agent and RESET per turn in stream_response_api. Injected into
    # CallContext.services so the migrated capabilities can mutate it without
    # touching `self`. The context provider re-reads the attribute on every tool
    # call, so the per-turn rebind is picked up automatically — do NOT cache
    # these objects anywhere. Lazy-create mirrors the legacy inline getattr().
    def _get_datalab_agg_timeout_tables(self) -> set:
        tables = getattr(self, "_datalab_agg_timeout_tables", None)
        if tables is None:
            tables = self._datalab_agg_timeout_tables = set()
        return tables

    def _get_datalab_job_poll_counts(self) -> dict:
        counts = getattr(self, "_job_poll_counts", None)
        if counts is None:
            counts = self._job_poll_counts = {}
        return counts

    # ── V2 capabilities layer (shared core) ────────────────────────────────
    def _datalab_tool_fn(self, name: str, *, clear_card: bool = True):
        """Return the capability-backed callable for a migrated Data Lab tool.

        Used directly in the tool registration (``function=self._datalab_tool_fn(...)``)
        so the inline ``_datalab_*`` method can be DELETED and the capability in
        ``capabilities/datalab.py`` is the single implementation (no flag, no
        inline duplicate). The Data Lab client / result store are resolved lazily
        per call via the context provider.

        ``clear_card=False`` is datalab_export_notebook's mode: it routes to the
        MINIMAL notebook context provider, which injects ONLY the notebook
        generator and does NOT clear last_run_result (the generator sets the
        notebook card — the deliverable). Crucially the minimal provider builds
        NO Data Lab client / image service / SVO client, so a pure notebook
        export never depends on (or triggers the os.makedirs side-effect of)
        unrelated Data Lab service initialization — byte-parity with the legacy
        method, which touched none of them. (docs/v2 P1; guard verify CX-02)"""
        from capabilities.datalab import CAPABILITIES
        from adapters.native import build_tool
        cap = next((c for c in CAPABILITIES if c.name == name), None)
        if cap is None:  # pragma: no cover - registration wiring guard
            raise KeyError(f"No migrated Data Lab capability named '{name}'")
        provider = self._datalab_ctx_provider if clear_card else self._datalab_notebook_ctx_provider
        return build_tool(cap, provider).function

    def _datalab_image_tool_fn(self, name: str, *, nested_key: Optional[str] = None):
        """Like _datalab_tool_fn, but for image/plot-producing Data Lab capabilities.

        The capability stays transport-pure: it returns the raw plot result (with
        image_base64/path/plotly_spec) as data. This wrapper applies the SSE/UI
        transport at the adapter boundary via the existing
        _datalab_attach_image_result — it sets last_run_result to a displayable
        image card and strips the heavy base64/figure spec out of the LLM-facing
        dict (adding image_attached). Byte-parity with the legacy inline plot
        methods.

        ``nested_key`` handles tools whose image is NESTED inside the result
        (density_vetting's out["cutout_grid"]) instead of being the result: the
        capability marks that nested dict with a private ``_caption`` when — and
        only when — the legacy attach condition holds; this wrapper pops the mark
        and attaches the card to the nested dict, leaving the top-level result
        untouched. (docs/v2 P1)"""
        from capabilities.datalab import CAPABILITIES
        from adapters.native import build_tool
        cap = next((c for c in CAPABILITIES if c.name == name), None)
        if cap is None:  # pragma: no cover - registration wiring guard
            raise KeyError(f"No migrated Data Lab capability named '{name}'")
        base_fn = build_tool(cap, self._datalab_ctx_provider).function

        def _fn(**kwargs):
            raw = base_fn(**kwargs)
            if not isinstance(raw, dict):
                return raw
            if nested_key is not None:
                nested = raw.get(nested_key)
                if isinstance(nested, dict):
                    caption = nested.pop("_caption", None)
                    if caption is not None:
                        raw[nested_key] = self._datalab_attach_image_result(nested, caption)
                return raw
            # A capability may pass a computed caption via the private "_caption"
            # key; pop it so it drives the image card WITHOUT leaking into the
            # LLM-facing dict (legacy SIA tools add no caption key).
            caption = (
                kwargs.get("title")
                or raw.pop("_caption", None)
                or raw.get("title")
                or raw.get("caption")
                or "Data Lab plot"
            )
            return self._datalab_attach_image_result(raw, caption)

        return _fn

    def _datalab_ctx_provider(self, clear_card: bool = True):
        """Build the per-call CallContext for Data Lab capabilities and apply the
        one transport concern the capability must not: clearing last_run_result
        so the streaming loop can't re-emit a stale data card (Data Lab tools
        emit summaries/result_ids, not cards). datalab_export_notebook opts OUT
        (clear_card=False): its injected notebook generator sets the card — the
        deliverable — mirroring the legacy method's deliberate non-clear."""
        from capabilities.base import CallContext
        if clear_card:
            self.last_run_result = None

        def _lazy(getter):
            # A failing constructor for a service THIS tool never touches must
            # not block the call (guard CX-03: a malformed SVO TTL or an
            # unwritable plots dir used to fail tiled_search/export_notebook
            # via the adapter's generic could-not-build-context error, which
            # the inline methods never did). None → CallContext.service()
            # raises its clear typed error only if the tool actually asks.
            try:
                return getter()
            except Exception:
                return None

        return CallContext(
            services={
                "datalab_client": _lazy(self._get_datalab_client),
                "svo_fps_client": _lazy(self._get_svo_fps_client),
                "datalab_image_service": _lazy(self._get_datalab_image_service),
                "datalab_job_service": _lazy(default_job_service),
                # Bound callable → byte-identical coordinate resolution (target
                # name → ra/dec via _resolve_target) as the inline tools use.
                "resolve_coordinates": self._datalab_coordinates,
                # Bound callable → the agent-owned notebook generator (it sets
                # the last_run_result notebook card, a transport concern that
                # stays agent-side; the export capability just calls it).
                "generate_notebook": self._generate_notebook,
                # Per-turn mutable state the capabilities own the semantics of but
                # not the lifetime: re-read (never cached) so the per-turn reset in
                # stream_response_api takes effect on the next tool call.
                "datalab_agg_timeout_tables": self._get_datalab_agg_timeout_tables(),
                "datalab_job_poll_counts": self._get_datalab_job_poll_counts(),
            },
            result_store=_lazy(self._get_datalab_result_store),
            user_id=(
                getattr(self._tls, "current_user_id", None)
                or getattr(getattr(self, "config", None), "user_id", None)
            ),
        )

    def _datalab_notebook_ctx_provider(self):
        """Minimal CallContext for datalab_export_notebook.

        Export needs ONLY the notebook generator (which itself sets the
        last_run_result notebook card — the deliverable). Unlike every other
        Data Lab tool it therefore constructs NO Data Lab client / result store
        / image service / SVO client: a pure notebook export must never depend
        on — or trigger the os.makedirs side-effect of — unrelated Data Lab
        service initialization (guard verify CX-02), and its legacy inline body
        touched none of them. It also does NOT clear last_run_result (the
        generator owns the card; the legacy method deliberately never cleared
        it). (docs/v2 P1)"""
        from capabilities.base import CallContext
        return CallContext(
            services={"generate_notebook": self._generate_notebook},
            user_id=getattr(getattr(self, "config", None), "user_id", None),
        )

    def _alma_tool_fn(self, name: str, *, log_name: Optional[str] = None):
        """Return the capability-backed callable for a migrated ALMA/archive tool.

        Same shape as _datalab_tool_fn: used directly in the registration
        (``function=self._alma_tool_fn(...)``) so the inline method is DELETED
        and capabilities/alma.py is the single implementation. ``log_name``
        re-applies the ``@log_tool`` decorator the legacy method carried
        (entry/exit logs + Langfuse span under the legacy method's name) at the
        adapter boundary, so observability doesn't drift. (docs/v2 P1)"""
        from capabilities.alma import CAPABILITIES as ALMA_CAPABILITIES
        from adapters.native import build_tool
        cap = next((c for c in ALMA_CAPABILITIES if c.name == name), None)
        if cap is None:  # pragma: no cover - registration wiring guard
            raise KeyError(f"No migrated ALMA capability named '{name}'")
        fn = build_tool(cap, self._alma_ctx_provider).function
        if log_name:
            fn.__name__ = log_name
            fn = log_tool(fn)
        return fn

    def _alma_ctx_provider(self):
        """Build the per-call CallContext for the ALMA/archive-search capabilities.

        Unlike the Data Lab provider this does NOT clear last_run_result — the
        legacy inline tools never cleared it (each sets its own data/image card
        or deliberately leaves the previous one), and the capabilities preserve
        that verbatim through the injected setters below. The get/set accessors
        are bound closures over the agent's thread-local properties, so the
        request-scoped TLS semantics stay agent-side."""
        from capabilities.base import CallContext
        if not hasattr(self, "_alma_tap_provenance_state"):
            # Mirrors the legacy plain instance attrs _last_alma_tap_query/_url:
            # instance-wide (NOT thread-local), last-write-wins across requests.
            self._alma_tap_provenance_state = {"query": None, "url": None}

        def _get_lsr():
            return self.last_search_results

        def _set_lsr(value):
            self.last_search_results = value

        def _get_lrr():
            return self.last_run_result

        def _set_lrr(value):
            self.last_run_result = value

        return CallContext(
            services={
                # getattr: a partially-constructed agent (tests build via
                # __new__) yields None here, and CallContext.service() raises
                # a clear typed error only if the tool actually needs it.
                "search_service": getattr(self, "search_service", None),
                "analysis_service": getattr(self, "analysis_service", None),
                "plotting_service": getattr(self, "plotting_service", None),
                "mast_client": getattr(self, "mast_client", None),
                # Bound callable → byte-identical SIMBAD resolution for the
                # (dead, preserved-verbatim) positional fallback + CADC fallback.
                "resolve_target": self._resolve_target,
                "get_last_search_results": _get_lsr,
                "set_last_search_results": _set_lsr,
                "get_last_run_result": _get_lrr,
                "set_last_run_result": _set_lrr,
                # The legacy [FILTER]/[MULTI]/[FALLBACK] print traces, unchanged
                # on stdout; the capability itself never calls print().
                "console_log": print,
                "alma_tap_provenance": self._alma_tap_provenance_state,
            },
            user_id=getattr(getattr(self, "config", None), "user_id", None),
        )

    def _archives_tool_fn(self, name: str):
        """Return the capability-backed callable for a migrated MAST/ESO/IRSA tool.

        Same shape as _alma_tool_fn (none of this family carried @log_tool, so
        there is no log_name re-wrap). (docs/v2 P1)"""
        from capabilities.archives import CAPABILITIES as ARCHIVES_CAPABILITIES
        from adapters.native import build_tool
        cap = next((c for c in ARCHIVES_CAPABILITIES if c.name == name), None)
        if cap is None:  # pragma: no cover - registration wiring guard
            raise KeyError(f"No migrated archives capability named '{name}'")
        return build_tool(cap, self._archives_ctx_provider).function

    def _archives_ctx_provider(self):
        """Build the per-call CallContext for the MAST/ESO/IRSA capabilities.

        Does NOT clear last_run_result (the legacy inline tools never did —
        each sets its own data card through the injected setters). Clients are
        plain getattr reads; the capabilities fetch them with
        ctx.services.get(...) inside the legacy try so a missing client yields
        the same caught-and-typed error dict."""
        from capabilities.base import CallContext

        def _get_lsr():
            return self.last_search_results

        def _set_lsr(value):
            self.last_search_results = value

        def _get_lrr():
            return self.last_run_result

        def _set_lrr(value):
            self.last_run_result = value

        return CallContext(
            services={
                "mast_client": getattr(self, "mast_client", None),
                "eso_client": getattr(self, "eso_client", None),
                "irsa_client": getattr(self, "irsa_client", None),
                "get_last_search_results": _get_lsr,
                "set_last_search_results": _set_lsr,
                "get_last_run_result": _get_lrr,
                "set_last_run_result": _set_lrr,
            },
            user_id=getattr(getattr(self, "config", None), "user_id", None),
        )

    def _vo_tool_fn(self, name: str):
        """Return the capability-backed callable for a migrated VO registry tool.

        Same shape as _alma_tool_fn (no @log_tool in this family). (docs/v2 P1)"""
        from capabilities.vo import CAPABILITIES as VO_CAPABILITIES
        from adapters.native import build_tool
        cap = next((c for c in VO_CAPABILITIES if c.name == name), None)
        if cap is None:  # pragma: no cover - registration wiring guard
            raise KeyError(f"No migrated VO capability named '{name}'")
        return build_tool(cap, self._vo_ctx_provider).function

    def _vo_ctx_provider(self):
        """Build the per-call CallContext for the VO registry capabilities.

        Every legacy method began with ``self.last_run_result = None`` — the
        clear happens here (per call, immediately before run()), equivalent
        ordering. The service GETTER is injected (not the service) so a failing
        VoRegistryService constructor keeps its legacy caught-in-try error
        locality; the table-card builder and coordinate resolver stay
        agent-side as bound callables."""
        from capabilities.base import CallContext
        self.last_run_result = None
        return CallContext(
            services={
                "get_vo_registry_service": self._get_vo_registry_service,
                "external_catalog_table_result": self._external_catalog_table_result,
                "live_imagery_coordinates": self._live_imagery_coordinates,
            },
            user_id=getattr(getattr(self, "config", None), "user_id", None),
        )

    def _spectra_tool_fn(self, name: str):
        """Return the capability-backed callable for a SPARCL spectra tool.

        Same shape as _vo_tool_fn (new family, not a legacy migration)."""
        from capabilities.spectra import CAPABILITIES as SPECTRA_CAPABILITIES
        from adapters.native import build_tool
        cap = next((c for c in SPECTRA_CAPABILITIES if c.name == name), None)
        if cap is None:  # pragma: no cover - registration wiring guard
            raise KeyError(f"No SPARCL spectra capability named '{name}'")
        return build_tool(cap, self._spectra_ctx_provider).function

    def _spectra_ctx_provider(self):
        """Build the per-call CallContext for the SPARCL spectra capabilities.

        Clears last_run_result (sparcl_search_spectra sets a fresh table card
        via the injected table-result helper; sparcl_get_spectrum emits no
        card). The result store is shared with the Data Lab family so stored
        spectra chain into datalab_get_result and stacking; a failing store
        constructor degrades to no persistence rather than blocking the call."""
        from capabilities.base import CallContext
        self.last_run_result = None

        def _lazy(getter):
            try:
                return getter()
            except Exception:
                return None

        return CallContext(
            services={
                "get_sparcl_spectra_service": self._get_sparcl_spectra_service,
                "get_sparcl_stacking_service": self._get_sparcl_stacking_service,
                "external_catalog_table_result": self._external_catalog_table_result,
                "live_imagery_coordinates": self._live_imagery_coordinates,
            },
            result_store=_lazy(self._get_datalab_result_store),
            user_id=getattr(getattr(self, "config", None), "user_id", None),
        )

    def _spectra_image_tool_fn(self, name: str):
        """Like _spectra_tool_fn, but for the image-producing stacking tool:
        attaches the plot as a UI image card and strips the heavy base64 out of
        the LLM-facing dict (same transport boundary as _datalab_image_tool_fn)."""
        base_fn = self._spectra_tool_fn(name)

        def _fn(**kwargs):
            raw = base_fn(**kwargs)
            if not isinstance(raw, dict):
                return raw
            caption = raw.pop("_caption", None) or kwargs.get("title") or "Stacked SPARCL spectra"
            return self._datalab_attach_image_result(raw, caption)

        return _fn

    def _viz_tool_fn(self, name: str):
        """Return the capability-backed callable for a migrated viz/FITS tool.

        Same shape as _alma_tool_fn (no @log_tool in this family). (docs/v2 P1)"""
        from capabilities.viz import CAPABILITIES as VIZ_CAPABILITIES
        from adapters.native import build_tool
        cap = next((c for c in VIZ_CAPABILITIES if c.name == name), None)
        if cap is None:  # pragma: no cover - registration wiring guard
            raise KeyError(f"No migrated viz capability named '{name}'")
        return build_tool(cap, self._viz_ctx_provider).function

    def _viz_ctx_provider(self):
        """Build the per-call CallContext for the viz/FITS capabilities.

        Does NOT clear last_run_result (the legacy inline tools never did —
        each sets its own image card, with image_url already a served /plots
        path, so no base64 stripping wrapper is needed)."""
        from capabilities.base import CallContext

        def _set_lrr(value):
            self.last_run_result = value

        return CallContext(
            services={
                "skyview_client": getattr(self, "skyview_client", None),
                "mast_client": getattr(self, "mast_client", None),
                "search_service": getattr(self, "search_service", None),
                "datalink_client": getattr(self, "datalink_client", None),
                "set_last_run_result": _set_lrr,
            },
            user_id=getattr(getattr(self, "config", None), "user_id", None),
        )

    def _calc_tool_fn(self, name: str):
        """Return the capability-backed callable for a migrated calculator tool.

        The calc family is pure computation — the CallContext carries no
        services. (docs/v2 P1)"""
        from capabilities.calc import CAPABILITIES as CALC_CAPABILITIES
        from adapters.native import build_tool
        cap = next((c for c in CALC_CAPABILITIES if c.name == name), None)
        if cap is None:  # pragma: no cover - registration wiring guard
            raise KeyError(f"No migrated calc capability named '{name}'")

        def _provider():
            from capabilities.base import CallContext
            return CallContext(
                user_id=getattr(getattr(self, "config", None), "user_id", None),
            )

        return build_tool(cap, _provider).function

    def _papers_tool_fn(self, name: str, *, log_name: Optional[str] = None):
        """Return the capability-backed callable for a migrated papers/ADS tool.

        Same shape as _alma_tool_fn: used directly in the registration
        (``function=self._papers_tool_fn(...)``) so the inline method is DELETED
        and capabilities/papers.py is the single implementation. ``log_name``
        re-applies the ``@log_tool`` decorator the legacy method carried at the
        adapter boundary. (docs/v2 P1)"""
        from capabilities.papers import CAPABILITIES as PAPERS_CAPABILITIES
        from adapters.native import build_tool
        cap = next((c for c in PAPERS_CAPABILITIES if c.name == name), None)
        if cap is None:  # pragma: no cover - registration wiring guard
            raise KeyError(f"No migrated papers capability named '{name}'")
        fn = build_tool(cap, self._papers_ctx_provider).function
        if log_name:
            fn.__name__ = log_name
            fn = log_tool(fn)
        return fn

    def _papers_ctx_provider(self):
        """Build the per-call CallContext for the papers/ADS/OpenAlex capabilities.

        Does NOT clear last_run_result (the legacy inline tools never did; each
        sets its own papers/consensus/text/code card via the injected setter).
        Services are plain getattr reads so a partially-constructed agent
        (tests build via __new__) yields None and the capability's own legacy
        guard/except paths handle it — the capabilities fetch these with
        ctx.services.get(...) at the legacy read position, never through the
        raising ctx.service()."""
        from capabilities.base import CallContext

        def _set_lrr(value):
            self.last_run_result = value

        return CallContext(
            services={
                "ads_client": getattr(self, "ads_client", None),
                "openalex_client": getattr(self, "openalex_client", None),
                "search_service": getattr(self, "search_service", None),
                "pdf_service": getattr(self, "pdf_service", None),
                # The agent's OpenAI client + config: evaluate_consensus and
                # reproduce_paper_methods call client.responses.create(
                # model=config.model, ...) exactly as the inline bodies did.
                "llm_client": getattr(self, "client", None),
                "agent_config": getattr(self, "config", None),
                "set_last_run_result": _set_lrr,
                # The legacy [OpenAlex]/[ADS identifier] print traces, unchanged.
                "console_log": print,
            },
            user_id=getattr(getattr(self, "config", None), "user_id", None),
        )

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

    def _get_sparcl_stacking_service(self):
        if not hasattr(self, "_sparcl_stacking_service_instance"):
            from services.sparcl_stacking import SparclStackingService

            self._sparcl_stacking_service_instance = SparclStackingService(
                spectra_service=self._get_sparcl_spectra_service()
            )
        return self._sparcl_stacking_service_instance

    def _get_sparcl_spectra_service(self):
        if not hasattr(self, "_sparcl_spectra_service_instance"):
            from services.sparcl_spectra import SparclSpectraService

            self._sparcl_spectra_service_instance = SparclSpectraService()
        return self._sparcl_spectra_service_instance

    # NOTE: _datalab_list_catalogs / _datalab_describe_table / _datalab_cone_count /
    # _datalab_select_catalog_rows were migrated to capabilities/datalab.py and are
    # now registered via _datalab_tool_fn(...). The inline copies were deleted so the
    # capability is the single implementation. (docs/v2 P1)

    # NOTE: _datalab_density_aggregate (incl. its auto-tiling + per-turn
    # sync-timeout table set) was migrated to capabilities/datalab.py and is
    # registered via _datalab_tool_fn("datalab_density_aggregate"). The per-turn
    # state it mutates is injected through CallContext.services — see
    # _get_datalab_agg_timeout_tables. Inline copy deleted. (docs/v2 P1)

    # NOTE: _datalab_q3c_crossmatch / _datalab_sql_query were migrated to
    # capabilities/datalab.py (registered via _datalab_tool_fn). Inline copies
    # deleted — the capability is the single implementation. (docs/v2 P1)

    # NOTE: _execute_datalab_sql + _datalab_query_summary were retired with the
    # last inline SQL tool (density_aggregate). Their single implementation now
    # lives in capabilities/datalab.py as execute_datalab_sql / _query_summary.
    # (docs/v2 P1)

    # NOTE: _datalab_error was retired with the last inline datalab tool. Its
    # single implementation lives in capabilities/datalab.py as datalab_error
    # (imported here as capability_datalab_error for the remaining non-datalab
    # caller, _svo_filter_wavelength). (docs/v2 P1)

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

    # NOTE: _datalab_fit_rows was retired — the single row-fitting
    # implementation lives in capabilities/datalab.py (imported here as
    # datalab_fit_rows for the MMU/external-catalog table summaries).
    # (docs/v2 P1)

    # NOTE: _datalab_get_result was migrated to capabilities/datalab.py
    # (registered via _datalab_tool_fn). Inline copy deleted. (docs/v2 P1)

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
        preview_rows, _ = datalab_fit_rows(compact_preview_frame(df), 10, char_budget=4000)
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
        preview_rows, _ = datalab_fit_rows(compact_preview_frame(df), 10, char_budget=4000)
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

    def _plot_sky_map_tool(self, **kw):
        """plot_sky_map tool adapter.

        Models habitually pass result_id (learned from the datalab_* plot
        tools); the raw **kw lambda forwarded it into
        PlottingService.plot_sky_map and crashed with an unexpected-keyword
        TypeError (live P9). Accept result_id properly — plot the stored Data
        Lab rows — and drop any other unknown kwargs instead of crashing.
        """
        allowed = {"ra_col", "dec_col", "color_by", "title", "dark_mode"}
        result_id = str(kw.pop("result_id", "") or "").strip()
        clean = {k: v for k, v in kw.items() if k in allowed}
        if result_id:
            from services.datalab_result_store import default_result_store

            frame = default_result_store().get(result_id).dataframe
            records = frame.to_dict("records")
            # Data Lab rows use ra/dec, not the archive default s_ra/s_dec.
            clean.setdefault("ra_col", "ra" if "ra" in frame.columns else "s_ra")
            clean.setdefault("dec_col", "dec" if "dec" in frame.columns else "s_dec")
        else:
            records = (
                self.last_search_results.to_dict("records")
                if self.last_search_results is not None and not self.last_search_results.empty
                else []
            )
        return self.plotting_service.plot_sky_map(data_records=records, **clean)

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
        preview_rows, _ = datalab_fit_rows(compact_preview_frame(df), 10, char_budget=4000)
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

    # Palette for MOC footprint overlays (Aurora accents, distinct on dark imagery).
    _MOC_OVERLAY_COLORS = ["#22d3ee", "#fbbf24", "#34d399", "#a78bfa"]

    def _survey_footprint(
        self,
        survey_ids: List[str],
        target_name: Optional[str] = None,
        ra: Optional[float] = None,
        dec: Optional[float] = None,
        fov_deg: float = 20.0,
        survey: str = "optical",
        order: int = 8,
    ) -> Dict[str, Any]:
        """Draw survey MOC footprints on the sky (Data Lab parity 2026-07):
        fetches MOC geometry from CDS MOCServer for the given dataset ids and
        renders a HiPS card whose interactive view overlays the footprints.
        Natural chain: survey_coverage -> survey_footprint(ids from its rows)."""
        self.last_run_result = None
        try:
            geometry = self._get_moc_coverage_service().moc_geometry(survey_ids, order=order)
            if not geometry.get("success"):
                return geometry
            ra_f, dec_f, label = self._live_imagery_coordinates(target_name=target_name, ra=ra, dec=dec)
            result = self._get_hips_image_service().cutout(
                ra_f, dec_f, fov_deg=fov_deg, survey=survey, width=512
            )
            mocs = []
            for k, moc in enumerate(geometry.get("mocs") or []):
                mocs.append({
                    "id": moc["id"],
                    "name": str(moc["id"]).rsplit("/", 1)[-1] or moc["id"],
                    "color": self._MOC_OVERLAY_COLORS[k % len(self._MOC_OVERLAY_COLORS)],
                    "order": moc.get("order"),
                    "n_cells": moc.get("n_cells"),
                    "moc_json": moc["moc_json"],
                })
            names = ", ".join(m["name"] for m in mocs)
            caption = f"Survey footprints at {label}: {names}"
            meta = {
                "kind": "hips",
                "ra": ra_f,
                "dec": dec_f,
                "fov_deg": result.get("fov_deg", fov_deg),
                "survey": result.get("survey_id") or result.get("survey") or survey,
                "mocs": mocs,
            }
            out = self._datalab_attach_image_result(result, caption, meta=meta)
            if isinstance(out, dict):
                out["footprints"] = [
                    {k: m[k] for k in ("id", "name", "color", "order", "n_cells")} for m in mocs
                ]
                out["warnings"] = list(out.get("warnings") or []) + list(geometry.get("warnings") or [])
                out["note"] = (
                    "Open the card's Interactive view to see the footprints drawn on the sky; "
                    "the static preview does not show them."
                )
            return out
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
            import math
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

    # NOTE: the F01 VO registry discovery chain (_vo_find_services /
    # _vo_list_tables / _vo_describe_table / _vo_adql_query / _vo_cone_search)
    # migrated to capabilities/vo.py (registered via _vo_tool_fn; the provider
    # clears last_run_result exactly as each method's first line did). (docs/v2 P1)


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
            # meta carries the /spectral-lines deep link + SPARCL identity for
            # the interactive card's cross-nav chip.
            return self._datalab_attach_image_result(
                result, f"SparCL spectrum: {sparcl_id}", meta=result.get("meta")
            )
        except Exception as e:
            return {"success": False, "error": str(e)}

    # NOTE: _datalab_confirm_sky_area migrated to capabilities/datalab.py
    # (registered via _datalab_tool_fn). Inline copy deleted. (docs/v2 P1)

    # NOTE: _datalab_density_vetting migrated to capabilities/datalab.py and is
    # registered via _datalab_image_tool_fn("datalab_density_vetting",
    # nested_key="cutout_grid") — the nested grid card is attached at the
    # adapter boundary. Inline copy deleted. (docs/v2 P1)

    # NOTE: _datalab_color_color_diagram / _datalab_color_magnitude_diagram migrated
    # to capabilities/datalab.py (registered via _datalab_image_tool_fn). Inline
    # copies deleted. (docs/v2 P1)

    # NOTE: _datalab_tiled_search migrated to capabilities/datalab.py
    # (registered via _datalab_tool_fn). The background job is started through
    # the CallContext-injected datalab_job_service (the same default_job_service
    # singleton). Inline copy deleted. (docs/v2 P1)

    # NOTE: _datalab_job_status / _datalab_job_results / _datalab_job_cancel
    # migrated to capabilities/datalab.py (registered via _datalab_tool_fn).
    # job_status's per-turn poll counter is injected through
    # CallContext.services — see _get_datalab_job_poll_counts. Inline copies
    # deleted. (docs/v2 P1)

    # NOTE: _datalab_export_notebook migrated to capabilities/datalab.py
    # (registered via _datalab_tool_fn("datalab_export_notebook",
    # clear_card=False) — the injected _generate_notebook sets the notebook
    # card, so the ctx provider must NOT clear last_run_result for this tool).
    # Inline copy deleted. (docs/v2 P1)

    @staticmethod
    def _datalab_card_fits_url(result: Dict[str, Any]) -> Optional[str]:
        """FITS download URL for an image card, if the result carries one.

        Only single-band SIA cutouts retain a single ``provenance.source_url`` (the
        selected tile's Data Lab FITS access URL); color composites, cutout grids,
        and plot results do not, so they get no download link here. Guarded to
        https(s) URLs so a stray local path never becomes a card link. (T7.1)"""
        prov = result.get("provenance")
        if not isinstance(prov, dict):
            return None
        src = prov.get("source_url")
        if isinstance(src, str) and src.startswith(("http://", "https://")):
            return src
        return None

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
                card_meta = dict(meta) if meta is not None else {}
                # In-card FITS download (T7.1): a single-band SIA cutout retains the
                # selected tile's Data Lab access URL in provenance.source_url — a raw
                # FITS URL the card can offer for download. It is NOT an almascience.*
                # host, so the frontend links to it directly (raw anchor), never through
                # /api/fits/preview (that endpoint allowlists ALMA hosts only). HiPS cards
                # build their own hips2fits format=fits URL client-side from the meta so
                # the download tracks the live survey/fov the user is viewing.
                fits_url = self._datalab_card_fits_url(result)
                if fits_url and "fits_url" not in card_meta:
                    card_meta["fits_url"] = fits_url
                if card_meta:
                    image_result["meta"] = card_meta
                self.last_run_result = image_result
        # Keep the heavy base64/figure spec OUT of the LLM-facing tool result: they bloat context
        # and get truncated mid-string by the 8000-char tool-output slice → malformed JSON → the
        # model emits an empty response. The visual goes to the UI card above.
        if result.get("image_base64") or result.get("plotly_spec"):
            result = {k: v for k, v in result.items() if k not in ("image_base64", "plotly_spec")}
            result["image_attached"] = True
        # Also strip server-side file paths so the model never sees "/plots/..." and cannot
        # embed ![](/plots/...) against the NO-IMAGE-URLS rule. The UI card already captured
        # the path into self.last_run_result above; these keys are UI-only, never model-facing.
        if any(k in result for k in ("path", "png_path", "pdf_path")):
            result = {k: v for k, v in result.items() if k not in ("path", "png_path", "pdf_path")}
        return result

    # NOTE: _datalab_image_cutout / _color_image / _cutout_grid were migrated to
    # capabilities/datalab.py (SIA image capabilities) and are registered via
    # _datalab_image_tool_fn(...). Inline copies deleted — one implementation
    # each. Coordinate resolution + the image service are injected via
    # CallContext; the band-substitution + image-card transport are preserved. (docs/v2 P1)

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
            # Same typed-error mapping the datalab family uses — the single
            # implementation now lives in capabilities/datalab.py; the native
            # dict is byte-identical to the retired agent copy for SVO errors
            # (no DatalabPolicyError can reach this path). (docs/v2 P1)
            return capability_datalab_error(e).to_native()

    # NOTE: _datalab_catalog_scatter / _sky_density_map / _period_fold / _sed_plot /
    # _lss_wedge were migrated to capabilities/datalab.py (PLOT_CAPABILITIES) and are
    # registered via _datalab_image_tool_fn(...). Inline copies deleted — one
    # implementation each. The image-card transport (_datalab_attach_image_result)
    # is applied by the image wrapper at the adapter boundary. (docs/v2 P1)

    # NOTE: _get_sky_image migrated to capabilities/viz.py; _download_mast_data
    # migrated to capabilities/archives.py (registered via _viz_tool_fn /
    # _archives_tool_fn). (docs/v2 P1)

    # NOTE: an earlier, richer `_filter_results` implementation lived here but was
    # dead code — the class defines `_filter_results` again further down
    # (the later definition wins in Python), so this copy never executed. Removed
    # to eliminate the duplicate-method smell flagged as C11. The live version is
    # the one below in this class.

    # NOTE: _get_observation_details migrated to capabilities/alma.py. (docs/v2 P1)

    # NOTE: _download_data (legacy stub) migrated to capabilities/alma.py. (docs/v2 P1)

    # NOTE: _analyze_uv_coverage migrated to capabilities/alma.py (analysis
    # service injected via CallContext). (docs/v2 P1)
            
    # NEW METHODS implemented from SearchService enhancements
    
    # NOTE: _search_alma_with_keywords migrated to capabilities/alma.py. (docs/v2 P1)

    # NOTE: _advanced_search migrated to capabilities/alma.py (incl. the Data Lab
    # schema guard + obscore check). (docs/v2 P1)

    # NOTE: _search_alma_co_in_redshift_range migrated to capabilities/alma.py. (docs/v2 P1)

    # NOTE: _tap_obscore_dataframe + _redshifted_line_where migrated to
    # capabilities/alma.py as module functions; the legacy _last_alma_tap_query/
    # _last_alma_tap_url instance attrs became the injected agent-owned
    # alma_tap_provenance dict (see _alma_ctx_provider). (docs/v2 P1)


    # NOTE: _query_alma_science_archive migrated to capabilities/alma.py. (docs/v2 P1)

    # NOTE: _match_cross_archive_sources + _match_perseus_protostars_alma_jwst
    # migrated to capabilities/alma.py (shared _match_cross_archive_sources_impl;
    # Perseus rewrites the card tool_name through the injected get_last_run_result).
    # (docs/v2 P1)


    # NOTE: _merged_plot_alma_results migrated to capabilities/alma.py as
    # plot_alma_results (search/plotting services injected). (docs/v2 P1)


    # NOTE: the visualization/FITS family (_render_fits_image /
    # _overlay_fits_images / _overlay_archive_images / _compute_moment_map /
    # _extract_spectrum / _fit_spectral_line / _generate_finding_chart) plus the
    # helpers _overlay_region_coordinates / _mast_product_access_url /
    # _pick_mast_fits_product / _find_alma_overlay_product migrated to
    # capabilities/viz.py; the astronomy calculators (_calculate_redshift /
    # _convert_coordinates / _calculate_beam / _calculate_alma_sensitivity)
    # migrated to capabilities/calc.py (registered via _viz_tool_fn/_calc_tool_fn);
    # _get_sky_image and _download_mast_data migrated to capabilities/archives.py
    # (registered via _archives_tool_fn). Deleted — one implementation each.
    # (docs/v2 P1; masterplan S11)

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

    # NOTE: _search_papers / _search_papers_by_observation_identifier migrated
    # to capabilities/papers.py (their registration lambdas' ads_client gates
    # included; @log_tool re-applied via _papers_tool_fn(log_name=...)). (docs/v2 P1)

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

    # NOTE: _derive_archive_identifiers_for_paper_search, _lookup_researcher,
    # _get_research_trends, _evaluate_consensus, _extract_paper_details and
    # _reproduce_paper_methods migrated to capabilities/papers.py. (docs/v2 P1)

    # NOTE: _download_alma_data migrated to capabilities/alma.py. (docs/v2 P1)

    # NOTE: _find_alma_line_coverage migrated to capabilities/alma.py. (docs/v2 P1)

    # NOTE: _check_line_coverage / _check_co_lines migrated to capabilities/alma.py
    # (they read/write the last-results state through the injected accessors).
    # (docs/v2 P1)


    # NOTE: _search_catalog migrated to capabilities/alma.py. (docs/v2 P1)

    def _resolve_target(self, target_name: str) -> Dict[str, Any]:
        """Resolve target name to RA/Dec using SIMBAD (Fix 3).

        Thin delegate: the single implementation is capabilities.calc.
        resolve_target (also the body of the resolve_target tool). Kept as a
        bound method because other families receive it by injection
        (_alma_ctx_provider's resolve_target service, _datalab_coordinates,
        _live_imagery_coordinates). (docs/v2 P1)"""
        from capabilities.calc import resolve_target
        return resolve_target(target_name)

    # NOTE: _filter_results migrated to capabilities/alma.py. (docs/v2 P1)


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

    def _sandbox_tool_bridge(
        self, tool_name: str, kwargs: dict, *, user_id: Optional[str] = None
    ) -> Any:
        """
        Bridge from sandbox call_tool(name, **kwargs) to registered Quasar tools.

        Any of the 140+ registered Quasar tools can be called from inside the
        sandbox Python environment:
          call_tool("search_by_target", target_name="Elias 2-27")
          call_tool("check_line_coverage", line_freq_ghz=230.538, z=0.0)
          call_tool("generate_casa_imaging_script", target="Elias 2-27", vis="x.ms")
        """
        if user_id:
            self._tls.current_user_id = user_id
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

    def _conductor_tool_executor(
        self,
        task_description: str,
        dep_context: str = "",
        subtask_model: str = "",
        user_id: str = "",
    ) -> str:
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
        if user_id:
            self._tls.current_user_id = user_id
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
        from core.router import is_live_data_query
        return is_live_data_query(self, query)

    def _detect_beyond_cutoff(self, query: str) -> Optional[str]:
        from core.router import detect_beyond_cutoff
        return detect_beyond_cutoff(self, query)

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
        from core.runner import stream_response_api as _stream_impl
        return _stream_impl(
            self, query, message_placeholder=message_placeholder, user_id=user_id,
            on_token=on_token, on_status=on_status, attachments=attachments,
            raw_query=raw_query, conversation_id=conversation_id,
            plan_feedback_queue=plan_feedback_queue, on_thought=on_thought,
            web_search=web_search, model=model, run_token=run_token,
            _history_recovery_attempted=_history_recovery_attempted,
        )

    
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
            # ADS now raises on missing key / archive failure (no fabricated
            # papers). This Optional[DataFrame] helper returns None on error
            # rather than propagating an unhandled exception. (C2 guard)
            try:
                papers = self.ads_client.search_by_target(source_name)
            except Exception as e:
                print(f"[search_papers] ADS lookup failed for '{source_name}': {e}")
                return None
            if papers:
                return pd.DataFrame(papers)

        return None






