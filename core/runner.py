"""core/runner.py — QuasarAgent.stream_response_api extracted verbatim (masterplan S14).
Byte-parity: the former method body with the instance receiver renamed to the explicit
`agent` parameter. The delegate on QuasarAgent keeps the public signature so every caller
(ui-pro/api/sse.py, stream_general_response, the internal recovery recursion) is
untouched."""
from __future__ import annotations

import json
import os
import re
import threading
import uuid
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from core.llm_client import detect_provider
from core.logger import logger
from core.token_budget import TokenBudget
from core.tool_budget import apply_tool_result_budget
# DUAL_SOURCE_SCAFFOLD: referenced at the RAG-context branch — its absence
# made every documentation-grounded query NameError into a misleading
# provider-error message (scan CAR-2). Conductor: its absence made the whole
# multi-agent branch dead code, silently swallowed by the complexity-detection
# except (scan CAR-1). Both were dropped in the S14 extraction from agent.py.
from core.agent import DUAL_SOURCE_SCAFFOLD, _run_result_is_new, _unescape_tool_args
from core.conductor import Conductor
from services.citation_verifier import append_citation_warning
from services.content_safety import FILTER_NOTICE, is_explicit_query, safe_assistant_text


def _stamp_request_on_results(agent, request: Dict[str, Any], tool_name: str,
                              *, since: int, primary: Any = None) -> None:
    """Attach one tool call's ``request`` to the run results it produced.

    The per-request tool trace lives in a thread-local on the agent worker
    thread, so the SSE generator (event-loop thread) can only see it via the
    end-of-run snapshot — too late for eagerly-emitted cards. Carrying the
    request on the result dict itself makes it available to BOTH paths, and it
    rides into `messages.metadata` with the persisted card for free.

    Only results this tool just produced are stamped (``since`` = the
    accumulator length before the call), so a later tool never relabels an
    earlier card. Never raises.
    """
    if not request:
        return
    try:
        payload = {"request": request, "requestTool": str(tool_name or "")}

        def _stamp(target: Any) -> None:
            if isinstance(target, dict) and not target.get("request"):
                target.update(payload)

        accumulated = getattr(agent, "_accumulated_run_results", None) or []
        for item in list(accumulated)[max(0, int(since)):]:
            _stamp(item)
        if primary is not None:
            _stamp(primary)
            _stamp(getattr(agent, "last_run_result", None))
    except Exception:  # pragma: no cover - provenance never breaks a tool call
        pass

if TYPE_CHECKING:
    from core.agent import QuasarAgent


def stream_response_api(
    agent,
    query,
    message_placeholder=None,
    user_id="user",
    on_token=None,
    on_status=None,
    attachments=None,
    raw_query=None,
    conversation_id=None,
    plan_feedback_queue=None,
    on_thought=None,
    web_search=False,
    model=None,
    run_token=None,
    _history_recovery_attempted=False,
):
        # Assign a unique conversation_id if none provided (isolates anonymous
        # concurrent requests so they never share OpenAI response state).
        if not conversation_id:
            conversation_id = f"anon_{uuid.uuid4().hex[:12]}"
        selected_model = model or agent.config.model
        agent._begin_response_run(conversation_id, selected_model, run_token)

        # Periodic cleanup to prevent unbounded memory growth
        agent._cleanup_conv_states()

        # Reset per-request thread-local state
        agent._accumulated_run_results = []
        agent._accumulated_tool_trace = []
        agent.last_run_result = None
        agent._job_poll_counts = {}  # per-turn datalab_job_status poll counter
        agent._datalab_agg_timeout_tables = set()  # tables whose aggregates sync-timed-out this turn
        agent.last_search_results = None
        agent._tls.current_conversation_id = conversation_id
        agent._tls.current_user_id = user_id
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
        agent._prune_session_if_needed(query, user_id)

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
        # The EXACT step label the pre-pass opened. The UI store matches steps
        # by text, so the close event must reuse this verbatim — rebuilding it
        # at close time left intent-detected searches spinning forever (live
        # 2026-07-18: "searching the web in parallel · 24s" long after the
        # search finished).
        _web_status_open_label = None

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
        _is_archive_or_paper = agent._is_live_data_query(_uq)

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
        )) and not agent._LIVE_DATA_KEYWORDS_RE.search(_bare_lower)

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
                _cutoff_match = agent._detect_beyond_cutoff(_user_query)
                if _cutoff_match:
                    _web_search_query = _cutoff_match
                    _web_search_reason = "cutoff"
                else:
                    # Run deepseek-v4-flash intent classification fallback
                    if agent._has_web_provider_key() and agent._detect_web_search_needed_via_llm(_user_query):
                        _web_search_query = _user_query
                        _web_search_reason = "intent_detection"

        if _web_search_query:
            if _web_search_reason == "researcher_supplement":
                _web_status_open_label = "Searching the web for researcher profile"
                if on_status:
                    on_status(_web_status_open_label, "running")
                # Extract the person's name for targeted search
                _person_name = re.sub(
                    r'\b(?:who is|who\'s|tell me about|look up|profile of)\b',
                    '', _user_query, flags=re.IGNORECASE,
                ).strip().strip('?').strip()

                # Thread 1: General context search
                def _bg_web_search_researcher():
                    try:
                        _web_result_holder["data"] = agent._tavily_web_search(
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
                        _email_result_holder["data"] = agent._tavily_web_search(
                            query=f"{_person_name} email contact professor astronomy",
                            max_results=3,
                            search_depth="basic",
                        )
                    except Exception as _e:
                        _email_result_holder["error"] = str(_e)

                _email_thread = threading.Thread(target=_bg_email_search, daemon=True)
                _email_thread.start()
            else:
                msg = "Searching the web in parallel"
                if _web_search_reason == "cutoff":
                    msg = "⚡ Time period beyond training knowledge cutoff detected — searching the web in parallel"
                elif _web_search_reason == "intent_detection":
                    msg = "🌐 Query requires real-time information — searching the web in parallel"
                _web_status_open_label = msg
                if on_status:
                    on_status(msg, "running")

                def _bg_web_search():
                    try:
                        _web_result_holder["data"] = agent._tavily_web_search(
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
            agent._has_pending_alma_project_picker(conversation_id)
            and agent._parse_project_picker_selection_index(_user_query) is not None
        )

        _is_data_product_triage_query = _is_alma_project_picker_selection_followup or bool(
            re.search(r'\b(?:fetch|get|list|show|inspect|triage|analy[sz]e)\b', _query_lower)
            and re.search(r'\b(?:alma|project\s+code|proposal\s+id|mous|member_ous|asdm|fits|data\s+products?|products?|files?)\b', _query_lower)
            and re.search(r'\b(?:fits|data\s+products?|products?|files?|mous|member_ous|asdm)\b', _query_lower)
        )
        if _is_data_product_triage_query:
            _should_rag = False

        _alma_science_route = agent._route_alma_science_archive_query(_user_query)
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

        _cross_archive_route = agent._route_cross_archive_source_match_query(_user_query)
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

        # Radio SED / spectral-index questions must run the radio_sed tool —
        # unforced, gpt-oss answers spectral indices from parametric memory
        # (T4.1: 2/2 live failures). The prompt rule alone is ignored; force
        # the round-0 tool call like the other data-fetch classes.
        _is_radio_sed_query = bool(re.search(
            r'\bspectral\s+ind(?:ex|ices)\b'
            r'|\bradio\s+(?:sed|spectrum|spectra|spectral\s+energy)\b',
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
                # TACC gpt-oss-120b: free/unmetered for this deployment, ~3 s —
                # the old gpt-4o-mini default rode the (quota-dead) OpenAI key.
                _intent_model = os.getenv("QUASAR_PAPER_INTENT_MODEL") or os.getenv("QUASAR_FAST_MODEL") or "gpt-oss-120b"
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
        )) and not agent._LIVE_DATA_KEYWORDS_RE.search(_bare_lower)
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
            if _is_researcher_query and _web_thread is None and agent._has_web_provider_key():
                _web_search_reason = "researcher_supplement"
                _web_status_open_label = "Searching the web for researcher profile"
                if on_status:
                    on_status(_web_status_open_label, "running")

                # Extract the person's name from the query for targeted searches
                _person_name = re.sub(
                    r'\b(?:who is|who\'s|tell me about|look up|profile of)\b',
                    '', _user_query, flags=re.IGNORECASE,
                ).strip().strip('?').strip()

                # Thread 1: General context search
                def _bg_web_search_researcher():
                    try:
                        _web_result_holder["data"] = agent._tavily_web_search(
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
                        _email_result_holder["data"] = agent._tavily_web_search(
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

                docs, _rag_diag = agent.rag_service.search_with_diagnostics(
                    _user_query,
                    min_year=_rag_min_year,
                    min_score=0.35,
                )

                # Defense-in-depth relevance gate: min_score above applies to the vector
                # similarity, but the reranked _score shown in citations can still be
                # near-zero for off-domain queries (live test P14 cited chunks at 0.02).
                # Junk context is worse than none — it invites answering from the docs.
                def _rag_doc_score(d):
                    # Gate on the semantic (cosine) relevance score, which lives
                    # on a 0–1 scale. NOT the RRF fusion `_score`: with rrf_k=60
                    # that caps near ~0.016, so comparing it against a 0–1
                    # threshold silently dropped EVERY reranked doc and disabled
                    # RAG context entirely. Upstream already filters
                    # `_semantic_score >= min_score`, so this stays a harmless
                    # secondary floor while restoring the retrieved context. (C5)
                    raw = d.metadata.get("_semantic_score", d.metadata.get("_score"))
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
                    if _web_thread is None and agent._has_web_provider_key():
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
                        _web_status_open_label = "Searching the web for updated information"
                        if on_status:
                            on_status(_web_status_open_label, "running")

                        def _bg_web_search_rag():
                            try:
                                _web_result_holder["data"] = agent._tavily_web_search(
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
        if agent.long_term_memory and not _is_anonymous:
            try:
                memories = agent.long_term_memory.search(query=query, user_id=user_id, limit=5)
                if memories and memories.get("results"):
                    memory_pieces = [f"- {m['memory']}" for m in memories["results"]]
                    memory_context = "\n\nUser Memories:\n" + "\n".join(memory_pieces)
            except Exception as e:
                print(f"[WARNING] mem0 search failed: {e}")
        
        # 3. Build tools list
        tools = agent._build_tools_for_responses_api()
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
                "3. WEB SECTION: ONLY if web search results are actually present in your context, summarize them under "
                "'🌐 Updated Information from the Web:', citing ONLY URLs that appear verbatim in those results — never "
                "invent or reconstruct a link. If no web results are present or none are relevant, omit the section "
                "entirely (a documentation-only answer is fine)."
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
        #     is already running, tell the LLM not to call web_search itagent.
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
            complexity = agent.complexity_detector.assess(_user_query).score
            if complexity > Conductor.COMPLEXITY_THRESHOLD:
                import asyncio, json as _json
                trace_id = agent.query_tracer.new_trace(_user_query, user_id=user_id)
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
                agent._last_on_status = on_status

                # ③ Run the async Conductor in a dedicated thread with its own event loop.
                #    We CANNOT use pool.submit(asyncio.run, coro) here because this
                #    function already runs inside a ThreadPoolExecutor thread, and some
                #    Python/asyncio combinations deadlock when nesting executors that way.
                conductor_answer = None
                conductor_run = None
                conductor_exc = None
                agent._conductor_images = []  # Accumulate images from sub-agents
                agent._conductor_images_lock = threading.Lock()  # Thread-safe — subtasks run in parallel
                _done = threading.Event()

                # A2 CX-01: the usage/quota accounting context lives in a
                # threading.local installed on THIS worker thread (sse.py).
                # The conductor thread and its executor threads never inherit
                # it, so every conductor-path LLM call went unrecorded and
                # unmetered. Capture it here; reinstall inside the thread.
                from core.llm_client import (
                    get_llm_request_context,
                    reinstall_llm_request_context,
                )
                _parent_llm_ctx = get_llm_request_context()

                # CX-11/CX-39: sub-agent threads collect their tool-call trace
                # into a collector that rides the REQUEST-scoped accounting
                # context (never a singleton agent attribute — two concurrent
                # conductor turns must not share or clobber collectors).
                # Without a context (no accounting installed, e.g. bare bot
                # channels) conductor traces stay untracked, as before.
                _conductor_trace_collector = None
                if _parent_llm_ctx is not None:
                    _conductor_trace_collector = {
                        "calls": [],
                        "lock": threading.Lock(),
                        "owner": threading.get_ident(),
                    }
                    _parent_llm_ctx.tool_trace_collector = _conductor_trace_collector

                def _run_conductor():
                    nonlocal conductor_answer, conductor_run, conductor_exc
                    try:
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        try:
                            with reinstall_llm_request_context(_parent_llm_ctx):
                                conductor_answer, conductor_run = loop.run_until_complete(
                                    agent.conductor.orchestrate(
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

                # CX-11: merge sub-agent tool calls into this request's trace
                # (even on failure/timeout — those calls really executed).
                # Copy under the lock: on timeout, executor threads may still
                # be appending. No detach is needed — the collector rides this
                # request's context object, so a later turn can never see it
                # (CX-39); post-timeout stragglers append harmlessly to a
                # collector nobody reads again.
                if _conductor_trace_collector is not None:
                    with _conductor_trace_collector["lock"]:
                        _collected = list(_conductor_trace_collector["calls"])
                    _parent_trace = agent._accumulated_tool_trace
                    _room = 200 - len(_parent_trace)
                    if _collected and _room > 0:
                        _parent_trace.extend(_collected[:_room])

                if conductor_exc:
                    print(f"[WARNING] Conductor orchestration failed: {conductor_exc}. Falling back to standard path.")
                    agent.query_tracer.end_trace(trace_id, "failed")
                elif conductor_answer is not None:
                    # Conductor ran — use its answer (even if some subtasks failed).
                    # Guard against empty synthesis: if the model returned blank,
                    # build a minimal fallback from the DAG status.
                    if not conductor_answer.strip():
                        dag_summary = conductor_run.dag.get_execution_summary()
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

                    agent.query_tracer.end_trace(trace_id, "completed")
                    # Append parallel web search results to conductor answer
                    if _web_thread is not None:
                        _web_thread.join(timeout=15)
                        if on_status:
                            # Close the EXACT label the pre-pass opened — the UI
                            # matches steps by text, so a rebuilt label leaves
                            # the original step spinning forever.
                            on_status(
                                _web_status_open_label or "Searching the web in parallel",
                                "completed",
                            )
                        web_data = _web_result_holder.get("data")
                        if web_data and web_data.get("success"):
                            # Synthesize a query-relevant summary instead of using
                            # the raw Tavily answer which is often generic.
                            tavily_answer = agent._synthesize_web_summary(
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
                                web_event = agent._build_web_sources_event(web_event_payload)
                                if web_event:
                                    on_event(web_event)
                    # Re-emit accumulated images via last_run_result so the SSE
                    # loop in main.py can emit them as inline image events.
                    # `self` was the pre-extraction receiver — a latent NameError
                    # inside the conductor block (scan CAR-3).
                    if hasattr(agent, '_conductor_images') and agent._conductor_images:
                        agent.last_run_result = {
                            "type": "conductor_result",
                            "images": agent._conductor_images,
                        }

                    # Companion notebook attachment has been disabled for Conductor tasks as per requirements.

                    # NO IMAGE URLS rule applies here too — this return bypasses
                    # the standard-path 7c-pre guard below (T6.2 Conductor hole).
                    if conductor_answer and "![" in conductor_answer:
                        _n_imgs = len(re.findall(r"!\[[^\]]*\]\([^)]*\)", conductor_answer))
                        if _n_imgs:
                            conductor_answer = re.sub(r"!\[([^\]]*)\]\([^)]*\)", r"\1", conductor_answer)
                            print(f"[GUARD] Stripped {_n_imgs} inline markdown image(s) from the Conductor answer")

                    return safe_assistant_text(conductor_answer)
                # Conductor returned None → not complex enough, fall through to standard path
        except Exception as e:
            print(f"[WARNING] Complexity detection failed: {e}. Using standard path.")
        
        try:
            
            # Smart token budget replaces hard MAX_TOOL_ROUNDS = 12
            _token_budget = TokenBudget(max_budget=100_000)
            last_id = agent._get_response_id(conversation_id, selected_model)
            output_text = ""
            _had_tool_calls = False
            _web_tool_results: List[Dict[str, Any]] = []
            _all_tool_results: List[Dict[str, Any]] = []  # every round's tool outputs (for the no-text safety net)
            tool_results: List[Dict[str, Any]] = []

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

            # The dual-source (📚 docs / 🌐 web) scaffold is only relevant when
            # documentation (RAG) context was actually retrieved THIS turn. It
            # must NOT key on _web_thread: a background web search that later
            # fails or returns nothing still forced the mandatory sections onto
            # pure Data Lab answers, and gpt-oss filled them with INVENTED
            # sources (live P3: fake blog slug, stackexchange /q/123456, fake
            # PDF; live P4: "sourced from ALMA Documentation" on an imaging
            # answer). Real web results are appended post-hoc from the actual
            # payload (step 7a), which needs no scaffold.
            _dual_source_active = bool(rag_context)
            _instructions = agent.system_prompt + ("\n\n" + DUAL_SOURCE_SCAFFOLD if _dual_source_active else "")

            # 5. Call Responses API with manual streaming loop
            for _round in range(_token_budget.HARD_MAX_ITERATIONS if hasattr(_token_budget, 'HARD_MAX_ITERATIONS') else 25):
                # A deadline-killed run invalidates the token and clears the
                # provider history mid-flight (live DS-P8: the next round then
                # 400'd against a broken chain). Stop instead of working into
                # the void.
                if _round > 0 and not agent._response_run_active(conversation_id, selected_model, run_token):
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
                    "instructions": _instructions,
                    "previous_response_id": last_id,
                    "tools": tools,
                    "temperature": agent.config.temperature,
                    "max_output_tokens": agent.config.max_tokens,
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
                    or _is_imagery_request or _is_radio_sed_query
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
                    response_stream = agent.client.responses.create(**request_kwargs)
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
                        agent.clear_response_state(conversation_id, selected_model, run_token)
                        response_stream = agent.client.responses.create(**request_kwargs)
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
                        agent._set_response_id(
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
                    step_label = agent._tool_status_label(tool_name, args)
                    if on_status:
                        on_status(step_label, "running")

                    _trace_result_obj = None
                    _tool_sidecar = None
                    tool = agent.tool_registry.get_tool(tool_name)
                    if not tool:
                        # gpt-oss habitually typos tool names ("datlab_density_vetting")
                        # and then gives up after the Unknown-tool error (live test P12).
                        # Resolve unambiguous near-misses automatically; keep the original
                        # name in the trace note so the correction is auditable.
                        import difflib as _difflib
                        _registered = [t.name for t in agent.tool_registry.list_tools()]
                        _fuzzy = _difflib.get_close_matches(tool_name, _registered, n=2, cutoff=0.75)
                        _unambiguous = len(_fuzzy) == 1 or (
                            len(_fuzzy) >= 2
                            and _difflib.SequenceMatcher(None, tool_name, _fuzzy[0]).ratio()
                            - _difflib.SequenceMatcher(None, tool_name, _fuzzy[1]).ratio() >= 0.08
                        )
                        if _fuzzy and _unambiguous:
                            print(f"[TOOL CALL] Auto-corrected unknown tool '{tool_name}' -> '{_fuzzy[0]}'")
                            tool_name = _fuzzy[0]
                            tool = agent.tool_registry.get_tool(tool_name)
                    if tool:
                        try:
                            _acc_len_before = len(agent._accumulated_run_results)
                            _rr_before = agent.last_run_result
                            result = agent._execute_tool_with_progress(
                                tool,
                                args,
                                tool_name=tool_name,
                                step_label=step_label,
                                on_status=on_status,
                            )
                            if _run_result_is_new(_rr_before, agent.last_run_result):
                                _primary_run_result = (
                                    agent.last_run_result.copy()
                                    if isinstance(agent.last_run_result, dict)
                                    else agent.last_run_result
                                )
                            else:
                                _primary_run_result = None
                            _auto_paper_result = None
                            if tool_name in {"search_by_target", "search_by_position"}:
                                try:
                                    if on_status:
                                        on_status("Searching papers linked to observation", "running")
                                    _auto_paper_result = agent._auto_link_project_papers_from_result(tool_name, result)
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
                            # Strip the adapter's provenance sidecar BEFORE the
                            # result is serialized for the model, then stamp the
                            # uniform request onto this tool's own run results so
                            # every card can show the exact query that produced
                            # it — on the eager path as well as the done path
                            # (the trace itself is thread-local to this worker
                            # and invisible to the SSE generator). Feature 1.
                            result, _tool_sidecar = agent._pop_provenance_sidecar(result)
                            _trace_result_obj = result if isinstance(result, dict) else None
                            try:
                                from core.provenance import build_tool_request
                                _tool_request = build_tool_request(
                                    tool_name, args,
                                    result_obj=_trace_result_obj,
                                    sidecar=_tool_sidecar,
                                )
                            except Exception:
                                _tool_request = None
                            if _tool_request:
                                _stamp_request_on_results(
                                    agent, _tool_request, tool_name,
                                    since=_acc_len_before,
                                    primary=_primary_run_result,
                                )
                            result_str = json.dumps(result, default=str)[:8000]  # increased for multi-step chains
                            _acc_len_after = len(agent._accumulated_run_results)

                            # If the tool itself already accumulated results
                            # (e.g. multi-target search appends per-target results),
                            # eagerly emit each new result for parallel data card rendering.
                            if _acc_len_after > _acc_len_before:
                                # Tool accumulated its own results — emit each new one
                                if on_status:
                                    for _new_idx in range(_acc_len_before, _acc_len_after):
                                        _new_rc = agent._accumulated_run_results[_new_idx]
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
                                agent._accumulated_run_results.append(_rc)
                                # "image" included so figures render AS TOOLS COMPLETE —
                                # a turn killed later (deadline) no longer loses them
                                # (live DS-P8: a density map rendered server-side but the
                                # end-of-turn emission never ran).
                                if on_status and isinstance(_rc, dict) and _rc.get("type") in ("data", "papers", "image"):
                                    _payload = json.dumps({"_eager_result": True, "_idx": len(agent._accumulated_run_results) - 1, "_inline": True})
                                    on_status(f"__data_ready__{_payload}", "ready")
                                    on_status(f"__eager_data__{json.dumps(_rc, default=str)}", "data")
                            if _auto_paper_result and _auto_paper_result.get("papers"):
                                _paper_rc = _auto_paper_result.copy()
                                _paper_rc["_result_id"] = id(_auto_paper_result)
                                agent._accumulated_run_results.append(_paper_rc)
                                if on_status:
                                    _payload = json.dumps({"_eager_result": True, "_idx": len(agent._accumulated_run_results) - 1, "_inline": True})
                                    on_status(f"__data_ready__{_payload}", "ready")
                                    on_status(f"__eager_data__{json.dumps(_paper_rc, default=str)}", "data")
                            # Record tool calls for session memory
                            agent.session_memory.record_tool_calls(1)

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
                                web_event = agent._build_web_sources_event(result)
                                if web_event:
                                    on_event(web_event)
                        except Exception as te:
                            result_str = json.dumps({"error": str(te)})
                    else:
                        import difflib as _difflib
                        _tool_names = [t.name for t in agent.tool_registry.list_tools()]
                        # TACC/gpt-oss sometimes emits an ELIDED tool name (live
                        # P14: literal "dat..." x8). difflib can't resolve a
                        # 3-letter stub — prefix-match the registry first.
                        _stub = str(tool_name or "").rstrip(".").strip().lower()
                        _prefix_hits = [n for n in _tool_names if _stub and n.lower().startswith(_stub)]
                        _unknown = {"error": f"Unknown tool: {tool_name}"}
                        if _prefix_hits:
                            _unknown["did_you_mean"] = _prefix_hits[:8]
                            _unknown["hint"] = (
                                f"'{tool_name}' looks like a truncated tool name. Re-call with the "
                                f"FULL name, e.g. {_prefix_hits[0]}"
                                + (f" (+{len(_prefix_hits) - 8} more with this prefix)." if len(_prefix_hits) > 8 else ".")
                            )
                        else:
                            _suggestions = _difflib.get_close_matches(tool_name, _tool_names, n=3, cutoff=0.55)
                            if _suggestions:
                                _unknown["did_you_mean"] = _suggestions
                                _unknown["hint"] = f"Retry with the exact tool name, e.g. {_suggestions[0]}."
                        result_str = json.dumps(_unknown)

                    if on_status:
                        on_status(step_label, "completed")

                    agent._record_tool_trace(tool_name, args, result_str,
                                            result_obj=_trace_result_obj,
                                            provenance=_tool_sidecar)
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
                getattr(agent, "_accumulated_run_results", None) or agent.last_run_result
            )
            if not output_text and _web_tool_results:
                output_text = agent._synthesize_web_tool_answer(_user_query, _web_tool_results)
                if output_text and on_token:
                    on_token(output_text)
            # Safety net: the model called tools but produced no final text. First ask the
            # model once more — without tools — to compose a real answer from the tool
            # results; only if that also yields nothing fall back to the mechanical
            # step summary.
            if (
                not output_text
                and _had_tool_calls
                and agent._response_run_active(conversation_id, selected_model, run_token)
            ):
                if on_status:
                    on_status("Composing final answer from tool results", "running")
                output_text = agent._compose_final_answer_from_tools(
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
                _tool_summary = agent._summarize_tool_outcomes(_all_tool_results)
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
                # Close off the web status indicator with the EXACT label the
                # pre-pass opened — the UI matches steps by text, so a rebuilt
                # label leaves the original step spinning forever.
                if on_status:
                    on_status(
                        _web_status_open_label or "Searching the web in parallel",
                        "completed",
                    )
                web_data = _web_result_holder.get("data")
                if web_data and web_data.get("success"):
                    # Synthesize a query-relevant summary instead of using
                    # the raw Tavily answer which is often generic.
                    tavily_answer = agent._synthesize_web_summary(
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
                        web_event = agent._build_web_sources_event(web_event_payload)
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
                rr for rr in (getattr(agent, "_accumulated_run_results", None) or [])
                if _is_visual_artifact(rr)
            ]
            if _is_visual_artifact(agent.last_run_result):
                _turn_visuals.append(agent.last_run_result)
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

            # 7c-pre. Inline image markdown is banned outright (NO IMAGE URLS
            # rule): a non-URL src like ![CMD 1](dlr_013e8bd2…) renders as a
            # broken image (live P15), and real figures attach as cards. Keep
            # the alt text, drop the image syntax.
            if output_text and "![" in output_text:
                _n_imgs = len(re.findall(r"!\[[^\]]*\]\([^)]*\)", output_text))
                if _n_imgs:
                    output_text = re.sub(r"!\[([^\]]*)\]\([^)]*\)", r"\1", output_text)
                    print(f"[GUARD] Stripped {_n_imgs} inline markdown image(s) from the answer")

            # 7c. Fabricated-link guard — strip external URLs that no tool, web
            # search, or documentation context produced this turn (live P3: the
            # scaffold induced invented blog/StackExchange/PDF links).
            try:
                output_text = agent._strip_unverified_urls(
                    output_text,
                    sources=[
                        rag_context or "",
                        json.dumps(_web_tool_results, default=str) if _web_tool_results else "",
                        json.dumps(_web_result_holder.get("data"), default=str)
                        if _web_result_holder.get("data") else "",
                        json.dumps(_all_tool_results, default=str) if _all_tool_results else "",
                    ],
                    on_token=on_token,
                )
            except Exception as _url_guard_err:
                print(f"[GUARD] URL guard failed (non-fatal): {_url_guard_err}")

            # 8. Update long-term memory — only for authenticated users
            output_text = safe_assistant_text(output_text)

            # R3: derive mechanical citation recall/precision from the same
            # verification pass; the SSE layer surfaces them in run_meta for
            # eval mode. Tool results are the numeric-evidence corpus.
            def _citation_metrics_sink(verification, _answer_ref=None):
                from services.citation_metrics import (
                    compute_citation_metrics,
                    record_citation_metrics_on_request_context,
                )

                evidence_texts = []
                try:
                    if _all_tool_results:
                        evidence_texts.append(json.dumps(_all_tool_results, default=str))
                except Exception:
                    pass
                try:
                    if rag_context:
                        evidence_texts.append(str(rag_context))
                except Exception:
                    pass
                metrics = compute_citation_metrics(
                    output_text, verification, evidence_texts=evidence_texts
                )
                metrics["path"] = "simple"
                record_citation_metrics_on_request_context(metrics)

            output_text = append_citation_warning(
                output_text,
                agent.ads_client,
                on_token=on_token,
                verification_sink=_citation_metrics_sink,
            )
            output_text = safe_assistant_text(output_text)
            if agent.long_term_memory and not _is_anonymous:
                try:
                    messages = [
                        {"role": "user", "content": query},
                        {"role": "assistant", "content": output_text}
                    ]
                    agent.long_term_memory.add(messages, user_id=user_id)
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
                agent.clear_response_state(conversation_id, selected_model, run_token)
                if (
                    detect_provider(selected_model) in {"tacc", "deepseek"}
                    and not _history_recovery_attempted
                    and not output_text
                    and agent._response_run_active(conversation_id, selected_model, run_token)
                ):
                    logger.warning(
                        "Retrying interrupted %s tool history without cached state "
                        "for conversation %s",
                        detect_provider(selected_model),
                        conversation_id,
                    )
                    return agent.stream_response_api(
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
            # Log the full traceback server-side; the user only ever sees the
            # friendly message from _user_facing_provider_error.
            logger.error(
                "Responses API request failed for conversation %s: %s",
                conversation_id,
                str(e),
                exc_info=True,
            )
            print(f"[ERROR] Error with Responses API: {str(e)}")
            return agent._user_facing_provider_error(e)
