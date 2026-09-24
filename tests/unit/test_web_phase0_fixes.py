"""Phase 0 of the web search redesign (tmp/web-search-redesign-2026-09-24/PLAN.md):
one test group per verified defect D1-D13."""

import json
import os
import re
import threading
from types import SimpleNamespace

import pytest

import core.runner as runner
from core import web_policy
from core.router import detect_beyond_cutoff, llm_knowledge_cutoff, policy_web_override
from services import web_search_service as wss
from services.web_search_service import WebSearchService, wants_exa_search, wants_image_search


@pytest.fixture
def isolated_usage_file(tmp_path, monkeypatch):
    path = tmp_path / "search_usage.json"
    monkeypatch.setattr(wss, "USAGE_FILE", str(path))
    return path


@pytest.fixture(autouse=True)
def _reset_web_policy():
    web_policy.set_web_allowed(None)
    yield
    web_policy.set_web_allowed(None)


# ── D1: the Web Search switch reaches every web path ────────────────────


def test_d1_web_policy_strips_and_refuses_web_tools():
    tools = [{"name": "web_search"}, {"name": "search_by_target"}, {"function": {"name": "web_extract_url"}}]
    assert [t.get("name") or t["function"]["name"] for t in web_policy.strip_web_tools(tools)] == ["search_by_target"]
    assert web_policy.web_allowed() is True  # unset = previous behaviour
    with web_policy.web_scope(False):
        assert web_policy.web_allowed() is False
    assert web_policy.web_allowed() is True


def test_d1_web_policy_is_per_thread():
    web_policy.set_web_allowed(False)
    seen = {}
    t = threading.Thread(target=lambda: seen.setdefault("v", web_policy.web_allowed()))
    t.start()
    t.join()
    assert seen["v"] is True and web_policy.web_allowed() is False


def test_d1_tool_guard_refuses_web_tools_when_switched_off():
    from core.agent import QuasarAgent

    agent = QuasarAgent.__new__(QuasarAgent)
    called = []
    web_policy.set_web_allowed(False)
    out = agent._execute_tool_guarded(lambda **kw: called.append(kw), {"query": "x"}, tool_name="web_search")
    assert out["success"] is False and out["web_search_disabled"] is True and not called


def test_d1_conductor_subagent_threads_get_the_switch(monkeypatch):
    """The Conductor's executor wrapper installs run.web_search on the worker thread."""
    import asyncio

    from core.conductor import Conductor, OrchestrationRun, TaskDAG, WorkflowMemory

    seen = {}

    def fake_executor(task, dep_context, model, *rest):
        seen["allowed"] = web_policy.web_allowed()
        return {"ok": True}

    cond = Conductor.__new__(Conductor)
    cond.tool_executor = fake_executor
    cond.sandbox_executor = None
    cond.model_router = None
    cond.conductor_model = "m"
    cond.verbose = False
    node = SimpleNamespace(id="n1", agent_type="data", description="task", depends_on=[], result=None, model_used=None)
    for flag in (False, True):
        seen.clear()
        run = OrchestrationRun(dag=TaskDAG(), workflow_memory=WorkflowMemory(), web_search=flag)
        asyncio.run(cond._execute_node(run, node))
        assert seen.get("allowed") is flag
    assert web_policy.web_allowed() is True  # the caller thread is untouched


def _run_prepass(monkeypatch, query, *, web_search=True, researcher=False, live=False, state=None):
    """Drive the runner far enough to see which pre-pass searches start.
    Returns the list of queries sent to _tavily_web_search; ``state`` (a dict)
    receives the turn's web-allowed flag as the runner set it.

    These tests pin the Phase 0 / Phase 1 path: the Phase 2 planner is off
    (QUASAR_WEB_PLANNER=0), which is exactly what that flag must restore."""
    monkeypatch.setenv("QUASAR_WEB_PLANNER", "0")
    calls = []

    class _Stop(Exception):
        pass

    class FakeAgent:
        _LIVE_DATA_KEYWORDS_RE = re.compile(r"\b(?:data|archive|observations?)\b")
        memory = SimpleNamespace(get_last_n_turns=lambda n: [])

        def __init__(self):
            self._tls = threading.local()
            self.config = SimpleNamespace(model="m")

        def _begin_response_run(self, *a, **k):
            pass

        def _cleanup_conv_states(self):
            pass

        def _prune_session_if_needed(self, *a, **k):
            pass

        def _is_live_data_query(self, q):
            return live

        def _detect_beyond_cutoff(self, q):
            return None

        def _has_web_provider_key(self):
            return True

        def _detect_web_search_needed_via_llm(self, q):
            return False

        def _tavily_web_search(self, query, max_results=10, search_depth="basic", want_images=None):
            calls.append({"query": query, "want_images": want_images})
            return {"success": True, "results": []}

    agent = FakeAgent()
    monkeypatch.setattr(runner, "is_explicit_query", lambda q: False)
    monkeypatch.setattr(runner, "_maybe_failover_model", lambda a, m, *x, **k: m)

    # Stop right after the pre-pass: the RAG step imports rag_service.
    import services.rag_service as rag

    def _boom(*a, **k):
        raise _Stop()

    monkeypatch.setattr(rag, "is_domain_relevant", _boom, raising=False)
    try:
        runner._stream_response_api_impl(agent, query, web_search=web_search, conversation_id="c1")
    except _Stop:
        pass
    except Exception:
        pass
    if state is not None:
        state["web_allowed"] = web_policy.web_allowed()
    for t in threading.enumerate():
        if t is not threading.current_thread() and t.daemon and t.name.startswith("Thread"):
            t.join(timeout=2)
    return calls


def test_d1_researcher_query_with_switch_off_starts_no_web_thread(monkeypatch):
    assert _run_prepass(monkeypatch, "Who is Crystal Brogan?", web_search=False) == []


def test_d1_researcher_query_with_switch_on_still_searches(monkeypatch):
    calls = _run_prepass(monkeypatch, "Who is Paola Caselli?", web_search=True)
    queries = [c["query"] for c in calls]
    assert "Who is Paola Caselli?" in queries
    assert any("email contact" in q for q in queries)
    assert all(c["want_images"] is False for c in calls)


def test_d1_second_researcher_path_is_guarded_in_source():
    src = open(runner.__file__, encoding="utf-8").read()
    block = src[src.index("# D1: this second researcher path"):][:400]
    assert "not _explicit_no_web" in block


# ── D2: researcher contact snippets read the normalized key ─────────────


def test_d2_contact_context_reads_snippet_key():
    src = open(runner.__file__, encoding="utf-8").read()
    assert '_r.get("snippet") or _r.get("content")' in src
    assert '_snippet = _r.get("content", "").strip()' not in src


# ── D3: Brave Web Search result fields ──────────────────────────────────

BRAVE_WEB_FIXTURE = {
    "web": {
        "results": [
            {
                "title": "ALMA <strong>Cycle 13</strong> Proposer's Guide",
                "url": "https://almascience.org/proposing/proposers-guide",
                "description": "The <strong>proprietary period</strong> is 12 months.",
                "extra_snippets": ["Starts when data are delivered to the PI.", "The proprietary period is 12 months."],
                "page_age": "2026-03-15T00:00:00",
                "age": "March 15, 2026",
            },
            {"title": "No text", "url": "https://example.org/x", "age": "2 days ago"},
        ]
    }
}


class _Resp:
    def __init__(self, status, payload):
        self.status_code = status
        self._payload = payload
        self.text = json.dumps(payload)

    def json(self):
        return self._payload


def test_d3_brave_web_search_reads_description_extra_snippets_and_age(monkeypatch, isolated_usage_file):
    monkeypatch.setenv("BRAVE_API_KEY", "k")
    WebSearchService._brave_down_until = 0.0
    seen = {}

    def fake_get(url, headers=None, params=None, timeout=None):
        if "llm/context" in url:
            return _Resp(200, {"grounding": {"generic": []}})
        seen["params"] = params
        return _Resp(200, BRAVE_WEB_FIXTURE)

    monkeypatch.setattr(wss.requests, "get", fake_get)
    res = WebSearchService().search_brave("alma cycle 13 proprietary period")
    assert res["success"] and res["provider"] == "Brave Web Search"
    first = res["results"][0]
    assert first["title"] == "ALMA Cycle 13 Proposer's Guide"
    assert first["snippet"] == "The proprietary period is 12 months.\nStarts when data are delivered to the PI."
    assert first["published_date"] == "2026-03-15T00:00:00"
    assert res["results"][1]["snippet"] == "" and res["results"][1]["published_date"] == "2 days ago"
    assert seen["params"]["extra_snippets"] == "true"


def test_d3_brave_image_fallback_reads_description(monkeypatch, isolated_usage_file):
    monkeypatch.setenv("BRAVE_API_KEY", "k")
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.setenv("QUASAR_WEB_IMAGES_ENABLED", "true")
    monkeypatch.setattr(wss.requests, "get", lambda *a, **k: _Resp(200, BRAVE_WEB_FIXTURE))
    res = WebSearchService().search_images("image of M87")
    assert res["results"][0]["snippet"].startswith("The proprietary period is 12 months.")


# ── D4: routing by word-boundary intent ─────────────────────────────────


@pytest.mark.parametrize(
    "query,image,exa",
    [
        ("ALMA imaging pipeline update 2026", False, False),
        ("spectrum of M87 news", False, False),
        ("CV review", False, False),
        ("compare VLA vs ALMA sensitivity", False, False),
        ("latest research on 3I/ATLAS", False, False),
        ("sky map of the Galactic plane", False, False),
        ("plot the light curve policy", False, False),
        ("show me an image of the Crab Nebula", True, False),
        ("picture of the VLA antennas", True, False),
        ("photo of ALMA at night", True, False),
        ("show me some pictures from the JWST first images release", True, False),
        ("recent papers on protoplanetary disks", False, True),
        ("CASA documentation for tclean", False, True),
        ("ALMA technical handbook sensitivity", False, True),
        ("literature on dust masses", False, True),
    ],
)
def test_d4_routing_table(query, image, exa):
    assert wants_image_search(query) is image
    assert wants_exa_search(query) is exa


def test_d4_advanced_depth_still_uses_exa():
    assert wants_exa_search("ALMA antenna setup", "advanced") is True


def test_d4_imaging_pipeline_question_does_not_take_image_route(monkeypatch, isolated_usage_file):
    monkeypatch.delenv("BRAVE_API_KEY", raising=False)
    monkeypatch.delenv("EXA_API_KEY", raising=False)
    monkeypatch.setenv("TAVILY_API_KEY", "t")
    monkeypatch.setattr(WebSearchService, "search_images", lambda self, *a, **k: pytest.fail("image route"))
    monkeypatch.setattr(
        WebSearchService, "search_tavily",
        lambda self, q, **k: {"success": True, "provider": "Tavily", "results": [], "images": []},
    )
    assert WebSearchService().route_and_search("ALMA imaging pipeline update 2026")["provider"] == "Tavily"


# ── D5: explicit "no web" phrasing ──────────────────────────────────────


@pytest.mark.parametrize(
    "text",
    [
        "don't search the web: explain the proprietary period",
        "don’t search the web please",
        "dont use web search",
        "do not use the internet for this",
        "without web search, what is a Jeans mass?",
        "answer without using the web",
        "no web search",
        "no internet search",
        "offline only: explain redshift",
    ],
)
def test_d5_negation_phrases(text):
    assert runner._explicit_no_web_requested(text.lower()) is True


@pytest.mark.parametrize("text", ["search the web for alma cycle 13", "what is on the web about 3i/atlas", "show me images without labels"])
def test_d5_non_negations(text):
    assert runner._explicit_no_web_requested(text) is False


def test_d5_dont_search_the_web_starts_no_prepass(monkeypatch):
    assert _run_prepass(monkeypatch, "Don't search the web: explain what the proprietary period means for ALMA data.") == []


# ── D6: policy override before live-data suppression ───────────────────


def test_d6_policy_override_phrases():
    assert policy_web_override("Cycle 13 proprietary period for archive data")
    assert policy_web_override("JWST Cycle 5 deadline")
    assert policy_web_override("ALMA proposer's guide changes")
    assert policy_web_override("call for proposals 2027")
    assert not policy_web_override("Find ALMA Cycle 12 observations of M87")
    assert not policy_web_override("plot the Band 7 data")


def test_d6_policy_question_with_data_words_searches(monkeypatch):
    calls = _run_prepass(monkeypatch, "Cycle 13 proprietary period for archive data", live=True)
    assert [c["query"] for c in calls] == ["Cycle 13 proprietary period for archive data"]
    assert calls[0]["want_images"] is False


def test_d6_plain_archive_query_is_still_suppressed(monkeypatch):
    assert _run_prepass(monkeypatch, "Find ALMA Cycle 12 observations of M87", live=True) == []


# ── D7: no Tavily image prefetch unless images are wanted ──────────────


def test_d7_prefetch_not_started_when_images_not_wanted(monkeypatch, isolated_usage_file):
    monkeypatch.setenv("BRAVE_API_KEY", "b")
    monkeypatch.setenv("TAVILY_API_KEY", "t")
    monkeypatch.setenv("QUASAR_WEB_IMAGES_ENABLED", "true")
    monkeypatch.delenv("EXA_API_KEY", raising=False)
    WebSearchService._brave_down_until = 0.0
    started = []
    monkeypatch.setattr(WebSearchService, "_start_image_prefetch", lambda self, q, **k: (started.append(q), (None, None))[1])
    monkeypatch.setattr(
        WebSearchService, "search_brave",
        lambda self, q, max_results=5: {"success": True, "provider": "Brave", "results": [{"url": "https://a.org"}], "images": []},
    )
    WebSearchService().route_and_search("current ALMA policy", want_images=False)
    WebSearchService().route_and_search("current ALMA policy")  # tool path: no picture request
    assert started == []
    WebSearchService().route_and_search("current ALMA policy", want_images=True)
    assert started == ["current ALMA policy"]


def test_d7_tavily_fallback_asks_for_no_images_when_not_wanted(monkeypatch, isolated_usage_file):
    monkeypatch.delenv("BRAVE_API_KEY", raising=False)
    monkeypatch.delenv("EXA_API_KEY", raising=False)
    monkeypatch.setenv("TAVILY_API_KEY", "t")
    monkeypatch.setenv("QUASAR_WEB_IMAGES_ENABLED", "true")
    seen = {}

    def fake_tavily(self, q, max_results=10, search_depth="basic", include_images=True, include_answer=False):
        seen["include_images"] = include_images
        return {"success": True, "provider": "Tavily", "results": [], "images": []}

    monkeypatch.setattr(WebSearchService, "search_tavily", fake_tavily)
    WebSearchService().route_and_search("current ALMA policy", want_images=False)
    assert seen["include_images"] is False


# ── D9: timed-out web join is not reported as completed ────────────────


def test_d9_timed_out_join_closes_step_as_timed_out():
    events = []
    gate = threading.Event()
    t = threading.Thread(target=gate.wait, daemon=True)
    t.start()
    finished = runner._join_web_thread(t, 0.05)
    runner._close_web_step(lambda label, state: events.append((label, state)), "🌐 Searching the web", finished, 30)
    gate.set()
    assert finished is False
    assert events[0] == ("🌐 Searching the web", "error")
    assert any(state == "completed" and "timed out" in label for label, state in events[1:])
    assert ("🌐 Searching the web", "completed") not in events


def test_d9_finished_join_closes_with_the_open_label():
    events = []
    t = threading.Thread(target=lambda: None)
    t.start()
    assert runner._join_web_thread(t, 2) is True
    runner._close_web_step(lambda label, state: events.append((label, state)), "Searching the web in parallel", True, 30)
    assert events == [("Searching the web in parallel", "completed")]


# ── D10: URL encoding, locked usage store, Tavily counts ────────────────


def test_d10_duckduckgo_query_is_url_encoded(monkeypatch):
    from services.browser import BrowserService

    visited = {}

    class FakePage:
        def goto(self, url, **k):
            visited["url"] = url

        def wait_for_timeout(self, ms):
            pass

        def evaluate(self, js):
            return []

    svc = BrowserService.__new__(BrowserService)
    svc._page = FakePage()
    monkeypatch.setattr(BrowserService, "_ensure_browser", lambda self: None, raising=False)
    svc.web_search("M87 & jets #1 ?x=y")
    assert visited["url"] == "https://duckduckgo.com/?q=M87+%26+jets+%231+%3Fx%3Dy&ia=web"


def test_d10_usage_counter_is_thread_safe_and_counts_tavily(isolated_usage_file):
    svc = WebSearchService()

    def bump():
        for _ in range(40):
            svc._increment_usage("tavily")
            svc._increment_usage("brave")

    threads = [threading.Thread(target=bump) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    usage = json.loads(isolated_usage_file.read_text(encoding="utf-8"))
    assert usage["tavily_count"] == 240 and usage["brave_count"] == 240
    assert usage["exa_count"] == 0 and usage["tavily_extract_count"] == 0
    leftovers = [p for p in os.listdir(isolated_usage_file.parent) if p.startswith(".search_usage.")]
    assert leftovers == []


def test_d10_tavily_search_increments_usage(monkeypatch, isolated_usage_file):
    monkeypatch.setenv("TAVILY_API_KEY", "t")
    monkeypatch.setattr(WebSearchService, "_get_tavily_client", lambda self: (_ for _ in ()).throw(RuntimeError("no sdk")))
    monkeypatch.setattr(WebSearchService, "_tavily_post", lambda self, e, p, timeout=60: {"results": [{"url": "https://a.org", "content": "x"}]})
    WebSearchService().search_tavily("q", include_images=False)
    assert json.loads(isolated_usage_file.read_text(encoding="utf-8"))["tavily_count"] == 1


# ── D11: knowledge cutoff from the environment ─────────────────────────


def test_d11_default_cutoff_is_2025_06(monkeypatch):
    monkeypatch.delenv("QUASAR_LLM_CUTOFF", raising=False)
    assert llm_knowledge_cutoff() == (2025, 6)


@pytest.mark.parametrize("raw,expected", [("2026-01", (2026, 1)), ("2024", (2024, 12)), ("junk", (2025, 6)), ("2025-13", (2025, 6))])
def test_d11_cutoff_env_parsing(monkeypatch, raw, expected):
    monkeypatch.setenv("QUASAR_LLM_CUTOFF", raw)
    assert llm_knowledge_cutoff() == expected


def test_d11_cutoff_detection_uses_env(monkeypatch):
    agent = SimpleNamespace(_is_live_data_query=lambda q: False)
    monkeypatch.delenv("QUASAR_LLM_CUTOFF", raising=False)
    assert detect_beyond_cutoff(agent, "What happened in 2025 at ALMA?") is None       # before 2025-06
    assert detect_beyond_cutoff(agent, "News from September 2025?") is not None      # after 2025-06
    monkeypatch.setenv("QUASAR_LLM_CUTOFF", "2024-10")
    assert detect_beyond_cutoff(agent, "What happened in 2025 at ALMA?") is not None


# ── D12: web_search is declared on every provider host ─────────────────


def test_d12_web_search_hosts_cover_all_providers():
    from services import tool_budgets

    hosts = set(tool_budgets.hosts_for("web_search"))
    assert {"api.search.brave.com", "api.exa.ai", "api.tavily.com"} <= hosts


def test_d12_open_tavily_circuit_alone_does_not_refuse_web_search(monkeypatch):
    from services import tool_budgets
    from services.host_breaker import HostBreaker

    hosts = tool_budgets.hosts_for("web_search")
    monkeypatch.setattr(HostBreaker, "open_hosts", classmethod(lambda cls, hs: [("api.tavily.com", {"retry_after": 60})]))
    open_ = HostBreaker.open_hosts(hosts)
    assert len(open_) < len(set(hosts))  # the guard refuses only when EVERY host is open


# ── D13: dead EXEMPT_TOOLS removed ──────────────────────────────────────


def test_d13_exempt_tools_removed():
    import core.tool_budget as tb

    assert not hasattr(tb, "EXEMPT_TOOLS")


# ── guard task-25bee13-16887 reconciliation ─────────────────────────────


@pytest.mark.parametrize(
    "query,web_search",
    [
        ("Don't search the web: who is Paola Caselli?", True),
        ("[GROUNDED_SUMMARY_MODE] Summarize the ALMA results", True),
        ("Who is Paola Caselli?", False),
    ],
)
def test_cx01_explicit_no_web_and_grounded_turn_off_web_tools_for_the_turn(monkeypatch, query, web_search):
    state = {}
    assert _run_prepass(monkeypatch, query, web_search=web_search, state=state) == []
    assert state["web_allowed"] is False  # tool guard + Conductor threads refuse web tools


def test_cx01_plain_turn_keeps_web_tools(monkeypatch):
    state = {}
    _run_prepass(monkeypatch, "Derive the Jeans mass", web_search=True, state=state)
    assert state["web_allowed"] is True


def test_cx01_tool_list_and_conductor_follow_the_turn_flag_in_source():
    src = open(runner.__file__, encoding="utf-8").read()
    assert "if not _web_allowed_turn:" in src
    assert "web_search=_web_allowed_turn," in src


@pytest.mark.parametrize(
    "query",
    [
        "Show me how image metadata is stored in FITS headers",
        "show me what image processing CASA does",
        "show me the image quality metrics for this run",
        # verify round 1 reopen cases
        "Show me image-processing techniques in CASA",
        "Show me image registration steps.",
        "show me images-per-second benchmarks",
        "show me images per second benchmarks",
        "show me image headers for this cube",
        "show me the high-resolution image reconstruction settings",
    ],
)
def test_cx02_show_me_questions_about_image_processing_are_not_picture_requests(query):
    assert wants_image_search(query) is False


@pytest.mark.parametrize(
    "query",
    [
        "show me an image",
        "show me some photos.",
        "show me the latest images of Jupiter from JWST",
        "Show me pictures from the Rubin first look",
        "show me images taken by Hubble",
        "show me a picture, please",
        # verify round 2 reopen cases (no punctuation / follow-up clause)
        "show me a picture please",
        "show me images and explain their processing",
        "show me an image then explain it",
        # verify round 3 reopen case + more modifier shapes
        "show me high-resolution images please",
        "show me the most recent JWST images",
        "show me JWST's newest images",
        "can you show me pictures of Saturn",
        "show me some nice photos!",
        "show me false-color photographs of the Pillars of Creation",
    ],
)
def test_cx02_show_me_picture_requests_still_route_to_images(query):
    assert wants_image_search(query) is True


def test_cx03_proposal_rules_override_live_data_suppression(monkeypatch):
    assert policy_web_override("What are the proposal rules for archive data?")
    calls = _run_prepass(monkeypatch, "What are the proposal rules for archive data?", live=True)
    assert [c["query"] for c in calls] == ["What are the proposal rules for archive data?"]


def test_cx04_explicit_picture_request_passes_want_images_true(monkeypatch):
    calls = _run_prepass(monkeypatch, "Search the web for an image of M87")
    assert calls and calls[0]["want_images"] is True


def test_cx06_month_rollover_does_not_lose_concurrent_increments(isolated_usage_file):
    isolated_usage_file.write_text(json.dumps({"month": "1999-01", "brave_count": 999, "tavily_count": 5}), encoding="utf-8")
    svc = WebSearchService()
    threads = [threading.Thread(target=lambda: [svc._increment_usage("tavily") for _ in range(25)]) for _ in range(6)]
    readers = [threading.Thread(target=lambda: [svc._get_usage() for _ in range(25)]) for _ in range(3)]
    for t in threads + readers:
        t.start()
    for t in threads + readers:
        t.join()
    usage = json.loads(isolated_usage_file.read_text(encoding="utf-8"))
    assert usage["tavily_count"] == 150 and usage["brave_count"] == 0 and usage["month"] != "1999-01"


def test_cx07_tavily_map_counts_usage(monkeypatch, isolated_usage_file):
    monkeypatch.setenv("TAVILY_API_KEY", "t")

    class FakeClient:
        def map(self, url, **kw):
            return {"results": ["https://a.org/x"]}

    monkeypatch.setattr(WebSearchService, "_get_tavily_client", lambda self: FakeClient())
    assert WebSearchService().map_tavily("https://a.org")["success"] is True
    assert json.loads(isolated_usage_file.read_text(encoding="utf-8"))["tavily_other_count"] == 1


def test_cx08_open_tavily_circuit_alone_leaves_web_search_executable(monkeypatch):
    """Exercise the real guard decision in _execute_tool_guarded."""
    from core.agent import QuasarAgent
    from services.host_breaker import HostBreaker

    agent = QuasarAgent.__new__(QuasarAgent)
    ran = []
    monkeypatch.setattr(
        HostBreaker, "open_hosts",
        classmethod(lambda cls, hs: [(h, {"retry_after": 60.0, "reason": "test"}) for h in hs if h == "api.tavily.com"]),
    )
    out = agent._execute_tool_guarded(SimpleNamespace(execute=lambda **kw: ran.append(kw) or {"success": True}), {"query": "q"},
                                      tool_name="web_search", timeout_seconds=5)
    assert ran and out == {"success": True}

    # every provider host open -> refused before it starts
    monkeypatch.setattr(HostBreaker, "open_hosts", classmethod(lambda cls, hs: [(h, {"retry_after": 600.0, "reason": "t"}) for h in hs]))
    ran.clear()
    out = agent._execute_tool_guarded(SimpleNamespace(execute=lambda **kw: ran.append(kw) or {"success": True}), {"query": "q"},
                                      tool_name="web_search", timeout_seconds=5)
    assert not ran and out.get("success") is False
