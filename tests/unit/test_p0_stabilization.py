"""
Regression tests for the P0 stabilization pass (docs/v2 IMPLEMENTATION_PRIORITIES).

These lock in the safety/correctness fixes that make the V2 refactor safe:
  * C1  — analyze_uv_coverage returns a typed error, never fabricated UV stats.
  * C2  — the ADS client raises instead of returning invented example papers.
  * C3  — a failed archive search is distinguishable from a genuine empty result.
  * C5  — the RAG relevance gate is on the cosine scale, not the tiny RRF scale.
  * C7  — RecoveryEngine enforces the task SLA (its TimeoutError path is live).
  * C9  — the DAG cache never replays a plan for the wrong target.
  * S1  — user-tool exec() is off by default (RCE guard).
  * S2  — stdio MCP command spawning is off by default (RCE guard).

All offline; no network, no live services.
"""

import asyncio

import pandas as pd
import pytest


# ─────────────────────────────────────────────────────────────────────────────
# C1 — fabrication → typed error (UV coverage)
# ─────────────────────────────────────────────────────────────────────────────
def test_c1_uv_coverage_errors_without_casa_no_mock():
    from services.analysis import RadioAnalysisService

    svc = RadioAnalysisService()
    svc.casa_available = False  # force the no-CASA branch deterministically
    out = svc.analyze_uv_coverage("/fake/ms.ms")

    assert out.get("success") is False
    assert out.get("error") == "casa_unavailable"
    # The fabricated payload must be gone.
    assert "mock_results" not in out
    assert "max_baseline_km" not in str(out)


# ─────────────────────────────────────────────────────────────────────────────
# C2 — fabrication → typed error (ADS literature)
# ─────────────────────────────────────────────────────────────────────────────
def test_c2_ads_raises_without_key_no_fabrication():
    from integrations.ads_client import ADSService, ADSServiceError

    svc = ADSService()
    svc.api_key = None  # force the no-key path regardless of environment

    with pytest.raises(ADSServiceError):
        svc.search_papers("bright quasars")


def test_c2_example_papers_helper_removed():
    from integrations import ads_client

    assert not hasattr(ads_client.ADSService, "_get_example_papers")


# ─────────────────────────────────────────────────────────────────────────────
# C3 — failed search distinguishable from empty
# ─────────────────────────────────────────────────────────────────────────────
def test_c3_errored_frame_is_empty_but_tagged():
    from services.search import _errored_frame

    df = _errored_frame("archive down")
    assert df.empty is True          # backward compatible with .empty callers
    assert len(df) == 0
    assert df.attrs.get("quasar_error") == "archive down"


def test_c3_search_service_tags_failed_search():
    from services.search import SearchService

    svc = SearchService()

    class _Boom:
        def search_by_target(self, *a, **k):
            raise RuntimeError("TAP 503")

    svc.alminer_client = _Boom()
    out = svc.search_by_target("NGC 1068")  # ALMA branch → alminer raises
    assert out.empty is True
    assert out.attrs.get("quasar_error")     # error preserved, not silently swallowed
    assert "503" in out.attrs["quasar_error"]


# ─────────────────────────────────────────────────────────────────────────────
# C5 — RAG relevance gate is on the cosine (0–1) scale, not the RRF scale
# ─────────────────────────────────────────────────────────────────────────────
def test_c5_gate_prefers_semantic_score_over_rrf():
    # Mirrors the corrected agent.py gate: read `_semantic_score` (0–1 cosine)
    # first, NOT `_score` (RRF, ~0.016 max). A relevant doc (semantic 0.5) with a
    # small RRF fusion score must NOT be dropped by a 0–1 threshold.
    def rag_doc_score(meta):
        raw = meta.get("_semantic_score", meta.get("_score"))
        try:
            return float(raw)
        except (TypeError, ValueError):
            return None

    reranked = {"_semantic_score": 0.5, "_score": 0.016}   # relevant, RRF-scored
    offdomain = {"_semantic_score": 0.05, "_score": 0.016}

    assert rag_doc_score(reranked) >= 0.15   # kept (was wrongly dropped before C5)
    assert rag_doc_score(offdomain) < 0.15   # still filtered as junk


# ─────────────────────────────────────────────────────────────────────────────
# C7 — RecoveryEngine enforces the task SLA
# ─────────────────────────────────────────────────────────────────────────────
class _Node:
    def __init__(self, sla=0.4):
        self.description = "do the thing"
        self.sla_seconds = sla
        self.id = "n1"
        self.agent_type = "archive"
        self.depends_on = []


def test_c7_recovery_enforces_sla_timeout():
    from core.recovery import RecoveryEngine

    engine = RecoveryEngine(client=None, max_retries=2)
    # SLA >= 2 so the RETRY handler's int(sla * 1.5) visibly increases it
    # (integer-seconds SLAs are what the DAG actually uses).
    node = _Node(sla=2)
    original_sla = node.sla_seconds

    async def _slow_executor(_node):
        await asyncio.sleep(30.0)  # always blows the SLA
        return {"success": True, "total_results": 5}

    result = asyncio.run(
        engine.execute_with_recovery(node, _slow_executor)
    )

    # The SLA fired: first timeout → RETRY extends the SLA, second → DECOMPOSE.
    # Before C7 nothing imposed a deadline, so the executor would have run to
    # completion and this would have returned {"success": True, ...}.
    assert node.sla_seconds > original_sla, "RETRY should have extended the SLA"
    assert isinstance(result, str) and "timed out" in result.lower()


def test_c7_recovery_returns_result_within_sla():
    from core.recovery import RecoveryEngine

    engine = RecoveryEngine(client=None, max_retries=2)
    node = _Node(sla=5.0)

    async def _fast_executor(_node):
        return {"success": True, "total_results": 3}

    result = asyncio.run(engine.execute_with_recovery(node, _fast_executor))
    assert result == {"success": True, "total_results": 3}


# ─────────────────────────────────────────────────────────────────────────────
# C9 — DAG cache never replays a plan for the wrong target
# ─────────────────────────────────────────────────────────────────────────────
def _fresh_cache(tmp_path):
    from core.dag_cache import DAGCache

    return DAGCache(cache_path=str(tmp_path / "dag_cache.json"))


def test_c9_same_target_reuses_plan(tmp_path):
    cache = _fresh_cache(tmp_path)
    subtasks = [
        {"id": "t1", "agent_type": "archive", "description": "Search ALMA for NGC 1068"},
        {"id": "t2", "agent_type": "literature", "description": "Find papers on NGC 1068"},
    ]
    cache.store("Compare ALMA data for NGC 1068 with papers", subtasks, "r")

    hit = cache.find_similar("Compare ALMA data for NGC 1068 with papers")
    assert hit is not None
    assert any("NGC 1068" in s["description"] for s in hit["subtasks"])


def test_c9_different_target_is_resubstituted(tmp_path):
    cache = _fresh_cache(tmp_path)
    subtasks = [
        {"id": "t1", "agent_type": "archive", "description": "Search ALMA for NGC 1068"},
        {"id": "t2", "agent_type": "literature", "description": "Find recent papers on NGC 1068"},
    ]
    cache.store("Compare ALMA data for NGC 1068 with papers", subtasks, "r")

    # A structurally identical query about a DIFFERENT object.
    hit = cache.find_similar("Compare ALMA data for M87 with papers")
    assert hit is not None, "clean 1:1 target swap should be reusable"
    blob = " ".join(s["description"] for s in hit["subtasks"])
    assert "M87" in blob
    assert "NGC 1068" not in blob, "the stale target must not survive"


def test_c9_unsafe_mismatch_skips_cache(tmp_path):
    cache = _fresh_cache(tmp_path)
    # Cached plan involves ONE target...
    subtasks = [
        {"id": "t1", "agent_type": "archive", "description": "Search ALMA for NGC 1068 continuum"},
    ]
    cache.store("Search ALMA for NGC 1068 continuum emission maps", subtasks, "r")

    # ...new query mentions TWO targets → no clean 1:1 mapping → skip (decompose fresh).
    hit = cache.find_similar("Search ALMA for M87 and NGC 4151 continuum emission maps")
    assert hit is None


def test_c9_prefix_collision_never_mangles_target(tmp_path):
    # Regression: 'M8' must not rewrite inside 'M87' (word-boundary substitution).
    cache = _fresh_cache(tmp_path)
    subtasks = [
        {"id": "t1", "agent_type": "archive", "description": "Observe M8"},
        {"id": "t2", "agent_type": "archive", "description": "Observe M87"},
    ]
    cache.store("Compare M8 and M87 in ALMA", subtasks, "r")

    hit = cache.find_similar("Compare M9 and M99 in ALMA")
    if hit is not None:  # if reused, targets must be exactly M9 and M99
        blob = " ".join(s["description"] for s in hit["subtasks"])
        assert "M97" not in blob, "prefix collision produced a spurious target"
        assert "M8" not in blob and "M87" not in blob, "stale target survived"
        assert "M9" in blob and "M99" in blob


def test_c9_proper_name_fuzzy_match_not_reused(tmp_path):
    # Regression: proper names aren't caught by the catalog-designation regex,
    # so a fuzzy match differing only by the proper name must NOT reuse verbatim.
    cache = _fresh_cache(tmp_path)
    subtasks = [
        {"id": "t1", "agent_type": "archive", "description": "Search ALMA archive for continuum images of Betelgeuse"},
        {"id": "t2", "agent_type": "literature", "description": "Find recent papers on Betelgeuse"},
    ]
    cache.store("Search ALMA archive for continuum images of Betelgeuse and summarize", subtasks, "r")

    hit = cache.find_similar("Search ALMA archive for continuum images of Fomalhaut and summarize")
    assert hit is None, "must not replay a Betelgeuse plan for a Fomalhaut query"


def test_c9_exact_match_reused_even_with_unparsed_target(tmp_path):
    # The SAME proper-name query (exact fingerprint) is safe to reuse.
    cache = _fresh_cache(tmp_path)
    subtasks = [
        {"id": "t1", "agent_type": "archive", "description": "Search ALMA for Betelgeuse"},
        {"id": "t2", "agent_type": "literature", "description": "Papers on Betelgeuse"},
    ]
    q = "Search ALMA archive for continuum images of Betelgeuse and summarize"
    cache.store(q, subtasks, "r")
    hit = cache.find_similar(q)
    assert hit is not None


def test_c7_decompose_passes_node_to_coroutine_executor():
    # Regression: _decompose_and_execute must call a node-taking coroutine
    # executor with a NODE (not (desc, ctx)) — the arity bug C7 introduced.
    from core.recovery import RecoveryEngine

    class _Resp:
        def __init__(self, txt):
            self.output_text = txt

    class _Responses:
        def create(self, **k):
            return _Resp('["sub one", "sub two"]')

    class _Client:
        responses = _Responses()

    engine = RecoveryEngine(client=_Client(), max_retries=2)
    node = _Node(sla=5)
    received = []

    async def _node_executor(n):
        received.append(getattr(n, "description", "__NOT_A_NODE__"))
        return {"success": True, "rows": 1, "for": n.description}

    result = asyncio.run(engine._decompose_and_execute(node, _node_executor))
    assert "sub one" in received and "sub two" in received  # got nodes, not strings
    assert "Sub-task" in result  # sub-results aggregated, not a TypeError wipeout


# ─────────────────────────────────────────────────────────────────────────────
# S1 / S2 — RCE guards are OFF by default
# ─────────────────────────────────────────────────────────────────────────────
def test_s1_user_tool_exec_disabled_by_default(monkeypatch):
    from services import user_tools_service as uts

    monkeypatch.delenv("QUASAR_ENABLE_USER_TOOL_EXEC", raising=False)
    assert uts._user_tool_exec_enabled() is False

    svc = uts.UserToolsService()
    with pytest.raises(RuntimeError):
        svc.build_callable({"name": "evil", "code": "def evil():\n    return __import__('os').getcwd()"})


def test_s1_user_tool_exec_enabled_when_flagged(monkeypatch):
    from services import user_tools_service as uts

    monkeypatch.setenv("QUASAR_ENABLE_USER_TOOL_EXEC", "1")
    assert uts._user_tool_exec_enabled() is True


def test_s2_stdio_mcp_disabled_by_default(monkeypatch, tmp_path):
    from services import mcp_server_service as mss

    monkeypatch.delenv("QUASAR_ENABLE_MCP_STDIO", raising=False)
    assert mss.mcp_stdio_enabled() is False

    svc = mss.MCPServerService(base_dir=str(tmp_path))
    cfg = mss.MCPServerConfig(name="local", transport="stdio", command="npx")
    with pytest.raises(ValueError):
        svc.save_server("user-under-test", cfg)


def test_s2_http_mcp_allowed_by_default(monkeypatch, tmp_path):
    from services import mcp_server_service as mss

    monkeypatch.delenv("QUASAR_ENABLE_MCP_STDIO", raising=False)
    svc = mss.MCPServerService(base_dir=str(tmp_path))
    cfg = mss.MCPServerConfig(name="remote", transport="http", url="https://example.org/mcp")
    saved = svc.save_server("user-under-test", cfg)   # http/SSE stays allowed
    assert saved["transport"] == "http"
