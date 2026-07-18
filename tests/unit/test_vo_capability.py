"""Unit tests for capabilities/vo.py (P1 family migration #5 — VO registry chain)."""

import threading

import pytest

from capabilities.vo import (
    CAPABILITIES,
    VoAdqlQuery,
    VoConeSearch,
    VoDescribeTable,
    VoFindServices,
    VoListTables,
)
from capabilities.base import CallContext


class _FakeVoService:
    """Recording fake for services.vo_registry.VoRegistryService."""

    def __init__(self, result=None):
        self.result = result if result is not None else {"success": True, "rows": []}
        self.calls = []

    def registry_search(self, keywords, service_type, waveband, max_rows):
        self.calls.append(("registry_search", keywords, service_type, waveband, max_rows))
        return dict(self.result)

    def list_tables(self, access_url, keyword, max_tables):
        self.calls.append(("list_tables", access_url, keyword, max_tables))
        return dict(self.result)

    def describe_table(self, access_url, table_name):
        self.calls.append(("describe_table", access_url, table_name))
        return dict(self.result)

    def run_adql(self, access_url, adql, max_rows):
        self.calls.append(("run_adql", access_url, adql, max_rows))
        return dict(self.result)

    def cone_search(self, access_url, ra, dec, radius_deg, max_rows):
        self.calls.append(("cone_search", access_url, ra, dec, radius_deg, max_rows))
        return dict(self.result)


class _TableRecorder:
    """Stands in for the agent's _external_catalog_table_result helper."""

    def __init__(self):
        self.kwargs = None

    def __call__(self, rows, **kwargs):
        self.kwargs = dict(kwargs, rows=rows)
        return {"success": True, "total_results": len(rows), "tool_name": kwargs["tool_name"]}


def _ctx(service=None, table=None, resolve=None):
    table = table or _TableRecorder()
    services = {
        "get_vo_registry_service": lambda: service,
        "external_catalog_table_result": table,
        "live_imagery_coordinates": resolve or (lambda **kw: (10.0, -5.0, "X")),
    }
    return CallContext(services=services), table


def _run(cap, ctx, **kwargs):
    return cap.run(cap.InputModel(**kwargs), ctx).to_native()


def test_find_services_projects_present_columns():
    svc = _FakeVoService({"success": True,
                          "rows": [{"short_name": "GLEAM", "access_url": "http://tap"}],
                          "warnings": ["w"], "provenance": {"p": 1}})
    ctx, table = _ctx(svc)
    out = _run(VoFindServices(), ctx, keywords="GLEAM", service_type="tap")

    assert out["success"] is True and out["tool_name"] == "vo_find_services"
    # Only the columns present in any row survive:
    assert table.kwargs["columns"] == ["short_name", "access_url"]
    assert table.kwargs["filter_label"] == "VO services for 'GLEAM' [tap]"
    assert table.kwargs["warnings"] == ["w"] and table.kwargs["provenance"] == {"p": 1}
    assert svc.calls[0] == ("registry_search", "GLEAM", "tap", None, 30)


def test_find_services_failure_passes_through_untouched():
    failure = {"success": False, "error": "registry down", "hint": "retry"}
    ctx, table = _ctx(_FakeVoService(failure))
    out = _run(VoFindServices(), ctx, keywords="x")
    assert out == failure and table.kwargs is None


def test_find_services_empty_rows_fall_back_to_all_columns():
    ctx, table = _ctx(_FakeVoService({"success": True, "rows": []}))
    _run(VoFindServices(), ctx, keywords="x")
    assert table.kwargs["columns"] == [
        "short_name", "title", "service_type", "waveband", "access_url"]


def test_find_services_constructor_failure_keeps_legacy_error():
    def _boom():
        raise RuntimeError("pyvo not installed")

    ctx, _ = _ctx()
    ctx.services["get_vo_registry_service"] = _boom
    out = _run(VoFindServices(), ctx, keywords="x")
    assert out == {"success": False, "error": "pyvo not installed"}


def test_list_tables_label_and_defaults():
    svc = _FakeVoService({"success": True, "rows": [{"table_name": "ivoa.obscore"}]})
    ctx, table = _ctx(svc)
    _run(VoListTables(), ctx, access_url="http://tap", keyword="obs")
    assert svc.calls[0] == ("list_tables", "http://tap", "obs", 50)
    assert table.kwargs["filter_label"] == "TAP tables matching 'obs'"
    assert table.kwargs["source"] == "http://tap"

    ctx2, table2 = _ctx(_FakeVoService({"success": True, "rows": []}))
    _run(VoListTables(), ctx2, access_url="http://tap")
    assert table2.kwargs["filter_label"] == "TAP tables"


def test_describe_table_source_and_columns():
    svc = _FakeVoService({"success": True, "rows": [{"name": "ra", "unit": "deg"}]})
    ctx, table = _ctx(svc)
    _run(VoDescribeTable(), ctx, access_url="http://tap", table_name="ivoa.obscore")
    assert table.kwargs["source"] == "Schema: ivoa.obscore"
    assert table.kwargs["columns"] == ["name", "unit"]


def test_adql_query_caps_columns_at_12_with_warning():
    cols = [f"c{i}" for i in range(15)]
    svc = _FakeVoService({"success": True, "rows": [{}], "columns": cols, "warnings": []})
    ctx, table = _ctx(svc)
    _run(VoAdqlQuery(), ctx, access_url="http://tap", adql="SELECT * FROM t")
    assert table.kwargs["columns"] == cols[:12]
    assert table.kwargs["warnings"] == ["Displaying the first 12 of the result's columns."]
    assert table.kwargs["filter_label"] == "SELECT * FROM t"
    assert svc.calls[0][3] == 200  # default max_rows


def test_adql_query_null_adql_flows_and_labels_empty():
    # CX-01: explicit null adql reaches the service call; the label uses (adql or "").
    svc = _FakeVoService({"success": True, "rows": [], "columns": []})
    ctx, table = _ctx(svc)
    out = _run(VoAdqlQuery(), ctx, access_url="http://tap", adql=None)
    assert out["success"] is True
    assert svc.calls[0] == ("run_adql", "http://tap", None, 200)
    assert table.kwargs["filter_label"] == ""
    assert table.kwargs["columns"] == ["result"]  # columns or ["result"]


def test_cone_search_resolves_then_labels_radius_from_provenance():
    svc = _FakeVoService({"success": True, "rows": [{}], "columns": ["ra"],
                          "provenance": {"radius_deg": 0.25}})
    resolved = []

    def _resolve(**kw):
        resolved.append(kw)
        return 150.1, 2.3, "COSMOS field"

    ctx, table = _ctx(svc, resolve=_resolve)
    _run(VoConeSearch(), ctx, access_url="http://scs", target_name="COSMOS")

    assert resolved == [{"target_name": "COSMOS", "ra": None, "dec": None}]
    assert svc.calls[0] == ("cone_search", "http://scs", 150.1, 2.3, 0.1, 100)
    assert table.kwargs["filter_label"] == "cone at COSMOS field, r=0.25 deg"
    assert table.kwargs["source"] == "SCS: http://scs"


def test_cone_search_resolution_failure_is_caught():
    def _resolve(**kw):
        raise ValueError("could not resolve 'Nowhere'")

    ctx, _ = _ctx(_FakeVoService(), resolve=_resolve)
    out = _run(VoConeSearch(), ctx, access_url="http://scs", target_name="Nowhere")
    assert out == {"success": False, "error": "could not resolve 'Nowhere'"}


# ─────────────────────────────────────────────────────────────────────────────
# agent wiring (real module surface)
# ─────────────────────────────────────────────────────────────────────────────
def _wiring_agent():
    from core.tools import ToolRegistry
    from tests.integration.test_agent_archive_tools import _load_agent_module

    module = _load_agent_module()
    agent = module.QuasarAgent.__new__(module.QuasarAgent)
    agent._tls = threading.local()
    agent.tool_registry = ToolRegistry()
    agent.last_search_results = None
    agent.last_run_result = None
    agent.ads_client = None
    agent.openalex_client = None
    return agent


def test_vo_registrations_keep_their_legacy_surface():
    agent = _wiring_agent()
    agent._register_tools()

    fs = agent.tool_registry.get_tool("vo_find_services")
    assert fs is not None and fs.category == "archive"
    assert fs.parameters["required"] == ["keywords"]
    assert fs.parameters["properties"]["service_type"]["enum"] == ["tap", "sia", "ssa", "scs"]

    aq = agent.tool_registry.get_tool("vo_adql_query")
    assert aq.parameters["required"] == ["access_url", "adql"]

    cs = agent.tool_registry.get_tool("vo_cone_search")
    assert cs.parameters["required"] == ["access_url"]
    assert cs.parameters["properties"]["radius_deg"]["default"] == 0.1


def test_vo_ctx_provider_clears_the_card_like_the_legacy_first_line():
    agent = _wiring_agent()
    agent.last_run_result = {"type": "data", "stale": True}
    ctx = agent._vo_ctx_provider()
    assert agent.last_run_result is None
    assert ctx.service("get_vo_registry_service") == agent._get_vo_registry_service


def test_vo_tool_fn_end_to_end_through_the_agent():
    agent = _wiring_agent()
    svc = _FakeVoService({"success": True,
                          "rows": [{"short_name": "GLEAM", "title": "t",
                                    "service_type": "tap", "waveband": "radio",
                                    "access_url": "http://tap"}]})
    agent._vo_registry_service_instance = svc  # pre-seed the lazy getter

    out = agent._vo_tool_fn("vo_find_services")(keywords="GLEAM")
    assert out["success"] is True
    # The real _external_catalog_table_result set the UI card:
    assert agent.last_run_result is not None
    assert agent.last_run_result.get("tool_name") == "vo_find_services"


def test_every_family_capability_is_registered():
    agent = _wiring_agent()
    agent._register_tools()
    for cap in CAPABILITIES:
        assert agent.tool_registry.get_tool(cap.name) is not None, cap.name


def test_unknown_capability_name_raises():
    agent = _wiring_agent()
    with pytest.raises(KeyError):
        agent._vo_tool_fn("not_a_tool")


def test_external_catalog_table_result_stamps_upstream_total():
    """f2-CX-22: a caller that KNOWS the remote total (lightkurve's
    total_available) stamps upstream_truncated/upstream_total structurally on
    the card run-result; an equal or unknown total stamps nothing."""
    agent = _wiring_agent()
    rows = [{"index": i, "mission": "TESS"} for i in range(3)]

    agent._external_catalog_table_result(
        rows, columns=["index", "mission"], source="S", filter_label="F",
        tool_name="search_space_lightcurves", upstream_total=50,
    )
    assert agent.last_run_result["upstream_truncated"] is True
    assert agent.last_run_result["upstream_total"] == 50

    agent._external_catalog_table_result(
        rows, columns=["index", "mission"], source="S", filter_label="F",
        tool_name="search_space_lightcurves", upstream_total=3,
    )
    assert "upstream_truncated" not in agent.last_run_result

    agent._external_catalog_table_result(
        rows, columns=["index", "mission"], source="S", filter_label="F",
        tool_name="search_space_lightcurves",
    )
    assert "upstream_truncated" not in agent.last_run_result


def test_capped_producers_wire_remote_totals_into_the_seam(monkeypatch):
    """f2-CX-22 sweep: every producer that KNOWS its remote total passes it to
    _external_catalog_table_result — MOCServer (total_matches) and ATNF
    (total_matches), alongside lightkurve (total_available)."""
    agent = _wiring_agent()
    agent._live_imagery_coordinates = lambda **k: (10.0, 20.0, "X")

    class _MocSvc:
        def coverage_at(self, *a, **k):
            return {"success": True,
                    "rows": [{"id": "a", "title": "T", "dataproduct_type": "image"}],
                    "count": 1, "total_matches": 40, "warnings": [],
                    "provenance": {"radius_deg": 0.5}}

    class _PsrSvc:
        def search_pulsars(self, *a, **k):
            return {"success": True,
                    "rows": [{"jname": "J0000+0000", "p0_s": 1.0, "sep_arcmin": 3.0}],
                    "count": 1, "total_matches": 12, "warnings": [],
                    "provenance": {"radius_deg": 1.0}}

    agent._get_moc_coverage_service = lambda: _MocSvc()
    agent._get_pulsar_catalog_service = lambda: _PsrSvc()

    out = agent._survey_coverage(ra=10.0, dec=20.0)
    assert out["success"] is True
    assert agent.last_run_result["upstream_truncated"] is True
    assert agent.last_run_result["upstream_total"] == 40

    out = agent._search_pulsars(ra=10.0, dec=20.0)
    assert out["success"] is True
    assert agent.last_run_result["upstream_truncated"] is True
    assert agent.last_run_result["upstream_total"] == 12
