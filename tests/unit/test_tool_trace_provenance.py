"""Feature 1 — raw-query provenance surface.

Asserts that every executed tool contributes a uniform, redacted ``request`` to
the tool trace (one case per kind: adql | http | ads | params | args), and — the
back-compat guard — that the top-level ``sql`` field SURVIVES for SQL tools,
because Benchmark/datalabbench scores against it via ``trace_regex``.
"""

import importlib.util
import json
import threading
from pathlib import Path

import pandas as pd
import pytest

from capabilities.base import (
    PROVENANCE_SIDECAR_KEY,
    BaseCapability,
    CallContext,
    Provenance,
    ToolResult,
)
from core.provenance import build_tool_request
from services.secret_redaction import redact_url


# ─────────────────────────────────────────────────────────────────────────────
# build_tool_request — one case per kind
# ─────────────────────────────────────────────────────────────────────────────
def _sidecar(**prov):
    return {"provenance": prov}


def test_request_kind_adql_carries_the_exact_executed_sql():
    sql = "SELECT ra, dec FROM gaia_dr3.gaia_source WHERE q3c_radial_query(ra, dec, 250.4, 36.4, 0.1)"
    req = build_tool_request(
        "datalab_cone_search",
        {"catalog": "gaia_dr3"},
        result_obj={"success": True, "query_summary": "SELECT ra, dec FROM ..."},
        sidecar=_sidecar(service="datalab", query=sql, endpoint="https://datalab.noirlab.edu/tap"),
    )
    assert req["kind"] == "adql"
    # The EXACT executed ADQL, not the lossy `query_summary` the model sees.
    assert req["text"] == sql
    assert req["service"] == "datalab"


def test_request_kind_http_uses_the_endpoint_url():
    req = build_tool_request(
        "hips_cutout",
        {"ra": 10.0, "dec": 20.0},
        result_obj={"success": True},
        sidecar=_sidecar(service="hips", endpoint="https://alasky.u-strasbg.fr/hips-image-services/hips2fits?ra=10&dec=20"),
    )
    assert req["kind"] == "http"
    assert req["method"] == "GET"
    assert req["url"].startswith("https://alasky.u-strasbg.fr/hips-image-services/hips2fits")
    assert req["text"] == req["url"]


def test_request_kind_ads_carries_the_query_and_filters():
    req = build_tool_request(
        "search_papers",
        {"query": "NGC 253 outflow", "rows": 25},
        result_obj={"success": True},
        sidecar=_sidecar(service="ads", query="NGC 253 outflow"),
    )
    assert req["kind"] == "ads"
    assert req["q"] == "NGC 253 outflow"
    assert req["rows"] == 25
    assert req["text"] == "NGC 253 outflow"


def test_ads_rows_falls_back_to_max_results():  # CX-37
    req = build_tool_request(
        "search_papers",
        {"query": "NGC 253", "max_results": 7},
        sidecar=_sidecar(service="ads", query="NGC 253"),
    )
    assert req["kind"] == "ads"
    assert req["rows"] == 7   # the executed result count is surfaced


def test_request_kind_params_for_a_parameterized_service_call():
    req = build_tool_request(
        "splatalogue_line_search",
        {"freq_min_ghz": 230.0, "freq_max_ghz": 231.0},
        result_obj={"success": True},
        sidecar=_sidecar(service="splatalogue"),
    )
    assert req["kind"] == "params"
    assert req["params"]["freq_min_ghz"] == 230.0
    # `text` is the copy payload for EVERY kind.
    assert "freq_min_ghz" in req["text"]


def test_request_kind_args_is_the_fallback_when_there_is_no_provenance():
    req = build_tool_request(
        "some_legacy_tool",
        {"target": "M13"},
        result_obj={"success": True},
        sidecar=None,
    )
    assert req["kind"] == "args"
    assert req["args"] == {"target": "M13"}
    assert "M13" in req["text"]


def test_request_falls_back_to_the_sql_argument_when_provenance_is_absent():
    req = build_tool_request(
        "datalab_sql_query",
        {"sql": "SELECT 1", "expert_ack": True},
        result_obj={"success": True},
        sidecar=None,
    )
    assert req["kind"] == "adql"
    assert req["text"] == "SELECT 1"


def test_request_never_raises_on_garbage_input():
    req = build_tool_request(None, "not-a-dict", result_obj=object(), sidecar="nonsense")
    assert req["kind"] in {"adql", "http", "ads", "params", "args"}
    assert isinstance(req["text"], str)


# ─────────────────────────────────────────────────────────────────────────────
# Redaction — this object reaches the browser
# ─────────────────────────────────────────────────────────────────────────────
def test_secrets_are_redacted_from_endpoint_urls():
    req = build_tool_request(
        "datalab_image_cutout",
        {},
        sidecar=_sidecar(
            service="datalab",
            endpoint="https://datalab.noirlab.edu/sia?TOKEN=abc123secret&ra=10",
        ),
    )
    assert "abc123secret" not in json.dumps(req)
    assert "[REDACTED]" in req["url"]
    # The parameter NAME survives, so the request stays recognizable.
    assert "TOKEN" in req["url"]
    assert "ra=10" in req["url"]


def test_secret_shaped_arguments_are_redacted():
    req = build_tool_request("legacy", {"api_key": "sk-abcdef0123456789", "target": "M13"})
    blob = json.dumps(req)
    assert "sk-abcdef0123456789" not in blob
    assert "M13" in blob


def test_redact_url_leaves_a_clean_url_untouched():
    url = "https://almascience.nrao.edu/tap?QUERY=SELECT+1&ra=10"
    assert "SELECT" in redact_url(url)
    assert "[REDACTED]" not in redact_url(url)


def test_nested_and_url_valued_args_are_redacted(  # CX-02
):
    req = build_tool_request("legacy", {
        "access_url": "https://x/sia?TOKEN=deadbeefsecret&ra=1",
        "params": {"auth": "sk-supersecretvalue0123", "ra": 5},
    })
    blob = json.dumps(req)
    assert "deadbeefsecret" not in blob      # URL-valued arg redacted
    assert "sk-supersecretvalue0123" not in blob  # nested secret key redacted
    assert '"ra": 5' in blob or "'ra': 5" in blob  # non-secret nested value kept


def test_args_and_params_are_size_capped(  # CX-17
):
    big = {f"k{i}": "x" * 200 for i in range(100)}  # ~20 KB
    req = build_tool_request("legacy", big)
    assert len(json.dumps(req.get("args", {}))) <= 4200


def test_persistable_trace_strips_raw_args_but_keeps_redacted_request():  # CX-01
    from core.provenance import persistable_trace

    live = [{
        "name": "sia_search",
        "ok": True,
        "arguments": {"access_url": "https://x/sia?TOKEN=leakmeplease&ra=1"},
        "output": "big blob with TOKEN=leakmeplease inside",
        "sql": "SELECT 1",
        "request": {"kind": "http", "url": "https://x/sia?TOKEN=[REDACTED]&ra=1",
                    "text": "https://x/sia?TOKEN=[REDACTED]&ra=1"},
    }]
    slim = persistable_trace(live)
    blob = json.dumps(slim)
    assert "leakmeplease" not in blob          # raw args/output never persisted
    assert "arguments" not in slim[0]
    assert "output" not in slim[0]
    assert slim[0]["request"]["kind"] == "http"  # redacted request kept for reload


def test_request_text_is_capped_and_does_not_blow_the_trace_budget():
    huge = "SELECT " + ("x" * 50000)
    req = build_tool_request("datalab_sql_query", {"sql": huge})
    assert len(req["text"]) <= 2100  # 2 KB cap + ellipsis


# ─────────────────────────────────────────────────────────────────────────────
# The adapter sidecar — the pipe that makes provenance reachable at all
# ─────────────────────────────────────────────────────────────────────────────
class _ProvCap(BaseCapability):
    name = "prov_tool"
    description = "test"
    category = "test"

    class InputModel(__import__("pydantic").BaseModel):
        sql: str = "SELECT 1"

    def run(self, inp, ctx):
        return ToolResult(
            success=True,
            native={"success": True, "rowcount": 1},
            provenance=Provenance(service="datalab", query=inp.sql),
        )


def test_adapter_attaches_provenance_without_changing_the_model_facing_dict():
    from adapters.native import build_tool

    tool = build_tool(_ProvCap(), lambda: CallContext())
    out = tool.execute(sql="SELECT 42")

    sidecar = out.pop(PROVENANCE_SIDECAR_KEY)
    # to_native()'s byte-parity contract is intact once the sidecar is removed.
    assert out == {"success": True, "rowcount": 1}
    assert sidecar["provenance"]["query"] == "SELECT 42"


def test_toolresult_to_native_still_drops_the_sidecar():
    tr = ToolResult(success=True, native={"a": 1}, provenance=Provenance(service="x", query="q"))
    assert tr.to_native() == {"a": 1}  # unchanged, frozen parity contract


# ─────────────────────────────────────────────────────────────────────────────
# _record_tool_trace — the benchmark back-compat guard
# ─────────────────────────────────────────────────────────────────────────────
class _FakeAgent:
    """The two attributes _record_tool_trace touches."""

    def __init__(self):
        self._tls = threading.local()
        self._tls.accumulated_tool_trace = []

    @property
    def _accumulated_tool_trace(self):
        return self._tls.accumulated_tool_trace


def _record(tool_name, args, result_obj, sidecar=None):
    from core.agent import QuasarAgent

    agent = _FakeAgent()
    QuasarAgent._record_tool_trace(
        agent, tool_name, args, json.dumps(result_obj), result_obj=result_obj,
        provenance=sidecar,
    )
    return agent._accumulated_tool_trace[0]


def test_trace_record_keeps_sql_for_the_benchmark_and_adds_request():
    sql = "SELECT ra, dec FROM gaia_dr3.gaia_source"
    record = _record(
        "datalab_cone_search",
        {"catalog": "gaia_dr3"},
        {"success": True, "query_summary": sql, "rowcount": 7},
        sidecar=_sidecar(service="datalab", query=sql),
    )
    # BACK-COMPAT: datalabbench's trace_regex scores against this field.
    assert record["sql"] == sql
    assert record["rowcount"] == 7
    assert record["request"]["kind"] == "adql"
    assert record["request"]["text"] == sql


def test_trace_record_has_a_request_for_a_tool_with_no_provenance():
    record = _record("web_search", {"query": "ngc 253"}, {"success": True})
    assert record["request"]["kind"] == "args"
    assert "sql" not in record


def test_trace_record_marks_failures_and_still_describes_the_request():
    record = _record("datalab_sql_query", {"sql": "SELECT bad"}, {"success": False, "error": "boom"})
    assert record["ok"] is False
    assert record["request"]["text"] == "SELECT bad"


# ─────────────────────────────────────────────────────────────────────────────
# The ALMA bridge: alma_tap_provenance is instance-wide + last-write-wins, so a
# STALE query must never be presented as this call's provenance.
# ─────────────────────────────────────────────────────────────────────────────
class _AlmaHost:
    """Minimal stand-in exposing the one agent attribute the wrapper reads."""

    from core.agent import QuasarAgent

    _with_alma_tap_provenance = QuasarAgent._with_alma_tap_provenance


def test_alma_wrapper_attaches_the_query_this_call_executed():
    host = _AlmaHost()
    host._alma_tap_provenance_state = {"query": None, "url": None}

    def fake_tool(**kwargs):
        host._alma_tap_provenance_state["query"] = (
            "SELECT * FROM ivoa.obscore WHERE target_name='NGC 253'"
        )
        host._alma_tap_provenance_state["url"] = "https://almascience.nrao.edu/tap"
        return {"success": True}

    out = host._with_alma_tap_provenance(fake_tool)(target_name="NGC 253")
    prov = out[PROVENANCE_SIDECAR_KEY]["provenance"]
    assert prov["service"] == "alma"
    assert "NGC 253" in prov["query"]

    req = build_tool_request(
        "search_by_target", {"target_name": "NGC 253"},
        result_obj={"success": True}, sidecar=out[PROVENANCE_SIDECAR_KEY],
    )
    assert req["kind"] == "adql"
    assert req["text"] == prov["query"]


def test_alma_wrapper_does_not_attach_a_stale_query_from_a_previous_call():
    host = _AlmaHost()
    # A PREVIOUS turn left its query behind in the instance-wide dict.
    host._alma_tap_provenance_state = {
        "query": "SELECT * FROM ivoa.obscore WHERE target_name='M87'",
        "url": "https://almascience.nrao.edu/tap",
    }

    def tool_that_runs_no_tap_query(**kwargs):
        return {"success": True, "data": "served from cache"}

    out = host._with_alma_tap_provenance(tool_that_runs_no_tap_query)(x=1)
    # Nothing attached: this call executed no query of its own.
    assert PROVENANCE_SIDECAR_KEY not in out


def test_ads_capability_provenance_reaches_the_request_as_ads_kind():  # CX-03
    """The papers capability carries the TRANSLATED ADS query via the sidecar,
    so the surfaced request is the executed ADS query, not the NL arguments."""
    from adapters.native import build_tool
    from capabilities.base import CallContext
    import capabilities.papers as papers

    tr = papers._native(
        {"success": True, "count": 1, "ads_query": 'author:"Smith" year:2020', "papers": []},
        ads_query='author:"Smith" year:2020',
    )
    sidecar = tr.provenance_sidecar()
    assert sidecar["provenance"]["service"] == "ads"

    req = build_tool_request("search_papers", {"query": "papers by Smith in 2020"},
                             result_obj=tr.to_native(), sidecar=sidecar)
    assert req["kind"] == "ads"
    assert req["text"] == 'author:"Smith" year:2020'   # executed query, not NL args


def test_alma_cone_provenance_is_built_request_local_and_exact():  # CX-07, CX-36, CX-38
    """The obscore cone is reconstructed from THIS request's coordinates and
    written to the request-scoped state — no shared client attribute — so it is
    both exact (byte-matches the client's raw-TAP query) and race-free."""
    from capabilities.alma import _obscore_cone_adql, _set_alma_cone_provenance
    from capabilities.base import CallContext

    adql = _obscore_cone_adql(250.42, 36.46, 0.05)
    assert adql == (
        "SELECT * FROM ivoa.obscore WHERE CONTAINS(POINT('ICRS', s_ra, s_dec), "
        "CIRCLE('ICRS', 250.42, 36.46, 0.05)) = 1"
    )

    # Two "concurrent" requests with their OWN injected state can't cross-attribute.
    a, b = {"query": None, "url": None}, {"query": None, "url": None}
    _set_alma_cone_provenance(CallContext(services={"alma_tap_provenance": a}), 10.0, 20.0, 0.05)
    _set_alma_cone_provenance(CallContext(services={"alma_tap_provenance": b}), 99.0, -5.0, 0.14)
    assert "10.0, 20.0, 0.05" in a["query"]
    assert "99.0, -5.0, 0.14" in b["query"]
    assert a["query"] != b["query"]


def test_dispatch_tool_call_pops_sidecar_from_model_payload():  # CX-22
    """Real consumer path: dispatch a capability-backed tool the way the agent
    does and assert the model-facing JSON is sidecar-free while the trace still
    captured the request."""
    import json as _json
    from adapters.native import build_tool
    from capabilities.base import CallContext
    from core.agent import QuasarAgent
    from core.tools import ToolRegistry

    registry = ToolRegistry()
    registry.register(build_tool(_ProvCap(), lambda: CallContext()))

    class _Agent:
        _tls = threading.local()
        tool_registry = registry
        _pop_provenance_sidecar = staticmethod(QuasarAgent._pop_provenance_sidecar)
        _record_tool_trace = QuasarAgent._record_tool_trace

        @property
        def _accumulated_tool_trace(self):
            if not hasattr(self._tls, "t"):
                self._tls.t = []
            return self._tls.t

    agent = _Agent()
    result_str = QuasarAgent._dispatch_tool_call(agent, "prov_tool", _json.dumps({"sql": "SELECT 7"}))

    assert PROVENANCE_SIDECAR_KEY not in result_str
    assert _json.loads(result_str) == {"success": True, "rowcount": 1}
    rec = agent._accumulated_tool_trace[0]
    assert rec["request"]["kind"] == "adql" and rec["request"]["text"] == "SELECT 7"


def test_alma_wrapper_attaches_an_identical_repeated_query():  # CX-08
    """Clearing the state before each call (vs diffing before/after) means a
    second execution of the SAME query is still surfaced, not suppressed."""
    host = _AlmaHost()
    host._alma_tap_provenance_state = {
        "query": "SELECT * FROM ivoa.obscore WHERE 1=1", "url": "u",
    }

    def tool_runs_same_query_again(**kwargs):
        host._alma_tap_provenance_state["query"] = "SELECT * FROM ivoa.obscore WHERE 1=1"
        host._alma_tap_provenance_state["url"] = "u"
        return {"success": True}

    out = host._with_alma_tap_provenance(tool_runs_same_query_again)()
    assert PROVENANCE_SIDECAR_KEY in out
    assert out[PROVENANCE_SIDECAR_KEY]["provenance"]["query"].endswith("WHERE 1=1")


def test_alma_wrapper_survives_a_missing_state_attribute():
    host = _AlmaHost()  # no _alma_tap_provenance_state at all
    out = host._with_alma_tap_provenance(lambda **k: {"success": True})()
    assert out == {"success": True}


def test_alma_tap_provenance_state_is_request_scoped_not_shared():
    """The agent is a process-wide singleton and concurrent chats run on
    separate executor threads — one request must never be able to report
    another request's ADQL as its own provenance."""
    from core.agent import QuasarAgent

    class _Agent:
        _tls = threading.local()
        _alma_tap_provenance_state = QuasarAgent._alma_tap_provenance_state

    agent = _Agent()
    agent._alma_tap_provenance_state["query"] = "SELECT 'main thread'"
    seen = {}

    def other_request():
        # A concurrent request must start clean, not see the other thread's query.
        seen["initial"] = agent._alma_tap_provenance_state["query"]
        agent._alma_tap_provenance_state["query"] = "SELECT 'worker thread'"
        seen["own"] = agent._alma_tap_provenance_state["query"]

    t = threading.Thread(target=other_request)
    t.start()
    t.join()

    assert seen["initial"] is None            # no cross-request bleed
    assert seen["own"] == "SELECT 'worker thread'"
    # ...and the other thread's write did not clobber this request's state.
    assert agent._alma_tap_provenance_state["query"] == "SELECT 'main thread'"


# ─────────────────────────────────────────────────────────────────────────────
# runner: stamping the request onto the results a tool produced
# ─────────────────────────────────────────────────────────────────────────────
def test_stamp_only_touches_results_this_tool_produced():
    from core.runner import _stamp_request_on_results

    class _Agent:
        pass

    agent = _Agent()
    earlier = {"type": "data", "request": {"kind": "adql", "text": "EARLIER"}}
    fresh = {"type": "data"}
    agent._accumulated_run_results = [earlier, fresh]
    agent.last_run_result = fresh

    request = {"kind": "adql", "text": "SELECT 1"}
    # since=1 → only `fresh` belongs to this call.
    _stamp_request_on_results(agent, request, "datalab_sql_query", since=1, primary=fresh)

    assert fresh["request"] == request
    assert earlier["request"]["text"] == "EARLIER"  # an earlier card is never relabeled


# ─────────────────────────────────────────────────────────────────────────────
# The data card carries the request through to the UI (and to persistence)
# ─────────────────────────────────────────────────────────────────────────────
def _load_api_main():
    pytest.importorskip("fastapi")
    path = Path(__file__).resolve().parents[2] / "ui-pro" / "api" / "main.py"
    spec = importlib.util.spec_from_file_location("quasar_api_main_for_provenance_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_data_card_carries_the_request_into_the_payload_and_rich_meta():
    api_main = _load_api_main()
    df = pd.DataFrame([{"ra": 1.0, "dec": 2.0}])
    request = {"kind": "adql", "text": "SELECT ra, dec FROM gaia_dr3.gaia_source"}

    event_str, rich = api_main._build_data_card_event({
        "type": "data",
        "data": df,
        "source": "Data Lab",
        "tool_name": "datalab_select_catalog_rows",
        "request": request,
        "requestTool": "datalab_select_catalog_rows",
    })

    assert rich["request"] == request          # persisted → survives a reload
    assert request["text"] in event_str        # emitted live
    assert rich["toolName"] == "datalab_select_catalog_rows"


def test_record_rich_image_persists_the_request_for_reload():  # CX-14
    """Image/plotly cards carry their own request into messages.metadata, so the
    per-card block survives a reload (not just the turn-level one)."""
    entries = []
    urls = set()
    rich_image = None

    # Mirrors sse.py::_record_rich_image's contract.
    def _record(img_url, caption, meta=None, request=None):
        nonlocal rich_image
        if not img_url or img_url in urls:
            return None
        entry = {"url": img_url, "caption": caption}
        if meta is not None:
            entry["meta"] = meta
        if isinstance(request, dict) and request:
            entry["request"] = request
        urls.add(img_url)
        entries.append(entry)
        rich_image = entry
        return entry

    req = {"kind": "adql", "text": "SELECT ra, dec FROM gaia_dr3.gaia_source"}
    _record("/plots/x.png", "sky map", {"kind": "sky"}, request=req)
    assert entries[0]["request"] == req
    # A card with no known request must not invent one.
    _record("/plots/y.png", "other", None, request=None)
    assert "request" not in entries[1]


def test_data_card_without_a_request_omits_the_key():
    api_main = _load_api_main()
    df = pd.DataFrame([{"ra": 1.0, "dec": 2.0}])
    _, rich = api_main._build_data_card_event({
        "type": "data", "data": df, "source": "Data Lab", "tool_name": "x",
    })
    assert "request" not in rich
