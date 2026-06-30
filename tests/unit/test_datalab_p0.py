from __future__ import annotations

import builtins
import json
import threading

import pandas as pd
import pytest

from core.tools import ToolRegistry
from integrations.datalab_client import DatalabClient, DatalabClientError, DatalabResult
from services.datalab_query_builders import (
    build_cone_select,
    build_q3c_crossmatch,
    build_sed_select,
    build_variable_star_select,
    build_zhistogram,
)
from services.datalab_result_store import DatalabResultStore
from services.datalab_sql_policy import DatalabPolicyError, validate
from tests.integration.test_agent_archive_tools import _load_agent_module


def test_client_sanitizes_trailing_semicolon_and_rejects_final_line_comment():
    assert DatalabClient.sanitize_query("SELECT * FROM gaia_dr3.gaia_source;  ") == "SELECT * FROM gaia_dr3.gaia_source"
    sql = "SELECT *\n-- inner comment is fine\nFROM gaia_dr3.gaia_source"
    assert "-- inner comment" in DatalabClient.sanitize_query(sql)
    with pytest.raises(ValueError, match="must not end"):
        DatalabClient.sanitize_query("SELECT * FROM gaia_dr3.gaia_source\n-- trailing")


def test_client_uses_anonymous_rest_transport_without_astro_datalab(monkeypatch):
    from integrations import datalab_client as dc

    # No astro-datalab dependency: defaults to the anonymous token + REST /query URL.
    client = dc.DatalabClient()
    assert client.token == dc.ANON_TOKEN
    assert client.service_url.endswith("/query")

    # Exactly one of sql/adql is required (checked before any network call).
    with pytest.raises(ValueError, match="exactly one"):
        client.query(sql="SELECT 1", adql="SELECT 1")
    with pytest.raises(ValueError, match="exactly one"):
        client.query()

    captured: dict = {}

    class _Resp:
        status_code = 200
        text = "ra,dec\n1.0,-1.0\n"

    def fake_get(url, headers=None, timeout=None):
        captured["url"] = url
        captured["headers"] = headers
        return _Resp()

    monkeypatch.setattr(dc.requests, "get", fake_get)
    res = client.query(sql="SELECT ra,dec FROM gaia_dr3.gaia_source WHERE q3c_radial_query(ra,dec,1,2,0.1)")
    assert isinstance(res, dc.DatalabResult)
    assert list(res.dataframe.columns) == ["ra", "dec"]
    assert "/query?" in captured["url"] and "ofmt=csv" in captured["url"] and "sql=" in captured["url"]
    assert captured["headers"]["X-DL-AuthToken"] == dc.ANON_TOKEN

    # Transport failures surface as DatalabClientError, not a raw requests error.
    def boom(url, headers=None, timeout=None):
        raise dc.requests.RequestException("network down")

    monkeypatch.setattr(dc.requests, "get", boom)
    with pytest.raises(dc.DatalabClientError):
        client.query(sql="SELECT 1 FROM gaia_dr3.gaia_source")


def test_crossmatch_builder_policy_accepts_planner_safe_form():
    sql, meta = build_q3c_crossmatch(
        ra=229.022,
        dec=-0.112,
        radius_deg=0.1,
        small_columns=["source_id", "ra", "dec", "phot_g_mean_mag"],
        big_columns=["id", "ra", "dec", "class_star"],
        limit=250,
    )

    assert sql.startswith("WITH g AS MATERIALIZED")
    assert "FROM gaia_dr3.gaia_source" in sql
    assert "JOIN nsc_dr2.object AS big" in sql
    assert "q3c_join(g.ra, g.dec, big.ra, big.dec," in sql
    validated = validate(sql, source="builder", meta=meta)
    assert validated.sql == sql


def test_policy_rejects_flat_or_unauthorized_q3c_join():
    flat = """
SELECT *
FROM nsc_dr2.object AS big
JOIN gaia_dr3.gaia_source AS g
  ON q3c_join(big.ra, big.dec, g.ra, g.dec, 0.000277777777777778)
WHERE big.ra BETWEEN 228 AND 230 AND big.dec BETWEEN -1 AND 1
LIMIT 50
"""
    with pytest.raises(DatalabPolicyError, match="BETWEEN"):
        validate(flat, source="expert")

    unauthorized = """
SELECT *
FROM gaia_dr3.gaia_source AS g
JOIN nsc_dr2.object AS big
  ON q3c_join(g.ra, g.dec, big.ra, big.dec, 0.000277777777777778)
WHERE q3c_radial_query(g.ra, g.dec, 229.022, -0.112, 0.1)
LIMIT 50
"""
    with pytest.raises(DatalabPolicyError, match="q3c_join"):
        validate(unauthorized, source="expert")


def test_policy_rejects_unsafe_queries_and_accepts_box_ok_table():
    with pytest.raises(DatalabPolicyError, match="Row-level"):
        validate("SELECT * FROM gaia_dr3.gaia_source")

    with pytest.raises(DatalabPolicyError, match="DDL/DML|Only SELECT"):
        validate("DELETE FROM gaia_dr3.gaia_source WHERE source_id = 1")

    with pytest.raises(DatalabPolicyError, match="BETWEEN"):
        validate("SELECT * FROM gaia_dr3.gaia_source WHERE ra BETWEEN 1 AND 2 AND dec BETWEEN -1 AND 1 LIMIT 10")

    validated = validate("SELECT * FROM sdss_dr17.specobj WHERE ra BETWEEN 1 AND 2 AND dec BETWEEN -1 AND 1 LIMIT 10")
    assert validated.sql.endswith("LIMIT 10")


def test_policy_injects_limit_for_spatial_row_level_select():
    sql, meta = build_cone_select("gaia_dr3", "gaia_source", ra=10, dec=0, radius_deg=0.1, limit=50)
    sql_without_limit = sql.rsplit("\nLIMIT", 1)[0]
    validated = validate(sql_without_limit, source="builder", meta=meta)
    assert validated.sql.endswith("LIMIT 500")
    assert any("Injected LIMIT" in warning for warning in validated.warnings)



def test_auxiliary_builders_keep_governor_metadata():
    sed_sql, sed_meta = build_sed_select(ra=10, dec=0, radius_deg=0.1)
    assert "ls_dr9.tractor" in sed_sql
    assert sed_meta["builder"] == "sed_select"

    var_sql, var_meta = build_variable_star_select(source_id="abc123")
    assert "id = 'abc123'" in var_sql
    assert validate(var_sql, source="builder", meta=var_meta).sql == var_sql

    z_sql, z_meta = build_zhistogram("desi_dr1", "zpix", bin=0.1)
    assert "/ 0.1" in z_sql
    assert validate(z_sql, source="builder", meta=z_meta).sql == z_sql

def test_result_store_round_trips_dataframe_without_diskcache():
    store = DatalabResultStore(enable_disk_cache=False)
    frame = pd.DataFrame([{"ra": 1.0, "dec": -1.0}])
    result_id = store.put(frame, {"catalog": "gaia_dr3", "table": "gaia_source"})
    result = store.get(result_id)
    pd.testing.assert_frame_equal(result.dataframe, frame)
    assert result.provenance["catalog"] == "gaia_dr3"


class _FakeDatalabClient:
    def __init__(self):
        self.sql = None

    def query(self, *, sql=None, adql=None, fmt="pandas", **kwargs):
        self.sql = sql
        return DatalabResult.from_dataframe(
            pd.DataFrame([{"row_count": 42}]),
            {"catalog": "gaia_dr3", "table": "gaia_source", "query": sql, "rowcount": 1},
        )


def _make_agent():
    module = _load_agent_module()
    agent = module.QuasarAgent.__new__(module.QuasarAgent)
    agent._tls = threading.local()
    agent.tool_registry = ToolRegistry()
    agent.last_search_results = None
    agent.last_run_result = None
    agent.ads_client = None
    agent.openalex_client = None
    agent._datalab_client_instance = _FakeDatalabClient()
    agent._datalab_result_store_instance = DatalabResultStore(enable_disk_cache=False)
    return agent


def test_agent_registers_and_dispatches_datalab_tools_without_full_rows():
    agent = _make_agent()
    agent._register_tools()

    for name in [
        "datalab_list_catalogs",
        "datalab_describe_table",
        "datalab_cone_count",
        "datalab_select_catalog_rows",
        "datalab_density_aggregate",
        "datalab_q3c_crossmatch",
        "datalab_sql_query",
        "datalab_get_result",
    ]:
        assert agent.tool_registry.get_tool(name) is not None

    payload = json.loads(
        agent._dispatch_tool_call(
            "datalab_cone_count",
            json.dumps({"catalog": "gaia_dr3", "table": "gaia_source", "ra": 10.0, "dec": 0.0, "radius_deg": 0.1}),
        )
    )
    assert payload["success"] is True
    assert payload["reported_count"] == 42
    assert payload["result_id"].startswith("dlr_")
    assert "results" not in payload


def test_agent_raw_sql_requires_expert_ack():
    agent = _make_agent()
    result = agent._datalab_sql_query("SELECT * FROM gaia_dr3.gaia_source LIMIT 1", expert_ack=False, reason="")
    assert result["success"] is False
    assert "expert" in result["error"]


# ── Regression tests for the @cx-guard P0 findings ─────────────────────────────

def test_cone_radius_is_capped():
    from services.datalab_query_builders import build_cone_count, MAX_CONE_RADIUS_DEG
    with pytest.raises(ValueError, match="max cone radius"):
        build_cone_count("gaia_dr3", "gaia_source", ra=10, dec=0, radius_deg=MAX_CONE_RADIUS_DEG + 1)


def test_zero_and_negative_limit_rejected():
    from services.datalab_query_builders import build_cone_select
    for bad in (0, -5):
        with pytest.raises(ValueError, match="limit must be positive"):
            build_cone_select("gaia_dr3", "gaia_source", ra=10, dec=0, radius_deg=0.1, limit=bad)


def test_density_aggregate_requires_region_or_all_sky():
    from services.datalab_query_builders import build_density_aggregate
    with pytest.raises(ValueError, match="requires a cone"):
        build_density_aggregate("nsc_dr2", "object", mode="grid")
    sql, meta = build_density_aggregate("nsc_dr2", "object", mode="grid", ra=185.41, dec=-31.98, radius_deg=1.0)
    assert "q3c_radial_query" in sql and meta["spatial_bound"] is True
    sql2, meta2 = build_density_aggregate("desi_dr1", "zpix", mode="grid", all_sky=True)
    assert "q3c_radial_query" not in sql2 and meta2["warnings"]


def test_policy_rejects_select_into():
    with pytest.raises(DatalabPolicyError):
        validate(
            "SELECT ra INTO TEMP t FROM gaia_dr3.gaia_source WHERE q3c_radial_query(ra,dec,10,0,0.1)",
            source="expert",
        )


def test_policy_accepts_non_radec_crossmatch():
    from services.datalab_query_builders import build_q3c_crossmatch
    sql, meta = build_q3c_crossmatch(
        small_catalog="desi_dr1", small_table="zpix",
        big_catalog="nsc_dr2", big_table="object",
        ra=180, dec=0, radius_deg=0.1,
    )
    assert "mean_fiber_ra" in sql  # small side uses DESI's registered coord columns
    validate(sql, source="builder", meta=meta)  # generic join check must not reject it


def test_get_result_payload_is_size_bounded_and_valid_json():
    import pandas as pd
    agent = _make_agent()
    wide = pd.DataFrame([{f"c{i}": i * 1.0 for i in range(20)} for _ in range(300)])
    # Put into the AGENT's own store (the test fixture uses a non-singleton store).
    rid = agent._get_datalab_result_store().put(wide, {"catalog": "x", "table": "y"})
    payload = agent._datalab_get_result(rid, max_rows=300)
    assert payload["success"] is True and payload["truncated"] is True
    assert payload["returned_rows"] < 300
    assert len(json.dumps(payload["rows"], default=str)) <= 6000
    json.loads(json.dumps(payload, default=str))  # must be valid JSON


def test_datalab_tool_clears_last_run_result():
    agent = _make_agent()
    agent.last_run_result = {"type": "data", "stale": True}
    agent._datalab_list_catalogs()
    assert agent.last_run_result is None


def test_result_store_memory_ttl_expiry(monkeypatch):
    import pandas as pd
    import services.datalab_result_store as rs
    store = rs.DatalabResultStore(enable_disk_cache=False, ttl_seconds=1)
    rid = store.put(pd.DataFrame([{"ra": 1.0}]), {})
    assert store.get(rid).dataframe.iloc[0]["ra"] == 1.0
    base = rs.time.time()
    monkeypatch.setattr(rs.time, "time", lambda: base + 10)
    with pytest.raises(KeyError):
        store.get(rid)
