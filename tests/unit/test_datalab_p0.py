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


pytestmark = pytest.mark.slow


def test_client_sanitizes_trailing_semicolon_and_rejects_final_line_comment():
    assert DatalabClient.sanitize_query("SELECT * FROM gaia_dr3.gaia_source;  ") == "SELECT * FROM gaia_dr3.gaia_source"
    sql = "SELECT *\n-- inner comment is fine\nFROM gaia_dr3.gaia_source"
    assert "-- inner comment" in DatalabClient.sanitize_query(sql)
    with pytest.raises(ValueError, match="must not end"):
        DatalabClient.sanitize_query("SELECT * FROM gaia_dr3.gaia_source\n-- trailing")


def test_client_uses_anonymous_rest_transport_without_astro_datalab(monkeypatch):
    from integrations import datalab_client as dc

    # Hermetic: earlier tests may import api.* which load_dotenv()s the owner's
    # real DATALAB_TOKEN into the process env; this test asserts the DEFAULT
    # (anonymous) transport, so the token must be absent.
    monkeypatch.delenv("DATALAB_TOKEN", raising=False)

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
        encoding = "utf-8"

        def iter_content(self, chunk_size=65536):
            yield self.text.encode("utf-8")

        def close(self):
            pass

    def fake_get(url, headers=None, timeout=None, stream=False):
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
    def boom(url, headers=None, timeout=None, stream=False):
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
    # Nearest-to-center first so a LIMIT-capped crossmatch keeps the cone CENTER
    # (live P9: the storage-order slice excluded Pal 5 itself).
    assert "ORDER BY q3c_dist(g.ra, g.dec, 229.022, -0.112)" in sql
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
    assert "smash_dr1.source" in var_sql
    assert validate(var_sql, source="builder", meta=var_meta).sql == var_sql

    # catalog is now parameterized (Data Lab parity 2026-07)
    var2_sql, _ = build_variable_star_select(catalog="smash_dr2", source_id="abc123")
    assert "smash_dr2.source" in var2_sql

    z_sql, z_meta = build_zhistogram("desi_dr1", "zpix", bin=0.1)
    assert "/ 0.1" in z_sql
    assert validate(z_sql, source="builder", meta=z_meta).sql == z_sql

def test_variability_rank_builder_emits_governed_aggregate():
    from services.datalab_query_builders import build_variability_rank

    sql, meta = build_variability_rank(
        "smash_dr1", ra=15.0, dec=-72.0, radius_deg=0.3, band="g", min_epochs=12, limit=50
    )
    assert "smash_dr1.source" in sql
    assert "q3c_radial_query(ra, dec, 15, -72, 0.3)" in sql
    assert "STDDEV(cmag)" in sql and "GROUP BY id" in sql
    # Distance from the cone center rides along so a target-position query can
    # pick the NEAREST candidate, not the most variable one (live P14). The
    # mean position is the wrap-safe circular mean, not AVG(ra) — a cone
    # straddling RA=0/360 averaged 359.999° and 0.001° to ~180°
    # (variability-avg-ra-wrap).
    assert "AVG(ra)" not in sql
    assert "atan2(AVG(sin(radians(ra))), AVG(cos(radians(ra))))" in sql
    assert "* 3600.0 AS dist_arcsec" in sql
    assert "AVG(dec) AS dec" in sql
    assert "HAVING COUNT(*) >= 12" in sql
    assert "filter = 'g'" in sql
    assert "ORDER BY var_snr DESC NULLS LAST" in sql
    assert meta["builder"] == "variability_rank"
    assert meta["aggregate"] is True and meta["spatial_bound"] is True
    assert meta["grouped_by_filter"] is False
    # the governor accepts it as a builder aggregate without touching the LIMIT
    assert validate(sql, source="builder", meta=meta).sql == sql

    all_bands_sql, all_bands_meta = build_variability_rank(
        "smash_dr1", ra=15.0, dec=-72.0, radius_deg=0.3, min_epochs=12, limit=50
    )
    assert "SELECT id,\n       filter," in all_bands_sql
    assert "GROUP BY id, filter" in all_bands_sql
    assert "filter =" not in all_bands_sql
    assert all_bands_meta["grouped_by_filter"] is True
    assert validate(all_bands_sql, source="builder", meta=all_bands_meta).sql == all_bands_sql


def test_variability_rank_builder_rejects_bad_inputs():
    from services.datalab_query_builders import build_variability_rank

    with pytest.raises(ValueError, match="multi-epoch"):
        build_variability_rank("gaia_dr3", ra=15.0, dec=-72.0, radius_deg=0.3)
    with pytest.raises(ValueError, match="min_epochs"):
        build_variability_rank("smash_dr1", ra=15.0, dec=-72.0, radius_deg=0.3, min_epochs=1)
    with pytest.raises(ValueError, match="Invalid band"):
        build_variability_rank("smash_dr1", ra=15.0, dec=-72.0, radius_deg=0.3, band="g'; DROP")


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
        "datalab_variable_candidates",
        "datalab_star_lightcurve",
    ]:
        assert agent.tool_registry.get_tool(name) is not None

    # Time-domain chain (Data Lab parity 2026-07): candidates dispatches the
    # governed aggregate through the same fake client path.
    var_payload = json.loads(
        agent._dispatch_tool_call(
            "datalab_variable_candidates",
            json.dumps({"catalog": "smash_dr1", "ra": 15.0, "dec": -72.0, "radius_deg": 0.2}),
        )
    )
    assert var_payload["success"] is True
    assert var_payload["result_id"].startswith("dlr_")
    assert "STDDEV" in agent._datalab_client_instance.sql

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
    # datalab_sql_query is now capability-backed (capabilities/datalab.py); dispatch
    # through the registry rather than the deleted inline method. (docs/v2 P1)
    agent = _make_agent()
    agent._register_tools()
    result = json.loads(
        agent._dispatch_tool_call(
            "datalab_sql_query",
            json.dumps({"sql": "SELECT * FROM gaia_dr3.gaia_source LIMIT 1", "expert_ack": False, "reason": ""}),
        )
    )
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
    # datalab_get_result is now capability-backed; dispatch through the registry.
    import pandas as pd
    agent = _make_agent()
    agent._register_tools()
    wide = pd.DataFrame([{f"c{i}": i * 1.0 for i in range(20)} for _ in range(300)])
    # Put into the AGENT's own store (the test fixture uses a non-singleton store).
    rid = agent._get_datalab_result_store().put(wide, {"catalog": "x", "table": "y"})
    payload = json.loads(
        agent._dispatch_tool_call("datalab_get_result", json.dumps({"result_id": rid, "max_rows": 300}))
    )
    assert payload["success"] is True and payload["truncated"] is True
    assert payload["returned_rows"] < 300
    assert len(json.dumps(payload["rows"], default=str)) <= 6000
    json.loads(json.dumps(payload, default=str))  # must be valid JSON


def test_datalab_tool_clears_last_run_result():
    # The capability's context provider clears last_run_result at the adapter
    # boundary (Data Lab tools emit summaries, not data cards). (docs/v2 P1)
    agent = _make_agent()
    agent._register_tools()
    agent.last_run_result = {"type": "data", "stale": True}
    agent._dispatch_tool_call("datalab_list_catalogs", json.dumps({}))
    assert agent.last_run_result is None


def test_datalab_image_wrapper_emits_card_and_strips_base64(monkeypatch):
    # The image wrapper (_datalab_image_tool_fn) applies the SSE/UI transport at
    # the adapter boundary: sets a displayable image card on last_run_result and
    # strips the heavy base64 out of the LLM-facing dict. (docs/v2 P1)
    import services.datalab_analysis as da
    monkeypatch.setattr(
        da, "catalog_scatter",
        lambda *a, **k: {"success": True, "image_base64": "BIGB64", "path": "/plots/s.png", "n": 5},
    )
    agent = _make_agent()
    agent._register_tools()
    agent.last_run_result = {"stale": True}
    out = json.loads(
        agent._dispatch_tool_call(
            "datalab_catalog_scatter",
            json.dumps({"result_id": "rid", "x_expr": "g - r", "y_expr": "g", "title": "My Plot"}),
        )
    )
    # LLM-facing output: base64 gone, image_attached added, other fields kept.
    assert out["success"] is True and out.get("image_attached") is True
    assert "image_base64" not in out
    assert out["n"] == 5
    # UI card set on last_run_result.
    assert agent.last_run_result["type"] == "image"
    assert agent.last_run_result["image_url"] == "/plots/s.png"
    assert agent.last_run_result["caption"] == "My Plot"


def test_datalab_image_cutout_wrapper_end_to_end():
    # Full SIA path: image service injected via CallContext, coordinate resolution
    # via _datalab_coordinates (ra/dec), image-card transport + base64 strip +
    # no _caption leak. (docs/v2 P1)
    agent = _make_agent()
    agent._register_tools()

    class _Svc:
        def cutout(self, ra, dec, fov, **k):
            return {"success": True, "image_base64": "B64", "path": "/plots/x.png"}

    agent._datalab_image_service_instance = _Svc()
    agent.last_run_result = {"stale": True}
    out = json.loads(
        agent._dispatch_tool_call(
            "datalab_image_cutout",
            json.dumps({"fov_deg": 0.1, "ra": 10.0, "dec": 20.0, "band": "g"}),
        )
    )
    assert out["success"] is True and out.get("image_attached") is True
    assert "image_base64" not in out and "_caption" not in out  # stripped + not leaked
    assert agent.last_run_result["type"] == "image"
    assert agent.last_run_result["image_url"] == "/plots/x.png"
    assert agent.last_run_result["caption"] == "Data Lab g-band cutout: RA=10.00000, Dec=20.00000"


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


def test_rectangular_region_wraps_through_ra_zero():
    """ra-wrap-rect-footprint-unsupported: ra_min > ra_max is a box wrapping
    through RA=0/360 and must query the two OR'd sub-boxes, not be rejected
    (or silently swapped into the 356° complement)."""
    from services.datalab_query_builders import build_rectangular_region_select

    sql, meta = build_rectangular_region_select(
        "gaia_dr3", "gaia_source",
        ra_min=358.0, ra_max=2.0, dec_min=-75.0, dec_max=-70.0, limit=100,
    )
    assert sql.count("q3c_poly_query") == 2
    assert "ARRAY[358, 360, 360, 358]" in sql
    assert "ARRAY[0, 2, 2, 0]" in sql
    assert any("wrap" in w.lower() for w in meta["warnings"])
    # The governor accepts the wrapped builder SQL.
    assert validate(sql, source="builder", meta=meta).sql == sql
    # A degenerate zero-width box is still rejected.
    with pytest.raises(ValueError, match="ra_min != ra_max"):
        build_rectangular_region_select(
            "gaia_dr3", "gaia_source", ra_min=10.0, ra_max=10.0, dec_min=0.0, dec_max=1.0
        )
    # Non-wrapping boxes keep the single-polygon form.
    sql2, _meta2 = build_rectangular_region_select(
        "gaia_dr3", "gaia_source", ra_min=10.0, ra_max=12.0, dec_min=0.0, dec_max=1.0
    )
    assert sql2.count("q3c_poly_query") == 1


def test_vhs_mag_template_resolves_apermag_columns():
    """vhs-mag-template-missing: without a template, VHS CMD/CCDs emitted
    nonexistent jmag/hmag/kmag and 400'd only at the server; band k must map
    to the VSA 'ks' spelling, and non-IR bands must fail locally."""
    from services import datalab_registry as reg

    assert reg.mag_column("vhs_dr5", "vhs_cat_v3", "j") == "japermag3"
    assert reg.mag_column("vhs_dr5", "vhs_cat_v3", "h") == "hapermag3"
    assert reg.mag_column("vhs_dr5", "vhs_cat_v3", "k") == "ksapermag3"
    assert reg.mag_column("vhs_dr5", "vhs_cat_v3", "ks") == "ksapermag3"
    with pytest.raises(ValueError, match="valid bands"):
        reg.mag_column("vhs_dr5", "vhs_cat_v3", "g")
