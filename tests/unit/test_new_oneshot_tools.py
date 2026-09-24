"""The one-shot ALMA (capabilities/alma_tools.py) and Data Lab
(capabilities/datalab_tools.py) tools — schemas, SQL/ADQL shapes, status
contract, and registration/budget declarations. Network is faked throughout.
"""
from __future__ import annotations

import json
import math
import re
from typing import Any, Dict, List

import pandas as pd
import pytest

from capabilities import alma_tools, datalab_tools
from capabilities.base import CallContext


# ── shared fakes ─────────────────────────────────────────────────────────


class _Tap:
    def __init__(self, answer):
        self.answer = answer
        self.queries: List[str] = []

    def search(self, q, maxrec=None):
        self.queries.append(q)
        df = self.answer(q)

        class _Res:
            quasar_tap_url = "https://almascience.nrao.edu/tap"
            query_status = "OK"

            def to_table(self_inner):
                return self_inner

            def to_pandas(self_inner):
                return df.copy()

        return _Res()


class _State:
    def __init__(self):
        self.last_run_result = None
        self.last_search_results = None
        self.cards: List[Dict[str, Any]] = []


def _alma_ctx(answer, **extra):
    state = _State()
    tap = _Tap(answer)
    svc = type("_Svc", (), {})()
    svc.alminer_client = type("_Alminer", (), {"_get_tap_service": lambda self: tap, "_standardize_columns": lambda self, df: df})()
    services = {
        "search_service": svc,
        "alma_tap_provenance": {"query": None, "url": None},
        "resolve_target": lambda name: {"success": True, "ra_deg": 204.2538, "dec_deg": -29.8658},
        "set_last_search_results": lambda v: setattr(state, "last_search_results", v),
        "set_last_run_result": lambda v: setattr(state, "last_run_result", v),
        "get_last_run_result": lambda: state.last_run_result,
        "console_log": lambda *a, **k: None,
    }
    services.update(extra)
    return CallContext(services=services), state, tap


def _run(cap, ctx, **kwargs):
    return cap.run(cap.InputModel(**kwargs), ctx).to_native()


# ── registration contract ────────────────────────────────────────────────


def test_every_new_tool_is_declared_in_the_budget_hierarchy_and_has_a_schema():
    from services import tool_budgets as tb
    from tests.unit.test_tool_budget_hierarchy import LOCAL_TOOLS

    for cap in alma_tools.CAPABILITIES + datalab_tools.CAPABILITIES:
        assert cap.name in tb.INNER_BUDGETS or cap.name in LOCAL_TOOLS, cap.name
        schema = cap.parameters_schema()
        assert schema.get("type") == "object" and "properties" in schema, cap.name
        assert cap.description and len(cap.description) > 80, cap.name
    assert tb.check_hierarchy([c.name for c in alma_tools.CAPABILITIES + datalab_tools.CAPABILITIES]) == []


def test_new_tools_are_part_of_the_datalab_capability_list():
    from capabilities.datalab import CAPABILITIES

    names = {c.name for c in CAPABILITIES}
    assert {"datalab_healpix_density_map", "datalab_stream_selection", "datalab_selection_diagram",
            "datalab_target_class_summary", "datalab_satellite_search"} <= names


# ── alma_project_census ──────────────────────────────────────────────────


def test_project_census_solar_dispatches_to_the_server_side_aggregate():
    def answer(q):
        if "GROUP BY proposal_id" in q:
            return pd.DataFrame([{"proposal_id": "2023.1.01040.S", "n_mous": 7, "n_eb": 7, "n_rows": 40, "target_name": "Sun",
                                  "pi_name": "White, Stephen", "obs_title": "Solar", "band_list": "3",
                                  "obs_release_date_min": "2024", "obs_release_date_max": "2025"}])
        return pd.DataFrame([{"proposal_id": "2023.1.01040.S", "target_name": "Sun", "band_list": "3",
                              "antenna_arrays": "A001:DV01 J501:CM01 N601:PM02", "schedblock_name": "x", "data_rights": "Public"}])

    ctx, state, tap = _alma_ctx(answer)
    out = _run(alma_tools.AlmaProjectCensus(), ctx, cycle=10, constraint={"type": "solar"}, public_only=True)
    assert out["success"] and out["status"] == "ok" and out["n_projects"] == 1
    assert "1 project(s)" in out["headline"] and out["complete"] is True
    assert all("data_rights = 'Public'" in q for q in tap.queries[:1])
    assert "science_observation = 'T'" in tap.queries[0]
    assert out["provenance"]["computation"] == "server-side" and len(out["provenance"]["adql"]) == 2
    assert state.last_run_result["type"] == "data" and state.last_run_result["tool_name"] == "alma_project_census"


def test_project_census_validates_inputs_and_types():
    ctx, _, _ = _alma_ctx(lambda q: pd.DataFrame())
    assert _run(alma_tools.AlmaProjectCensus(), ctx, constraint={"type": "solar"})["success"] is False
    assert _run(alma_tools.AlmaProjectCensus(), ctx, constraint={"type": "lines"})["success"] is False
    assert _run(alma_tools.AlmaProjectCensus(), ctx, constraint={"type": "redshift"})["success"] is False
    assert "unknown constraint.type" in _run(alma_tools.AlmaProjectCensus(), ctx, constraint={"type": "weird"})["error"]


def test_project_census_reports_infrastructure_failure_when_the_server_dies():
    def answer(q):
        raise RuntimeError("Read timed out")

    ctx, _, _ = _alma_ctx(answer)
    out = _run(alma_tools.AlmaProjectCensus(), ctx, cycle=9, constraint={"type": "arrays", "arrays": ["12m", "7m", "TP"]})
    assert out["success"] is False and out["status"] == "infrastructure_failure" and out["infrastructure_failure"] is True


# ── alma_source_summary ──────────────────────────────────────────────────


def test_source_summary_puts_filters_in_the_adql_and_summarises_per_mous():
    rows = []
    for uid, band, res, bw, freq in (("uid://A001/X1/X1", "7", 0.35, 1.875e9, 343.0), ("uid://A001/X1/X1", "7", 0.35, 1.875e9, 345.0),
                                     ("uid://A001/X2/X2", "7", 0.8, 2.0e9, 340.0)):
        rows.append({"member_ous_uid": uid, "proposal_id": "2019.1.00001.S", "target_name": "HH212", "band_list": band,
                     "spatial_resolution": res, "bandwidth": bw, "frequency": freq, "frequency_support": f"[{freq-1:.2f}..{freq+1:.2f}GHz,976.56kHz,XX YY]",
                     "cont_sensitivity_bandwidth": 0.05, "t_exptime": 3600.0, "obs_release_date": "2021-01-01T00:00:00", "data_rights": "Public",
                     "science_observation": "T", "qa2_passed": "T", "pi_name": "PI", "s_ra": 84.0, "s_dec": -1.0})
    ctx, state, tap = _alma_ctx(lambda q: pd.DataFrame(rows))
    out = _run(alma_tools.AlmaSourceSummary(), ctx, target="HH212", bands=[7], max_resolution_arcsec=1.0, public_only=True)
    assert out["success"] and out["status"] == "ok" and out["n_mous"] == 2
    q = tap.queries[0]
    assert "band_list = '7'" in q and "spatial_resolution <= 1" in q and "data_rights = 'Public'" in q and "science_observation = 'T'" in q
    best = out["results"][0]
    assert best["member_ous_uid"] == "uid://A001/X1/X1" and best["spatial_resolution_arcsec"] == 0.35
    assert best["aggregate_bandwidth_mhz"] == 4000.0  # two 2-GHz SPWs
    assert out["best_resolution_arcsec"] == 0.35 and out["filters_in_adql"]["band"] == [7]
    assert state.last_run_result["tool_name"] == "alma_source_summary"


def test_source_summary_empty_is_a_coverage_gap_not_a_failure():
    ctx, _, _ = _alma_ctx(lambda q: pd.DataFrame())
    out = _run(alma_tools.AlmaSourceSummary(), ctx, ra=10.0, dec=-20.0, bands=[10])
    assert out["success"] is True and out["status"] == "coverage_gap" and out["n_mous"] == 0


# ── alma_public_band_status ──────────────────────────────────────────────


def test_public_band_status_counts_tokens_and_honours_as_of():
    def answer(q):
        assert "data_rights = 'Public'" in q and "obs_release_date < '2026-03-26'" in q
        return pd.DataFrame([{"band_list": "3", "n_mous": 100, "n_projects": 40, "n_rows": 1000},
                             {"band_list": "6", "n_mous": 300, "n_projects": 90, "n_rows": 5000},
                             {"band_list": "5 10", "n_mous": 2, "n_projects": 1, "n_rows": 10}])

    ctx, _, tap = _alma_ctx(answer)
    out = _run(alma_tools.AlmaPublicBandStatus(), ctx, as_of="2026-03-25")
    assert out["success"] and out["bands_with_public_data"] == [3, 5, 6, 10] and out["n_bands_with_public_data"] == 4
    assert out["per_band"][5]["n_mous"] == 2 and out["band_to_band_combinations"] == ["5 10"]
    assert out["as_of"] == "2026-03-25" and "GROUP BY band_list" in out["provenance"]["adql"][0]
    assert _run(alma_tools.AlmaPublicBandStatus(), ctx, as_of="March 25")["success"] is False


# ── alma_archive_link ────────────────────────────────────────────────────


def test_archive_link_uses_documented_parameters_and_verifies(monkeypatch):
    import capabilities.alma_science_helpers as helpers

    probed = []
    monkeypatch.setattr(helpers, "head_verify", lambda url, timeout=6.0: (probed.append(url) or True, "HTTP 200"))
    ctx, _, _ = _alma_ctx(lambda q: pd.DataFrame())
    out = _run(alma_tools.AlmaArchiveLink(), ctx, target="M83", band=6)
    assert out["success"] and out["asa_url"].startswith("https://almascience.nrao.edu/aq/?")
    assert "sourceNameResolver=M83" in out["asa_url"] and "bandList=6" in out["asa_url"] and "target=" not in out["asa_url"]
    assert out["asa_url_verified"] is True and out["tap_sync_url"].startswith("https://almascience.nrao.edu/tap/sync?")
    assert "band_list+%3D+%276%27" in out["tap_sync_url"] or "band_list = '6'" in out["tap_adql"]
    assert len(probed) == 2 and out["links"][0] == out["asa_url"]


# ── alma_bibliography ────────────────────────────────────────────────────


def test_bibliography_queries_the_alma_bibgroup_and_extracts_project_codes():
    class _Ads:
        def __init__(self):
            self.calls = []

        def search_papers(self, query, max_results=10, sort="date desc", fields=None, filters=None, start=0):
            self.calls.append((query, max_results, sort, filters))
            return [{"bibcode": "2026ApJ...900..1X", "title": ["Outflows"], "author": ["Doe, J.", "Roe, R."], "year": "2026", "pub": "ApJ",
                     "abstract": "We use ALMA data (project 2019.1.00123.S) to study outflows. Second sentence. Third.", "doi": ["10.1/x"]}]

    ads = _Ads()
    ctx, state, _ = _alma_ctx(lambda q: pd.DataFrame(), ads_client=ads)
    out = _run(alma_tools.AlmaBibliography(), ctx, topic="protostellar outflows", n=10, since=2024)
    assert out["success"] and out["n_returned"] == 1
    q, n, sort, filters = ads.calls[0]
    assert q.startswith("bibgroup:ALMA") and n == 10 and sort == "date desc" and "property:refereed" in filters and "year:2024-" in filters
    row = out["results"][0]
    assert row["first_author"] == "Doe, J. et al." and row["alma_project_codes"] == "2019.1.00123.S" and row["summary"].endswith("Second sentence.")
    assert state.last_run_result["type"] == "papers"


def test_bibliography_without_ads_is_an_infrastructure_failure():
    ctx, _, _ = _alma_ctx(lambda q: pd.DataFrame())
    out = _run(alma_tools.AlmaBibliography(), ctx, topic="x")
    assert out["success"] is False and out["status"] == "infrastructure_failure"


# ── cross_archive_match ──────────────────────────────────────────────────


def test_cross_archive_match_forces_point_cones_and_adds_status_columns(monkeypatch):
    import capabilities.alma as alma

    seen = {}

    def fake_impl(ctx, **kw):
        seen.update(kw)
        return {"success": True, "partial": True, "sources_tested": 12, "matched_sources": 12, "archives": ["ALMA", "JWST"],
                "archive_errors": ["ALMA not queried for 4 source(s) -- tool budget exhausted: a, b, c, d"],
                "dead_hosts": [], "budget_exhausted": True, "footprint_mode": "point_only",
                "results": [{"source_name": "S1", "alma_observations": 2, "jwst_observations": 5, "mast_observations": 5},
                            {"source_name": "S2", "alma_observations": 0, "jwst_observations": 3, "mast_observations": 3}],
                "note": "n"}

    monkeypatch.setattr(alma, "_match_cross_archive_sources_impl", fake_impl)
    ctx, _, _ = _alma_ctx(lambda q: pd.DataFrame())
    out = _run(alma_tools.CrossArchiveMatch(), ctx, catalog_name="perseus_protostars", archives=["ALMA", "JWST"], require_all_archives=False)
    assert seen["alma_mode"] == "point_only"
    assert out["status"] == "partial" and out["alma_status_unknown_for"] == 4
    assert out["phase_status"]["ALMA"] == "timeout" and out["phase_status"]["JWST"] == "ok"
    assert out["results"][0]["alma_status"] == "ok" and out["results"][1]["alma_status"] == "timeout"
    assert out["results"][1]["jwst_status"] == "ok"
    assert "ALMA status unknown for 4 source(s)" in out["headline"] and "do not state that they lack ALMA data" in out["note"]


# ── archive_overlay ──────────────────────────────────────────────────────


def test_archive_overlay_reports_unknown_when_phases_do_not_complete(monkeypatch):
    import capabilities.viz as viz

    monkeypatch.setattr(viz, "overlay_region_coordinates", lambda region, ra_deg=None, dec_deg=None: (53.1625, -27.7914, "HUDF"))
    monkeypatch.setattr(alma_tools.ArchiveOverlay, "_covering_hips", classmethod(lambda cls, ra, dec, m: "CDS/P/JWST/F200W"))

    class _Mast:
        def search_by_position(self, *a, **k):
            df = pd.DataFrame()
            df.attrs["error"] = "MAST query did not answer within 30 s"
            return df

    def tap_answer(q):
        raise RuntimeError("Read timed out")

    ctx, _, _ = _alma_ctx(tap_answer, mast_client=_Mast(), datalink_client=object())
    out = _run(alma_tools.ArchiveOverlay(), ctx, field="Hubble Ultra Deep Field")
    assert out["success"] is False and out["status"] == "coverage_gap"
    assert out["missing"] == ["ALMA contour product"] and out["alma_status"].startswith("unknown")
    assert out["base_source"]["survey"] == "CDS/P/JWST/F200W"
    assert any("coverage UNKNOWN" in g for g in out["gaps"]), out["gaps"]
    assert not any("no JWST observations" in g or "lists no JWST" in g for g in out["gaps"]), "a MAST timeout is not 'no observations'"
    assert "UNKNOWN" in out["note"]


def test_archive_overlay_alma_lookup_uses_a_point_cone_without_order_by(monkeypatch):
    queries = []

    def tap_answer(q):
        queries.append(q)
        return pd.DataFrame([{"member_ous_uid": "uid://A001/X1/X1", "proposal_id": "2016.1.00324.L", "target_name": "HUDF",
                              "band_list": "6", "spatial_resolution": 1.5, "dataproduct_type": "image", "data_rights": "Public"}])

    class _DL:
        def __init__(self):
            self.calls = []

        def list_files(self, mous_uid, pattern=None):
            self.calls.append((mous_uid, pattern))
            return {"success": True, "files": [
                {"filename": "member.uid___A001_X1_X1.J0336-2644_chk.spw11.mfs.I.pbcor.fits", "size_mb": 0.4, "access_url": "https://almascience.org/cal.fits"},
                {"filename": "member.uid___A001_X1_X1.HUDF_sci.spw25.mfs.I.mask.fits", "size_mb": 1, "access_url": "https://almascience.org/mask.fits"},
                {"filename": "member.uid___A001_X1_X1.HUDF_sci.spw25.cont.I.pbcor.fits", "size_mb": 40, "access_url": "https://almascience.org/x.fits"},
                {"filename": "2016.1.00324.L_uid___A001_X1_X1_001_of_001.tar", "size_mb": 900, "access_url": "https://almascience.org/t.tar"}]}

    dl = _DL()
    ctx, _, _ = _alma_ctx(tap_answer, datalink_client=dl)
    out = alma_tools.ArchiveOverlay._find_alma_contour(ctx, 53.1625, -27.7914, 3.0, 150.0, "HUDF")
    assert out["product"]["access_url"].endswith("x.fits") and out["product"]["proposal_id"] == "2016.1.00324.L"
    assert "ORDER BY" not in queries[0] and "CONTAINS(POINT" in queries[0] and "science_observation = 'T'" in queries[0]
    assert dl.calls == [("uid://A001/X1/X1", None)]


# ── alma_reference ───────────────────────────────────────────────────────


def test_alma_reference_returns_tables_and_cited_passages():
    class _Doc:
        def __init__(self, text, meta):
            self.page_content, self.metadata = text, meta

    class _Rag:
        def search_with_diagnostics(self, query, k=5, **kw):
            return [_Doc("Bandwidth Switching (BWSW) ... 937.5 MHz", {"source_file": "alma-proposers-guide-cycle13.pdf", "page": 51, "_semantic_score": 0.9})], {}

    ctx, _, _ = _alma_ctx(lambda q: pd.DataFrame(), rag_service=_Rag())
    out = _run(alma_tools.AlmaReference(), ctx, topic="bandwidth switching")
    assert out["success"] and out["passages"][0]["cite_as"] == "[Source: alma-proposers-guide-cycle13.pdf, Page 51]"
    tables = _run(alma_tools.AlmaReference(), ctx, topic="bands")
    assert tables["success"] and tables["tables"]["topic"] == "bands" and tables["tables"]["receiver_band_count"] == 10


# ── Data Lab tools: SQL shapes ───────────────────────────────────────────


def test_stream_selection_sql_carries_pm_window_and_cmd_mask_and_passes_the_policy():
    from services import datalab_sql_policy

    sql, meta = datalab_tools.StreamSelection.build_sql(
        229.022, -0.112, 5.0, pm={"pmra_min": -3.7, "pmra_max": -1.7, "pmdec_min": -3.6, "pmdec_max": -1.6},
        cmd_mask={"color_min": 0.1, "color_max": 0.9, "g_min": 16, "g_max": 23}, match_arcsec=1.0, small_limit=50000, limit=5000,
    )
    assert "AND pmra BETWEEN -3.7000 AND -1.7000" in sql and "AND pmdec BETWEEN -3.6000 AND -1.6000" in sql
    assert "WHERE (big_gmag - big_rmag) BETWEEN 0.100 AND 0.900" in sql and "big_gmag BETWEEN 16.000 AND 23.000" in sql
    assert "big_class_star > 0.5" in sql and sql.count("AS MATERIALIZED") == 2 and "q3c_join" in sql
    # JOIN-FIRST (live 2026-09-23 L09: > 90 s -> 3-4 s): the PM window sits INSIDE
    # the Gaia CTE, the index join in its OWN materialized CTE, the CMD mask on
    # the joined set -- never in the join's WHERE.
    assert sql.index("pmra BETWEEN") < sql.index("ORDER BY q3c_dist(ra, dec") < sql.index("q3c_join") \
        < sql.index("SELECT * FROM m") < sql.index("big_gmag - big_rmag")
    join_cte = sql[sql.index("m AS MATERIALIZED"):sql.index("SELECT * FROM m")]
    assert "gmag" not in join_cte.split("q3c_join")[1], "no photometric predicate beside the index join"
    validated = datalab_sql_policy.validate(sql, source="builder", meta=meta)
    assert "pmra BETWEEN" in validated.sql and meta["stream_selection"]["cmd_mask"]["g_max"] == 23


def test_key_select_builder_bounds_by_an_indexed_key_without_a_cone():
    from services import datalab_query_builders as b
    from services import datalab_sql_policy

    sql, meta = b.build_key_select("smash_dr1", "object", key_column="fieldid", key_value=169, columns=["id", "ra", "dec", "gmag", "rmag"], limit=1000)
    assert "WHERE fieldid = 169" in sql and "q3c_radial_query" not in sql and sql.strip().endswith("LIMIT 1000")
    validated = datalab_sql_policy.validate(sql, source="builder", meta=meta)
    assert "fieldid = 169" in validated.sql
    with pytest.raises(ValueError):
        b.build_key_select("smash_dr1", "object", key_column="gmag", key_value=20)


def test_select_catalog_rows_accepts_the_key_or_requires_the_cone():
    from capabilities.datalab import SelectCatalogRows

    cap = SelectCatalogRows()
    inp = cap.InputModel(catalog="smash_dr1", table="object", key_column="fieldid", key_value=169)
    assert inp.ra is None and inp.key_value == 169
    bad = cap.run(cap.InputModel(catalog="smash_dr1", table="object"), CallContext()).to_native()
    assert bad["success"] is False and "key_column" in bad["error"]


def test_region_presets_prefer_validated_fields():
    ra, dec, r, label, key = datalab_tools._region(None, None, fallback="lmc")
    assert key == "lmc" and abs(ra - 80.894) < 0.01
    ra, dec, r, label, key = datalab_tools._region("galactic centre", None, fallback="delve_south")
    assert key == "delve_south", "an unknown/unsafe region name falls back to the validated field, never the Galactic Centre"
    ra, dec, r, label, key = datalab_tools._region({"ra": 10.0, "dec": -20.0, "radius": 3.0}, None, fallback="lmc")
    assert key is None and (ra, dec, r) == (10.0, -20.0, 3.0)
    assert datalab_tools.REGION_PRESETS["great_wall"] == {"ra_min": 150.0, "ra_max": 220.0, "dec_min": 0.0, "dec_max": 5.0, "z_max": 0.1,
                                                          "label": "SDSS Great Wall (RA 150-220, Dec 0-5, z <= 0.1)"}


def test_target_class_summary_decodes_the_bitmask_and_validates_expressions():
    from services import datalab_registry as reg

    cap = datalab_tools.TargetClassSummary()
    info = reg.describe_table("desi_dr1", "zpix")
    pred, text = cap._bitmask(info, "LRG", None)
    assert pred == "(desi_target & 1) != 0" and "bit 0" in text
    pred2, text2 = cap._bitmask(info, None, "(desi_target & 4) != 0")
    assert pred2 == "(desi_target & 4) != 0" and "[2]" in text2
    with pytest.raises(ValueError):
        cap._bitmask(info, None, "desi_target > 0; DROP TABLE x")
    with pytest.raises(ValueError):
        cap._bitmask(info, "NOPE", None)


def test_quality_presets_encode_the_gaia_astrometric_cuts():
    cuts = datalab_tools.QUALITY_PRESETS["gaia_astrometric"]
    assert {"column": "parallax_over_error", "op": ">", "value": 5} in cuts and {"column": "ruwe", "op": "<", "value": 1.4} in cuts


def test_satellite_search_background_and_significance():
    frame = pd.DataFrame({"source_count": [10, 11, 9, 12, 10, 40, 10, 11]})
    med, sigma = datalab_tools.SatelliteSearch._background(frame)
    assert med == 10.5 and sigma >= 1.0
    assert datalab_tools.SatelliteSearch._peak_count({"label": "n=40"}) == 40


def test_selection_diagram_puts_the_total_proper_motion_floor_in_the_sql(monkeypatch):
    from services import datalab_orchestration as orch

    captured = {}

    class _Stop(Exception):
        pass

    def fake_run(sql, meta, **kw):
        captured["sql"] = sql
        raise _Stop()

    monkeypatch.setattr(orch, "_run_builder_sql", fake_run)
    with pytest.raises(_Stop):
        orch.color_magnitude_diagram(
            "gaia_dr3", "gaia_source", 60.0, -50.0, 5.0,
            x_expr="bp_rp", y_expr="phot_g_mean_mag + 5*log10(parallax) - 10",
            value_cuts=datalab_tools.QUALITY_PRESETS["gaia_astrometric"], pm_total_min=100.0,
            client=object(), result_store=object(), plotting_service=object(),
        )
    sql = captured["sql"]
    assert "(pmra*pmra + pmdec*pmdec) > 10000" in sql
    assert "parallax_over_error > 5" in sql and "ruwe < 1.4" in sql
    with pytest.raises(ValueError):
        orch.color_magnitude_diagram("gaia_dr3", "gaia_source", 60.0, -50.0, 5.0, x_expr="bp_rp", y_expr="phot_g_mean_mag",
                                     pm_total_min=float("nan"), client=object(), result_store=object(), plotting_service=object())



def test_cross_archive_match_status_uses_the_matcher_columns_with_require_all(monkeypatch):
    """Live 2026-09-23 D18: the status column read invented keys and labelled
    real JWST matches 'no_match'; the model then denied a correct answer."""
    import capabilities.alma as alma

    monkeypatch.setattr(alma, "_match_cross_archive_sources_impl", lambda ctx, **kw: {
        "success": True, "sources_tested": 12, "matched_sources": 1, "archives": ["ALMA", "JWST"], "archive_errors": [],
        "dead_hosts": [], "budget_exhausted": False,
        "results": [{"source_name": "SVS 13", "alma_observations": 252, "jwst_observations": 30, "mast_observations": 30}],
    })
    ctx, _, _ = _alma_ctx(lambda q: pd.DataFrame())
    out = _run(alma_tools.CrossArchiveMatch(), ctx, archives=["ALMA", "JWST"], require_all_archives=True)
    row = out["results"][0]
    assert row["alma_status"] == "ok" and row["jwst_status"] == "ok"
    assert row["jwst_n_observations"] == 30 and out["status"] == "ok"


def test_point_only_mode_queries_every_source_concurrently_without_the_footprint_query():
    import threading
    import time as _t

    from capabilities.alma import _match_cross_archive_sources_impl
    from services.tool_budgets import begin_tool_deadline, end_tool_deadline

    sources = [{"source_name": f"S{i}", "ra": 50.0 + i * 0.1, "dec": 30.0} for i in range(9)]
    active = {"now": 0, "max": 0}
    lock = threading.Lock()

    def answer(q):
        assert "INTERSECTS" not in q, "point_only must never send the N-way footprint query"
        with lock:
            active["now"] += 1
            active["max"] = max(active["max"], active["now"])
        _t.sleep(0.05)
        with lock:
            active["now"] -= 1
        ra = float(re.search(r"CIRCLE\('ICRS', ([\d.]+),", q).group(1))
        return pd.DataFrame([{"target_name": "t", "proposal_id": "2019.1.00001.S", "s_ra": ra, "s_dec": 30.0, "member_ous_uid": f"uid://{ra}"}])

    class _Mast:
        def search_by_position(self, *a, **k):
            return pd.DataFrame()

    ctx, _, tap = _alma_ctx(answer, mast_client=_Mast())
    begin_tool_deadline("cross_archive_match", 60.0)
    try:
        out = _match_cross_archive_sources_impl(ctx, catalog_name="inline", sources=sources, archives=["ALMA"],
                                                radius_arcsec=5.0, max_sources=20, alma_mode="point_only")
    finally:
        end_tool_deadline()
    assert out["success"] and out["matched_sources"] == 9 and out["footprint_mode"] == "point_only"
    assert len(tap.queries) == 9 and active["max"] > 1, "cones must run concurrently"
    assert not any("not queried" in e for e in out["archive_errors"])



def test_satellite_aperture_peaks_rank_a_compact_overdensity_first_and_ignore_edges():
    import numpy as np

    rng = np.random.default_rng(3)
    rows = []
    step = 0.05
    for i in range(-24, 25):
        for j in range(-24, 25):
            if (i * step) ** 2 + (j * step) ** 2 > 1.2 ** 2:
                continue  # circular field: nothing outside the footprint
            rows.append({"ra_bin": 185.0 + i * step / 0.85, "dec_bin": -31.5 + j * step, "source_count": int(rng.poisson(70))})
    df = pd.DataFrame(rows)
    # a compact dwarf: +120 stars spread over the centre cell and its 4 neighbours
    hot = (np.isclose(df.ra_bin, 185.0 + 10 * step / 0.85)) & (np.isclose(df.dec_bin, -31.5 - 6 * step))
    df.loc[hot, "source_count"] += 60
    for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        m = (np.isclose(df.ra_bin, 185.0 + (10 + di) * step / 0.85)) & (np.isclose(df.dec_bin, -31.5 + (-6 + dj) * step))
        df.loc[m, "source_count"] += 15
    peaks = datalab_tools.SatelliteSearch.aperture_peaks(df, step)
    assert peaks and abs(peaks[0]["ra"] - (185.0 + 10 * step / 0.85)) < 1e-3 and abs(peaks[0]["dec"] - (-31.5 - 6 * step)) < 1e-3
    assert peaks[0]["significance"] >= 5 and all(p["significance"] < peaks[0]["significance"] for p in peaks[1:])



def test_satellite_aperture_peaks_find_a_dwarf_beside_a_brighter_smooth_host():
    import numpy as np

    step = 0.05
    rows = []
    for i in range(-30, 31):
        for j in range(-30, 31):
            x, y = i * step, j * step
            # smooth, bright "host" (e.g. a galaxy body) in the west half: its
            # cells out-count the dwarf but have no LOCAL excess
            host = 400.0 * math.exp(-((x + 3.0) ** 2 + y ** 2) / (2 * 2.0 ** 2))
            rows.append({"ra_bin": 80.0 + x, "dec_bin": y, "source_count": int(round(50 + host))})
    df = pd.DataFrame(rows)
    hot = np.isclose(df.ra_bin, 80.0 + 0.9) & np.isclose(df.dec_bin, 0.3)
    df.loc[hot, "source_count"] += 80
    peaks = datalab_tools.SatelliteSearch.aperture_peaks(df, step, max_candidates=5)
    assert abs(peaks[0]["ra"] - 80.9) < 1e-3 and abs(peaks[0]["dec"] - 0.3) < 1e-3
    assert len(peaks) <= 5


def test_satellite_search_custom_wide_region_is_gated_before_any_query(monkeypatch):
    from services import datalab_orchestration as orch

    def _boom(*a, **k):
        raise AssertionError("no scan may run before confirmation")

    monkeypatch.setattr(orch, "tiled_density_aggregate", _boom)
    monkeypatch.setattr(orch, "_run_builder_sql", _boom)
    cap = datalab_tools.SatelliteSearch()
    out = cap.run(cap.InputModel(survey="delve", region={"ra": 30.0, "dec": -50.0, "radius": 5.0}), CallContext()).to_native()
    assert out["status"] == "needs_confirmation" and out["scanned"] is False
    assert out["sky_area"]["area_deg2"] == pytest.approx(math.pi * 25.0)


def test_satellite_search_preset_area_is_reported_not_gated(monkeypatch):
    from services import datalab_orchestration as orch

    seen = {}

    def _tiled(*a, **k):
        seen["scanned"] = True
        return {"result_id": None, "error": "stop here"}

    monkeypatch.setattr(orch, "tiled_density_aggregate", _tiled)
    cap = datalab_tools.SatelliteSearch()
    out = cap.run(cap.InputModel(survey="delve", preset="delve_south"), CallContext(services={"datalab_client": object()})).to_native()
    # The 5 deg preset exceeds the unconfirmed cap but is curated: it scans
    # once, with no "re-call with confirm=true" round trip.
    assert seen.get("scanned") is True and out.get("status") != "needs_confirmation"


def test_bright_star_verdict_uses_the_legacy_surveys_mask_radius():
    cand = {"ra": 30.0, "dec": -50.0}
    verdict = datalab_tools.SatelliteSearch.bright_star_verdict
    assert datalab_tools.SatelliteSearch.bright_star_mask_radius_deg(8.0) * 60 == pytest.approx(1.89, abs=0.02)
    inside = pd.DataFrame({"ra": [30.0], "dec": [-50.02], "g": [8.2]})        # 1.2' < 1.77' mask
    assert "G = 8.2" in verdict(cand, inside) and "artefact" in verdict(cand, inside)
    # Hydra II (live): a G = 8.1 star 5.3' away must NOT veto a real dwarf.
    assert verdict(cand, pd.DataFrame({"ra": [30.0], "dec": [-50.0883], "g": [8.1]})) is None
    assert verdict(cand, pd.DataFrame({"ra": [30.0], "dec": [-50.02], "g": [10.5]})) is None   # 1.2' > 0.82'
    assert verdict(cand, pd.DataFrame({"ra": [30.0], "dec": [-50.008], "g": [10.5]})) is not None
    assert verdict(cand, pd.DataFrame()) is None


def test_legacy_mask_verdict_flags_sga_galaxies_and_reports_coverage():
    verdict = datalab_tools.SatelliteSearch.legacy_mask_verdict
    # Live L15 values (2026-09-23): the 4.8 sigma peak vs a clean peak / outside LS.
    covered, note = verdict({"n": 763, "n_galaxy": 10, "n_bright": 0})
    assert covered and "large galaxy" in note
    assert verdict({"n": 957, "n_galaxy": 0, "n_bright": 0}) == (True, None)
    assert verdict({"n": 0, "n_galaxy": float("nan"), "n_bright": float("nan")}) == (False, None)
    assert "bright-star mask" in verdict({"n": 300, "n_galaxy": 0, "n_bright": 5})[1]


def test_wedge_selection_thins_uniformly_instead_of_truncating_in_storage_order():
    from services import datalab_orchestration as orch

    # Live 2026-09-23: 17 162 galaxies in RA 150-220 / Dec 0-5 / z <= 0.1; a
    # bare LIMIT 5000 covered only RA 150-172.
    assert orch.wedge_thinning(17162, 5000) == 4 and orch.wedge_thinning(4999, 5000) == 1 and orch.wedge_thinning(None, 5000) == 1
    sql, meta = orch.build_wedge_selection(thin=2)
    assert "MOD(fiberid, 2) = 0" in sql and "dec BETWEEN -1.25 AND 1.25" in sql and meta["thinning"] == 2
    assert "MOD(" not in orch.build_wedge_selection()[0]
    csql, cmeta = orch.build_wedge_count()
    assert csql.startswith("SELECT COUNT(*) AS n FROM sdss_dr17.specobj") and cmeta["aggregate"]
    with pytest.raises(ValueError):
        orch.build_wedge_selection(dec_min=0, dec_max=10)


def test_satellite_search_output_says_which_ranks_were_vetted():
    import inspect

    src = inspect.getsource(datalab_tools.SatelliteSearch.run)
    # every ranked row carries its evidence cards; the summary names the ranks
    assert '"evidence_cards"' in src and '"vetting_summary": vetting_summary' in src
    assert "has NOT been vetted" in src


def test_satellite_search_reports_phase_timings():
    """L15 re-run 2 (2026-09-23) spent ~283 s with no way to tell which Data
    Lab phase was slow; the result now carries per-phase seconds."""
    import inspect

    src = inspect.getsource(datalab_tools.SatelliteSearch.run)
    assert '"phase_seconds": phase_s' in src
    for key in ('"density"', '"map_peaks_screen"', '"cutouts"', '"cmds"'):
        assert key in src
