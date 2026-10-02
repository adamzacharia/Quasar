"""Fixes A-E from the 2026-10-01 MANNA-evals UI benchmark (tmp/manna-evals-2026-10-01).

Stems in each test name map to tmp/manna-evals-2026-10-01/QUESTIONS.md. Live
reference values come from references/references.md (queried 2026-10-01);
the rewritten Data Lab SQL was also run live on 2026-10-02 (T07 rows match,
T14 box = 1,846, T17 cone = 1,584).
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import pandas as pd
import pytest

from capabilities.base import CallContext
from services.datalab_sql_policy import DatalabPolicyError, validate


# ── Fix B: Data Lab SQL rewrites instead of rejections ─────────────────────

T07_SQL = "SELECT TOP 5 ra, dec FROM smash_dr2.object WHERE ra BETWEEN 185 AND 185.01 ORDER BY ra"
T14_GPTOSS_SQL = (
    "SELECT COUNT(*) AS n_objects FROM nsc_dr2.object WHERE ra BETWEEN 187.705931-0.05 AND "
    "187.705931+0.05 AND dec BETWEEN 12.391123-0.05 AND 12.391123+0.05"
)
# references.md 1d: side 0.1 deg, plain RA half-width 0.05 (1,846 objects live)
T14_REF_SQL = (
    "SELECT COUNT(*) AS n FROM nsc_dr2.object WHERE ra BETWEEN 187.65593077 AND 187.75593077 "
    "AND dec BETWEEN 12.34112325 AND 12.44112325"
)


def test_t07_exact_user_query_runs_as_index_ordered_band():
    v = validate(T07_SQL, source="expert")
    assert "TOP" not in v.sql.upper().split("FROM")[0]
    assert v.sql.rstrip().endswith("LIMIT 5")
    assert "WHERE ra BETWEEN 185 AND 185.01 ORDER BY ra" in v.sql
    assert "q3c" not in v.sql  # no invented cone (DeepSeek's T07 failure)
    assert any("Translated ADQL 'SELECT TOP 5'" in w for w in v.warnings)
    assert any("BETWEEN band without a cone" in w for w in v.warnings)
    assert v.meta["row_limit"] == 5


def test_t14_box_with_arithmetic_bounds_becomes_padded_poly_and_keeps_exact_box():
    v = validate(T14_GPTOSS_SQL, source="expert")
    assert "q3c_poly_query(ra, dec, ARRAY[" in v.sql
    # the original predicates stay, so the count is exactly the box
    assert "(ra BETWEEN 187.705931-0.05 AND 187.705931+0.05 AND dec BETWEEN 12.391123-0.05 AND 12.391123+0.05)" in v.sql
    assert any("Rewrote the RA/Dec BETWEEN box" in w for w in v.warnings)


def test_t14_reference_box_polygon_is_a_superset_of_the_box():
    v = validate(T14_REF_SQL, source="expert")
    arr = v.sql.split("ARRAY[", 1)[1].split("]", 1)[0]
    vals = [float(x) for x in arr.split(",")]
    ras, decs = vals[0::2], vals[1::2]
    assert min(ras) < 187.65593077 and max(ras) > 187.75593077
    assert min(decs) < 12.34112325 and max(decs) > 12.44112325


@pytest.mark.parametrize("dec0", [-70.0, -30.0, 0.0, 12.39, 60.0, 80.0])
def test_box_padding_covers_great_circle_bow(dec0):
    """q3c polygon edges are great circles; the padded polygon must contain the
    constant-Dec edges of the box at their midpoint (the worst case)."""
    from services.datalab_sql_policy import _gc_bulge_deg

    lo, hi, span = dec0 - 1.0, dec0 + 1.0, 10.0
    sql = (f"SELECT COUNT(*) FROM nsc_dr2.object WHERE ra BETWEEN 100 AND {100 + span} "
           f"AND dec BETWEEN {lo} AND {hi}")
    v = validate(sql, source="expert")
    arr = [float(x) for x in v.sql.split("ARRAY[", 1)[1].split("]", 1)[0].split(",")]
    p_lo, p_hi = min(arr[1::2]), max(arr[1::2])
    for edge, padded in ((lo, p_lo), (hi, p_hi)):
        # the polygon's own edge at `padded` bows poleward by its bulge; its
        # equatorward-most point must still be beyond the box edge
        bow = _gc_bulge_deg(padded, span)
        if padded < edge:  # southern padded edge
            inner = padded + (bow if padded > 0 else 0.0)
            assert inner < edge
        else:
            inner = padded - (bow if padded < 0 else 0.0)
            assert inner > edge


def test_t17_contains_circle_becomes_q3c_radial():
    sql = ("SELECT COUNT(*) AS n_rows FROM nsc_dr2.object WHERE "
           "1=CONTAINS(POINT('ICRS', ra, dec), CIRCLE('ICRS', 180, -30, 0.1))")
    v = validate(sql, source="expert")
    assert "WHERE q3c_radial_query(ra, dec, 180, -30, 0.1)" in v.sql
    assert "CONTAINS" not in v.sql.upper()


def test_q3c_equals_one_is_stripped_for_postgres():
    v = validate("SELECT ra, dec FROM nsc_dr2.object WHERE q3c_radial_query(ra, dec, 180, -30, 0.1) = 1", source="expert")
    assert "q3c_radial_query(ra, dec, 180, -30, 0.1)\n" in v.sql or v.sql.endswith(")") or "= 1" not in v.sql
    assert "= 1" not in v.sql
    v2 = validate("SELECT ra, dec FROM nsc_dr2.object WHERE 1 = q3c_radial_query(ra, dec, 180, -30, 0.1)", source="expert")
    assert "1 =" not in v2.sql


def test_ra_only_band_without_order_still_refused_with_fix_hint():
    with pytest.raises(DatalabPolicyError) as err:
        validate("SELECT ra, dec FROM smash_dr2.object WHERE ra BETWEEN 185 AND 185.01", source="expert")
    assert "RA-only or Dec-only" in str(err.value)
    assert "ORDER BY" in err.value.fix_hint


def test_ra_only_band_with_large_limit_or_wide_band_refused():
    with pytest.raises(DatalabPolicyError):
        validate("SELECT ra, dec FROM smash_dr2.object WHERE ra BETWEEN 185 AND 185.01 ORDER BY ra LIMIT 2000", source="expert")
    with pytest.raises(DatalabPolicyError):
        validate("SELECT ra, dec FROM smash_dr2.object WHERE ra BETWEEN 0 AND 360 ORDER BY ra LIMIT 5", source="expert")
    with pytest.raises(DatalabPolicyError):  # ordered by a different column
        validate("SELECT ra, dec FROM smash_dr2.object WHERE ra BETWEEN 185 AND 185.01 ORDER BY gmag LIMIT 5", source="expert")


def test_ra_only_count_is_aggregate_safe():
    v = validate("SELECT COUNT(*) AS n FROM smash_dr2.object WHERE ra BETWEEN 185 AND 185.01", source="expert")
    assert v.sql.startswith("SELECT COUNT(*)")


@pytest.mark.parametrize("sql", [
    # OR could let a branch escape the bound
    "SELECT ra FROM nsc_dr2.object WHERE ra BETWEEN 1 AND 2 AND dec BETWEEN 1 AND 2 OR gmag < 20 LIMIT 5",
    # RA wrap / out of range
    "SELECT ra FROM nsc_dr2.object WHERE ra BETWEEN 359 AND 361 AND dec BETWEEN 1 AND 2 LIMIT 5",
    # too large
    "SELECT ra FROM nsc_dr2.object WHERE ra BETWEEN 10 AND 50 AND dec BETWEEN 1 AND 2 LIMIT 5",
    # touches the pole
    "SELECT ra FROM nsc_dr2.object WHERE ra BETWEEN 10 AND 11 AND dec BETWEEN 89.5 AND 89.99 LIMIT 5",
    # non-numeric bound
    "SELECT ra FROM nsc_dr2.object WHERE ra BETWEEN gmag AND 2 AND dec BETWEEN 1 AND 2 LIMIT 5",
])
def test_unprovable_boxes_still_refused(sql):
    with pytest.raises(DatalabPolicyError, match="BETWEEN"):
        validate(sql, source="expert")


def test_box_rewrite_keeps_row_cap_nan_guard_and_limit_ceiling():
    v = validate("SELECT ra, dec, gmag FROM nsc_dr2.object WHERE ra BETWEEN 10 AND 10.1 AND dec BETWEEN -5 AND -4.9 AND gmag > 18",
                 source="expert")
    assert "LIMIT 500 /* platform row cap, not a science cut */" in v.sql
    assert "gmag < 'Infinity'::float8" in v.sql
    with pytest.raises(DatalabPolicyError, match="exceeds"):
        validate("SELECT ra FROM nsc_dr2.object WHERE ra BETWEEN 10 AND 10.1 AND dec BETWEEN -5 AND -4.9 LIMIT 9000",
                 source="expert")


def test_box_rewrite_keeps_registry_quality_cuts():
    v = validate("SELECT ra, dec FROM des_dr1.main WHERE ra BETWEEN 10 AND 10.1 AND dec BETWEEN -5 AND -4.9 LIMIT 10",
                 source="expert")
    assert "q3c_poly_query" in v.sql
    # des_dr1.main registry defaults (if any) are still injected around the bound
    from services import datalab_registry as reg

    for cut in reg.default_quality_cuts("des_dr1", "main"):
        assert str(cut["column"]) in v.sql


def test_top_with_existing_limit_keeps_limit():
    v = validate("SELECT TOP 5 ra FROM nsc_dr2.object WHERE q3c_radial_query(ra, dec, 1, 1, 0.01) LIMIT 3", source="expert")
    assert "TOP" not in v.sql.upper()
    assert v.meta["row_limit"] == 3


def test_q3c_join_rule_unchanged_by_dialect_pass():
    with pytest.raises(DatalabPolicyError, match="q3c_join"):
        validate("SELECT * FROM gaia_dr3.gaia_source AS g JOIN nsc_dr2.object AS big ON q3c_join(g.ra, g.dec, big.ra, big.dec, 0.0003) "
                 "WHERE q3c_radial_query(g.ra, g.dec, 1, 1, 0.1) LIMIT 5", source="expert")


def test_literal_text_is_never_rewritten():
    sql = "SELECT ra, dec, 'CONTAINS(POINT(x), CIRCLE(1,2,3)) = 1' AS note FROM nsc_dr2.object WHERE q3c_radial_query(ra, dec, 1, 1, 0.01)"
    v = validate(sql, source="expert")
    assert "'CONTAINS(POINT(x), CIRCLE(1,2,3)) = 1'" in v.sql


# ── Fix A: NRAO archive knowledge ─────────────────────────────────────────

def test_nrao_profile_reaches_prompt_index_with_endpoint_table_async_and_no_lower():
    from services.archive_profiles import prompt_index_lines

    line = next(l for l in prompt_index_lines() if l.startswith("- nrao:"))
    assert "https://data-query.nrao.edu/tap" in line
    assert "tap_schema.obscore" in line and "ivoa.obscore does not exist" in line
    assert "LOWER()" in line and "UPPER()" in line
    assert "mode='auto' or 'async'" in line


def test_nrao_profile_notes_and_endpoint_matching():
    from services.archive_profiles import error_hints, get_profile, profile_for_endpoint

    p = get_profile("vla")
    assert p.archive == "nrao"
    ids = {x.id for x in p.pitfalls}
    assert {"obscore_at_tap_schema", "no_string_functions", "async_and_selective",
            "almascience_nrao_is_alma", "instrument_values"} <= ids
    inst = next(x for x in p.pitfalls if x.id == "instrument_values").summary
    for value in ("'EVLA'", "'VLA'", "'VLBA'", "'GBT'"):
        assert value in inst
    assert profile_for_endpoint("https://data-query.nrao.edu/tap/async").archive == "nrao"
    # the ALMA North American mirror is ALMA, never NRAO
    assert profile_for_endpoint("https://almascience.nrao.edu/tap").archive == "alma"
    hints = error_hints("https://data-query.nrao.edu/tap", "SELECT * FROM tap_schema.obscore WHERE UPPER(target_name)='M87'")
    assert any("LOWER()" in h for h in hints)


@pytest.mark.parametrize("bundle", ["v2", "legacy"])
def test_archive_rules_reach_both_prompt_paths(bundle):
    from core.agent import QuasarAgent

    prompt = QuasarAgent._build_system_prompt(SimpleNamespace(prompt_bundle=bundle))
    assert "ARCHIVE ACCURACY RULES:" in prompt
    assert "https://data-query.nrao.edu/tap" in prompt
    assert "almascience.nrao.edu is ALMA's mirror, never the NRAO archive" in prompt
    assert "lower bound" in prompt  # row-cap count rule (Fix C)
    assert "browse_schema(<archive>)" in prompt  # tool-first capability rule (Fix D)
    assert "file:// link" in prompt  # no local paths as URLs (Fix E)
    assert "- nrao: call browse_schema('nrao')" in prompt


def test_alma_tools_not_described_as_vla_archive(tool_registry=None):
    from capabilities.alma import AdvancedSearch
    from core.prompts.tool_description_overrides import OVERRIDES

    assert "VLA" not in AdvancedSearch.description.split("ALMA ONLY")[0]
    assert "nrao" in OVERRIDES["vo_adql_query"]


def _vo_ctx(service):
    table = lambda rows, **kw: {"success": True, "rowcount": len(rows), "warnings": kw.get("warnings")}
    return CallContext(services={
        "get_vo_registry_service": lambda: service,
        "external_catalog_table_result": table,
        "live_imagery_coordinates": lambda **kw: (10.0, -5.0, "X"),
    })


class _RecordingVo:
    def __init__(self, result=None):
        self.calls = []
        self.result = result or {"success": True, "rows": [{"a": 1}], "columns": ["a"],
                                 "provenance": {"maxrec": 200}}

    def run_adql(self, access_url, adql, max_rows, mode="sync", owner=None):
        self.calls.append((access_url, adql, mode))
        return dict(self.result)


def _vo_run(svc, **kw):
    from capabilities.vo import VoAdqlQuery

    cap = VoAdqlQuery()
    return cap.run(cap.InputModel(**kw), _vo_ctx(svc)).to_native()


def test_t18_nrao_lower_is_refused_before_any_network_call():
    svc = _RecordingVo()
    out = _vo_run(svc, access_url="https://data-query.nrao.edu/tap",
                  adql="SELECT TOP 10 * FROM tap_schema.obscore WHERE LOWER(target_name) LIKE '%m87%'", mode="async")
    assert out["success"] is False and out["query_sent"] is False
    assert "LIKE" in out["hint"] and "3C274" in out["hint"]
    assert svc.calls == []


def test_nrao_ivoa_obscore_rewritten_and_disclosed():
    svc = _RecordingVo()
    out = _vo_run(svc, access_url="https://data-query.nrao.edu/tap",
                  adql="SELECT TOP 25 * FROM ivoa.obscore WHERE instrument_name = 'EVLA'", mode="auto")
    assert svc.calls[0][1] == "SELECT TOP 25 * FROM tap_schema.obscore WHERE instrument_name = 'EVLA'"
    assert any("tap_schema.obscore" in w for w in out["warnings"])


def test_t26_nrao_unfiltered_obscore_read_refused_cleanly():
    svc = _RecordingVo()
    out = _vo_run(svc, access_url="https://data-query.nrao.edu/tap",
                  adql="SELECT TOP 10 * FROM tap_schema.obscore", mode="sync")
    assert out["success"] is False and svc.calls == []
    assert "Traceback" not in str(out) and "C:\\" not in str(out)
    assert "selective WHERE" in out["hint"]


def test_nrao_metadata_query_passes_and_sync_data_read_warns():
    svc = _RecordingVo()
    _vo_run(svc, access_url="https://data-query.nrao.edu/tap",
            adql="SELECT column_name FROM tap_schema.columns WHERE table_name = 'tap_schema.obscore'")
    assert len(svc.calls) == 1
    out = _vo_run(svc, access_url="https://data-query.nrao.edu/tap",
                  adql="SELECT obs_id FROM tap_schema.obscore WHERE instrument_name = 'GBT'", mode="sync")
    assert any("mode='auto'" in w for w in out["warnings"])


def test_t10_vla_filter_on_alma_mirror_points_to_nrao():
    svc = _RecordingVo()
    out = _vo_run(svc, access_url="https://almascience.nrao.edu/tap",
                  adql="SELECT TOP 100 * FROM ivoa.obscore WHERE facility = 'VLA'", mode="async")
    assert out["success"] is False and svc.calls == []
    assert "https://data-query.nrao.edu/tap" in out["hint"]
    # an ordinary ALMA query on the same mirror is untouched
    _vo_run(svc, access_url="https://almascience.nrao.edu/tap", adql="SELECT TOP 5 * FROM ivoa.obscore WHERE target_name = 'M87'")
    assert len(svc.calls) == 1


def test_wrong_nrao_host_gets_the_real_endpoint():
    svc = _RecordingVo()
    out = _vo_run(svc, access_url="https://data.nrao.edu/tap", adql="SELECT TOP 1 * FROM ivoa.obscore")
    assert out["success"] is False and "https://data-query.nrao.edu/tap" in out["hint"]


def test_advanced_search_refuses_vla_filters():
    from capabilities.alma import AdvancedSearch

    called = []
    ctx = CallContext(services={"search_service": SimpleNamespace(advanced_search=lambda q: called.append(q))})
    cap = AdvancedSearch()
    out = cap.run(cap.InputModel(query="SELECT DISTINCT instrument_name FROM ivoa.obscore WHERE instrument_name = 'EVLA'"),
                  ctx).to_native()
    assert out["success"] is False and called == []
    assert "data-query.nrao.edu" in out["hint"]


# Data Lab TAP route (vo_adql_query against datalab.noirlab.edu/tap)

@pytest.mark.parametrize("adql,expected", [
    ("SELECT COUNT(*) AS n FROM nsc_dr2.object WHERE q3c_radial_query(ra, dec, 180, -30, 0.1)",
     "q3c_radial_query(ra, dec, 180, -30, 0.1) = 't'"),
    ("SELECT COUNT(*) AS n FROM nsc_dr2.object WHERE q3c_radial_query(ra, dec, 180, -30, 0.1) = true",
     "q3c_radial_query(ra, dec, 180, -30, 0.1) = 't'"),
    ("SELECT COUNT(*) AS n FROM nsc_dr2.object WHERE q3c_radial_query(ra, dec, 180, -30, 0.1) = 1",
     "q3c_radial_query(ra, dec, 180, -30, 0.1) = 't'"),
    ("SELECT COUNT(*) AS n FROM nsc_dr2.object WHERE 1=CONTAINS(POINT('ICRS', ra, dec), CIRCLE('ICRS', 180, -30, 0.1))",
     "WHERE q3c_radial_query(ra, dec, 180, -30, 0.1) = 't'"),
])
def test_t17_datalab_tap_dialect_normalised_to_what_runs(adql, expected):
    from services.adql_preflight import preflight

    out, warnings = preflight("https://datalab.noirlab.edu/tap", adql)
    assert expected in out
    assert "= 1" not in out and "= true" not in out
    assert warnings


def test_datalab_tap_q3c_t_literal_untouched_and_array_refused():
    from services.adql_preflight import PreflightRejection, preflight

    ok = "SELECT COUNT(*) FROM nsc_dr2.object WHERE q3c_radial_query(ra, dec, 1, 1, 0.1) = 't'"
    assert preflight("https://datalab.noirlab.edu/tap", ok) == (ok, [])
    with pytest.raises(PreflightRejection, match="ARRAY"):
        preflight("https://datalab.noirlab.edu/tap/sync",
                  "SELECT COUNT(*) FROM nsc_dr2.object WHERE q3c_poly_query(ra, dec, ARRAY[1,1,2,1,2,2,1,2]) = 't'")


def test_unknown_archives_pass_through_untouched():
    from services.adql_preflight import preflight

    q = "SELECT TOP 5 * FROM ivoa.obscore WHERE LOWER(target_name) = 'm87'"
    assert preflight("https://gea.esac.esa.int/tap-server/tap", q) == (q, [])
    assert preflight("https://archive.eso.org/tap_obs", q) == (q, [])
    assert preflight(None, None) == (None, [])


# ── Fix C: row caps are never counts ──────────────────────────────────────

def test_row_cap_fields_shapes():
    from services.row_cap import row_cap_fields

    assert row_cap_fields(10, 500) == {}
    assert row_cap_fields(500, 500, 500) == {}  # exactly the cap exists
    exact = row_cap_fields(500, 500, 4386)
    assert exact["truncated"] and exact["total"] == 4386 and not exact["count_is_lower_bound"]
    unknown = row_cap_fields(500, 500)
    assert unknown["total"] == "unknown" and unknown["count_is_lower_bound"]
    assert "at least 500" in unknown["row_cap_note"]


class _EsoClient:
    def __init__(self, df):
        self.df = df

    def search_by_position(self, **kw):
        return self.df

    search_by_target = search_by_position


def _eso_run(df, **kw):
    from capabilities.archives import SearchEso

    ctx = CallContext(services={"eso_client": _EsoClient(df), "set_last_search_results": lambda v: None,
                                "set_last_run_result": lambda v: None})
    cap = SearchEso()
    return cap.run(cap.InputModel(**kw), ctx).to_native()


def _eso_frame(n, total=None):
    df = pd.DataFrame({"instrument_name": ["GIRAFFE"] * n, "dataproduct_type": ["spectrum"] * n})
    df.attrs["row_cap"] = 500
    if total is not None:
        df.attrs["total_count"] = total
    return df


def test_mq17_eso_cap_reports_exact_count_not_500():
    out = _eso_run(_eso_frame(500, 4386), ra=83.8, dec=-5.4, radius_arcmin=6)
    assert out["total_results"] == 4386
    assert out["truncated"] is True and out["returned"] == 500 and out["total"] == 4386
    assert "4386" in out["note"] and "Found 500 ESO observations" not in out["note"]


def test_mq17_eso_cap_with_unknown_total_is_a_lower_bound():
    out = _eso_run(_eso_frame(500), ra=83.8, dec=-5.4, radius_arcmin=6)
    assert out["total"] == "unknown" and out["count_is_lower_bound"] is True
    assert "at least 500" in out["note"]


def test_eso_below_cap_is_byte_identical_and_error_is_not_no_data():
    out = _eso_run(_eso_frame(12), ra=83.8, dec=-5.4, radius_arcmin=6)
    assert "truncated" not in out and out["total_results"] == 12
    failed = pd.DataFrame()
    failed.attrs["quasar_error"] = "ESO TAP search failed: timeout"
    err = _eso_run(failed, ra=83.8, dec=-5.4, radius_arcmin=6)
    assert err["success"] is False and "not a 'no observations' result" in err["note"]


def test_eso_client_runs_count_companion_when_capped(monkeypatch):
    from integrations import eso_tap_client as eso

    client = eso.ESOTapClient() if hasattr(eso, "ESOTapClient") else None
    if client is None:
        cls = next(v for v in vars(eso).values() if isinstance(v, type) and hasattr(v, "search_by_position"))
        client = cls()
    rows = pd.DataFrame({"s_ra": [1.0] * 3, "s_dec": [2.0] * 3, "instrument_name": ["X"] * 3, "t_min": [1.0] * 3})

    class _Res:
        def to_table(self):
            from astropy.table import Table

            return Table.from_pandas(rows)

    monkeypatch.setattr(client, "_get_tap_service", lambda: SimpleNamespace(search=lambda adql: _Res()))
    seen = {}
    monkeypatch.setattr(client, "_count_where", lambda where, timeout_s=45.0: seen.setdefault("where", where) and 4386)
    df = client.search_by_position(83.8, -5.4, radius_arcmin=6, max_results=3)
    assert df.attrs["row_cap"] == 3 and df.attrs["total_count"] == 4386
    assert "CONTAINS(POINT('ICRS', s_ra, s_dec)" in seen["where"]


def test_mast_and_irsa_report_pre_cap_total():
    from capabilities.archives import _apply_row_cap

    df = pd.DataFrame({"a": range(500)})
    df.attrs.update(row_cap=500, total_count=1234)
    out = _apply_row_cap({"success": True, "total_results": 500, "note": "Found 500 MAST observations."},
                         df, "MAST observations")
    assert out["total_results"] == 1234 and out["truncated"] and "1234" in out["note"]
    small = pd.DataFrame({"a": range(5)})
    small.attrs.update(row_cap=500, total_count=5)
    assert _apply_row_cap({"total_results": 5}, small, "x") == {"total_results": 5}


def test_vo_adql_query_flags_a_result_that_filled_its_top():
    svc = _RecordingVo({"success": True, "rows": [{"a": i} for i in range(10)], "columns": ["a"],
                        "provenance": {"maxrec": 200}})
    out = _vo_run(svc, access_url="https://gea.esac.esa.int/tap-server/tap",
                  adql="SELECT TOP 10 source_id FROM gaiadr3.gaia_source WHERE 1=CONTAINS(POINT(ra,dec),CIRCLE(1,1,1))")
    assert out["truncated"] is True and out["total"] == "unknown"
    count = _vo_run(_RecordingVo({"success": True, "rows": [{"n": 51}], "columns": ["n"], "provenance": {"maxrec": 200}}),
                    access_url="https://gea.esac.esa.int/tap-server/tap", adql="SELECT COUNT(*) AS n FROM gaiadr3.gaia_source")
    assert "truncated" not in count


def test_advanced_search_flags_a_result_that_filled_its_top():
    from capabilities.alma import AdvancedSearch

    df = pd.DataFrame({"member_ous_uid": [f"uid://A001/X1/X{i}" for i in range(5)]})
    ctx = CallContext(services={
        "search_service": SimpleNamespace(advanced_search=lambda q: df),
        "set_last_search_results": lambda v: None, "set_last_run_result": lambda v: None,
        "alma_tap_provenance": {},
    })
    cap = AdvancedSearch()
    out = cap.run(cap.InputModel(query="SELECT TOP 5 member_ous_uid FROM ivoa.obscore WHERE target_name = 'M83'"), ctx).to_native()
    assert out["truncated"] is True and out["count_is_lower_bound"] is True


# ── Fix D: identifiers no tool returned ────────────────────────────────────

def _summary(*outputs):
    from core.answer_verifier import build_trace_summary

    return build_trace_summary([{"output": o} for o in outputs])


def test_mq18_source_id_not_in_any_tool_output_is_flagged():
    from core.answer_verifier import verify_answer

    s = _summary('{"success": true, "count": 198}')
    rep = verify_answer("There are 198 sources. An example source ID is 1315356426404373760.", s,
                        user_text="List Gaia DR3 sources within 0.02 degrees of RA=201.365, Dec=-43.019")
    kinds = [c.kind for c in rep.unsupported]
    assert "identifier" in kinds
    assert any("1315356426404373760" in c.text for c in rep.unsupported)


def test_source_id_returned_by_a_tool_or_typed_by_user_is_not_flagged():
    from core.answer_verifier import verify_answer

    s = _summary('{"success": true, "rows": [{"source_id": 6088704552603060224, "pmra": -7.74}]}')
    rep = verify_answer("Example: source_id 6088704552603060224 (pmra -7.74).", s)
    assert not [c for c in rep.unsupported if c.kind == "identifier"]
    rep2 = verify_answer("Gaia DR3 3907709439453756032 is the M87 nucleus.", _summary('{"success": true}'),
                         user_text="What is Gaia DR3 3907709439453756032?")
    assert not [c for c in rep2.unsupported if c.kind == "identifier"]


def test_mq09_file_url_is_always_flagged():
    from core.answer_verifier import format_verification_block, verify_answer

    s = _summary('{"success": true, "fits_path": "C:\\\\Users\\\\adama\\\\quasar_data\\\\sky_images\\\\M51_DSS2_Red.fits"}')
    rep = verify_answer("Direct access URL: file:///C:/Users/adama/quasar_data/sky_images/M51_DSS2_Red.fits", s)
    assert [c.kind for c in rep.unsupported] == ["local_url"]
    assert "local file path presented as a URL" in format_verification_block(rep)


def test_data_urls_checked_base_urls_not():
    from core.answer_verifier import verify_answer

    s = _summary('{"success": true, "fits_url": "https://cadc-west-01.canfar.net/raven/files/mast:HST/product/jcmx06t8q_flc.fits"}')
    good = verify_answer("FITS: https://cadc-west-01.canfar.net/raven/files/mast:HST/product/jcmx06t8q_flc.fits.", s)
    assert not good.unsupported
    bad = verify_answer("FITS: https://ws.cadc-ccda.hia-iha.nrc-cnrc.gc.ca/data/pub/HST/made_up.fits", s)
    assert [c.kind for c in bad.unsupported] == ["identifier"]
    base = verify_answer("The Gaia TAP base URL is https://gea.esac.esa.int/tap-server/tap.", _summary("{}"))
    assert not base.unsupported


def test_percent_encoded_datalink_matches_decoded_answer():
    from core.answer_verifier import verify_answer

    enc = "https://ws.cadc-ccda.hia-iha.nrc-cnrc.gc.ca/caom2ops/datalink?ID=ivo%3A%2F%2Fcadc.nrc.ca%2FHST%3Fj1%2Fj1_flc"
    s = _summary('{"success": true, "access_url": "%s"}' % enc)
    rep = verify_answer("DataLink: https://ws.cadc-ccda.hia-iha.nrc-cnrc.gc.ca/caom2ops/datalink?ID=ivo://cadc.nrc.ca/HST?j1/j1_flc", s)
    assert not rep.unsupported


def test_bibcode_and_alma_uid_checks():
    from core.answer_verifier import verify_answer

    s = _summary('{"success": true, "bibcode": "2019ApJ...875L...1E", "member_ous_uid": "uid://A001/X12a3/X45"}')
    ok = verify_answer("See 2019ApJ...875L...1E for MOUS uid://A001/X12a3/X45.", s)
    assert not ok.unsupported
    bad = verify_answer("See 2021A&A...650A..12Z and uid://A001/X99/X1.", s)
    assert len([c for c in bad.unsupported if c.kind == "identifier"]) == 2


def test_failed_call_output_is_not_evidence():
    from core.answer_verifier import build_trace_summary, verify_answer

    s = build_trace_summary([], [{"name": "x", "ok": False, "output": '{"source_id": 1315356426404373760}'}])
    rep = verify_answer("source_id 1315356426404373760", s)
    assert any(c.kind == "identifier" for c in rep.unsupported)


def test_no_tool_turn_identifier_check_in_finalize(monkeypatch):
    from core import runner

    agent = SimpleNamespace(_accumulated_run_results=[], last_run_result=None, _accumulated_tool_trace=[],
                            get_conversation_history=lambda: [])
    streamed = []
    out = runner._finalize_answer_text(agent, "Try this FITS: file:///C:/tmp/x.fits", on_token=streamed.append,
                                       user_query="Give me a FITS URL for M51", had_tool_calls=False)
    assert "local file path presented as a URL" in out
    clean = runner._finalize_answer_text(agent, "M87 is at RA 187.70593, Dec +12.39112.", user_query="Where is M87?",
                                         had_tool_calls=False)
    assert "Verification" not in clean


# ── Fix E: direct access URLs ─────────────────────────────────────────────

_DATALINK_VOTABLE = b"""<?xml version="1.0" encoding="UTF-8"?>
<VOTABLE version="1.3" xmlns="http://www.ivoa.net/xml/VOTable/v1.3">
 <RESOURCE type="results"><TABLE>
  <FIELD name="ID" datatype="char" arraysize="*"/>
  <FIELD name="access_url" datatype="char" arraysize="*"/>
  <FIELD name="service_def" datatype="char" arraysize="*"/>
  <FIELD name="error_message" datatype="char" arraysize="*"/>
  <FIELD name="semantics" datatype="char" arraysize="*"/>
  <FIELD name="content_type" datatype="char" arraysize="*"/>
  <DATA><TABLEDATA>
   <TR><TD>ivo://x</TD><TD>https://example.org/preview.png</TD><TD></TD><TD></TD><TD>#preview</TD><TD>image/png</TD></TR>
   <TR><TD>ivo://x</TD><TD>https://example.org/files/jcmx06t8q_flc.fits</TD><TD></TD><TD></TD><TD>#this</TD><TD>application/fits</TD></TR>
  </TABLEDATA></DATA>
 </TABLE></RESOURCE>
</VOTABLE>"""


def test_datalink_this_row_is_picked():
    from services.datalink_resolve import parse_datalink, pick_this_row

    assert pick_this_row(parse_datalink(_DATALINK_VOTABLE)) == "https://example.org/files/jcmx06t8q_flc.fits"
    assert pick_this_row([{"semantics": "#this", "access_url": "", "error_message": ""}]) is None


def test_resolve_many_follows_only_datalinks():
    from services.datalink_resolve import resolve_many

    class _Session:
        def __init__(self):
            self.urls = []

        def get(self, url, timeout=None, **kw):
            self.urls.append(url)
            return SimpleNamespace(status_code=200, content=_DATALINK_VOTABLE,
                                   iter_content=lambda chunk_size: iter([_DATALINK_VOTABLE]))

    sess = _Session()
    link = "https://ws.cadc-ccda.hia-iha.nrc-cnrc.gc.ca/caom2ops/datalink?ID=ivo%3A%2F%2Fx"
    out = resolve_many([link, "https://example.org/direct.fits"], session=sess)
    assert out == {link: "https://example.org/files/jcmx06t8q_flc.fits"}
    assert sess.urls == [link]


def test_get_sky_image_says_it_has_no_archive_url():
    from capabilities.viz import GetSkyImage

    assert "no archive access URL" in GetSkyImage.description
    ctx = CallContext(services={
        "skyview_client": SimpleNamespace(get_image=lambda **kw: {
            "success": True, "survey": "DSS2 Red", "fits_path": "C:\\x\\M51.fits", "preview_path": ""}),
        "set_last_run_result": lambda v: None,
    })
    cap = GetSkyImage()
    out = cap.run(cap.InputModel(target_name="M51"), ctx).to_native()
    assert out["paths_are_local"] is True
    assert "NOT archive URLs" in out["note"] and "file://" in out["note"]


# ── Codex guard round 1 (task-5314487-39390): CX-01 .. CX-10 ───────────────

def _alma_position(df, **kw):
    from capabilities.alma import SearchByPosition

    class _Svc:
        def cone_search(self, ra, dec, radius, facility, max_results, **extra):
            return df.copy()

    ctx = CallContext(services={
        "search_service": _Svc(), "console_log": lambda *a: None,
        "set_last_search_results": lambda v: None, "set_last_run_result": lambda v: None,
        "alma_tap_provenance": {"query": None, "url": None},
    })
    cap = SearchByPosition()
    return cap.run(cap.InputModel(**kw), ctx).to_native()


def _capped_alma_frame():
    df = pd.DataFrame({"member_ous_uid": [f"uid://A001/X1/X{i}" for i in range(4)],
                       "scan_intent": ["TARGET", "BANDPASS", "TARGET", "PHASE"],
                       "band_list": ["3", "3", "6", "6"], "proposal_id": ["2019.1.00001.S"] * 4})
    df.attrs.update(truncated=True, total_count=913, total_mous=174)
    return df


def test_cx01_capped_alma_cone_reports_exact_totals_without_post_filter():
    out = _alma_position(_capped_alma_frame(), ra=204.25, dec=-29.87, radius=0.1, max_results=4)
    assert out["truncated"] is True and out["total"] == 913 and out["total_mous"] == 174
    assert out["count_is_lower_bound"] is False


def test_cx01_post_filtered_capped_cone_total_is_unknown_not_the_cone_total():
    out = _alma_position(_capped_alma_frame(), ra=204.25, dec=-29.87, radius=0.1, max_results=4,
                         scan_intent="TARGET")
    assert out["total"] == "unknown" and out["count_is_lower_bound"] is True
    assert out["total_unfiltered_cone_rows"] == 913
    assert "total_mous" not in out
    assert any("UNFILTERED cone holds 913" in w for w in out["warnings"])


def test_cx02_long_failed_output_never_substantiates_an_identifier():
    from core.answer_verifier import build_trace_summary, verify_answer

    long_failed = '{"success": false, "error": "' + ("x" * 2100) + ' 1315356426404373760"}'
    trace = [{"name": "q", "ok": False, "output": long_failed[:2000]}]
    s = build_trace_summary([{"output": long_failed}], trace)
    assert any(c.kind == "identifier" for c in verify_answer("source_id 1315356426404373760", s).unsupported)
    # a success:false output without a trace record is excluded too
    s2 = build_trace_summary([{"output": '{"success": false, "rows": [{"source_id": 1315356426404373760}]}'}])
    assert any(c.kind == "identifier" for c in verify_answer("source_id 1315356426404373760", s2).unsupported)


def test_cx03_overlapping_ids_and_truncated_urls_do_not_match():
    from core.answer_verifier import verify_answer

    s = _summary('{"success": true, "source_id": 1234567890123456789, '
                 '"fits_url": "https://example.org/files/a.fits.gz"}')
    assert any("234567890123456789" in c.text for c in verify_answer("id 234567890123456789", s).unsupported)
    assert not verify_answer("id 1234567890123456789", s).unsupported
    assert verify_answer("FITS https://example.org/files/a.fits", s).unsupported  # prefix of a.fits.gz
    assert not verify_answer("FITS https://example.org/files/a.fits.gz", s).unsupported


def test_cx04_datalink_resolution_is_sequential_and_bounded(monkeypatch):
    import threading

    from services import datalink_resolve as dr

    threads, calls = set(), []

    def fake_resolve(url, timeout=None, session=None):
        threads.add(threading.get_ident())
        calls.append(timeout)
        return None

    monkeypatch.setattr(dr, "resolve_this", fake_resolve)
    links = [f"https://h/caom2ops/datalink?ID={i}" for i in range(3)]
    dr.resolve_many(links, timeout=8.0, overall=15.0)
    assert threads == {threading.get_ident()}  # the tool's own thread, so its deadline applies
    assert calls and all(t <= 8.0 for t in calls)
    monkeypatch.undo()

    # a body that trickles past the wall clock is abandoned
    clock = [0.0]
    monkeypatch.setattr(dr.time, "monotonic", lambda: clock[0])

    def chunks(chunk_size):
        for _ in range(5):
            clock[0] += 5.0
            yield b"x"

    sess = SimpleNamespace(get=lambda url, timeout=None, stream=False: SimpleNamespace(
        status_code=200, iter_content=chunks, close=lambda: None))
    assert dr.resolve_this("https://h/caom2ops/datalink?ID=1", timeout=8.0, session=sess) is None


def test_cx05_projected_telescope_label_on_alma_passes():
    from services.adql_preflight import PreflightRejection, preflight

    ok = "SELECT 'VLA' AS label, target_name FROM ivoa.obscore WHERE target_name = 'M87'"
    assert preflight("https://almascience.eso.org/tap", ok) == (ok, [])
    ok2 = "SELECT obs_id FROM ivoa.obscore WHERE target_name = 'VLA 1623-243'"
    assert preflight("https://almascience.eso.org/tap", ok2) == (ok2, [])
    with pytest.raises(PreflightRejection):
        preflight("https://almascience.eso.org/tap",
                  "SELECT * FROM ivoa.obscore WHERE instrument_name IN ('ALMA', 'VLBA')")


def test_cx06_user_supplied_file_url_is_still_flagged():
    from core.answer_verifier import verify_answer

    rep = verify_answer("Direct FITS URL: file:///C:/tmp/example.fits", _summary("{}"),
                        user_text="Is file:///C:/tmp/example.fits an archive URL?")
    assert [c.kind for c in rep.unsupported] == ["local_url"]


def test_cx07_vo_cone_search_full_result_gets_the_row_cap_shape():
    from capabilities.vo import VoConeSearch

    svc = SimpleNamespace(cone_search=lambda *a, **k: {"success": True, "rows": [{"a": 1}] * 3, "columns": ["a"],
                                                       "truncated": True, "provenance": {}})
    cap = VoConeSearch()
    out = cap.run(cap.InputModel(access_url="https://x/scs", ra=1.0, dec=2.0, max_rows=3), _vo_ctx(svc)).to_native()
    assert out["truncated"] is True and out["returned"] == 3 and out["total"] == "unknown"


def test_cx08_box_with_is_not_null_or_not_in_is_rewritten_but_bare_not_refused():
    v = validate("SELECT ra, dec FROM nsc_dr2.object WHERE ra BETWEEN 10 AND 10.1 AND dec BETWEEN -5 AND -4.9 "
                 "AND gmag IS NOT NULL AND id NOT IN ('a', 'b') LIMIT 10", source="expert")
    assert "q3c_poly_query" in v.sql
    with pytest.raises(DatalabPolicyError, match="BETWEEN"):
        validate("SELECT ra FROM nsc_dr2.object WHERE NOT (ra BETWEEN 10 AND 10.1 AND dec BETWEEN -5 AND -4.9) LIMIT 10",
                 source="expert")


def test_cx09_contains_compared_to_zero_is_left_alone():
    from services.adql_preflight import preflight

    sql = ("SELECT COUNT(*) FROM nsc_dr2.object WHERE q3c_radial_query(ra, dec, 1, 1, 1) AND "
           "CONTAINS(POINT('ICRS', ra, dec), CIRCLE('ICRS', 1, 1, 0.1)) = 0")
    v = validate(sql, source="expert")
    assert "CONTAINS(POINT('ICRS', ra, dec), CIRCLE('ICRS', 1, 1, 0.1)) = 0" in v.sql
    adql = "SELECT COUNT(*) FROM nsc_dr2.object WHERE 0 = CONTAINS(POINT('ICRS', ra, dec), CIRCLE('ICRS', 1, 1, 0.1))"
    out, _ = preflight("https://datalab.noirlab.edu/tap", adql)
    assert "CONTAINS(" in out


def test_cx10_documentation_download_links_are_not_policed():
    from core.answer_verifier import verify_answer

    rep = verify_answer("Docs: https://www.cosmos.esa.int/web/gaia/download and "
                        "https://example.org/dataset/overview", _summary("{}"))
    assert not rep.unsupported
    # generic ?id= and /files/ documentation links (verify round 1 reopen)
    rep2 = verify_answer("See https://docs.example.org/article?id=123 and https://example.org/files/guide",
                         _summary("{}"))
    assert not rep2.unsupported
    # IVOA dataset ids are still data URLs
    rep3 = verify_answer("FITS: https://h/caom2ops/pkg?ID=ivo://cadc.nrc.ca/HST?x", _summary("{}"))
    assert rep3.unsupported


# ── Codex guard verify round 1 reopens: CX-02/03/04/05 + new CX-11/12 ─────

def test_cx02_runner_context_does_not_readmit_failed_tool_results():
    import json as _json

    from core import runner
    from core.answer_verifier import build_trace_summary, verify_answer

    failed = '{"success": false, "error": "' + ("x" * 2100) + ' 1315356426404373760"}'
    results = [{"output": failed}]
    agent = SimpleNamespace(get_conversation_history=lambda: [])
    ctx = runner._identifier_context(agent, "q", ["docs text", _json.dumps(results, default=str)], results)
    assert "1315356426404373760" not in ctx and "docs text" in ctx
    s = build_trace_summary(results, [{"name": "q", "ok": False, "output": failed[:2000]}])
    assert any(c.kind == "identifier" for c in verify_answer("id 1315356426404373760", s, user_text=ctx).unsupported)


def test_cx03_comma_inside_a_returned_url_does_not_end_it():
    from core.answer_verifier import verify_answer

    s = _summary('{"success": true, "note": "files: https://example.org/files/a.fits,version2 and '
                 'https://example.org/files/b.fits, https://example.org/files/c.fits"}')
    assert verify_answer("FITS https://example.org/files/a.fits", s).unsupported
    assert not verify_answer("FITS https://example.org/files/b.fits and https://example.org/files/c.fits", s).unsupported


def test_cx04_each_read_waits_at_most_a_quarter_of_the_link_budget():
    from services import datalink_resolve as dr

    seen = {}

    def get(url, timeout=None, stream=False):
        seen["timeout"], seen["stream"] = timeout, stream
        return SimpleNamespace(status_code=200, iter_content=lambda chunk_size: iter([_DATALINK_VOTABLE]),
                               close=lambda: seen.setdefault("closed", True))

    assert dr.resolve_this("https://h/caom2ops/datalink?ID=1", timeout=8.0,
                           session=SimpleNamespace(get=get)) == "https://example.org/files/jcmx06t8q_flc.fits"
    assert seen["timeout"] == 2.0 and seen["stream"] is True and seen["closed"] is True


def test_cx11_non_200_streamed_response_is_closed():
    from services import datalink_resolve as dr

    closed = []
    sess = SimpleNamespace(get=lambda url, timeout=None, stream=False: SimpleNamespace(
        status_code=503, close=lambda: closed.append(True)))
    assert dr.resolve_this("https://h/caom2ops/datalink?ID=1", timeout=8.0, session=sess) is None
    assert closed == [True]


def test_cx05_telescope_name_in_a_comment_does_not_trip_the_alma_guard():
    from services.adql_preflight import PreflightRejection, preflight

    ok = "SELECT obs_id FROM ivoa.obscore WHERE target_name = 'M87' -- instrument_name = 'VLA'"
    assert preflight("https://almascience.eso.org/tap", ok) == (ok, [])
    ok2 = "SELECT obs_id FROM ivoa.obscore WHERE target_name = 'M87' /* facility = 'GBT' */"
    assert preflight("https://almascience.eso.org/tap", ok2) == (ok2, [])
    with pytest.raises(PreflightRejection):
        preflight("https://almascience.eso.org/tap", "SELECT obs_id FROM ivoa.obscore WHERE facility_name = 'GBT' -- x")


def test_cx01_r2_band_fallback_without_server_side_band_makes_total_unknown():
    from capabilities.alma import SearchByPosition

    df = _capped_alma_frame()

    class _OldSvc:  # an older facade that cannot take band=
        def cone_search(self, ra, dec, radius, facility, max_results, **extra):
            if "band" in extra:
                raise TypeError("cone_search() got an unexpected keyword argument 'band'")
            return df.copy()

    ctx = CallContext(services={
        "search_service": _OldSvc(), "console_log": lambda *a: None,
        "set_last_search_results": lambda v: None, "set_last_run_result": lambda v: None,
        "alma_tap_provenance": {"query": None, "url": None},
    })
    cap = SearchByPosition()
    out = cap.run(cap.InputModel(ra=204.25, dec=-29.87, radius=0.1, max_results=4, band=3), ctx).to_native()
    assert out["total"] == "unknown" and out["count_is_lower_bound"] is True
    assert out["total_unfiltered_cone_rows"] == 913


def test_cx02_r2_conductor_trace_in_url_sources_is_pruned_of_failed_calls():
    import json as _json

    from core import runner

    trace = [{"name": "gaia", "ok": False, "output": '{"success": false, "source_id": 1315356426404373760}'},
             {"name": "simbad", "ok": True, "output": '{"success": true, "main_id": "M 87", "bibcode": "2009ApJS..182..543A"}'},
             {"name": "x", "output": '{"success": false, "obs_id": "jw01234001001_02101_00001_nrca1"}'}]
    agent = SimpleNamespace(get_conversation_history=lambda: [])
    ctx = runner._identifier_context(agent, "q", [_json.dumps(trace)], [])
    assert "1315356426404373760" not in ctx and "jw01234001001" not in ctx
    assert "2009ApJS..182..543A" in ctx
    assert runner._without_failed_records("plain RAG text 1315356426404373760") == "plain RAG text 1315356426404373760"


def test_cx04_r2_watchdog_closes_a_stalled_body():
    import threading
    import time as _t

    from services import datalink_resolve as dr

    released = threading.Event()

    def stalled(chunk_size):
        yield b"<?xml"
        released.wait(5.0)  # blocks until the watchdog closes the response
        raise ConnectionError("closed")

    sess = SimpleNamespace(get=lambda url, timeout=None, stream=False: SimpleNamespace(
        status_code=200, iter_content=stalled, close=released.set))
    t0 = _t.monotonic()
    assert dr.resolve_this("https://h/caom2ops/datalink?ID=1", timeout=0.4, session=sess) is None
    assert _t.monotonic() - t0 < 1.0  # bounded by the link timeout, not the 5 s stall


def test_cx13_compact_url_list_still_supports_each_url():
    from core.answer_verifier import verify_answer

    s = _summary('{"success": true, "note": "https://example.org/files/a.fits,https://example.org/files/b.fits"}')
    assert not verify_answer("FITS https://example.org/files/a.fits", s).unsupported
    assert not verify_answer("FITS https://example.org/files/b.fits", s).unsupported


def test_cx14_escaped_quote_with_dashes_does_not_hide_a_vla_filter():
    from services.adql_preflight import PreflightRejection, preflight

    with pytest.raises(PreflightRejection):
        preflight("https://almascience.eso.org/tap",
                  "SELECT obs_id FROM ivoa.obscore WHERE target_name = 'O''--' AND instrument_name = 'VLA'")
    ok = "SELECT obs_id FROM ivoa.obscore WHERE target_name = 'it''s VLA' -- facility = 'GBT'"
    assert preflight("https://almascience.eso.org/tap", ok) == (ok, [])


def test_cx01_r3_band_dropped_below_the_capability_is_detected_from_the_executed_cone():
    # The service silently dropped band (services/search._call_with_optional_kwargs):
    # the executed cone's recorded filters carry no band, so the total is unknown.
    df = _capped_alma_frame()
    df.attrs["quasar_cone"] = {"filters": {"science_only": True}}
    out = _alma_position(df, ra=204.25, dec=-29.87, radius=0.1, max_results=4, band=3)
    assert out["total"] == "unknown" and out["total_unfiltered_cone_rows"] == 913


def test_cx04_r3_watchdog_also_bounds_a_request_that_never_returns_headers():
    import threading
    import time as _t

    from services import datalink_resolve as dr

    closed = threading.Event()

    class _HangingSession:
        def get(self, url, timeout=None, stream=False):
            if closed.wait(5.0):  # the watchdog closes the session mid-request
                raise ConnectionError("session closed")
            raise AssertionError("watchdog never fired")

        def close(self):
            closed.set()

    t0 = _t.monotonic()
    assert dr.resolve_this("https://h/caom2ops/datalink?ID=1", timeout=0.4, session=_HangingSession()) is None
    assert _t.monotonic() - t0 < 1.0


def test_cx12_band_filter_keeps_the_exact_server_side_totals():
    df = _capped_alma_frame()
    df.attrs["quasar_cone"] = {"filters": {"band": 3}}  # the client executed the band
    df["band_list"] = ["3", "3", "3", "3"]
    out = _alma_position(df, ra=204.25, dec=-29.87, radius=0.1, max_results=4, band=3)
    assert out["total"] == 913 and out["total_mous"] == 174 and out["count_is_lower_bound"] is False
