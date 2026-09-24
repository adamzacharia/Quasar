"""Offline tests for the fixes to existing archive tools made while closing the
ArchiveBench dev-set gaps (2026-09): failures must not read as empty results,
the model must get values (not only counts), and four specific defects.

Targets and values here are generic examples, not benchmark questions.
"""
from __future__ import annotations

import pandas as pd
import pytest


# ── MAST ────────────────────────────────────────────────────────────────────

def test_mast_instrument_family_wildcard_and_explicit_mode():
    from integrations.mast_client import MASTClient

    assert MASTClient._instrument_criterion("nirspec") == "NIRSPEC*"
    assert MASTClient._instrument_criterion("NIRSPEC/IFU") == "NIRSPEC/IFU"
    assert MASTClient._instrument_criterion("WFC3*") == "WFC3*"
    assert abs(MASTClient._to_mjd("2000-01-01T12:00:00") - 51544.5) < 1e-6
    assert MASTClient._to_mjd(59000) == 59000.0


def test_mast_failed_frame_is_reported_as_failure_not_zero():
    from capabilities.archives import SearchMast, _frame_error
    from integrations.mast_client import MASTClient
    from capabilities.base import CallContext

    failed = MASTClient._failed("TimeoutError: MAST query timed out")
    assert _frame_error(failed) == "TimeoutError: MAST query timed out"

    class _M:
        def search_by_target(self, **kw):
            return failed.copy() if False else MASTClient._failed("resolver failed")

    state = {}
    ctx = CallContext(services={"mast_client": _M(), "set_last_search_results": lambda v: None,
                                "set_last_run_result": lambda v: state.update(lrr=v)})
    out = SearchMast().execute({"target_name": "Some Galaxy", "mission": "JWST"}, ctx).native
    assert out["success"] is False and "resolver failed" in out["error"]
    assert "not an empty result" in out["note"]


def test_mast_program_summary_lists_programs_dates_and_tess_sectors():
    from capabilities.archives import _mast_programs

    df = pd.DataFrame([
        {"project_code": "1234", "telescope": "JWST", "instrument_name": "MIRI/IMAGE", "filters": "F770W",
         "t_min": 59800.5, "t_exptime": 100.0, "t_obs_release": 59801.0, "target_name": "T", "proposal_pi": "Doe, J."},
        {"project_code": "1234", "telescope": "JWST", "instrument_name": "MIRI/IMAGE", "filters": "F1800W",
         "t_min": 59700.2, "t_exptime": 50.0, "t_obs_release": 59701.0, "target_name": "T", "proposal_pi": "Doe, J."},
        {"project_code": "G0", "telescope": "TESS", "instrument_name": "Photometer", "filters": "TESS",
         "t_min": 58600.0, "t_exptime": 120.0, "sequence_number": 11, "t_obs_release": 58700.0, "target_name": "S"},
        {"project_code": "G0", "telescope": "TESS", "instrument_name": "Photometer", "filters": "TESS",
         "t_min": 59600.0, "t_exptime": 20.0, "sequence_number": 38, "t_obs_release": 59700.0, "target_name": "S"},
    ])
    out = _mast_programs(df)
    progs = {p["program"]: p for p in out["programs"]}
    assert progs["1234"]["first_obs_date"] == "2022-05-01" and progs["1234"]["filters"] == ["F1800W", "F770W"]
    assert progs["1234"]["all_public"] is True and progs["1234"]["pi"] == "Doe, J."
    assert out["tess_sectors"] == [11, 38]
    assert out["tess_sectors_by_exptime"] == {"120s": [11], "20s": [38]}


# ── IRSA ────────────────────────────────────────────────────────────────────

def test_irsa_error_and_nearest_rows():
    from capabilities.archives import SearchIrsa, _nearest_rows
    from capabilities.base import CallContext
    from integrations import irsa_client

    bad = irsa_client._failed("DALQueryError: unknown table: foo")

    class _I:
        def search_by_position(self, **kw):
            return bad

    ctx = CallContext(services={"irsa_client": _I(), "set_last_search_results": lambda v: None,
                                "set_last_run_result": lambda v: None})
    out = SearchIrsa().execute({"ra": 10.0, "dec": 10.0, "catalog": "foo"}, ctx).native
    assert out["success"] is False and "unknown table" in out["error"]

    df = pd.DataFrame([{"s_ra": 10.01, "s_dec": 10.0, "w1mpro": 9.5}, {"s_ra": 10.0, "s_dec": 10.0, "w1mpro": 8.1}])
    df.attrs["center"] = (10.0, 10.0)
    rows = _nearest_rows(df, n=2)
    assert rows[0]["w1mpro"] == 8.1 and rows[0]["sep_arcsec"] == 0.0 and rows[1]["sep_arcsec"] > 30


# ── NED-D ───────────────────────────────────────────────────────────────────

def test_ned_picks_measurement_table_not_summary():
    from services.distance_service import _pick_ned_measurement_table

    summary = pd.DataFrame({"Statistic": ["Mean", "Median", "Std"], "Distance Modulus": [29.1, 29.2, 0.1]})
    measurements = pd.DataFrame({"Distance Modulus": [29.0 + i * 0.01 for i in range(12)],
                                 "Distance Modulus Error": [0.1] * 12, "Method": ["TRGB"] * 12,
                                 "Refcode": ["2020X"] * 12})
    assert _pick_ned_measurement_table([summary, measurements], "X") is measurements
    with pytest.raises(ValueError):
        _pick_ned_measurement_table([pd.DataFrame({"a": [1]})], "X")


# ── ADS fallback ────────────────────────────────────────────────────────────

def test_ads_heuristic_fields_instead_of_one_phrase():
    from integrations.ads_client import heuristic_ads_query as h

    q = h("How many refereed papers did Jane Q Doe publish in the Astronomical Journal between 2011 and 2013?")
    assert 'year:[2011 TO 2013]' in q and 'bibstem:"AJ"' in q and 'author:"Q Doe, Jane"' in q
    assert '"How many' not in q
    q2 = h("first detection of a molecule in a comet coma")
    assert q2 == "abs:(detection AND molecule AND comet AND coma)"
    assert h("") == "*:*"
    assert 'bibstem:"MNRAS"' in h("stellar streams MNRAS 2019") and "year:2019" in h("stellar streams MNRAS 2019")


# ── ALMA ────────────────────────────────────────────────────────────────────

def test_alma_observation_details_returns_pi_and_title():
    from services.search import SearchService

    frame = pd.DataFrame([{"proposal_id": "2099.1.00001.S", "member_ous_uid": "uid://A001/X1/X1",
                           "asdm_uid": "uid://A002/X1/X1", "pi_name": "Doe, Jane", "obs_title": "A survey",
                           "target_name": "T", "band_list": "6"}])

    class _Svc:
        def search(self, q):
            class _R:
                def to_table(self_inner):
                    class _T:
                        def to_pandas(self_t):
                            return frame
                    return _T()
            return _R()

    svc = SearchService.__new__(SearchService)
    svc.alminer_client = type("A", (), {"_get_tap_service": lambda self: _Svc()})()
    out = svc.get_observation_details("2099.1.00001.S")
    assert out["pi_names"] == ["Doe, Jane"] and out["project_titles"] == ["A survey"]


def test_alma_redshifted_line_cone_skips_category_filter():
    from services.alma_server_side import redshifted_line_projects_server_side

    seen = []

    def run_query(sql, **kw):
        seen.append(sql)
        return pd.DataFrame()

    try:
        redshifted_line_projects_server_side(run_query, rest_species=["CO(3-2)"], z_min=0.0, z_max=0.001, cycle=7,
                                             extra_where="1=CONTAINS(POINT('ICRS', s_ra, s_dec), CIRCLE('ICRS', 1, 2, 0.01))",
                                             skip_category=True)
    except Exception:
        pass
    assert seen, "no query issued"
    where = seen[0].split(" WHERE ", 1)[1]
    assert "CIRCLE('ICRS', 1, 2, 0.01)" in where and "scientific_category" not in where


# ── Data Lab ────────────────────────────────────────────────────────────────

def test_desi_default_cut_is_zcat_primary():
    from services import datalab_registry as reg

    merged, _ = reg.merge_default_quality_cuts("desi_dr1", "zpix", None)
    assert [c["column"] for c in merged] == ["zwarn", "zcat_primary"]


def test_cone_count_accepts_predicates():
    from services import datalab_query_builders as B

    preds = B.build_catalog_predicates("desi_dr1", "zpix", value_cuts=[{"column": "zwarn", "op": "=", "value": 0}])
    sql, meta = B.build_cone_count("desi_dr1", "zpix", ra=10.0, dec=10.0, radius_deg=0.5, predicates=preds)
    assert "COUNT(*)" in sql and "zwarn = 0" in sql and "q3c_radial_query" in sql
