"""Regression tests for the ALMA skill integration (report 2026-09-02, INT-1..INT-8).

Every fixture uses REAL archive-shaped strings (Pad:Antenna pairs, space
delimited band_list, U-joined frequency_support) — the previous suite passed
on fabricated shapes ('12m Array', comma-split bands) that never occur in
ivoa.obscore, which is how the defects survived CI.
"""

from __future__ import annotations

import io
import math

import pandas as pd
import pytest

from services import alma_science_queries as asq
from services.alma_science_queries import (
    aggregate_counts,
    alma_cone_adql,
    alma_cone_where,
    band_tokens,
    counts_note,
    cycle_project_prefixes,
    exclude_sunyaev,
    filter_band,
    infer_arrays,
    infer_arrays_from_schedblock,
    project_prefix_where,
    projects_with_array_combo,
    requested_bands,
    row_matches_band,
    select_obscore_query,
    solar_where,
    summarize_projects,
    bandwidth_switching_candidates,
    truncation_info,
    wavelength_overlap_where,
)


# ─────────────────────────────────────────────────────────────────────────────
# INT-1 · archive-query correctness
# ─────────────────────────────────────────────────────────────────────────────
def test_wavelength_overlap_where_uses_metres_and_correct_inequalities():
    where = wavelength_overlap_where(230.0, 240.0)
    # em_min is the SHORTEST wavelength (highest frequency): em_min <= c/nu_lo
    assert where.startswith("(em_min <= 0.0013034")
    assert "em_max >= 0.0012491" in where
    # swapped bounds are tolerated
    assert wavelength_overlap_where(240.0, 230.0) == where
    with pytest.raises(ValueError):
        wavelength_overlap_where(0.0, 10.0)


def test_footprint_cone_is_the_union_of_intersects_and_point_test():
    where = alma_cone_where(10.68, 41.27, 0.1)
    assert "INTERSECTS(CIRCLE('ICRS', 10.68000000, 41.27000000, 0.10000000), s_region) = 1" in where
    assert "CONTAINS(POINT('ICRS', s_ra, s_dec), CIRCLE('ICRS', 10.68000000, 41.27000000, 0.10000000)) = 1" in where
    assert " OR " in where
    # point-only fallback when a service rejects INTERSECTS
    point = alma_cone_where(10.68, 41.27, 0.1, footprint=False)
    assert "INTERSECTS" not in point and "CONTAINS(POINT" in point
    adql = alma_cone_adql(10.68, 41.27, 0.1, public=True, top=500)
    assert adql.startswith("SELECT TOP 500 ") and adql.endswith("AND data_rights = 'Public'")
    assert "s_region" in adql and "asdm_uid" in adql


def test_band_tokens_keep_band_to_band_rows():
    assert band_tokens("5 10") == ["5", "10"]
    assert band_tokens("BAND 6") == ["6"]
    assert band_tokens("6,7") == ["6", "7"]
    assert band_tokens("B7") == ["7"]
    assert requested_bands("6 and 7") == ["6", "7"]
    assert requested_bands([6, "Band 7"]) == ["6", "7"]
    assert requested_bands("band 12") == []  # out of range
    assert row_matches_band("5 10", ["10"]) and row_matches_band("5 10", [5])
    assert not row_matches_band("1", ["10"])   # '1' must not match Band 10
    assert not row_matches_band("10", ["1"])


def test_filter_band_retains_multi_band_rows():
    df = pd.DataFrame({"band_list": ["6", "5 10", "3", "BAND 10"]})
    assert filter_band(df, 10)["band_list"].tolist() == ["5 10", "BAND 10"]
    assert filter_band(df, "5")["band_list"].tolist() == ["5 10"]
    assert len(filter_band(df, None)) == 4


def test_cycle_predicates_cover_supplemental_and_ddt_periods():
    assert cycle_project_prefixes(7) == ["2019.1.", "2019.2.", "2019.A."]
    where = project_prefix_where(7)
    for code in ("2019.1.00001.S", "2019.2.00001.S", "2019.A.00001.T"):
        prefix = code[:7]
        assert f"proposal_id LIKE '{prefix}%'" in where
    assert project_prefix_where(0) == "proposal_id LIKE '2011.0.%'"          # Cycle 0 special case
    assert project_prefix_where(7, include_supplemental=False, include_ddt=False) == "proposal_id LIKE '2019.1.%'"
    assert asq.cycle_to_project_prefix(12) == "2025.1." and asq.cycle_to_project_prefix(13) == "2026.1."
    disclosure = asq.cycle_periods_disclosure(7)
    assert "2019.2" in disclosure and "DDT" in disclosure


def test_truncation_info_flags_a_capped_fetch():
    df = pd.DataFrame({"x": range(5000)})
    info = truncation_info(df, 5000)
    assert info["truncated"] is True and "INCOMPLETE" in info["warning"]
    assert truncation_info(df.head(10), 5000)["truncated"] is False
    assert truncation_info(None, 5000)["rows_fetched"] == 0


def test_solar_predicate_excludes_sunyaev_zeldovich():
    where = solar_where()
    assert "LOWER(scientific_category) = 'sun'" in where
    assert "NOT LIKE '%sunyaev%'" in where
    assert "LIKE '%sun%'" not in where  # the bare substring that matched Sunyaev
    assert "'%solar%'" not in where              # would match ALMA's 'Solar system' category
    assert "NOT LIKE '%solar system%'" in where
    df = pd.DataFrame([
        {"proposal_id": "2023.1.00001.S", "science_keyword": "Sunyaev-Zel'dovich effect", "obs_title": "SZ clusters",
         "scientific_category": "Cosmology", "target_name": "RXJ"},
        {"proposal_id": "2023.1.00002.S", "science_keyword": "The Sun", "obs_title": "Solar flares",
         "scientific_category": "Sun", "target_name": "Sun"},
        {"proposal_id": "2023.1.00003.S", "science_keyword": "Solar system - Comets", "obs_title": "Sun-grazing comet",
         "scientific_category": "Solar system", "target_name": "C/2023 A3"},
    ])
    kept = exclude_sunyaev(df)
    assert kept["proposal_id"].tolist() == ["2023.1.00002.S"]


def test_standard_projection_selects_grain_footprint_and_rights_columns():
    q = select_obscore_query("1=1")
    for column in ("asdm_uid", "group_ous_uid", "schedblock_name", "s_region", "is_mosaic",
                   "data_rights", "science_observation", "qa2_passed", "access_url",
                   "sensitivity_10kms", "cont_sensitivity_bandwidth", "t_min", "t_max"):
        assert column in q, column
    assert "SELECT TOP 5000" in q


# ─────────────────────────────────────────────────────────────────────────────
# INT-2 · MOUS/EB aggregation
# ─────────────────────────────────────────────────────────────────────────────
_GRAIN_DF = pd.DataFrame([
    # one MOUS observed in 3 EBs, two fields -> 6 rows
    {"proposal_id": "2021.1.00001.S", "member_ous_uid": "uid://A001/X1/X1", "asdm_uid": "uid://A002/X1/Xa",
     "target_name": "F1", "band_list": "6", "data_rights": "Public", "antenna_arrays": "A001:DV01 A002:DA41"},
    {"proposal_id": "2021.1.00001.S", "member_ous_uid": "uid://A001/X1/X1", "asdm_uid": "uid://A002/X1/Xa",
     "target_name": "F2", "band_list": "6", "data_rights": "Public", "antenna_arrays": "A001:DV01 A002:DA41"},
    {"proposal_id": "2021.1.00001.S", "member_ous_uid": "uid://A001/X1/X1", "asdm_uid": "uid://A002/X1/Xb",
     "target_name": "F1", "band_list": "6", "data_rights": "Public", "antenna_arrays": "A001:DV01 A002:DA41"},
    {"proposal_id": "2021.1.00001.S", "member_ous_uid": "uid://A001/X1/X1", "asdm_uid": "uid://A002/X1/Xc",
     "target_name": "F1", "band_list": "6", "data_rights": "Public", "antenna_arrays": "A001:DV01 A002:DA41"},
    {"proposal_id": "2021.1.00001.S", "member_ous_uid": "uid://A001/X1/X2", "asdm_uid": "uid://A002/X1/Xd",
     "target_name": "F1", "band_list": "6", "data_rights": "Proprietary", "antenna_arrays": "J501:CM01 J502:CM02"},
    {"proposal_id": "2021.1.00001.S", "member_ous_uid": "uid://A001/X1/X2", "asdm_uid": "uid://A002/X1/Xd",
     "target_name": "F2", "band_list": "6", "data_rights": "Proprietary", "antenna_arrays": "J501:CM01 J502:CM02"},
])


def test_aggregate_counts_reports_rows_mous_eb_projects_separately():
    counts = aggregate_counts(_GRAIN_DF)
    assert counts["rows"] == 6 and counts["n_mous"] == 2 and counts["n_eb"] == 4 and counts["n_projects"] == 1
    assert counts["n_public_rows"] == 4 and counts["n_proprietary_rows"] == 2
    note = counts_note(counts)
    assert "6 archive rows" in note and "2 datasets" in note and "4 execution blocks" in note
    assert "observations" not in note.split("do not call rows observations")[0]
    # columns absent -> None, never a fabricated count
    assert aggregate_counts(pd.DataFrame({"x": [1, 2]}))["n_mous"] is None


def test_summarize_projects_uses_the_right_grain_and_union_bands():
    df = _GRAIN_DF.copy()
    df.loc[4:, "band_list"] = "5 10"
    df["obs_release_date"] = ["2023-01-01T00:00:00", "2023-01-01T00:00:00", "2023-01-01T00:00:00",
                              "2023-01-01T00:00:00", "3000-01-01T00:00:00", "3000-01-01T00:00:00"]
    out = summarize_projects(df)
    row = out.iloc[0]
    assert row["rows"] == 6 and row["n_mous"] == 2 and row["n_eb"] == 4
    assert "observations" not in out.columns
    assert row["band_list"] == "5 6 10"                     # union of tokens, not an arbitrary row
    assert row["arrays"] == "12m, 7m"                        # detected across all rows
    assert row["obs_release_date_min"].startswith("2023") and row["obs_release_date_max"].startswith("3000")
    assert row["n_public_rows"] == 4 and row["n_proprietary_rows"] == 2


def test_bandwidth_switching_counts_each_mous_setup_once():
    support = "[216.90..218.88GHz,31250.00kHz,XX YY] U [218.90..220.88GHz,31250.00kHz,XX YY] U [230.00..231.90GHz,31250.00kHz,XX YY] U [232.00..233.90GHz,31250.00kHz,XX YY]"
    one = pd.DataFrame([{"proposal_id": "2023.1.00003.S", "member_ous_uid": "uid://A001/X1/X1",
                         "frequency": 225.0, "bandwidth": 7.5e9, "frequency_support": support, "target_name": "sci"}])
    two = pd.concat([one, one], ignore_index=True)  # same MOUS setup repeated per EB
    r1 = bandwidth_switching_candidates(one)
    r2 = bandwidth_switching_candidates(two)
    # identical setup twice must not double the SPW-interval evidence
    s1 = r1.iloc[0]["score"] if not r1.empty else 0
    s2 = r2.iloc[0]["score"] if not r2.empty else 0
    assert s1 == s2


# ─────────────────────────────────────────────────────────────────────────────
# INT-1/3 · array inference from live antenna_arrays strings (A-01)
# ─────────────────────────────────────────────────────────────────────────────
def test_infer_arrays_from_pad_antenna_pairs():
    assert infer_arrays("A004:DV07 A025:CM03 J505:PM03") == ["12m", "7m", "TP"]
    assert infer_arrays("A109:DV09 J504:DV02 J505:DV05") == ["12m"]
    assert infer_arrays("J501:CM01 J502:CM02 J503:CM03") == ["7m"]
    assert infer_arrays("N601:PM02 N602:PM03") == ["TP"]
    assert infer_arrays("") == []
    # prose fallback still recognised
    assert infer_arrays("12m Array") == ["12m"] and infer_arrays("Total Power") == ["TP"]
    assert infer_arrays_from_schedblock("NGC253_a_06_7M") == ["7m"]
    assert infer_arrays_from_schedblock("NGC253_a_06_TM1") == ["12m"]
    assert infer_arrays_from_schedblock("NGC253_a_06_TP") == ["TP"]


def test_projects_with_array_combo_on_live_shaped_rows():
    df = pd.DataFrame([
        {"proposal_id": "2023.1.00010.S", "antenna_arrays": "A001:DV01 A002:DA41", "schedblock_name": "X_TM1",
         "member_ous_uid": "uid://A001/X1/X1", "asdm_uid": "uid://A002/X1/X1"},
        {"proposal_id": "2023.1.00010.S", "antenna_arrays": "J501:CM01 J502:CM02", "schedblock_name": "X_7M",
         "member_ous_uid": "uid://A001/X1/X2", "asdm_uid": "uid://A002/X1/X2"},
        {"proposal_id": "2023.1.00010.S", "antenna_arrays": "N601:PM02", "schedblock_name": "X_TP",
         "member_ous_uid": "uid://A001/X1/X3", "asdm_uid": "uid://A002/X1/X3"},
        {"proposal_id": "2023.1.00011.S", "antenna_arrays": "A001:DV01", "schedblock_name": "Y_TM1",
         "member_ous_uid": "uid://A001/X2/X1", "asdm_uid": "uid://A002/X2/X1"},
    ])
    out = projects_with_array_combo(df, ["12m", "7m", "TP"])
    assert out["proposal_id"].tolist() == ["2023.1.00010.S"]
    assert out.iloc[0]["n_mous"] == 3 and out.iloc[0]["n_eb"] == 3 and out.iloc[0]["rows"] == 3
    assert "heuristic" in out.iloc[0]["array_inference"]


# ─────────────────────────────────────────────────────────────────────────────
# INT-1 · frequency coverage decided from frequency_support, not bandwidth/2
# ─────────────────────────────────────────────────────────────────────────────
def test_line_coverage_uses_spw_windows_not_aggregate_bandwidth():
    from integrations.alminer_client import ALminerClient

    client = ALminerClient()
    # 4 x 1.875 GHz SPWs at 216-218/218-220/230-232/232-234: bandwidth 7.5 GHz,
    # frequency 225 -> the contiguous idiom spans 221.25-228.75 (wrong).
    support = "[216.00..218.00GHz,31250.00kHz,XX YY] U [218.00..220.00GHz,31250.00kHz,XX YY] U [230.00..232.00GHz,31250.00kHz,XX YY] U [232.00..234.00GHz,31250.00kHz,XX YY]"
    df = pd.DataFrame([{"target_name": "G", "frequency": 225.0, "bandwidth": 7.5e9, "frequency_support": support}])
    # CO(2-1) at z=0.045 -> 220.61 GHz falls in the LSB/USB gap: NOT covered
    assert client.get_line_coverage(df, 230.538, z=0.045).empty
    # 233 GHz sits in the outer USB SPW: covered, with the SPW named
    hit = client.get_line_coverage(df, 233.0, z=0.0, line_name="X")
    assert len(hit) == 1
    assert hit.iloc[0]["covering_spw_ghz"] == "232.000-234.000 GHz"
    assert hit.iloc[0]["coverage_method"].startswith("frequency_support")
    # without frequency_support the fallback is labelled approximate
    approx = client.get_line_coverage(df.drop(columns=["frequency_support"]), 226.0)
    assert len(approx) == 1 and "APPROXIMATE" in approx.iloc[0]["coverage_method"]
    # no way to decide -> errored frame, never a silent "not covered"
    err = client.get_line_coverage(pd.DataFrame([{"target_name": "G"}]), 230.0)
    assert err.empty and "quasar_error" in err.attrs


def test_co_redshift_tool_prefilters_on_wavelength_and_verifies_spw(monkeypatch):
    import capabilities.alma as alma
    from tests.unit.test_alma_capability import _ctx, _fake_pyvo, _run

    support = "[216.00..218.00GHz,31250.00kHz,XX YY] U [232.00..234.00GHz,31250.00kHz,XX YY]"
    df = pd.DataFrame([
        {"target_name": "gap", "proposal_id": "2023.1.00001.S", "member_ous_uid": "uid://A001/X1/X1",
         "asdm_uid": "uid://A002/X1/X1", "frequency": 225.0, "bandwidth": 4e9, "frequency_support": support,
         "band_list": "6", "scientific_category": "Galaxy evolution", "science_keyword": "",
         "s_ra": 1.0, "s_dec": 2.0, "t_exptime": 10.0, "s_resolution": 0.5, "data_rights": "Public",
         "obs_release_date": "2024-01-01"},
    ])
    _fake_pyvo(df, monkeypatch)
    ctx, state = _ctx()
    # CO(2-1) at z=0.04-0.05 -> 219.6-221.7 GHz: in the gap -> row dropped as a prefilter false positive
    out = _run(alma.SearchAlmaCoInRedshiftRange(), ctx, z_min=0.04, z_max=0.05)
    assert out["success"] is True and out["count"] == 0 and out["prefilter_rows"] == 1
    assert "em_min <=" in _fake_pyvo.last_query and "0.5*bandwidth" not in _fake_pyvo.last_query
    # CO(2-1) at z=-0.01..0.0 is not physical; use z where 232-234 covers a CO line: CO(2-1)/(1+z) in [232,234]
    # -> z in [-0.015, -0.006]; instead test CO(3-2) 345.796/(1+z) in [232,234] -> z 0.478-0.49
    out = _run(alma.SearchAlmaCoInRedshiftRange(), ctx, z_min=0.478, z_max=0.490)
    assert out["success"] is True and out["count"] == 1
    assert "CO(3-2)" in out["results"][0]["CO_transitions_covered"]
    assert out["results"][0]["coverage_method"].startswith("frequency_support")
    assert out["counts"]["n_mous"] == 1 and out["counts"]["n_eb"] == 1


# ─────────────────────────────────────────────────────────────────────────────
# INT-5 · failure states preserved, advertised controls honoured
# ─────────────────────────────────────────────────────────────────────────────
def test_alminer_client_failures_are_tagged_not_empty(monkeypatch):
    from integrations import alminer_client as mod

    client = mod.ALminerClient()
    client.SOURCE_TIMEOUT_S = 5

    class _Boom:
        def search(self, q, *, maxrec):
            raise RuntimeError("TAP down")

    monkeypatch.setattr(client, "_get_tap_service", lambda: _Boom())
    monkeypatch.setattr(mod, "ALMINER_AVAILABLE", False)
    df = client._parallel_search(1.0, 2.0, radius=0.05)
    assert df.empty and "TAP down" in df.attrs["quasar_error"]

    # a real (empty) TAP answer is a genuine empty result, not an error
    class _Empty:
        def search(self, q, *, maxrec):
            class _R:
                def to_table(self):
                    return self

                def to_pandas(self):
                    return pd.DataFrame()
            return _R()

    monkeypatch.setattr(client, "_get_tap_service", lambda: _Empty())
    df = client._parallel_search(1.0, 2.0, radius=0.05, public=True, max_results=100)
    assert df.empty and "quasar_error" not in df.attrs
    assert df.attrs["quasar_cone"]["footprint_mode"] == "intersects_or_point"
    assert df.attrs["quasar_cone"]["public"] is True
    assert "data_rights = 'Public'" in df.attrs["quasar_cone"]["adql"]
    assert "TOP 100" in df.attrs["quasar_cone"]["adql"]


def test_frequency_search_uses_wavelength_overlap_top_and_public(monkeypatch):
    from integrations import alminer_client as mod

    client = mod.ALminerClient()
    captured = {}

    class _Svc:
        def search(self, q, *, maxrec):
            captured["q"] = q

            class _R:
                def to_table(self):
                    return self

                def to_pandas(self):
                    return pd.DataFrame([{"target_name": "x", "member_ous_uid": "uid://A001/X1/X1"}])
            return _R()

    monkeypatch.setattr(client, "_get_tap_service", lambda: _Svc())
    df = client.search_by_frequency(230.0, 240.0, public=True, max_results=50)
    assert "em_min <=" in captured["q"] and "em_max >=" in captured["q"]
    assert "frequency >=" not in captured["q"]
    assert "TOP 50" in captured["q"] and "data_rights = 'Public'" in captured["q"]
    assert df.attrs["truncated"] is False and df.attrs["quasar_adql"] == captured["q"]


def test_standardize_columns_keeps_access_url_and_sensitivity(monkeypatch):
    from integrations.alminer_client import ALminerClient

    df = pd.DataFrame([{"member_ous_uid": "uid://A001/X1/X1", "access_url": "https://almascience.nrao.edu/datalink/sync?ID=uid://A001/X1/X1",
                        "sensitivity_10kms": 1.5, "s_resolution": 0.3,
                        "frequency_support": "[216.0..218.0GHz,1kHz] U [232.0..234.0GHz,1kHz]"}])
    out = ALminerClient()._standardize_columns(df)
    assert out.iloc[0]["access_url"].startswith("https://almascience.nrao.edu/datalink")   # DataLink URL kept
    assert out.iloc[0]["archive_url"].startswith("https://almascience.nrao.edu/aq/")
    assert out.iloc[0]["sensitivity_10kms"] == 1.5 and out.iloc[0]["sensitivity"] == 1.5     # alias, not rename
    assert out.iloc[0]["s_resolution"] == 0.3 and out.iloc[0]["resolution"] == 0.3
    assert out.iloc[0]["freq_min_ghz"] == 216.0 and out.iloc[0]["freq_max_ghz"] == 234.0
    # the R2 sensitivity ranking survives standardisation (A-22)
    ranked = asq.summarize_sensitivity(out.assign(proposal_id="2019.1.00001.S"), "sensitivity_10kms")
    assert "best_sensitivity_10kms_mjy_beam" in ranked.columns


def test_search_by_target_honours_public_only_and_date_range():
    from capabilities.alma import SearchByTarget
    from tests.unit.test_alma_capability import _ctx, _run

    calls = []

    class _Svc:
        def search_by_target(self, name, facility=None, date_range=None, max_results=100, **kw):
            calls.append((name, date_range, kw))
            df = pd.DataFrame([{"target_name": name, "proposal_id": "2019.1.00001.S", "band_list": "5 10",
                                "member_ous_uid": "uid://A001/X1/X1", "asdm_uid": "uid://A002/X1/X1",
                                "access_url": "https://x/datalink", "data_rights": "Public",
                                "t_min": 58500.0, "t_max": 58501.0}])
            df.attrs["quasar_cone"] = {"ra": 1.0, "dec": 2.0, "radius_deg": 0.05, "public": bool(kw.get("public_only")),
                                       "footprint_mode": "intersects_or_point",
                                       "adql": "SELECT ... INTERSECTS(...)"}
            if date_range:
                df.attrs["date_range"] = {"requested": date_range, "applied": True}
            return df

    ctx, state = _ctx(search_service=_Svc())
    out = _run(SearchByTarget(), ctx, target_name="M87", public_only=True, band="10", date_range="2019")
    assert calls[0][2] == {"public_only": True} and calls[0][1] == "2019"
    assert out["success"] is True and out["total_results"] == 1      # band-to-band row kept for band 10
    assert out["public_only"] is True
    assert out["counts"] == {"rows": 1, "n_mous": 1, "n_eb": 1, "n_projects": 1, "n_public_rows": 1, "n_proprietary_rows": 0}
    assert "1 archive row" in out["note"] and "observations" not in out["note"].split("do not call")[0]
    assert any("more than one band" in w for w in out["warnings"])
    assert out["footprint_mode"] == "intersects_or_point"
    assert out["date_range"]["applied"] is True
    # provenance carries the EXACT executed cone (no more kind:"args" fallback for name search)
    assert ctx.service("alma_tap_provenance")["query"] == "SELECT ... INTERSECTS(...)"


def test_scan_intent_filter_tokenises_and_supports_exclusion():
    from capabilities.alma import filter_by_scan_intent

    df = pd.DataFrame([
        {"target_name": "sci", "scan_intent": "TARGET"},
        {"target_name": "sci-wvr", "scan_intent": "TARGET WVR"},
        {"target_name": "cal", "scan_intent": "BANDPASS FLUX WVR"},
        {"target_name": "phase", "scan_intent": "PHASE WVR"},
    ])
    kept, label = filter_by_scan_intent(df, "exclude calibrators")
    assert kept["target_name"].tolist() == ["sci"] and "excluded" in label
    kept, label = filter_by_scan_intent(df, "not PHASE")
    assert kept["target_name"].tolist() == ["sci", "sci-wvr", "cal"]
    same, label = filter_by_scan_intent(df, "something odd")
    assert label == "" and len(same) == 4 and "not understood" in same.attrs["scan_intent_warning"]


def test_observation_details_is_a_real_lookup(monkeypatch):
    from services.search import SearchService

    svc = SearchService.__new__(SearchService)
    captured = {}

    class _Svc:
        def search(self, q):
            captured["q"] = q

            class _R:
                def to_table(self):
                    return self

                def to_pandas(self):
                    return pd.DataFrame([
                        {"member_ous_uid": "uid://A001/X1/X1", "asdm_uid": "uid://A002/X1/Xa", "proposal_id": "2019.1.00001.S",
                         "target_name": "M87", "band_list": "6", "data_rights": "Public", "obs_release_date": "2021-01-01", "qa2_passed": "T"},
                        {"member_ous_uid": "uid://A001/X1/X1", "asdm_uid": "uid://A002/X1/Xb", "proposal_id": "2019.1.00001.S",
                         "target_name": "M87", "band_list": "6", "data_rights": "Public", "obs_release_date": "2021-01-01", "qa2_passed": "T"},
                    ])
            return _R()

    class _Client:
        def _get_tap_service(self):
            return _Svc()

    svc.alminer_client = _Client()
    out = svc.get_observation_details("uid://A001/X1/X1")
    assert out["success"] and out["found"] and out["matched_as"] == ["member_ous_uid"]
    assert out["counts"]["rows"] == 2 and out["counts"]["n_eb"] == 2 and out["counts"]["n_mous"] == 1
    assert "member_ous_uid = 'uid://A001/X1/X1'" in captured["q"]
    assert svc.get_observation_details("")["success"] is False


# ─────────────────────────────────────────────────────────────────────────────
# INT-3 · typed DataLink inventory + triage categories
# ─────────────────────────────────────────────────────────────────────────────
_VOT_HEAD = """<?xml version="1.0"?><VOTABLE version="1.4" xmlns="http://www.ivoa.net/xml/VOTable/v1.3">
<RESOURCE type="results"><INFO name="QUERY_STATUS" value="OK"/><TABLE>
<FIELD name="ID" datatype="char" arraysize="*"/><FIELD name="access_url" datatype="char" arraysize="*"/>
<FIELD name="service_def" datatype="char" arraysize="*"/><FIELD name="error_message" datatype="char" arraysize="*"/>
<FIELD name="semantics" datatype="char" arraysize="*"/><FIELD name="description" datatype="char" arraysize="*"/>
<FIELD name="content_type" datatype="char" arraysize="*"/><FIELD name="content_length" datatype="long"/>
<DATA><TABLEDATA>"""
_VOT_TAIL = "</TABLEDATA></DATA></TABLE></RESOURCE></VOTABLE>"


def _row(access="", service="", error="", semantics="#this", desc="", ctype="", length=""):
    return (f"<TR><TD>uid://A001/X1/X1</TD><TD>{access}</TD><TD>{service}</TD><TD>{error}</TD>"
            f"<TD>{semantics}</TD><TD>{desc}</TD><TD>{ctype}</TD><TD>{length}</TD></TR>")


def test_datalink_rows_are_typed_and_sizes_may_be_unknown():
    from integrations.datalink import DataLinkClient, partition_entries, size_summary

    xml = _VOT_HEAD + "".join([
        _row(access="https://almascience.nrao.edu/dataPortal/member.uid___A001_X1_X1.NGC_253_sci.spw25.cube.I.pbcor.fits",
             ctype="application/fits", length="1048576"),
        _row(access="https://almascience.nrao.edu/dataPortal/2019.1.00001.S_uid___A001_X1_X1_auxiliary.tar",
             semantics="#auxiliary", ctype="application/x-tar"),
        _row(service="soda-sync", semantics="#cutout"),
        _row(error="NotFoundFault: no such file", semantics="#this"),
        _row(access="https://almascience.nrao.edu/datalink/sync?ID=uid://A002/X1/X1", semantics="#progenitor",
             ctype="application/x-votable+xml;content=datalink"),
    ]) + _VOT_TAIL
    client = DataLinkClient.__new__(DataLinkClient)
    entries, partial, fault = client._parse_votable_response(xml)
    assert fault is None and partial is False
    groups = partition_entries(entries)
    assert len(groups["files"]) == 2 and len(groups["services"]) == 1
    assert len(groups["errors"]) == 1 and len(groups["nested"]) == 1
    fits = groups["files"][0]
    assert fits["size_mb"] == 1.0 and fits["size_known"] is True and fits["semantics"] == "#this"
    tar = groups["files"][1]
    assert tar["size_mb"] is None and tar["size_known"] is False        # unknown, not 0
    summary = size_summary(groups["files"])
    assert summary["n_size_unknown"] == 1 and summary["total_known_bytes"] == 1048576


def test_datalink_empty_and_not_found_are_distinct_states(monkeypatch):
    from integrations.datalink import DataLinkClient, STATE_EMPTY, STATE_NOT_FOUND, STATE_UNAVAILABLE

    client = DataLinkClient.__new__(DataLinkClient)
    client.alma = None
    empty_xml = _VOT_HEAD + _VOT_TAIL
    monkeypatch.setattr(client, "_list_via_http", lambda uid: (client._parse_votable_response(empty_xml)[0], False, None, []))
    out = client.list_files("uid___A001_X1_X1")
    assert out["success"] is True and out["state"] == STATE_EMPTY and out["total_files"] == 0
    assert out["mous_uid"] == "uid://A001/X1/X1"                          # sanitized form normalised
    assert "does NOT mean the UID is invalid" in out["message"]
    assert "authorization" in out["message"]

    fault_xml = ('<?xml version="1.0"?><VOTABLE><RESOURCE type="results"><INFO name="QUERY_STATUS" value="ERROR">'
                 'NotFoundFault: uid://A001/X9/X9 not found</INFO></RESOURCE></VOTABLE>')
    monkeypatch.setattr(client, "_list_via_http", lambda uid: (None, False, client._votable_fault(fault_xml), []))
    out = client.list_files("uid://A001/X9/X9")
    assert out["success"] is False and out["state"] == STATE_NOT_FOUND and "NotFoundFault" in out["error"]

    monkeypatch.setattr(client, "_list_via_http", lambda uid: (None, False, None, ["nrao: timeout", "eso: timeout"]))
    out = client.list_files("uid://A001/X9/X9")
    assert out["success"] is False and out["state"] == STATE_UNAVAILABLE
    assert "transport failure" in out["error"].lower() and "not evidence" in out["error"]


def test_datalink_uid_normalisation_accepts_pasted_forms():
    from integrations.datalink import DataLinkClient

    n = DataLinkClient._normalize_uid
    assert n("uid://A001/X1590/X30a8") == "uid://A001/X1590/X30a8"
    assert n("uid___A001_X1590_X30a8") == "uid://A001/X1590/X30a8"
    assert n("member.uid___A001_X1590_X30a8") == "uid://A001/X1590/X30a8"
    assert n("uid://A001/X1590/X30a8/") == "uid://A001/X1590/X30a8"
    assert n("2019.1.00001.S_uid___A001_X1590_X30a8_auxiliary.tar") == "uid://A001/X1590/X30a8"
    assert n("A001/X1590/X30a8") == "uid://A001/X1590/X30a8"


def test_product_classification_covers_roles_intents_and_tar_parts():
    from services.data_product_triage import (
        build_product_row, category_counts, classify_product, product_intent, product_rank, product_role,
    )

    files = [
        {"filename": "member.uid___A001_X1_X1.NGC_253_sci.spw25.cube.I.pbcor.fits", "size_mb": 300, "content_type": "application/fits"},
        {"filename": "member.uid___A001_X1_X1.J0423-0120_ph.spw25.mfs.I.pbcor.fits", "size_mb": 20, "content_type": "application/fits"},
        {"filename": "member.uid___A001_X1_X1.NGC_253_sci.spw25.cube.I.pb.fits.gz", "size_mb": None, "content_type": ""},
        {"filename": "2019.1.00001.S_uid___A001_X1_X1_001_of_003.tar", "size_mb": None},
        {"filename": "2019.1.00001.S_uid___A001_X1_X1_auxiliary.tar", "size_mb": 900},
        {"filename": "member.uid___A001_X1_X1.README.txt", "size_mb": 0.01},
        {"filename": "member.uid___A001_X1_X1.qa2_report.pdf", "size_mb": 2},
        {"filename": "uid___A002_X1_Xa.asdm.sdm.tar", "size_mb": 40000, "semantics": "#progenitor"},
    ]
    roles = category_counts(files)
    assert roles["science FITS"] == 2 and roles["primary beam"] == 1 and roles["product tar (split)"] == 1
    assert roles["auxiliary tar"] == 1 and roles["README"] == 1 and roles["QA report"] == 1
    assert roles["raw ASDM (per-EB, restore only)"] == 1
    assert product_intent(files[1]) == "phase calibrator" and product_intent(files[0]) == "science"
    assert classify_product(files[0]) == "primary-beam-corrected cube"
    assert classify_product(files[2]) == "primary beam FITS"
    ranked = sorted(files, key=product_rank)
    assert ranked[0]["filename"].endswith("cube.I.pbcor.fits")            # science pbcor first
    assert ranked[1]["filename"].endswith("cube.I.pb.fits.gz")            # science products before calibrator images
    assert "J0423-0120_ph" in ranked[2]["filename"]                        # calibrator (_ph) after science
    row = build_product_row(files[3], member_ous_uid="uid://A001/X1/X1")
    assert row["tar_part"] == 1 and row["tar_parts_total"] == 3 and row["size_mb"] == "" and row["size_known"] is False
    assert product_role(files[7]).startswith("raw ASDM")


# ─────────────────────────────────────────────────────────────────────────────
# INT-4 · QA2 three-state
# ─────────────────────────────────────────────────────────────────────────────
def test_qa2_flag_labels_never_invent_semipass():
    from services.alma_qa2 import qa2_label_from_flag

    assert qa2_label_from_flag("T") == "Pass" and qa2_label_from_flag(True) == "Pass"
    assert qa2_label_from_flag("F") == "Not Pass" and qa2_label_from_flag(False) == "Not Pass"
    assert qa2_label_from_flag(None) == "Unknown"
    assert "SemiPass" not in {qa2_label_from_flag(v) for v in ("T", "F", "0", "1", None)}


def test_data_card_qa2_normalizer_three_state():
    import importlib.util
    from pathlib import Path

    import sys

    pytest.importorskip("fastapi")
    ui_pro = Path(__file__).resolve().parents[2] / "ui-pro"
    if str(ui_pro) not in sys.path:
        sys.path.insert(0, str(ui_pro))
    path = ui_pro / "api" / "serializers" / "data_card.py"
    spec = importlib.util.spec_from_file_location("quasar_data_card_for_qa2", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    n = mod._normalize_qa2_value
    assert n("Fail") == "Fail" and n("QA2 FAILED") == "Fail"
    assert n("SemiPass") == "SemiPass" and n("Semi Pass") == "SemiPass"
    assert n("F") == "Not Pass" and n(False) == "Not Pass"
    assert n("T") == "Pass" and n("Pass") == "Pass"
    assert n("Fail") != "SemiPass" and n(False) != "SemiPass"


def test_qa2_cache_overlay_and_async_warm_are_non_blocking(monkeypatch, tmp_path):
    from services import alma_qa2

    cache = tmp_path / "qa2.sqlite3"
    alma_qa2._write_cached_statuses(
        [alma_qa2.QA2StatusResult(normalized_uid="A001_X12a3_X407", status="SemiPass", report_url="u")],
        cache_path=cache,
    )
    cached = alma_qa2.cached_statuses(["uid://A001/X12a3/X407", "uid://A001/X1/X1"], cache_path=cache)
    assert cached == {"A001_X12a3_X407": "SemiPass"}
    # under pytest the warm is disabled (no network) and reports 0 handed off
    assert alma_qa2.warm_qa2_cache_async(["uid://A001/X1/X1"], cache_path=cache) == 0


# ─────────────────────────────────────────────────────────────────────────────
# INT-7 · CASA generator: no unsafe ALMA manual recipe
# ─────────────────────────────────────────────────────────────────────────────
def test_casa_calibration_refuses_generic_alma_recipe_and_points_to_restore():
    from services.casa_generator import CASAScriptGenerator

    gen = CASAScriptGenerator()
    out = gen.generate_casa_calibration_script("NGC 253", "x.ms", "J1517-2422", "J0006-0623", "DA41")
    assert out["success"] is False and "scriptForPI" in out["error"]
    assert any("casa --pipeline -c scriptForPI.py" in step for step in out["restore_guidance"]["steps"])
    assert "Butler-JPL-Horizons" not in json_dumps(out)
    # forced bespoke recipe: a priori steps present, no solar-system standard on a quasar, calwt=True
    forced = gen.generate_casa_calibration_script("NGC 253", "x.ms", "J1517-2422", "J0006-0623", "DA41", force_manual=True)
    assert forced["success"] is True and "gencal(vis=vis, caltable='tsys.cal', caltype='tsys')" in forced["script"]
    assert "wvrgcal(" in forced["script"] and "caltype='antpos'" in forced["script"]
    assert "calwt=True" in forced["script"] and "flux_is_solar_system = False" in forced["script"]
    # VLA keeps a manual recipe with the catalogue standard
    vla = gen.generate_casa_calibration_script("3C48 field", "v.ms", "3C48", "J0137", "ea10", telescope="VLA")
    assert vla["success"] is True and "standard='Perley-Butler 2017'" in vla["script"]
    assert "standard='Butler-JPL-Horizons" not in vla["script"]
    img = gen.generate_casa_imaging_script("NGC 253", "calibrated/uid___A002_X1.ms")
    assert "flagdata(" not in img["script"] and "savemodel='none'" in img["script"]
    assert "scriptForPI" in img["prerequisite"] and img["restore_guidance"]["steps"]


def json_dumps(value):
    import json

    return json.dumps(value)


# ─────────────────────────────────────────────────────────────────────────────
# INT-8 · sensitivity calculator per Technical Handbook eq. 9.8 / 9.11
# ─────────────────────────────────────────────────────────────────────────────
def test_sensitivity_matches_handbook_equation_9_8_arithmetic():
    from services.astro_calculators import calculate_alma_sensitivity as f

    out = f(band=6, bandwidth_ghz=7.5, t_integration_s=60.0, n_antennas=43, n_polarizations=2, tsys_k=120.0)
    k = 1.380649e-23
    a_eff = 0.68 * 113.1
    expected_jy = 2 * k * 120.0 / (0.96 * 0.88 * a_eff * math.sqrt(43 * 42 * 2 * 7.5e9 * 60.0)) * 1e26
    assert out["continuum_sensitivity_jy_beam"] == pytest.approx(expected_jy, rel=1e-9)
    assert out["continuum_sensitivity_ujy_beam"] == pytest.approx(126.5, rel=1e-2)
    assert out["equation"].endswith("9.8") and out["n_baselines"] == 903
    # w_r and shadowing scale as eq. 9.8 says: w_r / (1 - f_s)
    scaled = f(band=6, n_antennas=43, tsys_k=120.0, robust_weighting_factor=1.2, shadowing_fraction=0.1)
    assert scaled["continuum_sensitivity_jy_beam"] == pytest.approx(expected_jy * 1.2 / 0.9, rel=1e-9)
    # 7-m array: 10 antennas, 38.5 m^2, Table 9.3 eta 0.69
    aca = f(band=6, array="7m", tsys_k=120.0)
    aca_expected = 2 * k * 120.0 / (0.96 * 0.88 * 0.69 * 38.5 * math.sqrt(10 * 9 * 2 * 7.5e9 * 60.0)) * 1e26
    assert aca["continuum_sensitivity_jy_beam"] == pytest.approx(aca_expected, rel=1e-9) and aca["n_antennas"] == 10
    # Total Power: eq. 9.11 with sqrt(N), N = 3, no shadowing term
    tp = f(band=6, array="TP", tsys_k=120.0, channel_width_khz=976.56)
    tp_expected = 2 * k * 120.0 / (0.96 * 0.88 * 0.68 * 113.1 * math.sqrt(3 * 2 * 976.56e3 * 60.0)) * 1e26
    assert tp["line_sensitivity_jy_beam"] == pytest.approx(tp_expected, rel=1e-9) and tp["equation"].endswith("9.11")
    # Bands 1 and 2 supported; the returned formula matches the arithmetic (no /2)
    assert f(band=1)["success"] and f(band=2)["success"] and f(band=2)["warnings"]
    assert "/2" not in out["formula"] and "N*(N-1)" in out["formula"]
    assert f(band=11)["success"] is False and f(band=6, array="ACA-x")["success"] is False


def test_beam_and_doppler_helpers_label_honestly():
    from services.astro_calculators import bands_for_frequency, calculate_beam, calculate_doppler_shift

    beam = calculate_beam(230.0, array_config="C-6")
    assert "approximate" in beam["formula"] and "diffraction limit" not in beam["note"].lower().replace("not a diffraction limit", "")
    # 0.574 lambda/L80 reproduces the Handbook's tabulated C-6 Band 6 resolution (~0.13").
    assert beam["l80_m"] == 1172.5 and beam["beam_l80_arcsec"] == pytest.approx(0.1316, abs=1e-3)
    assert bands_for_frequency(100.0) == [3, 2] and bands_for_frequency(120.0) == []   # gap + Band 3 preferred
    assert calculate_beam(120.0, max_baseline_m=1000)["alma_band"] is None
    dop = calculate_doppler_shift(rest_frequency_ghz=230.538, redshift=0.045)
    assert dop["observed_frequency_ghz"] == pytest.approx(230.538 / 1.045, rel=1e-9)
    assert dop["velocity_kms_by_convention"]["optical"] == pytest.approx(0.045 * 299792.458, rel=1e-9)
    back = calculate_doppler_shift(observed_frequency_ghz=dop["observed_frequency_ghz"], redshift=0.045)
    assert back["rest_frequency_ghz"] == pytest.approx(230.538, rel=1e-9)
    assert calculate_doppler_shift(rest_frequency_ghz=230.538)["success"] is False


# ─────────────────────────────────────────────────────────────────────────────
# INT-6 · knowledge kernel + profile + guidance tool
# ─────────────────────────────────────────────────────────────────────────────
def test_alma_kernel_is_generated_current_and_within_budget():
    import subprocess
    import sys
    from pathlib import Path

    from core.prompts import ALMA_TAP_SCHEMA
    from core.prompts.alma_kernel import ALMA_KERNEL_TOKEN_BUDGET

    for needle in ("member_ous_uid", "asdm_uid", "DataLink", "INTERSECTS", "bandwidth Hz", "METERS",
                   "qa2_passed", "scriptForPI", "browse_alma_guidance", "s_region"):
        assert needle in ALMA_TAP_SCHEMA, needle
    for wrong in ("Data download URL", "'F' for SEMIPASS", "cont_sens_bandwidth", "Total bandwidth (GHz)", "km/s"):
        assert wrong not in ALMA_TAP_SCHEMA, wrong
    # the name-resolver PRINCIPLE survives; its point-only example does not
    assert "string-match PI target_name" in ALMA_TAP_SCHEMA
    assert "SELECT * FROM ivoa.obscore\n  WHERE CONTAINS(POINT('ICRS', s_ra, s_dec)" not in ALMA_TAP_SCHEMA
    tiktoken = pytest.importorskip("tiktoken")
    enc = tiktoken.get_encoding("o200k_base")
    assert len(enc.encode(ALMA_TAP_SCHEMA)) <= ALMA_KERNEL_TOKEN_BUDGET
    repo = Path(__file__).resolve().parents[2]
    check = subprocess.run([sys.executable, str(repo / "scripts" / "gen_alma_kernel.py"), "--check"],
                           capture_output=True, text=True, cwd=str(repo))
    assert check.returncode == 0, check.stdout + check.stderr


def test_alma_profile_units_grain_and_ranked_pitfalls():
    from services import archive_profiles

    profile = archive_profiles.get_profile("alma")
    table = profile.tables["ivoa.obscore"]
    cols = {c.name: c for c in table.columns}
    assert cols["bandwidth"].unit == "Hz" and cols["velocity_resolution"].unit == "m/s"
    assert cols["em_min"].unit == "m" and cols["access_estsize"].unit == "kbyte"
    assert "member_ous_uid" in table.grain and "asdm_uid" in table.grain
    for name in ("asdm_uid", "group_ous_uid", "s_region", "is_mosaic", "data_rights", "obs_release_date",
                 "access_estsize", "schedblock_name", "antenna_arrays", "spatial_scale_max", "frequency_support"):
        assert name in cols, name
    assert "DataLink" in cols["access_url"].description and "download" in cols["access_url"].description.lower()
    assert "SEMIPASS" in cols["qa2_passed"].description and "'F' = SEMIPASS" not in cols["qa2_passed"].description
    ranked = profile.prompt_pitfalls()
    assert len(ranked) == 3
    assert "member_ous_uid" in ranked[0] and "asdm_uid" in ranked[0]
    assert "bandwidth Hz" in ranked[1] and "INTERSECTS" in ranked[1]
    assert "qa2_passed" in ranked[2] and "DataLink" in ranked[2]
    assert "cont_sens_bandwidth" not in str(profile.dump())


def test_browse_alma_guidance_serves_dated_sections_with_a_cap():
    from capabilities.alma_guidance import TOPICS, BrowseAlmaGuidance, load_topic, skill_available
    from capabilities.base import CallContext

    assert skill_available()
    for topic in TOPICS:
        payload = load_topic(topic)
        assert payload["review_date"] == "2026-07-18", topic
        assert payload["chars"] <= 6100 and payload["content"], topic
        assert payload["section_heading_drift"] is False, topic
    restore = load_topic("products-qa-restore")
    assert "scriptForPI" in restore["content"] and "calibrated MeasurementSet" in restore["content"]
    out = BrowseAlmaGuidance().run(BrowseAlmaGuidance.InputModel(topic="query-units-footprints"), CallContext(services={}))
    assert out.success and "INTERSECTS" in out.data["content"] and out.provenance.service == "alma_guidance"
    bad = BrowseAlmaGuidance().run(BrowseAlmaGuidance.InputModel(topic="nope"), CallContext(services={}))
    assert bad.success is False and "Unknown topic" in bad.error


def test_guidance_and_qa2_tools_are_registered_with_plain_schemas():
    from tests.unit.test_alma_capability import _wiring_agent

    agent = _wiring_agent()
    agent._register_tools()
    guidance = agent.tool_registry.get_tool("browse_alma_guidance")
    assert guidance is not None and guidance.parameters["required"] == ["topic"]
    assert "products-qa-restore" in guidance.parameters["properties"]["topic"]["enum"]
    qa2 = agent.tool_registry.get_tool("get_alma_qa2_status")
    assert qa2 is not None and qa2.parameters["required"] == ["mous_uid"]
    doppler = agent.tool_registry.get_tool("calculate_doppler_shift")
    assert doppler is not None
    sens = agent.tool_registry.get_tool("calculate_alma_sensitivity")
    assert "array" in sens.parameters["properties"] and "1-10" in sens.parameters["properties"]["band"]["description"]
    tgt = agent.tool_registry.get_tool("search_by_target")
    assert "data_rights" in tgt.parameters["properties"]["public_only"]["description"]
    files = agent.tool_registry.get_tool("list_alma_files")
    assert "EMPTY" in files.description and "calibrated MS" in files.description
    cal = agent.tool_registry.get_tool("generate_casa_calibration_script")
    assert "force_manual" in cal.parameters["properties"] and "scriptForPI" in cal.description


def test_vendored_skill_manifest_is_current():
    import subprocess
    import sys
    from pathlib import Path

    repo = Path(__file__).resolve().parents[2]
    check = subprocess.run([sys.executable, str(repo / "scripts" / "sync_alma_skill.py"), "--check"],
                           capture_output=True, text=True, cwd=str(repo))
    assert check.returncode == 0, check.stdout + check.stderr
