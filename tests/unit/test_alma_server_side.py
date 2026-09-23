"""services/alma_server_side.py — the ALMA named query types compute at the
server (UI benchmark 2026-09-22: D09, D10, D11, D21, D22 were TOP-5000 slivers
sorted by proposal_id).

ADQL string assertions per query type, recorded-response aggregation tests
with a query-aware fake TAP runner, completeness/partial flags, and the
capability wiring (server-side first, legacy pull as the disclosed fallback).
"""
from __future__ import annotations

import re
from typing import Callable, Dict, List

import pandas as pd
import pytest

from services import alma_server_side as ss
from services.alma_science_queries import LINE_REST_FREQ_GHZ


class _Recorder:
    """A ``run_query`` that answers by matching the ADQL against regex rules."""

    def __init__(self, rules: Dict[str, Callable[[str], pd.DataFrame]]):
        self.rules = rules
        self.queries: List[str] = []

    def __call__(self, query: str) -> pd.DataFrame:
        self.queries.append(query)
        for pattern, answer in self.rules.items():
            if re.search(pattern, query, re.S):
                return answer(query)
        raise AssertionError(f"no recorded response for query: {query[:200]}")


# ── ADQL shapes ──────────────────────────────────────────────────────────


def test_coverage_where_uses_an_indexable_range_and_exact_spw_arithmetic():
    w = ss.coverage_where(230.538)
    assert "frequency BETWEEN 229.3380 AND 231.7380" in w
    assert "(frequency - bandwidth/2.0e9) <= 230.538000" in w and "(frequency + bandwidth/2.0e9) >= 230.538000" in w


def test_solar_projects_aggregate_is_grouped_by_project_and_science_only():
    rec = _Recorder({
        r"GROUP BY proposal_id": lambda q: pd.DataFrame([{"proposal_id": "2023.1.01040.S", "n_mous": 7, "n_eb": 7, "n_rows": 40,
                                                          "target_name": "Sun", "pi_name": "White, Stephen", "obs_title": "Solar flares",
                                                          "band_list": "3", "obs_release_date_min": "2024-01-01", "obs_release_date_max": "2024-06-01"}]),
        r"SELECT DISTINCT proposal_id, target_name": lambda q: pd.DataFrame([
            {"proposal_id": "2023.1.01040.S", "target_name": "Sun", "band_list": "3", "antenna_arrays": "A001:DV01 J501:CM01 N601:PM02", "schedblock_name": "Sun_a_03_TM1", "data_rights": "Public"},
        ]),
    })
    out = ss.solar_projects_server_side(rec, 10)
    agg = rec.queries[0]
    assert "COUNT(DISTINCT member_ous_uid) AS n_mous" in agg and "GROUP BY proposal_id" in agg
    assert "science_observation = 'T'" in agg, "calibrators such as J0238+1636 must be excluded (D10)"
    assert "proposal_id LIKE '2023.1.%'" in agg and "sunyaev" in agg.lower()
    assert out.complete and out.unit == "projects" and out.n_units == 1
    row = out.frame.iloc[0]
    assert row["proposal_id"] == "2023.1.01040.S" and row["target_name"] == "Sun" and row["n_mous"] == 7
    assert "12m" in row["arrays"] and "7m" in row["arrays"] and "TP" in row["arrays"]
    assert "complete" in out.completeness_note()


def test_array_combo_intersects_one_distinct_query_per_array():
    def distinct(pattern_projects):
        return lambda q: pd.DataFrame({"proposal_id": pattern_projects})

    rec = _Recorder({
        r"antenna_arrays LIKE '%:DV%'": distinct(["2022.1.00005.S", "2022.1.00007.S", "2022.1.00012.S"]),
        r"antenna_arrays LIKE '%:CM%'": distinct(["2022.1.00007.S", "2022.1.00049.S"]),
        r"antenna_arrays LIKE '%:PM%'": distinct(["2022.1.00007.S", "2022.1.00005.S"]),
        r"GROUP BY proposal_id": lambda q: pd.DataFrame([
            {"proposal_id": p, "n_mous": i + 2, "n_eb": i + 3, "n_rows": 10 * (i + 1), "target_name": f"T{i}", "pi_name": "PI", "obs_title": "t", "band_list": "6"}
            for i, p in enumerate(["2022.1.00005.S", "2022.1.00007.S", "2022.1.00012.S", "2022.1.00049.S"])
        ]),
    })
    out = ss.array_combo_projects_server_side(rec, 9, ["12m", "7m", "TP"])
    assert len(rec.queries) == 4
    assert sum("SELECT DISTINCT proposal_id" in q for q in rec.queries) == 3
    assert all("(proposal_id LIKE '2022.1.%' OR proposal_id LIKE '2022.2.%' OR proposal_id LIKE '2022.A.%')" in q for q in rec.queries)
    assert list(out.frame["proposal_id"]) == ["2022.1.00007.S"], "only the project in ALL three sets"
    assert out.frame.iloc[0]["n_mous"] == 3
    assert out.extras["projects_in_window"] == 4 and out.extras["matching_projects"] == 1
    assert out.extras["per_array_project_counts"] == {"12m": 3, "7m": 2, "TP": 2}
    assert out.complete


def test_array_combo_rejects_unknown_arrays():
    with pytest.raises(ValueError):
        ss.array_combo_projects_server_side(_Recorder({}), 9, ["12m", "ngVLA"])


def test_line_set_runs_one_grouped_coverage_query_per_line_and_intersects_mous():
    def mous(rows):
        return lambda q: pd.DataFrame(rows)

    common = {"band_list": "6", "pi_name": "PI", "obs_title": "disks", "science_keyword": "Disks around low-mass stars"}
    rec = _Recorder({
        r"<= 230\.538000": mous([{"member_ous_uid": "uid://A001/X1/X1", "proposal_id": "2023.1.00001.S", "target_name": "Disk A", "n_spw_hits": 1, **common},
                                 {"member_ous_uid": "uid://A001/X2/X2", "proposal_id": "2023.1.00002.S", "target_name": "Disk B", "n_spw_hits": 1, **common}]),
        r"<= 220\.398684": mous([{"member_ous_uid": "uid://A001/X1/X1", "proposal_id": "2023.1.00001.S", "target_name": "Disk A", "n_spw_hits": 1, **common}]),
        r"<= 219\.560354": mous([{"member_ous_uid": "uid://A001/X1/X1", "proposal_id": "2023.1.00001.S", "target_name": "Disk A", "n_spw_hits": 1, **common}]),
    })
    out = ss.line_set_projects_server_side(rec, ["12CO", "13CO", "C18O"], band=6)
    assert len(rec.queries) == 3
    for q in rec.queries:
        assert "GROUP BY member_ous_uid, proposal_id" in q and "science_observation = 'T'" in q
        assert "band_list = '6'" in q and "frequency BETWEEN" in q and "bandwidth/2.0e9" in q
        assert "TOP" not in q.upper().split("FROM")[0], "no TOP-capped row pull"
    assert list(out.frame["proposal_id"]) == ["2023.1.00001.S"]
    assert out.frame.iloc[0]["n_mous_covering_all_lines"] == 1
    assert "12CO(2-1) 230.538" in out.frame.iloc[0]["rest_frequencies_ghz"]
    assert out.extras["rest_frequencies_ghz"]["C18O(2-1)"] == LINE_REST_FREQ_GHZ["C18O(2-1)"]
    assert out.extras["mous_per_line"] == {"12CO(2-1)": 2, "13CO(2-1)": 1, "C18O(2-1)": 1}
    assert out.complete and out.scanned == "whole archive"


def test_line_set_marks_partial_when_a_page_truncates():
    big = pd.DataFrame({"member_ous_uid": [f"uid://A001/X{i}/X1" for i in range(ss.AGG_MAXREC)],
                        "proposal_id": ["2023.1.00001.S"] * ss.AGG_MAXREC, "target_name": ["T"] * ss.AGG_MAXREC,
                        "band_list": ["6"] * ss.AGG_MAXREC, "pi_name": ["PI"] * ss.AGG_MAXREC, "obs_title": ["t"] * ss.AGG_MAXREC,
                        "science_keyword": [""] * ss.AGG_MAXREC, "n_spw_hits": [1] * ss.AGG_MAXREC})
    rec = _Recorder({r".": lambda q: big})
    out = ss.line_set_projects_server_side(rec, ["12CO", "13CO"], band=6)
    assert not out.complete and "partial" in out.completeness_note()


def test_redshifted_line_projects_scan_cycles_and_label_coverage_compatible_ranges(monkeypatch):
    monkeypatch.setattr(ss, "all_cycles", lambda newest_first=True: [10, 9])
    agg_rows = {
        "2023.1.": [{"proposal_id": "2023.1.00026.S", "n_mous": 73, "n_eb": 80, "n_rows": 500, "target_name": "SPT0311", "pi_name": "PI",
                     "obs_title": "high-z", "scientific_category": "Galaxy evolution", "science_keyword": "Starburst galaxies", "band_list": "3"}],
        "2022.1.": [{"proposal_id": "2022.1.00300.S", "n_mous": 2, "n_eb": 2, "n_rows": 8, "target_name": "Neptune-not", "pi_name": "PI",
                     "obs_title": "AGN", "scientific_category": "Active galaxies", "science_keyword": "AGN", "band_list": "6"}],
    }

    def agg(q):
        for prefix, rows in agg_rows.items():
            if f"proposal_id LIKE '{prefix}%'" in q:
                return pd.DataFrame(rows)
        return pd.DataFrame()

    detail = pd.DataFrame([
        # CO(2-1) 230.538 GHz observed at z=1.5 -> 92.2 GHz; SPW 91.2-93.2
        {"proposal_id": "2023.1.00026.S", "member_ous_uid": "uid://A001/X1/X1", "target_name": "SPT0311", "frequency": 92.2, "bandwidth": 2e9, "band_list": "3"},
        # CO(3-2) 345.796 at z=1.0 -> 172.9 GHz; SPW 172.0-174.0
        {"proposal_id": "2022.1.00300.S", "member_ous_uid": "uid://A001/X2/X1", "target_name": "AGN-1", "frequency": 173.0, "bandwidth": 2e9, "band_list": "5"},
    ])
    rec = _Recorder({r"GROUP BY proposal_id": agg, r"proposal_id IN \(": lambda q: detail})
    out = ss.redshifted_line_projects_server_side(rec, rest_species="CO", z_min=1.0, z_max=2.0)
    per_cycle = [q for q in rec.queries if "GROUP BY proposal_id" in q]
    assert len(per_cycle) == 2 and all("science_observation = 'T'" in q for q in per_cycle)
    assert all("galax" in q.lower() and "frequency BETWEEN" in q and "bandwidth/2.0e9" in q for q in per_cycle)
    assert out.complete and out.scanned == "cycles 9–10" and out.unit == "projects"
    top = out.frame.iloc[0]
    assert top["proposal_id"] == "2023.1.00026.S" and top["n_mous"] == 73
    assert "CO(2-1)" in top["transitions"]
    assert "coverage_compatible_z_range" in out.frame.columns and re.search(r"CO\(2-1\): z 1\.\d\d–1\.\d\d", top["coverage_compatible_z_range"])
    assert "not a measured redshift" in top["redshift_interpretation"] or "COULD detect" in top["redshift_interpretation"]
    assert out.extras["windows"][0]["transition"] == "CO(1-0)"


def test_redshifted_scan_stops_when_the_budget_runs_low(monkeypatch):
    # Cycles are scanned newest-first in concurrent batches of 4; once the
    # tool budget runs low the remaining (older) cycles are reported unscanned.
    monkeypatch.setattr(ss, "all_cycles", lambda newest_first=True: [10, 9, 8, 7, 6, 5])
    monkeypatch.setattr(ss, "_budget_allows", lambda min_seconds: False)
    rec = _Recorder({r"GROUP BY proposal_id": lambda q: pd.DataFrame([{"proposal_id": "2023.1.1.S", "n_mous": 1, "n_eb": 1, "n_rows": 1,
                                                                        "target_name": "t", "pi_name": "p", "obs_title": "o",
                                                                        "scientific_category": "Galaxy evolution", "science_keyword": "", "band_list": "3"}]),
                     r"proposal_id IN \(": lambda q: pd.DataFrame()})
    out = ss.redshifted_line_projects_server_side(rec, rest_species="CO", z_min=1, z_max=2)
    assert not out.complete
    assert out.unscanned == ["Cycle 5", "Cycle 6"] and "not scanned: Cycle 5, Cycle 6" in out.completeness_note()
    assert out.scanned == "cycles 7–10"
    assert sum("GROUP BY proposal_id" in q for q in rec.queries) == 4, "the first batch always runs"


def test_bandwidth_switching_uses_the_handbook_threshold_and_exact_aggregate(monkeypatch):
    monkeypatch.setattr(ss, "all_cycles", lambda newest_first=True: [9])
    narrow = "[216.90..217.13GHz,31250.00kHz,XX YY] U [218.90..219.13GHz,31250.00kHz,XX YY] U [229.50..229.73GHz,31250.00kHz,XX YY]"  # 3 x 234 MHz
    wide_sum = "[216.90..217.83GHz,31250.00kHz,XX YY] U [218.90..219.83GHz,31250.00kHz,XX YY]"  # 2 x 937.5 = 1875 MHz aggregate
    rec = _Recorder({
        r"HAVING MAX\(bandwidth\) < 937500000": lambda q: pd.DataFrame([
            {"member_ous_uid": "uid://A001/X1/X1", "proposal_id": "2022.1.00100.S", "n_spw": 3, "max_bw_hz": 234375000.0},
            {"member_ous_uid": "uid://A001/X2/X2", "proposal_id": "2022.1.00200.S", "n_spw": 2, "max_bw_hz": 937000000.0},
        ]),
        r"SELECT DISTINCT member_ous_uid, target_name": lambda q: pd.DataFrame([
            {"member_ous_uid": "uid://A001/X1/X1", "target_name": "G205", "band_list": "6", "pi_name": "PI", "obs_title": "narrow", "frequency_support": narrow},
            {"member_ous_uid": "uid://A001/X2/X2", "target_name": "wide", "band_list": "6", "pi_name": "PI", "obs_title": "wide", "frequency_support": wide_sum},
        ]),
    })
    out = ss.bandwidth_switching_server_side(rec)
    q = rec.queries[0]
    assert "GROUP BY member_ous_uid, proposal_id HAVING MAX(bandwidth) < 937500000" in q and "science_observation = 'T'" in q
    assert "MIN(target_name)" not in q, "string aggregates are fetched for the candidates only (4x faster scan)"
    assert any(dq.startswith("SELECT DISTINCT member_ous_uid, target_name") for dq in rec.queries)
    assert list(out.frame["proposal_id"]) == ["2022.1.00100.S"], "the 1875 MHz aggregate setup is NOT a BWSW candidate"
    row = out.frame.iloc[0]
    assert abs(row["min_aggregate_bandwidth_mhz"] - 690.0) < 1.0 and row["n_mous_narrow_setup"] == 1
    assert "937.5" in row["criterion"] and "Technical Handbook" in out.extras["citation"]
    assert out.complete and out.unit == "candidate MOUS" and out.n_units == 1


def test_bandwidth_switching_single_cycle_scans_only_that_cycle():
    rec = _Recorder({r"HAVING": lambda q: pd.DataFrame()})
    out = ss.bandwidth_switching_server_side(rec, cycle=10)
    assert len(rec.queries) == 1 and "proposal_id LIKE '2023.1.%'" in rec.queries[0]
    assert out.complete and out.scanned == "Cycle 10" and out.frame.empty


# ── capability wiring ────────────────────────────────────────────────────


class _QueryAwareTap:
    def __init__(self, answer: Callable[[str], pd.DataFrame]):
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


def _capability_ctx(answer):
    from tests.unit.test_alma_capability import _FakeSearchService, _ctx

    svc = _FakeSearchService()
    tap = _QueryAwareTap(answer)
    svc.alminer_client = type("_Alminer", (), {"_get_tap_service": lambda self: tap, "_standardize_columns": lambda self, df: df})()
    ctx, state = _ctx(search_service=svc)
    return ctx, state, tap


def test_capability_solar_uses_the_server_side_aggregate_and_reports_completeness():
    from capabilities.alma import QueryAlmaScienceArchive
    from tests.unit.test_alma_capability import _run

    def answer(q):
        if "GROUP BY proposal_id" in q:
            return pd.DataFrame([{"proposal_id": "2023.1.01040.S", "n_mous": 7, "n_eb": 7, "n_rows": 40, "target_name": "Sun",
                                  "pi_name": "White, Stephen", "obs_title": "Solar", "band_list": "3",
                                  "obs_release_date_min": "2024-01-01", "obs_release_date_max": "2024-02-01"}])
        return pd.DataFrame([{"proposal_id": "2023.1.01040.S", "target_name": "Sun", "band_list": "3",
                              "antenna_arrays": "A001:DV01 J501:CM01 N601:PM02", "schedblock_name": "x", "data_rights": "Public"}])

    ctx, state, tap = _capability_ctx(answer)
    out = _run(QueryAlmaScienceArchive(), ctx, query_type="cycle_solar_projects", cycle=10)
    assert out["success"] is True and out["computation"] == "server-side" and out["status"] == "ok"
    assert out["unique_projects"] == 1 and out["results"][0]["target_name"] == "Sun"
    assert "GROUP BY proposal_id" in out["provenance"]["adql"] and "science_observation = 'T'" in out["provenance"]["adql"]
    assert out["counts"]["n_mous"] == 7 and out["counts"]["computed"] == "server-side aggregate"
    assert any("Completeness: complete" in w for w in out["warnings"])
    assert "1 project(s)" in out["headline_count_note"]
    assert state.last_run_result["tool_name"] == "query_alma_science_archive"


def test_capability_array_combo_headline_is_the_matching_count_not_the_window_count():
    from capabilities.alma import QueryAlmaScienceArchive
    from tests.unit.test_alma_capability import _run

    def answer(q):
        if "SELECT DISTINCT proposal_id" in q:
            if ":DV%" in q:
                return pd.DataFrame({"proposal_id": ["A", "B", "C"]})
            if ":CM%" in q:
                return pd.DataFrame({"proposal_id": ["B", "C"]})
            return pd.DataFrame({"proposal_id": ["C"]})
        return pd.DataFrame([{"proposal_id": p, "n_mous": 2, "n_eb": 2, "n_rows": 5, "target_name": "t", "pi_name": "p", "obs_title": "o", "band_list": "6"} for p in "ABCDEFGH"])

    ctx, _, _ = _capability_ctx(answer)
    out = _run(QueryAlmaScienceArchive(), ctx, query_type="cycle_array_combo_projects", cycle=9)
    assert out["unique_projects"] == 1 and out["n_projects_in_window"] == 8
    assert out["results"][0]["proposal_id"] == "C"
    assert "1 project(s) used ALL requested arrays; 8 projects exist in the cycle window" in out["headline_count_note"]


def test_capability_falls_back_to_the_row_pull_when_the_server_rejects_aggregates(monkeypatch):
    import capabilities.alma as alma
    from capabilities.alma import QueryAlmaScienceArchive
    from tests.unit.test_alma_capability import _run

    def answer(q):
        raise RuntimeError("ADQL: GROUP BY not supported")

    ctx, _, _ = _capability_ctx(answer)
    fallback_rows = pd.DataFrame([{"proposal_id": "2023.1.01040.S", "target_name": "Sun", "member_ous_uid": "uid://A001/X1/X1",
                                   "asdm_uid": "uid://A002/X1/X1", "antenna_arrays": "A001:DV01", "schedblock_name": "s", "band_list": "3",
                                   "pi_name": "PI", "obs_title": "t", "scientific_category": "Sun", "science_keyword": "Sun"}])
    monkeypatch.setattr(alma, "_tap_obscore_dataframe", lambda where, *, max_results=5000, order_by="proposal_id", ctx: fallback_rows.copy())
    out = _run(QueryAlmaScienceArchive(), ctx, query_type="cycle_solar_projects", cycle=10)
    assert out["success"] is True and out["unique_projects"] == 1
    assert "computation" not in out
    assert any("Server-side aggregation failed" in w and "fell back" in w for w in out["warnings"])


def test_search_by_target_filters_are_in_the_adql_with_a_count_companion(monkeypatch):
    from integrations import alminer_client as mod
    from services.alma_science_queries import alma_cone_adql, alma_cone_count_adql

    q = alma_cone_adql(204.25, -29.87, 1 / 60, public=False, top=500, band=[6], max_resolution_arcsec=1.0, science_only=True)
    assert "band_list = '6'" in q and "spatial_resolution <= 1" in q and "science_observation = 'T'" in q and q.startswith("SELECT TOP 500 ")
    c = alma_cone_count_adql(204.25, -29.87, 1 / 60, public=False, band=[6])
    assert c.startswith("SELECT COUNT(*) AS total_rows, COUNT(DISTINCT member_ous_uid) AS total_mous FROM ivoa.obscore WHERE")
    assert "band_list = '6'" in c

    client = mod.ALminerClient()
    client.SOURCE_TIMEOUT_S = 5
    seen: List[str] = []

    class _Tap:
        def search(self, query, *, maxrec):
            seen.append(query)

            class _Res:
                quasar_tap_url = "https://almascience.nrao.edu/tap"
                query_status = "OVERFLOW" if "TOP" in query else "OK"

                def to_table(self_inner):
                    return self_inner

                def to_pandas(self_inner):
                    if query.startswith("SELECT COUNT(*)"):
                        return pd.DataFrame([{"total_rows": 1234, "total_mous": 56}])
                    return pd.DataFrame([{"proposal_id": "2019.1.00001.S", "band_list": "6", "member_ous_uid": "uid://A001/X1/X1",
                                          "target_name": "M83", "s_ra": 204.25, "s_dec": -29.87}] * 3)

            return _Res()

    monkeypatch.setattr(client, "_get_tap_service", lambda: _Tap())
    monkeypatch.setattr(mod, "ALMINER_AVAILABLE", False)
    df = client._parallel_search(204.25, -29.87, radius=1 / 60, public=False, max_results=3, band=[6], max_resolution_arcsec=1.0)
    assert len(seen) == 2 and "band_list = '6'" in seen[0] and seen[1].startswith("SELECT COUNT(*)")
    assert df.attrs["truncated"] is True and df.attrs["total_count"] == 1234 and df.attrs["total_mous"] == 56
    assert df.attrs["filters_in_adql"] == {"band": [6], "max_resolution_arcsec": 1.0}


def test_fine_structure_and_dense_gas_species_resolve_and_unknown_species_never_become_co():
    """Guard CX-29: [C II] & co. used to fall back silently to the CO ladder
    while the answer kept the [C II] label."""
    from services.alma_science_queries import UnsupportedSpecies, line_names_for_species, redshifted_line_windows

    assert line_names_for_species("[C II]") == line_names_for_species("CII") == ["[CII]158um"]
    assert line_names_for_species("[OIII]") == ["[OIII]88um", "[OIII]52um"]
    assert line_names_for_species("HCN") == ["HCN(1-0)", "HCN(2-1)", "HCN(3-2)", "HCN(4-3)"]
    assert line_names_for_species("")[0] == "CO(1-0)"  # no species -> documented CO default
    w = redshifted_line_windows("[CII]", 4.0, 6.0)
    assert len(w) == 1 and 271.0 < w[0]["observed_min_ghz"] < 272.0 and 380.0 < w[0]["observed_max_ghz"] < 381.0  # Band 7
    with pytest.raises(UnsupportedSpecies) as info:
        line_names_for_species("HeII")
    assert "NOT replaced by CO" in str(info.value)
    with pytest.raises(UnsupportedSpecies):
        ss.redshift_windows("HeII", 1.0, 2.0)
