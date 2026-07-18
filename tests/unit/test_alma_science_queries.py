import pandas as pd

from services.alma_science_queries import (
    annotate_line_coverage,
    bandwidth_switching_candidates,
    cycle_to_project_prefix,
    line_names_for_input,
    normalize_target_alias,
    parse_frequency_support_intervals,
    projects_covering_all_lines,
    projects_with_array_combo,
    redshifted_line_projects,
)


def test_cycle_to_project_prefix_maps_current_cycles():
    assert cycle_to_project_prefix(9) == "2022.1."
    assert cycle_to_project_prefix(10) == "2023.1."


def test_cycle_to_project_prefix_irregular_pre_cycle8_mapping():
    # alma-cycle-prefix-wrong-pre-cycle8: explicit table, not year=cycle+2013.
    expected = {
        0: "2011.0.",  # Early Science call uses the .0. suffix
        1: "2012.1.",
        2: "2013.1.",
        3: "2015.1.",  # Cycle 2 spanned longer: no 2014 call
        4: "2016.1.",
        5: "2017.1.",
        6: "2018.1.",
        7: "2019.1.",
        8: "2021.1.",  # COVID-cancelled 2020 call: no 2020 proposal year
        9: "2022.1.",
        10: "2023.1.",
        11: "2024.1.",
        12: "2025.1.",  # linear extrapolation resumes from Cycle 8 onward
    }
    for cycle, prefix in expected.items():
        assert cycle_to_project_prefix(cycle) == prefix, cycle
    # 2014 and 2020 were never ALMA proposal years.
    all_prefixes = {cycle_to_project_prefix(c) for c in range(0, 31)}
    assert not any(p.startswith(("2014.", "2020.")) for p in all_prefixes)


def test_line_names_for_input_is_case_insensitive():
    # line-name-case-sensitive-silent-drop: lower/mixed-case names with an
    # explicit transition must resolve, not silently drop.
    assert line_names_for_input(["co(1-0)"]) == ["CO(1-0)"]
    assert line_names_for_input(["C18o(2-1)"]) == ["C18O(2-1)"]
    assert line_names_for_input(["13co(3-2)"]) == ["13CO(3-2)"]
    assert line_names_for_input(["co"]) == ["CO(2-1)"]  # alias default


def test_projects_with_array_combo_groups_by_project():
    df = pd.DataFrame([
        {"proposal_id": "2022.1.00001.S", "antenna_arrays": "12m Array", "target_name": "A"},
        {"proposal_id": "2022.1.00001.S", "antenna_arrays": "7m ACA", "target_name": "A"},
        {"proposal_id": "2022.1.00001.S", "antenna_arrays": "Total Power", "target_name": "A"},
        {"proposal_id": "2022.1.00002.S", "antenna_arrays": "12m Array", "target_name": "B"},
    ])

    result = projects_with_array_combo(df, ["12m", "7m", "TP"])

    assert list(result["proposal_id"]) == ["2022.1.00001.S"]
    assert "TP" in result.iloc[0]["arrays_found"]


def test_line_coverage_and_project_line_set():
    df = pd.DataFrame([
        {"proposal_id": "2023.1.00001.S", "target_name": "Disk A", "frequency": 230.538, "bandwidth": 2e9},
        {"proposal_id": "2023.1.00001.S", "target_name": "Disk A", "frequency": 220.399, "bandwidth": 2e9},
        {"proposal_id": "2023.1.00001.S", "target_name": "Disk A", "frequency": 219.560, "bandwidth": 2e9},
        {"proposal_id": "2023.1.00002.S", "target_name": "Disk B", "frequency": 230.538, "bandwidth": 2e9},
    ])

    annotated = annotate_line_coverage(df, ["12CO", "13CO", "C18O"])
    grouped = projects_covering_all_lines(df, ["12CO", "13CO", "C18O"])

    assert set(line_names_for_input(["12CO", "13CO", "C18O"])) == {"12CO(2-1)", "13CO(2-1)", "C18O(2-1)"}
    assert len(annotated) == 4
    assert list(grouped["proposal_id"]) == ["2023.1.00001.S"]
    assert "C18O(2-1)" in grouped.iloc[0]["covered_lines"]


def test_frequency_support_parser_drives_line_coverage():
    df = pd.DataFrame([
        {
            "proposal_id": "2024.1.00001.S",
            "target_name": "Disk A",
            "frequency_support": "spw0 218.000..221.000GHz; spw1 229.900..231.000GHz",
            "frequency": 100.0,
            "bandwidth": 1e6,
        },
        {
            "proposal_id": "2024.1.00001.S",
            "target_name": "Disk A",
            "frequency_support": "spw2 219500..219700MHz",
        },
    ])

    intervals = parse_frequency_support_intervals(df.iloc[0]["frequency_support"])
    annotated = annotate_line_coverage(df, ["12CO", "13CO", "C18O"])
    grouped = projects_covering_all_lines(df, ["12CO", "13CO", "C18O"])

    assert intervals == [(218.0, 221.0), (229.9, 231.0)]
    assert len(annotated) == 2
    assert list(grouped["proposal_id"]) == ["2024.1.00001.S"]


def test_redshifted_co_projects_summarize_project_hits():
    df = pd.DataFrame([
        {
            "proposal_id": "2023.1.00010.S",
            "target_name": "z galaxy",
            "frequency_support": "172.8..173.1GHz",
            "band_list": "5",
        },
        {
            "proposal_id": "2023.1.00011.S",
            "target_name": "foreground",
            "frequency_support": "30.0..31.0GHz",
        },
    ])

    result = redshifted_line_projects(df, rest_species="CO", z_min=1.0, z_max=2.0)

    assert list(result["proposal_id"]) == ["2023.1.00010.S"]
    assert "CO(3-2)" in result.iloc[0]["transitions"]
    assert "1." in result.iloc[0]["inferred_redshift_ranges"]


def test_hh212_alias_normalization():
    assert normalize_target_alias("HH212") == "HH 212"
    assert normalize_target_alias("HH 212") == "HH 212"


def test_bandwidth_switching_candidates_rank_diagnostic_rows():
    rows = []
    for idx, freq in enumerate([90, 92, 94, 96, 98, 100, 102, 104, 106, 108, 110, 112]):
        rows.append({
            "proposal_id": "2023.1.00003.S",
            "frequency": freq,
            "bandwidth": [1e9, 2e9, 4e9][idx % 3],
            "frequency_support": f"spw-{idx % 4}",
            "target_name": "cal-target",
        })
    rows.append({
        "proposal_id": "2023.1.00004.S",
        "frequency": 230,
        "bandwidth": 2e9,
        "frequency_support": "simple",
        "target_name": "science",
    })

    result = bandwidth_switching_candidates(pd.DataFrame(rows))

    assert list(result["proposal_id"]) == ["2023.1.00003.S"]
    assert result.iloc[0]["confidence"] == "likely"
    assert "diagnostic" in result.iloc[0]["diagnostic_warning"].lower()


def test_bandwidth_switching_no_signal_returns_empty():
    df = pd.DataFrame([
        {
            "proposal_id": "2023.1.00005.S",
            "frequency": 230.0,
            "bandwidth": 2e9,
            "frequency_support": "229.0..231.0GHz",
            "target_name": "science",
        }
    ])

    assert bandwidth_switching_candidates(df).empty


# ─────────────────────────────────────────────────────────────────────────────
# R2 — sensitivity-driven discovery + archive↔literature join helpers
# ─────────────────────────────────────────────────────────────────────────────
from services.alma_science_queries import (  # noqa: E402
    band_token_where,
    collect_publications,
    looks_like_bibcode,
    publication_join_where,
    select_obscore_query_extended,
    sensitivity_where,
    split_bibcodes,
    summarize_publication_links,
    summarize_sensitivity,
)


def test_looks_like_bibcode_accepts_standard_forms():
    assert looks_like_bibcode("2018ApJ...869L..41A")
    assert looks_like_bibcode("2024A&A...685A...1A")
    assert not looks_like_bibcode("2019.1.00123.S")
    assert not looks_like_bibcode("uid://A001/X1465/X9c6")
    assert not looks_like_bibcode("")
    assert not looks_like_bibcode("2018ApJ...869L..41")   # 18 chars


def test_split_bibcodes_parses_space_separated_blob():
    blob = "2024A&A...685A...1A 2024A&A...688A..55M not-a-bibcode 2024A&A...685A...1A"
    assert split_bibcodes(blob) == ["2024A&A...685A...1A", "2024A&A...688A..55M"]
    assert split_bibcodes(None) == []


def test_band_token_where_matches_exact_tokens_only():
    where = band_token_where(1)
    assert "band_list = '1'" in where
    assert "LIKE '1 %'" in where and "LIKE '% 1'" in where and "LIKE '% 1 %'" in where
    # No bare substring form that would also match Band 10.
    assert "LIKE '%1%'" not in where


def test_sensitivity_where_line_and_continuum():
    where, col_name = sensitivity_where(0.5)
    assert col_name == "sensitivity_10kms"
    assert "sensitivity_10kms <= 0.5" in where and "sensitivity_10kms > 0" in where

    where_c, col_c = sensitivity_where(1.5, continuum=True, band=6, science_category="Disks")
    assert col_c == "cont_sensitivity_bandwidth"
    assert "cont_sensitivity_bandwidth <= 1.5" in where_c
    assert "band_list = '6'" in where_c
    assert "scientific_category" in where_c and "disks" in where_c


def test_sensitivity_where_rejects_nonpositive():
    import pytest
    with pytest.raises(ValueError):
        sensitivity_where(0)
    with pytest.raises(ValueError):
        sensitivity_where(-1)


def test_publication_join_where_classifies_identifiers():
    w, kind = publication_join_where("2019.1.00123.S")
    assert kind == "project_code" and w == "proposal_id = '2019.1.00123.S'"
    w, kind = publication_join_where("uid://A001/X1465/X9c6")
    assert kind == "mous_uid" and w == "member_ous_uid = 'uid://A001/X1465/X9c6'"
    w, kind = publication_join_where("2018ApJ...869L..41A")
    assert kind == "bibcode" and w == "bib_reference LIKE '%2018ApJ...869L..41A%'"


def test_publication_join_where_rejects_unknown():
    import pytest
    with pytest.raises(ValueError):
        publication_join_where("HL Tau")
    with pytest.raises(ValueError):
        publication_join_where("")


def test_select_obscore_query_extended_appends_columns():
    q = select_obscore_query_extended("proposal_id = 'X'",
                                      extra_columns=("bib_reference", "pub_title"))
    assert "bib_reference, pub_title" in q
    assert "FROM ivoa.obscore" in q and "proposal_id = 'X'" in q
    # Base columns still present.
    assert "member_ous_uid" in q and "target_name" in q


def test_summarize_sensitivity_best_per_project_sorted():
    df = pd.DataFrame([
        {"proposal_id": "A", "target_name": "t1", "band_list": "6",
         "sensitivity_10kms": 0.4},
        {"proposal_id": "A", "target_name": "t1", "band_list": "6",
         "sensitivity_10kms": 0.2},
        {"proposal_id": "B", "target_name": "t2", "band_list": "7",
         "sensitivity_10kms": 0.1},
    ])
    out = summarize_sensitivity(df, "sensitivity_10kms")
    assert list(out["proposal_id"]) == ["B", "A"]
    row_a = out[out["proposal_id"] == "A"].iloc[0]
    assert row_a["best_sensitivity_10kms_mjy_beam"] == 0.2


def test_summarize_publication_links_dedupes_bibcodes():
    df = pd.DataFrame([
        {"proposal_id": "P1", "target_name": "t", "band_list": "6",
         "member_ous_uid": "uid://A/X1/X1",
         "bib_reference": "2024A&A...685A...1A 2024A&A...688A..55M"},
        {"proposal_id": "P1", "target_name": "t", "band_list": "6",
         "member_ous_uid": "uid://A/X1/X2",
         "bib_reference": "2024A&A...685A...1A"},
    ])
    out = summarize_publication_links(df)
    assert len(out) == 1
    row = out.iloc[0]
    assert row["n_publications"] == 2
    assert row["observations"] == 2
    assert "uid://A/X1/X1" in row["member_ous_uids"] and "uid://A/X1/X2" in row["member_ous_uids"]


def test_collect_publications_flat_list_with_ads_urls():
    df = pd.DataFrame([
        {"proposal_id": "P1", "bib_reference": "2024A&A...685A...1A 2024A&A...688A..55M"},
        {"proposal_id": "P2", "bib_reference": "2024A&A...685A...1A"},
    ])
    pubs = collect_publications(df)
    assert [p["bibcode"] for p in pubs] == ["2024A&A...685A...1A", "2024A&A...688A..55M"]
    assert pubs[0]["ads_url"].endswith("/abs/2024A&A...685A...1A")
    assert collect_publications(pd.DataFrame()) == []
