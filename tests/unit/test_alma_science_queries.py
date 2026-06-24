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
