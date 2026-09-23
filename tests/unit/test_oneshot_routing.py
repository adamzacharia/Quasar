"""core/oneshot_routing.py over the 37 UI-benchmark questions (2026-09-22).

Each benchmark question that one deterministic tool answers must route to it
(D08, D13, D14, D15, D17, D20 via detect_oneshot_intent; D09-D12, D21, D22 via
the router + census_arguments); every other question must not be hijacked.
"""
from __future__ import annotations

import types

import pytest

from core.oneshot_routing import (
    census_arguments,
    census_directive,
    cross_archive_directive,
    detect_oneshot_intent,
    overlay_directive,
)
from core.router import route_alma_science_archive_query
from tests.unit.test_rag_routing_override import DATALAB_QUESTIONS
from tests.unit.test_router_query_type_hints import DOMAIN_QUESTIONS

EXPECTED_ONESHOT = {
    "D08": ("alma_public_band_status", {"as_of": "2026-03-25"}),
    "D13": ("code_recipe", {"task": "query by object name", "library": "astroquery"}),
    "D14": ("code_recipe", {"task": "query by object name", "library": "astroquery"}),
    "D15": ("code_recipe", None),
    "D17": ("alma_archive_link", {"target": "M83", "band": [6]}),
    "D20": ("alma_bibliography", {"n": 10, "topic": "protostellar outflows"}),
}


@pytest.mark.parametrize("qid", sorted(DOMAIN_QUESTIONS))
def test_domain_questions_route_to_the_right_oneshot_tool(qid):
    intent = detect_oneshot_intent(DOMAIN_QUESTIONS[qid])
    expected = EXPECTED_ONESHOT.get(qid)
    if expected is None:
        assert intent is None, f"{qid} hijacked by {intent and intent['tool']}"
        return
    tool, args = expected
    assert intent is not None and intent["tool"] == tool, (qid, intent)
    if args is not None:
        assert intent["args"] == args, (qid, intent["args"])
    assert f"`{tool}`" in intent["directive"] and "MANDATORY INSTRUCTION" in intent["directive"]


def test_d15_asks_for_one_recipe_per_library():
    intent = detect_oneshot_intent(DOMAIN_QUESTIONS["D15"])
    libs = [c["library"] for c in intent["calls"]]
    assert libs == ["alminer", "pyvo"]
    assert all(c["task"] == "ADQL via TAP" or c["library"] == "alminer" for c in intent["calls"])


@pytest.mark.parametrize("qid", sorted(DATALAB_QUESTIONS))
def test_datalab_questions_are_never_routed_to_alma_oneshot_tools(qid):
    intent = detect_oneshot_intent(DATALAB_QUESTIONS[qid])
    assert intent is None or intent["tool"].startswith("datalab_")


_AGENT = types.SimpleNamespace()


@pytest.mark.parametrize(
    "qid, tool, args",
    [
        ("D09", "alma_project_census", {"constraint": {"type": "bandwidth_switching"}}),
        ("D10", "alma_project_census", {"cycle": 10, "constraint": {"type": "solar"}}),
        ("D11", "alma_project_census", {"cycle": 9, "constraint": {"type": "arrays", "arrays": ["12m", "7m", "TP"]}}),
        ("D12", "alma_source_summary", {"bands": [7], "max_resolution_arcsec": 1.0}),
        ("D21", "alma_project_census", {"constraint": {"type": "lines", "lines": ["12CO", "13CO", "C18O"], "band": [6]}}),
        ("D22", "alma_project_census", {"constraint": {"type": "redshift", "zmin": 1.0, "zmax": 2.0, "species": "CO"}}),
    ],
)
def test_science_routes_map_to_the_census_tools(qid, tool, args):
    route = route_alma_science_archive_query(_AGENT, DOMAIN_QUESTIONS[qid])
    plan = census_arguments(route)
    assert plan == {"tool": tool, "args": args}
    text = census_directive(route)
    assert f"`{tool}`" in text and "headline" in text and "calibrator" in text


def test_census_directive_without_a_route_lists_both_tools():
    text = census_directive(None)
    assert "`alma_project_census`" in text and "`alma_source_summary`" in text


def test_cross_archive_and_overlay_directives_name_the_oneshot_tools():
    assert "`cross_archive_match`" in cross_archive_directive(None) and "UNKNOWN" in cross_archive_directive(None)
    text = overlay_directive(DOMAIN_QUESTIONS["D19"])
    assert "`archive_overlay`" in text and '"field": "Hubble Ultra Deep Field"' in text and '"mission": "JWST"' in text
    assert "coverage_gap" in text


DLB_TEXT = {
    "L06": ("Find high-proper-motion white-dwarf candidates in Gaia DR3 in a 5°-radius patch of the southern sky (say around "
            "RA = 60, Dec = −50): significant parallax, large total proper motion, and absolute magnitudes on the WD sequence. "
            "Give me the HR diagram."),
    "L07": ("Help me look for a stellar overdensity — a possible dwarf companion — in SMASH DR1 field 169. Select blue "
            "main-sequence stars and find where they clump on the sky."),
    "L08": "Make a stellar density map of a ~20° × 20° region from NSC DR2 to reveal Milky Way structure — bin by HEALPix and show log counts.",
    "L09": ("Combine NSC DR2 photometry with Gaia DR3 proper motions around Palomar 5 to trace its tidal tails — select stream "
            "stars by proper motion and CMD, then plot their on-sky distribution."),
    "L10": ("Build optical-to-mid-infrared SEDs for a small sample (a few hundred) of red galaxies within 1° of the Coma cluster "
            "(RA = 194.95, Dec = +27.98) by combining Legacy Surveys DR9 grz photometry with the survey's forced WISE (W1/W2) "
            "photometry. Give me magnitude vs wavelength."),
    "L11": ("From the DESI DR1 redshift catalog, select luminous red galaxies (LRG target class) between z = 0.4 and 0.8, and "
            "show their redshift distribution and sky footprint."),
    "L15": ("I want to discover new Milky Way satellite dwarf-galaxy candidates. Devise and carry out a search strategy using "
            "Data Lab's deep imaging catalogs, and give me a ranked list of candidate positions with supporting CMDs and image cutouts."),
}


@pytest.mark.parametrize(
    "qid,tool,expected",
    [
        ("L06", "datalab_selection_diagram", {"ra": 60.0, "dec": -50.0, "radius_deg": 5.0, "overlay_locus": "white_dwarf",
                                              "abs_mag_from_parallax": True, "pm_total_min_mas_yr": 50.0}),
        ("L07", "datalab_satellite_search", {"smash_field": 169}),
        ("L08", "datalab_healpix_density_map", {"catalog": "nsc_dr2", "preset": "lmc"}),
        ("L09", "datalab_stream_selection", {"cluster_name": "Palomar 5"}),
        ("L11", "datalab_target_class_summary", {"target_class": "LRG", "z_range": [0.4, 0.8]}),
        ("L15", "datalab_satellite_search", {"survey": "delve", "preset": "delve_south"}),
    ],
)
def test_datalab_questions_route_to_their_oneshot_tool(qid, tool, expected):
    route = detect_oneshot_intent(DLB_TEXT[qid])
    assert route and route["tool"] == tool
    for key, value in expected.items():
        assert route["args"][key] == value
    assert f"`{tool}`" in route["directive"] and "MANDATORY" in route["directive"]


def test_datalab_routing_leaves_sed_questions_and_named_regions_alone():
    assert detect_oneshot_intent(DLB_TEXT["L10"]) is None
    route = detect_oneshot_intent("Show a HEALPix stellar density map of the anticenter from NSC DR2")
    assert route["args"] == {"catalog": "nsc_dr2", "preset": "anticenter"}
    assert "clarifying" not in route["directive"]
    # open-ended regions default to the LMC preset and must say so, not ask
    assert "Do NOT ask a clarifying question" in detect_oneshot_intent(DLB_TEXT["L08"])["directive"]
    # the strategy must be stated for the open-ended discovery question
    assert "STATE THE STRATEGY" in detect_oneshot_intent(DLB_TEXT["L15"])["directive"]
    # ALMA questions never take a Data Lab route
    assert detect_oneshot_intent("Which ALMA projects observed a stellar density peak in the LMC?") is None or \
        not detect_oneshot_intent("Which ALMA projects observed a stellar density peak in the LMC?")["tool"].startswith("datalab_")
