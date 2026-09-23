"""core/router.py — the pre-extracted ALMA query_type hint over the 22 domain
benchmark questions (tmp/ui-bench-2026-09-22).

D22 ("galaxies at a redshift between z=1 and z=2 ... Carbon Monoxide rest
frequency in their spectral setup") was routed to bandwidth_switching_candidates
because "spectral setup" was tested before the redshift branch; the model
obeyed, then self-corrected at the cost of a tool round. Precedence is now:
redshift interval + species -> redshifted_line_projects; >= 2 species ->
line_set_projects; literal "bandwidth switching" -> bandwidth_switching_candidates.
"""
from __future__ import annotations

import types

import pytest

from core.router import route_alma_science_archive_query

_AGENT = types.SimpleNamespace()

DOMAIN_QUESTIONS = {
    "D01": "what can you tell me about ALMA Band 6 sensitivity?",
    "D02": "ok, explain the ALMA correlator",
    "D03": "great question - what is the ALMA proprietary period policy?",
    "D04": "How many observing bands does ALMA have, and what are their approximate frequency ranges?",
    "D05": "What are the different ALMA configurations and their corresponding angular resolutions?",
    "D06": "How do I find out the information that indicates which CASA version was used to process the data from a given ALMA project?",
    "D07": "What types of data can I download for observations made with ALMA?",
    "D08": "How many observing bands are available with ALMA? How many bands have public data available in the ALMA Science Archive as of March 25, 2026?",
    "D09": "Which projects in the ALMA Science Archive likely needed Bandwidth Switching for calibration?",
    "D10": "How many projects in Cycle 10 observed the sun?",
    "D11": "How many projects in Cycle 9 used the 12m, 7m, and total power array for their observations?",
    "D12": "I'm interested in the source HH212. Give me a summary of the Band 7 data from any project that can be used for creating a deep, high-resolution (better than 1 arcsec) image of the continuum.",
    "D13": "Is there a way to make a query of the ALMA Science Archive to retrieve ALMA data for a specific object through Python?",
    "D14": "How do I use Astroquery to find ALMA observations of the source M83?",
    "D15": "Show me two ways to query for a source in the ALMA Science Archive, with one using TAP and the other using ALMiner.",
    "D16": "Find ALMA observations of the source M83 using Band 6.",
    "D17": "Generate the URL for the ALMA Science Archive for the query of M83 using Band 6.",
    "D18": "Show me locations of protostars in Perseus that have been observed with ALMA and JWST.",
    "D19": "I want to see two images of the Hubble Ultra Deep Field, one with ALMA and one with JWST. Ideally, overlap the two images in a way that I can see ALMA in contours and JWST in colorscale.",
    "D20": "I'm interested in studying protostellar outflows. Please make a table of the 10 most recent publications on this topic that used data from the ALMA Science Archive, and summarize them. Use the ALMA Science Archive to get this information.",
    "D21": "List the protostellar disks in the ALMA Science Archive for which 12CO, 13CO, and C18O in Band 6 have been observed in the same project.",
    "D22": "Show me all galaxies at a redshift between z=1 and z=2 that were observed with ALMA and included the Carbon Monoxide rest frequency in their spectral setup.",
}

EXPECTED_QUERY_TYPE = {
    "D09": "bandwidth_switching_candidates",
    "D10": "cycle_solar_projects",
    "D11": "cycle_array_combo_projects",
    "D12": "high_resolution_band_data",
    "D21": "line_set_projects",
    "D22": "redshifted_line_projects",
}


@pytest.mark.parametrize("qid", sorted(DOMAIN_QUESTIONS))
def test_domain_question_query_type_hint(qid):
    route = route_alma_science_archive_query(_AGENT, DOMAIN_QUESTIONS[qid])
    expected = EXPECTED_QUERY_TYPE.get(qid)
    if expected is None:
        assert route is None, f"{qid}: unexpected hint {route}"
    else:
        assert route is not None, f"{qid}: expected {expected}, got no hint"
        assert route["query_type"] == expected, f"{qid}: {route}"


def test_d22_redshift_interval_and_co_win_over_spectral_setup():
    route = route_alma_science_archive_query(_AGENT, DOMAIN_QUESTIONS["D22"])
    assert route == {
        "query_type": "redshifted_line_projects",
        "redshift_min": 1.0,
        "redshift_max": 2.0,
        "rest_species": "CO",
    }


def test_d21_line_set_carries_the_species_and_band():
    route = route_alma_science_archive_query(_AGENT, DOMAIN_QUESTIONS["D21"])
    assert route["lines"] == ["12CO", "13CO", "C18O"] and route["band"] == [6]


def test_d12_resolution_and_band_extracted():
    route = route_alma_science_archive_query(_AGENT, DOMAIN_QUESTIONS["D12"])
    assert route["band"] == [7] and route["max_resolution_arcsec"] == 1.0


def test_d10_and_d11_cycle_queries():
    assert route_alma_science_archive_query(_AGENT, DOMAIN_QUESTIONS["D10"]) == {"query_type": "cycle_solar_projects", "cycle": 10}
    r11 = route_alma_science_archive_query(_AGENT, DOMAIN_QUESTIONS["D11"])
    assert r11["cycle"] == 9 and r11["arrays"] == ["12m", "7m", "TP"]


@pytest.mark.parametrize(
    "question, expected",
    [
        ("Which ALMA projects observed CO at z = 2.5 - 3.5?", "redshifted_line_projects"),
        ("ALMA projects with HCN and HCO+ lines toward Orion", "line_set_projects"),
        ("Which Cycle 8 ALMA projects needed bandwidth switching?", "bandwidth_switching_candidates"),
        ("Which ALMA projects used a spectral setup with [CII] at redshift 6 to 7?", "redshifted_line_projects"),
        ("List ALMA archive projects with an unusual spectral setup", "bandwidth_switching_candidates"),
    ],
)
def test_precedence_examples(question, expected):
    route = route_alma_science_archive_query(_AGENT, question)
    assert route is not None and route["query_type"] == expected, (question, route)


def test_redshift_species_other_than_co_is_kept():
    route = route_alma_science_archive_query(_AGENT, "ALMA projects covering HCN at z=1 to 1.5 in their spectral setup")
    assert route["query_type"] == "redshifted_line_projects" and route["rest_species"] == "HCN"
