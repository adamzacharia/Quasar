"""Opt-in live smoke tests for Splatalogue and ALMA TAP."""

import os

import pytest

from services.spectral_line_explorer import find_alma_line_coverage
from services.splatalogue import SpectralLineQuery, SplatalogueTool


pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        os.getenv("RUN_LIVE_SPECTRAL_LINE_TESTS") != "1",
        reason="Set RUN_LIVE_SPECTRAL_LINE_TESTS=1 to query Splatalogue and ALMA TAP.",
    ),
]


def test_live_splatalogue_co_21():
    result = SplatalogueTool().query_catalog(
        SpectralLineQuery.basic(
            {
                "windows": [
                    {"minimum": 230.53, "maximum": 230.55, "unit": "GHz"}
                ],
                "species_ids": [204],
                "transition": "2-1",
            }
        )
    )
    assert result["backend"] in {
        "splatalogue_threaded_advanced",
        "splatalogue_advanced",
        "slap",
    }
    assert any(
        abs(float(line["frequency_ghz"]) - 230.538) < 0.001
        for line in result["lines"]
    )


def test_live_m87_co_21_exact_alma_coverage():
    result = find_alma_line_coverage(
        target_name="M87",
        species="CO",
        transition="2-1",
        redshift=0.00436,
    )
    assert result["success"] is True
    assert float(result["selected_line"]["frequency_ghz"]) == pytest.approx(
        230.538, abs=0.001
    )
    assert float(result["selected_line"]["observed_frequency_ghz"]) == pytest.approx(
        229.537, abs=0.002
    )
    for project in result["projects"]:
        for observation in project["observations"]:
            assert observation["matching_lines"]
