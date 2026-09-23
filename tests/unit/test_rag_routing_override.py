"""core/rag_routing.py — documentation RAG must fire on how-to / reference
questions about ALMA tooling (UI benchmark 2026-09-22: it ran on 9/22 domain
turns; invented specifics clustered where it did not — D06, D07, D08, D13,
D14, D15, D17) and must stay off for pure data requests.
"""
from __future__ import annotations

import pytest

from core.rag_routing import documentation_rag_override
from tests.unit.test_router_query_type_hints import DOMAIN_QUESTIONS

DATALAB_QUESTIONS = {
    "L01": "Which Data Lab catalogs cover the Large Magellanic Cloud, and which of those include near-infrared photometry? List the relevant table names.",
    "L02": "How many Gaia DR3 sources lie within 10 arcminutes of Palomar 5 (RA = 229.022, Dec = −0.112)?",
    "L03": "Get g and r magnitudes for point sources within 0.4° of the Draco dwarf (RA = 260.06, Dec = +57.92) from NSC DR2 and plot a g vs (g−r) CMD.",
    "L04": "Show me a color image of the center of M31 from the DECam Legacy Surveys.",
    "L05": "Using DES DR1 around RA = 30, Dec = −50, separate stars from galaxies and show me g−r vs r−i color–color diagrams for each population.",
    "L06": "Find high-proper-motion white-dwarf candidates in Gaia DR3 in a 5°-radius patch of the southern sky (say around RA = 60, Dec = −50): significant parallax, large total proper motion, and absolute magnitudes on the WD sequence. Give me the HR diagram.",
    "L07": "Help me look for a stellar overdensity — a possible dwarf companion — in SMASH DR1 field 169. Select blue main-sequence stars and find where they clump on the sky.",
    "L08": "Make a stellar density map of a ~20° × 20° region from NSC DR2 to reveal Milky Way structure — bin by HEALPix and show log counts.",
    "L09": "Combine NSC DR2 photometry with Gaia DR3 proper motions around Palomar 5 to trace its tidal tails — select stream stars by proper motion and CMD, then plot their on-sky distribution.",
    "L10": "Build optical-to-mid-infrared SEDs for a small sample (a few hundred) of red galaxies within 1° of the Coma cluster (RA = 194.95, Dec = +27.98) by combining Legacy Surveys DR9 grz photometry with the survey's forced WISE (W1/W2) photometry. Give me magnitude vs wavelength.",
    "L11": "From the DESI DR1 redshift catalog, select luminous red galaxies (LRG target class) between z = 0.4 and 0.8, and show their redshift distribution and sky footprint.",
    "L12": "Search NSC DR2 for the densest stellar clump within 1° of the Hydra II dwarf region, then pull DECam image cutouts of the top few candidate locations so I can eyeball them.",
    "L13": "Select galaxies from SDSS/BOSS in a thin redshift slice and make a cone/wedge plot to show the cosmic web — pick a region like the SDSS Great Wall.",
    "L14": "I have a candidate variable star at RA = 185.4311, Dec = −31.9953 (an RR Lyrae in the Hydra II field). Find its multi-epoch SMASH photometry, phase-fold the light curve to get the period, and pull an image cutout of the field.",
    "L15": "I want to discover new Milky Way satellite dwarf-galaxy candidates. Devise and carry out a search strategy using Data Lab's deep imaging catalogs, and give me a ranked list of candidate positions with supporting CMDs and image cutouts.",
}

# Knowledge / how-to / reference questions -> documentation search ON.
EXPECT_RAG = {"D01", "D02", "D03", "D04", "D05", "D06", "D07", "D08", "D09", "D13", "D14", "D15", "D17"}
# Pure data requests -> the override stays quiet (tools answer them).
EXPECT_NO_RAG = {"D10", "D11", "D12", "D16", "D18", "D19", "D20", "D21", "D22"}


@pytest.mark.parametrize("qid", sorted(DOMAIN_QUESTIONS))
def test_domain_question_rag_override(qid):
    decision = documentation_rag_override(DOMAIN_QUESTIONS[qid])
    if qid in EXPECT_RAG:
        assert decision is True, f"{qid} should reach the documentation"
    else:
        assert qid in EXPECT_NO_RAG and decision is False, f"{qid} is a data request"


@pytest.mark.parametrize("qid", sorted(DATALAB_QUESTIONS))
def test_datalab_questions_never_trigger_alma_documentation(qid):
    assert documentation_rag_override(DATALAB_QUESTIONS[qid]) is False


def test_override_only_turns_rag_on_for_alma_tooling_topics():
    assert documentation_rag_override("How do I bake sourdough bread?") is False
    assert documentation_rag_override("How do I list the DataLink files of an ALMA MOUS with astroquery?") is True
    assert documentation_rag_override("Which ALMA configuration gives 0.1 arcsec resolution in Band 7?") is True
    assert documentation_rag_override("What does the QA2 weblog contain?") is True


@pytest.mark.parametrize("q", [
    "How do I download HST FITS files from MAST?",
    "What is the angular resolution of JWST NIRCam and how do I get the data products?",
    "How can I find Chandra archive data for Cas A?",
    "Explain how to download SDSS spectra FITS files",
])
def test_generic_topics_about_other_facilities_do_not_trigger_alma_rag(q):
    """Guard CX-28: FITS / resolution / archive / download are generic; with
    another facility named and nothing ALMA-specific they are not ALMA-manual
    questions."""
    assert documentation_rag_override(q) is False


@pytest.mark.parametrize("q", [
    "How do I download the FITS products for an ALMA project?",
    "How do I compare ALMA and JWST resolution for my proposal?",
    "What is the angular resolution I can reach in Band 7?",
])
def test_alma_specific_questions_still_trigger_rag_even_with_other_facilities(q):
    assert documentation_rag_override(q) is True
