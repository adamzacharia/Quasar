"""Scientific species/transition matching: isotopologue & case distinction and
structural quantum-number transition matching.

Guards the fixes for the substring-matching hazards: CO != 13CO != C18O != Co,
and '2-1' must not match '12-11'.
"""

import pytest

from services.splatalogue import (
    formula_matches,
    normalize_formula,
    transition_matches,
)
from services.spectral_line_explorer import _transition_matches


# -- species / formula matching ----------------------------------------------

def test_formula_exact_and_state_stripping():
    assert formula_matches("CO", "CO")
    assert formula_matches("CO", "CO v=0")
    assert formula_matches("CO", 'CO <font color="red">v = 0</font>')
    assert normalize_formula('CO <font color="red">v = 0</font>') == "CO"


@pytest.mark.parametrize(
    "query,candidate",
    [
        ("CO", "Co"),       # carbon monoxide vs cobalt (case!)
        ("CO", "13CO"),     # isotopologue
        ("CO", "C18O"),     # isotopologue
        ("CO", "C17O"),
        ("CO", "HCO+"),
        ("CO", "HCOOH"),
        ("CO", "OCS"),
        ("CO", "CO2"),
        ("13CO", "CO"),
        ("13CO", "C18O"),
        ("C18O", "CO"),
        ("C18O", "13CO"),
        ("CS", "Cs"),       # carbon monosulfide vs caesium
        ("NO", "No"),       # nitric oxide vs nobelium
    ],
)
def test_formula_does_not_conflate(query, candidate):
    assert not formula_matches(query, candidate)


@pytest.mark.parametrize(
    "formula",
    ["CO", "13CO", "C18O", "Co", "CS", "Cs", "HCO+", "CH3OH"],
)
def test_formula_matches_itself(formula):
    assert formula_matches(formula, formula)


def test_formula_empty_is_not_a_match():
    assert not formula_matches("", "CO")
    assert not formula_matches("CO", "")
    assert not formula_matches("CO", None)


# -- transition matching ------------------------------------------------------

@pytest.mark.parametrize(
    "candidate",
    [
        "2-1",
        "J=2-1",
        "2 - 1",
        "2–1",          # en dash
        "2—1",          # em dash
        "2→1",
        "2(0,2)-1(0,1)",            # asymmetric top: leading J pair is 2-1
        "N=2-1,J=5/2-3/2,F=3-2",    # hyperfine listing containing the 2-1 pair
    ],
)
def test_transition_2_1_matches_real_forms(candidate):
    assert transition_matches("2-1", candidate)


@pytest.mark.parametrize(
    "candidate",
    ["12-11", "32-31", "20-19", "21-20", "1-0", "3-2", "22-21"],
)
def test_transition_2_1_rejects_other_rotational_lines(candidate):
    assert not transition_matches("2-1", candidate)


@pytest.mark.parametrize("candidate", ["10-9", "11-10", "100-99"])
def test_transition_1_0_not_substring_of_higher_J(candidate):
    assert not transition_matches("1-0", candidate)


def test_transition_1_0_matches_real_forms():
    assert transition_matches("1-0", "1-0")
    assert transition_matches("1-0", "J=1-0")
    assert transition_matches("1-0", "1(0,1)-0(0,0)")


def test_transition_underscore_delimited_cdms_qns():
    # CDMS-style underscore-delimited resolved QNs should still find the 2-1 pair,
    # and must not match 1-0.
    assert transition_matches("2-1", "2_1_2-1_0_1")
    assert not transition_matches("1-0", "2_1_2-1_0_1")


def test_transition_halfinteger_hyperfine():
    assert transition_matches("5/2-3/2", "J=5/2-3/2")
    assert not transition_matches("5/2-3/2", "3/2-1/2")


def test_transition_empty_query_matches_anything():
    assert transition_matches("", "anything")
    assert transition_matches(None, "5-4")


def test_transition_empty_candidate_does_not_match():
    assert not transition_matches("2-1", "")
    assert not transition_matches("2-1", None)


def test_transition_named_fallback_is_whole_string():
    # No numeric pair in the query -> whole-string normalized match, not substring.
    assert transition_matches("alpha", "Alpha")
    assert not transition_matches("alpha", "alphabeta")


# -- spectral_line_explorer._transition_matches delegation (arg order) --------

def test_explorer_transition_matches_delegation():
    # _transition_matches(value=candidate, requested=query)
    assert _transition_matches("2-1", "2-1") is True
    assert _transition_matches("12-11", "2-1") is False   # candidate 12-11, query 2-1
    assert _transition_matches("J=2-1", "2-1") is True
    assert _transition_matches("2-1", "1-0") is False
