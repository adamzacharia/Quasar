"""Round 2 of the Phase 2 review: the reopened findings (P2-02, 03, 09, 11, 12, 13)
and the new ones (P2-20 catalog-id rule swallowing numbers, P2-21 carry cap)."""
from __future__ import annotations

import inspect
import json

import pytest

from core import web_planner as wp
from core import web_revise as wr
from services.web_evidence import EvidenceRegistry, _Facts, split_sentences, unsupported_citation_claims


def _reg(*snips):
    reg = EvidenceRegistry(query="q")
    reg.add_from_payload({"success": True, "results": [
        {"url": f"https://a.org/{i}", "title": f"T{i}", "snippet": s} for i, s in enumerate(snips)]})
    reg.render_prompt_block()
    return reg


# P2-20: "at 230 GHz", "closes at 17:00", "1 hr 30 min" keep their numbers; real ids stay names


def test_p2_20_common_words_before_numbers_are_not_catalog_prefixes():
    reg = _reg("Band 7 observations at 345 GHz. The call closes at 15:00 UTC on 3 March.")
    assert unsupported_citation_claims("The observations were at 230 GHz [W1].", reg)
    assert unsupported_citation_claims("The call closes at 17:00 UTC [W1].", reg)
    assert unsupported_citation_claims("The call opens at 10 March [W1].", reg)
    found = unsupported_citation_claims("The call closes at 16:00 UTC on 2 May [W1].", reg)
    assert found and "16:00" in found[0]["missing"] and "2 May" in found[0]["missing"]
    assert unsupported_citation_claims("It lasts 1 hr 30 min, i.e. 90 minutes [W1].", reg)   # 90 unsupported
    f = _Facts("observations at 345 GHz and 1 hr 30 min and 12 m dishes and 5 g")
    assert {"345", "30", "12", "5"} <= f.numbers


def test_p2_20_real_catalog_ids_are_still_names():
    reg = _reg("The proprietary period is 12 months.")
    for name in ("M 87", "M87", "NGC 1068", "SN 2023ixf", "AT 2024abc", "L1544", "G34.3+0.2", "HR 8799", "3C 273"):
        assert unsupported_citation_claims(f"For {name} the proprietary period is 12 months [W1].", reg) == [], name


# P2-02 round 2: moving the value out of the tagged sentence is rejected


def _dates_reg():
    return _reg("Proposals close 15:00 UTC. No date given here.")


@pytest.mark.parametrize("revised", [
    "Proposals are due 23 April 2026. [W1]",
    "Per the guide [W1]. Proposals are due 23 April 2026.",
    "Proposals are due 23 April 2026.",
])
def test_p2_02_round2_value_cannot_leave_the_checked_claim(revised):
    reg = _dates_reg()
    text = "Proposals are due 23 April 2026 [W1]."
    out, info = wr.revise_unsupported(text, reg, llm_call=lambda p, m, t: json.dumps({"revisions": [{"original": text, "revised": revised}]}), budget_s=2)
    assert out == text and info["revised"] == 0


# P2-03 round 2: comma clauses and rows with a period inside a cell


def test_p2_03_round2_comma_clause_about_tool_data_is_not_a_web_claim():
    reg = _reg("The proprietary period is 12 months.")
    assert unsupported_citation_claims("ALMA's proprietary period is 12 months [W1], and the archive lists 37 projects.", reg) == []
    assert unsupported_citation_claims("ALMA's proprietary period is 12 months [W1], but 37 projects are listed.", reg) == []
    # a plain continuation of the SAME claim still counts
    assert unsupported_citation_claims("ALMA's proprietary period is 12 months [W1], starting 3 March 2027.", reg)


def test_p2_03_round2_table_row_is_one_unit_even_with_a_period_inside_a_cell():
    reg = _reg("The proprietary period is 12 months.")
    row = "| Cycle 13 | Deadline 23 April 2026. Late ok | 12 months [W1] |"
    assert split_sentences(row) == [row]
    found = unsupported_citation_claims(row, reg)
    assert found and "23 April 2026" in found[0]["missing"]


# P2-09 round 2: the planner-off follow-up heuristic


@pytest.mark.parametrize("q", [
    "When did its LSST survey officially begin?",
    "What about Cycle 13?",
    "Does that apply to Large programs too?",
    "And when does it move to the next one?",
    "Its current construction status?",
])
def test_p2_09_round2_follow_ups(q):
    assert wp.looks_like_follow_up(q)


@pytest.mark.parametrize("q", [
    "Is it true that JWST Cycle 5 closed?",
    "Derive the Jeans mass for an isothermal cloud.",
    "Find ALMA observations of NGC 1068 in Band 7.",
    "What is the proprietary period for ALMA Cycle 13 data?",
])
def test_p2_09_round2_not_follow_ups(q):
    assert not wp.looks_like_follow_up(q)


# P2-11 round 2: star and object names


@pytest.mark.parametrize("q", [
    "Tell me about Alpha Centauri", "Tell me about Proxima Centauri", "Tell me about Eta Carinae",
    "Tell me about Omega Centauri", "Tell me about Large Magellanic Cloud", "Tell me about Las Campanas",
    "Who is Sagittarius A*?",
])
def test_p2_11_round2_objects_are_not_people(q):
    assert not wp.looks_like_researcher_query(q)


def test_p2_11_round2_people_still_pass():
    for q in ("Who is Paola Caselli?", "Tell me about Ewine van Dishoeck", "Where does Crystal Brogan work?"):
        assert wp.looks_like_researcher_query(q), q


# P2-12 round 2: the route half also excludes policy questions


def test_p2_12_round2_route_half_excludes_policy_questions():
    import core.runner as runner
    src = inspect.getsource(runner._stream_response_api_impl)
    assert "bool(_alma_science_route) and (_census_intent or not _policy_web_override(_user_query))" in src


# P2-13 round 2: a swapped link is rejected


def test_p2_13_round2_swapped_link_is_rejected():
    reg = _dates_reg()
    text = "Proposals are due 23 April 2026, see the [guide](https://almascience.org/pg) [W1]."
    revised = "Proposals are due on a date not stated, see the [guide](https://phish.example/pg) [W1]."
    out, info = wr.revise_unsupported(text, reg, llm_call=lambda p, m, t: json.dumps({"revisions": [{"original": text, "revised": revised}]}), budget_s=2)
    assert out == text and info["revised"] == 0


# P2-21: the carry cap registered equals the cap stored


def test_p2_21_carry_caps_match():
    import core.runner as runner
    src = inspect.getsource(runner._stream_response_api_impl)
    assert 'add_carried(_carried_state["evidence"][:8])' in src
    assert "[ev.as_carry() for ev in _ordered][:8]" in src
