"""Round 3 of the Phase 2 review: P2-02 (value moved into an unchecked clause),
P2-30 (dates in a new clause after the tag are still checked), P2-31 (a census
question with a policy word keeps its census), P2-32 (survey names before
quantities)."""
from __future__ import annotations

import inspect
import json

import pytest

from core import web_revise as wr
from services.web_evidence import EvidenceRegistry, unsupported_citation_claims


def _reg(*snips):
    reg = EvidenceRegistry(query="q")
    reg.add_from_payload({"success": True, "results": [
        {"url": f"https://a.org/{i}", "title": f"T{i}", "snippet": s} for i, s in enumerate(snips)]})
    reg.render_prompt_block()
    return reg


# P2-30: a date or time in a new clause after the tag is a web claim; a plain number is tool data


@pytest.mark.parametrize("text", [
    "The period is 12 months [W1], and it starts 3 March 2027.",
    "The period is 12 months [W1] while the Cycle 13 deadline is 23 April 2026.",
    "The period is 12 months [W1]; proposals close at 15:00 UTC on 23 April.",
    "The period is 12 months [W1], which was announced on 19 March 2026 together with the call for proposals and the observing tool.",
])
def test_p2_30_dates_after_the_tag_are_checked(text):
    assert unsupported_citation_claims(text, _reg("The proprietary period is 12 months."))


def test_p2_30_plain_numbers_in_a_new_clause_are_not_web_claims():
    reg = _reg("The proprietary period is 12 months.")
    assert unsupported_citation_claims("The period is 12 months [W1], and the archive lists 37 projects.", reg) == []
    assert unsupported_citation_claims("The period is 12 months [W1]; 37 projects match.", reg) == []


# P2-02 round 3: moving the value into the unchecked clause is rejected


@pytest.mark.parametrize("revised", [
    "Proposals are due [W1], and the date is 23 April 2026.",
    "Per the guide [W1], and proposals are due 23 April 2026.",
    "Proposals are due [W1]; 23 April 2026.",
    "Proposals are due [W1] which is 23 April 2026.",
    "Proposals are due [W1], and the count is 37.",
])
def test_p2_02_round3_value_cannot_move_into_an_unchecked_clause(revised):
    reg = _reg("Proposals close 15:00 UTC. No date given here.")
    text = "Proposals are due 23 April 2026, 37 in total [W1]."
    out, info = wr.revise_unsupported(text, reg, llm_call=lambda p, m, t: json.dumps({"revisions": [{"original": text, "revised": revised}]}), budget_s=2)
    assert out == text and info["revised"] == 0


def test_p2_02_round3_a_legitimate_revision_still_passes():
    reg = _reg("Proposals close 15:00 UTC. No date given here.")
    text = "Proposals are due 23 April 2026 [W1]."
    good = "Proposals are due on a date not stated [W1]."
    out, info = wr.revise_unsupported(text, reg, llm_call=lambda p, m, t: json.dumps({"revisions": [{"original": text, "revised": good}]}), budget_s=2)
    assert out == good and info["revised"] == 1


# P2-31: a census question that uses a policy word keeps the forced census


def test_p2_31_census_intent_beats_the_policy_exclusion():
    import core.runner as runner
    src = inspect.getsource(runner._stream_response_api_impl)
    assert "bool(_alma_science_route) and (_census_intent or not _policy_web_override(_user_query))" in src
    from core.router import policy_web_override
    assert policy_web_override("List ALMA solar observations from Cycle 10 whose proprietary period has ended")


# P2-32: survey and mission names before quantities keep the numbers; designations stay names


def test_p2_32_survey_names_before_quantities_keep_their_numbers():
    reg = _reg("Nothing numeric here.")
    for text in ("WISE 22 micron [W1].", "IRAS 100 micron [W1].", "The ESO 2026 call [W1].", "The Kepler 2009 launch [W1].",
                 "The TIC 2026 update [W1].", "1 HR 30 min [W1].", "Gaia DR3 2022 release [W1]."):
        assert unsupported_citation_claims(text, reg), text


def test_p2_32_designations_are_still_names():
    reg = _reg("The proprietary period is 12 months.")
    for name in ("IRAS 16293-2422", "2MASS J12345678+1234567", "SDSS J1030+0524", "Kepler-452b", "TOI-700", "WASP-12b", "K2-18", "HR 8799", "GRB 250702B"):
        assert unsupported_citation_claims(f"For {name} the proprietary period is 12 months [W1].", reg) == [], name
