"""web_citation_support verifier check (PLAN 1.6)."""

from core.answer_verifier import format_verification_block, verify_web_citations
from services.web_evidence import EvidenceRegistry


def _registry(*pages):
    reg = EvidenceRegistry()
    reg.add_from_payload({
        "success": True,
        "results": [{"title": t, "url": f"https://site{i}.org/p", "snippet": s} for i, (t, s) in enumerate(pages)],
    })
    return reg


def test_supported_citations_pass():
    reg = _registry(
        ("JWST Cycle 5 Call for Proposals", "The Cycle 5 proposal deadline is October 15, 2025 at 8:00 pm EDT."),
        ("ALMA Users' Policies", "Data are subject to a proprietary period of 12 months; 1,234 projects."),
    )
    text = (
        "The JWST Cycle 5 deadline was 15 October 2025 [W1]. ALMA keeps data proprietary for 12 months [W2], "
        "across 1234 projects [W2]."
    )
    rep = verify_web_citations(text, reg)
    assert rep.ok and rep.checked["web_citation"] == 0
    assert format_verification_block(rep) == ""


def test_unsupported_number_and_date_are_findings():
    reg = _registry(("JWST Cycle 5", "The Cycle 5 proposal deadline is October 15, 2025."))
    rep = verify_web_citations("The JWST Cycle 5 deadline is June 7, 2025 [W1].", reg)
    assert not rep.ok
    claim = rep.unsupported[0]
    assert claim.kind == "web_citation"
    assert "June" in claim.detail and "W1" in claim.detail
    block = format_verification_block(rep)
    assert "cited web source does not show" in block
    assert "not in the cited web source" in block
    assert "Show-query" not in block


def test_grouped_citations_check_against_all_cited_sources():
    reg = _registry(("A", "The array moves to C configuration on 2026 Oct 29."), ("B", "Currently in D configuration."))
    assert verify_web_citations("It is in D now and moves to C on October 29, 2026 [W1, W2].", reg).ok
    assert not verify_web_citations("It moves to C on October 29, 2026 [W2].", reg).ok


def test_code_blocks_and_uncited_sentences_are_not_checked():
    reg = _registry(("A", "nothing numeric"))
    text = "An uncited claim of 999 units.\n```\nvalue = 42  # [W1]\n```\nDone."
    assert verify_web_citations(text, reg).ok


def test_mixed_tool_and_web_findings_use_the_tool_header():
    from core.answer_verifier import Claim, VerificationReport

    rep = VerificationReport(unsupported=[Claim("count", "19 projects"), Claim("web_citation", "x", "not in cited source W1: 24")])
    block = format_verification_block(rep)
    assert "what the tools actually ran" in block and "not in the cited web source" in block


def test_no_registry_or_empty_answer_is_ok():
    assert verify_web_citations("Twelve months [W1].", None).ok
    assert verify_web_citations("", _registry(("A", "x"))).ok
