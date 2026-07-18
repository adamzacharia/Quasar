"""R3 — mechanical citation recall/precision metrics + wiring."""

import pytest

from services.citation_metrics import (
    METRICS_METHOD,
    compute_citation_metrics,
    is_claim_sentence,
    record_citation_metrics_on_request_context,
    split_sentences,
)
from services.citation_verifier import append_citation_warning


# ─────────────────────────────────────────────────────────────────────────────
# Sentence machinery
# ─────────────────────────────────────────────────────────────────────────────
def test_split_sentences_drops_headings_tables_code():
    text = (
        "# Heading\n\n"
        "The flux density is 2.3 mJy at 230 GHz. This matches prior work.\n\n"
        "| col | val |\n|---|---|\n| a | 1 |\n\n"
        "```python\nx = 42\n```\n"
        "- The disk radius is 120 au.\n"
    )
    sentences = split_sentences(text)
    assert any("flux density" in s for s in sentences)
    assert any("disk radius" in s for s in sentences)
    assert not any(s.startswith("#") for s in sentences)
    assert not any("x = 42" in s for s in sentences)
    assert not any("| col |" in s for s in sentences)


def test_is_claim_sentence():
    assert is_claim_sentence("The flux density is 2.3 mJy.")
    assert is_claim_sentence("See 2018ApJ...869L..41A for details.")
    assert is_claim_sentence("Reported in doi:10.3847/1538-4357/abc123 recently.")
    assert not is_claim_sentence("What is the flux density of 42 sources?")
    assert not is_claim_sentence("The morphology appears disturbed and asymmetric.")


def test_bare_years_are_not_claims():
    # A publication year alone is bibliographic context, not a measurement.
    assert not is_claim_sentence("This was first proposed back in 2015 by the team.")


# ─────────────────────────────────────────────────────────────────────────────
# Metrics computation
# ─────────────────────────────────────────────────────────────────────────────
def _verification(resolved_ids=(), unresolved_ids=()):
    def entry(i, ok):
        return {"type": "bibcode", "id": i, "resolved": ok}
    return {
        "bibcodes": [entry(i, True) for i in resolved_ids]
        + [entry(i, False) for i in unresolved_ids],
        "dois": [],
        "unresolved": [entry(i, False) for i in unresolved_ids],
        "checked": len(resolved_ids) + len(unresolved_ids),
        "skipped": 0,
        "skipped_items": [],
    }


def test_metrics_resolved_citation_supports_sentence():
    text = "The disk mass is 0.01 solar masses (2018ApJ...869L..41A). It is large."
    metrics = compute_citation_metrics(
        text, _verification(resolved_ids=["2018ApJ...869L..41A"]))
    assert metrics["method"] == METRICS_METHOD
    assert metrics["claim_sentences"] == 1
    assert metrics["supported_claim_sentences"] == 1
    assert metrics["citation_recall"] == 1.0
    assert metrics["citations_total"] == 1
    assert metrics["citations_resolved"] == 1
    assert metrics["citation_precision"] == 1.0


def test_metrics_unresolved_citation_does_not_support():
    text = "The disk mass is 0.01 solar masses (2099Fake...123X...9Z)."
    metrics = compute_citation_metrics(
        text, _verification(unresolved_ids=["2099Fake...123X...9Z"]))
    assert metrics["supported_claim_sentences"] == 0
    assert metrics["citation_recall"] == 0.0
    assert metrics["citation_precision"] == 0.0


def test_metrics_numbers_supported_by_tool_evidence():
    text = "The query returned 1523 sources with a median magnitude of 18.4."
    metrics = compute_citation_metrics(
        text, None, evidence_texts=['{"count": 1523, "median_mag": 18.4}'])
    assert metrics["claim_sentences"] == 1
    assert metrics["supported_claim_sentences"] == 1
    assert metrics["citation_recall"] == 1.0
    # No citations attached → precision undefined.
    assert metrics["citation_precision"] is None


def test_metrics_numbers_without_evidence_unsupported():
    text = "The query returned 1523 sources with a median magnitude of 18.4."
    metrics = compute_citation_metrics(text, None, evidence_texts=None)
    assert metrics["supported_claim_sentences"] == 0
    assert metrics["citation_recall"] == 0.0


def test_metrics_majority_number_rule():
    # 1 of 3 numbers found in evidence → below majority → unsupported.
    text = "We found 1523 sources at 230.5 GHz brighter than 18.4 mag."
    metrics = compute_citation_metrics(text, None, evidence_texts=["1523"])
    assert metrics["supported_claim_sentences"] == 0
    # 2 of 3 → majority → supported.
    metrics = compute_citation_metrics(text, None, evidence_texts=["1523 230.5"])
    assert metrics["supported_claim_sentences"] == 1


def test_metrics_no_claims_yields_none_recall():
    metrics = compute_citation_metrics("A qualitative description only, nothing testable here.", None)
    assert metrics["claim_sentences"] == 0
    assert metrics["citation_recall"] is None
    assert metrics["citation_precision"] is None


def test_metrics_never_raises_on_garbage():
    metrics = compute_citation_metrics(None, {"bibcodes": "not-a-list"})  # type: ignore[arg-type]
    assert metrics["method"] == METRICS_METHOD


# ─────────────────────────────────────────────────────────────────────────────
# Wiring: verification sink + request-context recording
# ─────────────────────────────────────────────────────────────────────────────
class _AdsOk:
    api_key = "k"

    def get_paper_details(self, bibcode):
        return {"title": "Real paper", "bibcode": bibcode}


def test_append_citation_warning_fires_verification_sink():
    captured = {}

    def sink(verification):
        captured["verification"] = verification

    text = "The mass is 5.2 Msun (2018ApJ...869L..41A)."
    out = append_citation_warning(text, _AdsOk(), verification_sink=sink)
    assert "verification" in captured
    assert captured["verification"]["checked"] >= 1
    assert out.startswith(text)


def test_append_citation_warning_sink_failure_is_nonfatal():
    def sink(_v):
        raise RuntimeError("sink boom")

    text = "The mass is 5.2 Msun (2018ApJ...869L..41A)."
    out = append_citation_warning(text, _AdsOk(), verification_sink=sink)
    assert out.startswith(text)


def test_record_metrics_on_request_context():
    from core.llm_client import get_llm_request_context, llm_request_context

    with llm_request_context(user_id="u1"):
        record_citation_metrics_on_request_context({"citation_recall": 0.5})
        ctx = get_llm_request_context()
        assert ctx is not None
        assert ctx.citation_metrics == {"citation_recall": 0.5}
    # Outside a context: silently a no-op.
    record_citation_metrics_on_request_context({"citation_recall": 1.0})
