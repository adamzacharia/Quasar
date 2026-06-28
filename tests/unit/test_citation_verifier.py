from services.citation_verifier import (
    append_citation_warning,
    extract_citations,
    verify_citations,
)


class FakeADSClient:
    def __init__(self, details=None, doi_hits=None, api_key="test-key"):
        self.api_key = api_key
        self.details = details or {}
        self.doi_hits = doi_hits or {}
        self.calls = []

    def get_paper_details(self, bibcode):
        self.calls.append(("bibcode", bibcode))
        if bibcode == "RAISE":
            raise RuntimeError("ADS failure")
        return self.details.get(bibcode)

    def search_papers(self, query, max_results=10, sort="date desc", fields=None, filters=None, start=0):
        self.calls.append(("doi", query, max_results, sort))
        doi = query.split('"')[1].lower()
        return self.doi_hits.get(doi, [])


def test_extract_citations_finds_bibcodes_dois_and_encoded_ads_urls():
    text = (
        "See 2018ApJ...869L..41A and DOI https://doi.org/10.3847/2041-8213/aaf275). "
        "Encoded ADS URL: https://ui.adsabs.harvard.edu/abs/2024A%26A...688A...1B/abstract. "
        "Ignore plain year in 2018, arXiv:1812.04040v2, and short 2018ApJ...869L..41."
    )

    citations = extract_citations(text)

    assert citations["bibcodes"] == [
        "2018ApJ...869L..41A",
        "2024A&A...688A...1B",
    ]
    assert citations["dois"] == ["10.3847/2041-8213/aaf275"]


def test_extract_citations_deduplicates_dois_case_insensitively_and_allows_plus():
    text = (
        "DOI 10.1002/ANIE.202400123+SUPP and "
        "https://doi.org/10.1002/anie.202400123+supp."
    )

    assert extract_citations(text)["dois"] == ["10.1002/anie.202400123+supp"]


def test_verify_citations_reports_resolved_and_unresolved_without_network():
    good_bibcode = "2018ApJ...869L..41A"
    bad_bibcode = "2099ApJ...999Z...9X"
    doi = "10.3847/2041-8213/aaf275"
    fake = FakeADSClient(
        details={good_bibcode: {"title": "A resolved ADS paper", "bibcode": good_bibcode}},
        doi_hits={doi: [{"title": "A resolved DOI paper", "doi": doi}]},
    )
    text = f"Good {good_bibcode}, bad {bad_bibcode}, DOI {doi}."

    result = verify_citations(text, fake, deadline_seconds=None)

    assert result["checked"] == 3
    assert result["skipped"] == 0
    assert result["bibcodes"][0]["resolved"] is True
    assert result["bibcodes"][1]["resolved"] is False
    assert result["dois"][0]["resolved"] is True
    assert [item["id"] for item in result["unresolved"]] == [bad_bibcode]


def test_verify_citations_respects_max_lookups_and_reports_skipped():
    text = " ".join([
        "2018ApJ...869L..41A",
        "2020PASP..132c5001L",
        "2023ApJ...950..123C",
    ])
    fake = FakeADSClient(details={"2018ApJ...869L..41A": {"title": "Resolved"}})

    result = verify_citations(text, fake, max_lookups=1, deadline_seconds=None)

    assert result["checked"] == 1
    assert result["skipped"] == 2
    assert len(fake.calls) == 1
    assert [item["reason"] for item in result["skipped_items"]] == [
        "skipped_max_lookups",
        "skipped_max_lookups",
    ]
    assert result["bibcodes"][1]["resolved"] is None


def test_verify_citations_ads_client_none_returns_unknowns_without_exception():
    text = "Reference 2018ApJ...869L..41A and 10.3847/2041-8213/aaf275."

    result = verify_citations(text, None)

    assert result["checked"] == 0
    assert result["skipped"] == 0
    assert result["unresolved"] == []
    assert result["bibcodes"][0]["resolved"] is None
    assert result["bibcodes"][0]["reason"] == "no_ads_client"
    assert result["dois"][0]["resolved"] is None


def test_verify_citations_empty_ads_key_returns_unknown_and_does_not_call_client():
    fake = FakeADSClient(api_key="")

    result = verify_citations("Reference 2018ApJ...869L..41A.", fake)

    assert result["checked"] == 0
    assert fake.calls == []
    assert result["bibcodes"][0]["resolved"] is None
    assert result["bibcodes"][0]["reason"] == "no_ads_key"


def test_verify_citations_uses_cache_without_incrementing_checked():
    good_bibcode = "2018ApJ...869L..41A"
    fake = FakeADSClient(details={good_bibcode: {"title": "Resolved"}})
    cache = {}

    first = verify_citations(good_bibcode, fake, cache=cache, deadline_seconds=None)
    second = verify_citations(good_bibcode, fake, cache=cache, deadline_seconds=None)

    assert first["checked"] == 1
    assert second["checked"] == 0
    assert len(fake.calls) == 1
    assert second["bibcodes"][0]["resolved"] is True


def test_append_citation_warning_streams_warning_token_for_unresolved():
    bad_bibcode = "2099ApJ...999Z...9X"
    fake = FakeADSClient()
    tokens = []

    answer = append_citation_warning(
        f"This cites {bad_bibcode}.",
        fake,
        on_token=tokens.append,
        deadline_seconds=None,
    )

    assert len(tokens) == 1
    assert "Unverified citations" in tokens[0]
    assert bad_bibcode in tokens[0]
    assert answer.endswith(tokens[0])
