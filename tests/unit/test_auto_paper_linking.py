from services.ads_auto_link import build_exact_project_paper_links


class FakeADSClient:
    def __init__(self, papers_by_identifier=None):
        self.api_key = "test-key"
        self.calls = []
        self.papers_by_identifier = papers_by_identifier or {}

    def search_by_observation_identifier(self, identifier, max_results=20, facility="ALMA"):
        self.calls.append({
            "identifier": identifier,
            "max_results": max_results,
            "facility": facility,
        })
        papers = self.papers_by_identifier.get(identifier, [])
        return {
            "query": f'bibgroup:ALMA AND identifier:"{identifier}"',
            "identifier": identifier,
            "identifier_type": "project_code",
            "papers": papers,
        }


def test_auto_linking_searches_top_exact_project_codes_only(monkeypatch):
    monkeypatch.setenv("QUASAR_AUTO_LINK_PROJECT_PAPERS", "true")
    fake_ads = FakeADSClient({
        "2017.1.00001.S": [{
            "title": "Paper using exact ALMA project code",
            "bibcode": "2024ApJ...000...1A",
            "observation_links": [{
                "identifier": "2017.1.00001.S",
                "identifier_type": "project_code",
                "relation": "explicit_identifier_search",
                "confidence": "explicit",
            }],
        }],
        "2021.1.00473.S": [{
            "title": "Second exact project paper",
            "bibcode": "2024ApJ...000...2A",
            "observation_links": [{
                "identifier": "2021.1.00473.S",
                "identifier_type": "project_code",
                "relation": "explicit_identifier_search",
                "confidence": "explicit",
            }],
        }],
    })
    result = build_exact_project_paper_links(
        fake_ads,
        "search_by_target",
        {
            "success": True,
            "target": "Sz65",
            "top_project_codes": [
                "2017.1.00001.S",
                "not-a-project",
                "2021.1.00473.S",
                "2015.1.00222.S",
                "2018.1.00000.S",
            ],
        },
    )

    assert result["papers"]
    assert [call["identifier"] for call in fake_ads.calls] == [
        "2017.1.00001.S",
        "2021.1.00473.S",
        "2015.1.00222.S",
    ]
    assert all(call["facility"] == "ALMA" for call in fake_ads.calls)
    assert "Sz65" not in [call["identifier"] for call in fake_ads.calls]
    assert result["paper_provenance"]["auto_linked"] is True


def test_auto_linking_returns_clear_warning_for_no_exact_matches(monkeypatch):
    monkeypatch.setenv("QUASAR_AUTO_LINK_PROJECT_PAPERS", "true")
    fake_ads = FakeADSClient()

    result = build_exact_project_paper_links(
        fake_ads,
        "search_by_target",
        {
            "success": True,
            "target": "Sz65",
            "top_project_codes": ["2017.1.00001.S"],
        },
    )

    assert result["papers"] == []
    assert result["warnings"] == [
        "No NASA ADS papers were found with exact ALMA project-code links for 2017.1.00001.S."
    ]


def test_auto_linking_requires_real_ads_key(monkeypatch):
    monkeypatch.setenv("QUASAR_AUTO_LINK_PROJECT_PAPERS", "true")
    fake_ads = FakeADSClient()
    fake_ads.api_key = ""

    result = build_exact_project_paper_links(
        fake_ads,
        "search_by_target",
        {
            "success": True,
            "top_project_codes": ["2017.1.00001.S"],
        },
    )

    assert result is None
    assert fake_ads.calls == []
