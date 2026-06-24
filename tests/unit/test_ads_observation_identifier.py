from integrations.ads_client import ADSService


class CapturingADSService(ADSService):
    def __init__(self):
        super().__init__(api_key="test-key")
        self.calls = []

    def search_papers(self, query, max_results=10, sort="date desc", fields=None, filters=None, start=0):
        self.calls.append({
            "query": query,
            "max_results": max_results,
            "sort": sort,
            "filters": filters,
        })
        return [{
            "title": "Paper using exact ALMA project code",
            "bibcode": "2024ApJ...000...1A",
            "citations": 3,
        }]


def test_classifies_alma_project_code():
    assert ADSService.classify_observation_identifier("2019.1.00123.S") == "project_code"


def test_classifies_mous_uid():
    assert ADSService.classify_observation_identifier("uid://A001/X1/X2") == "mous_uid"


def test_identifier_search_uses_exact_ads_fields_and_adds_provenance():
    service = CapturingADSService()

    result = service.search_by_observation_identifier("2019.1.00123.S", max_results=5)

    call = service.calls[0]
    assert call["max_results"] == 5
    assert call["sort"] == "score desc"
    assert call["filters"] == ["property:refereed"]
    assert "bibgroup:ALMA" in call["query"]
    assert 'body:"2019.1.00123.S"' in call["query"]
    assert 'identifier:"2019.1.00123.S"' in call["query"]
    assert result["identifier_type"] == "project_code"
    assert result["papers"][0]["observation_links"][0]["identifier"] == "2019.1.00123.S"
    assert result["papers"][0]["observation_links"][0]["confidence"] == "explicit"
