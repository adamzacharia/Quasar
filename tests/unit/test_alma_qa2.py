import requests

from services import alma_qa2
from services.alma_qa2 import (
    QA2_UNKNOWN,
    QA2StatusResult,
    fetch_qa2_status,
    fetch_qa2_statuses,
    normalize_mous_uid,
    parse_qa2_status,
    qa2_report_url,
)


def test_normalizes_mous_uid_and_builds_report_url():
    assert normalize_mous_uid("uid://A001/X12a3/X407") == "A001_X12a3_X407"
    assert normalize_mous_uid("uid___A001_X12a3_X407") == "A001_X12a3_X407"
    assert normalize_mous_uid("member.uid___A001_X12a3_X407.qa2_report.pdf") == "A001_X12a3_X407"

    assert qa2_report_url("uid://A001/X12a3/X407") == (
        "https://almascience.nrao.edu/dataPortal/"
        "member.uid___A001_X12a3_X407.qa2_report.pdf"
    )


def test_parse_qa2_status_from_report_text():
    assert parse_qa2_status("QA2 Status\nSemiPass\nMember OUS Status ID") == "SemiPass"
    assert parse_qa2_status("QA2 Status: Pass") == "Pass"
    assert parse_qa2_status("QA2 Status\nSemi Pass") == "SemiPass"
    assert parse_qa2_status("QA2 Status\nFail") == "Fail"
    assert parse_qa2_status("No status in this document") == QA2_UNKNOWN


def test_fetch_qa2_status_returns_unknown_on_network_error():
    def failing_get(*args, **kwargs):
        raise requests.Timeout("too slow")

    result = fetch_qa2_status("uid://A001/X12a3/X407", http_get=failing_get)

    assert result.normalized_uid == "A001_X12a3_X407"
    assert result.status == QA2_UNKNOWN
    assert "too slow" in result.error


def test_fetch_qa2_status_uses_pdf_extractor(monkeypatch):
    class FakeResponse:
        headers = {"Content-Type": "application/pdf"}
        content = b"%PDF fake"

        def raise_for_status(self):
            return None

    def fake_get(url, **kwargs):
        assert url.endswith("member.uid___A001_X12a3_X407.qa2_report.pdf")
        return FakeResponse()

    monkeypatch.setattr(alma_qa2, "extract_qa2_status_from_pdf_bytes", lambda raw: "SemiPass")

    result = fetch_qa2_status("uid://A001/X12a3/X407", http_get=fake_get)

    assert result.status == "SemiPass"
    assert result.error == ""


def test_fetch_qa2_statuses_deduplicates_and_caps(monkeypatch):
    calls = []

    def fake_fetch(uid, timeout_s=6.0):
        calls.append(uid)
        return QA2StatusResult(normalized_uid=normalize_mous_uid(uid), status="Pass")

    monkeypatch.setattr(alma_qa2, "fetch_qa2_status", fake_fetch)

    result = fetch_qa2_statuses(
        [
            "uid://A001/X1/X1",
            "uid___A001_X1_X1",
            "uid://A001/X2/X1",
            "uid://A001/X3/X1",
        ],
        max_lookup=2,
        overall_timeout_s=2.0,
    )

    assert result.requested == 3
    assert result.looked_up == 2
    assert result.capped is True
    assert calls == ["A001_X1_X1", "A001_X2_X1"]
    assert result.statuses["A001_X1_X1"] == "Pass"
    assert result.statuses["A001_X2_X1"] == "Pass"
    assert result.statuses["A001_X3_X1"] == QA2_UNKNOWN
