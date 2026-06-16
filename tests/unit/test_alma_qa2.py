import sys
import tempfile
import threading
import time
from pathlib import Path

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


def _local_temp_cache_path():
    base = Path("test_results") / "qa2_cache_tests"
    base.mkdir(parents=True, exist_ok=True)
    tmp = tempfile.TemporaryDirectory(dir=base, ignore_cleanup_errors=True)
    return tmp, Path(tmp.name) / "qa2.sqlite3"


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
        cache_ttl_days=0,
    )

    assert result.requested == 3
    assert result.looked_up == 2
    assert result.capped is True
    assert calls == ["A001_X1_X1", "A001_X2_X1"]
    assert result.statuses["A001_X1_X1"] == "Pass"
    assert result.statuses["A001_X2_X1"] == "Pass"
    assert result.statuses["A001_X3_X1"] == QA2_UNKNOWN


def test_fetch_qa2_statuses_uses_persistent_cache(monkeypatch):
    calls = []
    tmp, cache_path = _local_temp_cache_path()

    def fake_fetch(uid, timeout_s=6.0):
        calls.append(uid)
        return QA2StatusResult(
            normalized_uid=normalize_mous_uid(uid),
            status="SemiPass",
            report_url="https://example.test/report.pdf",
        )

    monkeypatch.setattr(alma_qa2, "fetch_qa2_status", fake_fetch)

    try:
        first = fetch_qa2_statuses(
            ["uid://A001/X12a3/X407"],
            max_lookup=1,
            cache_ttl_days=30,
            cache_path=cache_path,
        )
        second = fetch_qa2_statuses(
            ["uid://A001/X12a3/X407"],
            max_lookup=1,
            cache_ttl_days=30,
            cache_path=cache_path,
        )
    finally:
        tmp.cleanup()

    assert first.statuses["A001_X12a3_X407"] == "SemiPass"
    assert second.statuses["A001_X12a3_X407"] == "SemiPass"
    assert second.cache_hits == 1
    assert calls == ["A001_X12a3_X407"]


def test_fetch_qa2_statuses_does_not_cache_timeout(monkeypatch):
    calls = []
    tmp, cache_path = _local_temp_cache_path()

    def fake_fetch(uid, timeout_s=6.0):
        calls.append(uid)
        return QA2StatusResult(
            normalized_uid=normalize_mous_uid(uid),
            status=QA2_UNKNOWN,
            error="QA2 lookup timed out",
        )

    monkeypatch.setattr(alma_qa2, "fetch_qa2_status", fake_fetch)

    try:
        for _ in range(2):
            result = fetch_qa2_statuses(
                ["uid://A001/X12a3/X407"],
                max_lookup=1,
                cache_ttl_days=30,
                cache_path=cache_path,
            )
            assert result.statuses["A001_X12a3_X407"] == QA2_UNKNOWN
            assert result.cache_hits == 0
    finally:
        tmp.cleanup()

    assert calls == ["A001_X12a3_X407", "A001_X12a3_X407"]


def test_fetch_qa2_statuses_uses_parallel_workers(monkeypatch):
    active = 0
    max_active = 0
    lock = threading.Lock()

    def fake_fetch(uid, timeout_s=6.0):
        nonlocal active, max_active
        with lock:
            active += 1
            max_active = max(max_active, active)
        time.sleep(0.05)
        with lock:
            active -= 1
        return QA2StatusResult(normalized_uid=normalize_mous_uid(uid), status="Pass")

    monkeypatch.setattr(alma_qa2, "fetch_qa2_status", fake_fetch)

    result = fetch_qa2_statuses(
        [f"uid://A001/X{i}/X1" for i in range(6)],
        max_lookup=6,
        max_workers=6,
        overall_timeout_s=2.0,
        cache_ttl_days=0,
    )

    assert result.looked_up == 6
    assert max_active > 1


def test_pdf_extraction_checks_page_one_before_page_two(monkeypatch):
    read_pages = []

    class FakePage:
        def __init__(self, index, text):
            self.index = index
            self.text = text

        def get_text(self, mode):
            read_pages.append(self.index)
            return self.text

    class FakeDoc:
        def __init__(self):
            self.pages = [
                FakePage(0, "QA2 Status\nPass"),
                FakePage(1, "QA2 Status\nSemiPass"),
            ]

        def __len__(self):
            return len(self.pages)

        def __getitem__(self, index):
            return self.pages[index]

        def close(self):
            return None

    class FakeFitz:
        @staticmethod
        def open(stream, filetype):
            return FakeDoc()

    monkeypatch.setitem(sys.modules, "fitz", FakeFitz)

    assert alma_qa2.extract_qa2_status_from_pdf_bytes(b"%PDF") == "Pass"
    assert read_pages == [0]


def test_pdf_extraction_falls_back_to_page_two(monkeypatch):
    class FakePage:
        def __init__(self, text):
            self.text = text

        def get_text(self, mode):
            return self.text

    class FakeDoc:
        pages = [
            FakePage("No status on first page"),
            FakePage("QA2 Status\nSemiPass"),
        ]

        def __len__(self):
            return len(self.pages)

        def __getitem__(self, index):
            return self.pages[index]

        def close(self):
            return None

    class FakeFitz:
        @staticmethod
        def open(stream, filetype):
            return FakeDoc()

    monkeypatch.setitem(sys.modules, "fitz", FakeFitz)

    assert alma_qa2.extract_qa2_status_from_pdf_bytes(b"%PDF") == "SemiPass"
