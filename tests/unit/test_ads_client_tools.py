"""integrations/ads_client.py — the ADS tool backends and the bounded request core.

Until 2026-09-21 the tools get_author_papers, get_paper_metrics,
get_author_metrics, export_bibtex, list_ads_libraries, get_ads_library_papers,
create_ads_library and add_to_ads_library (core/tool_registrations.py) called
ADSService methods that did not exist, so every call raised AttributeError
before any network call. These tests drive the implementations through a fake
requests session (no network) and pin the latency contract shared with
search_papers: the client's default timeout x attempts, tool-budget clamp,
host circuit breaker, fast failure on a long 429 Retry-After.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
import requests

from integrations import ads_client as ads_mod
from integrations.ads_client import ADSService
from services import http_budget_hook as hook
from services import tool_budgets as tb
from services.host_breaker import HostBreaker, HostCircuitOpen

REPO = Path(__file__).resolve().parents[2]
BASE = "https://api.adsabs.harvard.edu/v1"
HOST = "api.adsabs.harvard.edu"
BIB = "2018ApJ...869L..41A"
BIB2 = "2020PASP..132c5001L"
ATTEMPTS = ADSService.DEFAULT_RETRY_ATTEMPTS

ADS_TOOLS_ONE_CALL = {
    "get_author_papers", "get_paper_metrics", "get_paper_abstract", "export_bibtex",
    "list_ads_libraries", "create_ads_library", "add_to_ads_library",
}
ADS_TOOLS_TWO_CALLS = {"get_author_metrics", "get_ads_library_papers"}


# ── fakes ────────────────────────────────────────────────────────────────


class FakeResponse:
    def __init__(self, status_code=200, payload=None, *, text=None, headers=None):
        self.status_code = status_code
        self._payload = payload
        self.headers = headers or {}
        if text is not None:
            self.text = text
        else:
            self.text = json.dumps(payload) if payload is not None else ""

    def json(self):
        if self._payload is None:
            raise ValueError("no JSON body")
        return self._payload


class FakeSession:
    """Stands in for requests.Session: records every call, answers from a
    queue of FakeResponse instances or exceptions (which are raised)."""

    def __init__(self, *responses):
        self.queue = list(responses)
        self.calls = []

    def _answer(self, method, url, headers, params, json_body, timeout):
        self.calls.append({
            "method": method, "url": url, "headers": dict(headers or {}),
            "params": params, "json": json_body, "timeout": timeout,
        })
        if not self.queue:
            raise AssertionError(f"unexpected {method} {url}")
        item = self.queue.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item

    def get(self, url, headers=None, params=None, timeout=None):
        return self._answer("GET", url, headers, params, None, timeout)

    def post(self, url, headers=None, params=None, json=None, timeout=None):
        return self._answer("POST", url, headers, params, json, timeout)


def _svc(*responses, api_key="test-key", **kw):
    return ADSService(api_key=api_key, base_url=BASE, session=FakeSession(*responses), **kw)


def _search_payload(docs, num_found=None):
    return {"response": {"numFound": len(docs) if num_found is None else num_found, "start": 0, "docs": docs}}


def _doc(bibcode=BIB, title="Paper", year="2018", cites=900):
    return {
        "bibcode": bibcode, "title": [title], "author": ["Andrews, S.", "Huang, J."], "year": year,
        "citation_count": cites, "read_count": 10, "pub": "ApJL", "doi": ["10.3847/x"],
        "property": ["REFEREED"], "keyword": ["disks"], "doctype": "article",
    }


# Shape probed live 2026-09-21 (keys carry spaces).
METRICS = {
    "skipped bibcodes": [],
    "basic stats": {"number of papers": 1, "total number of reads": 1234, "recent number of reads": 56,
                    "total number of downloads": 78},
    "citation stats": {"number of citing papers": 900, "total number of citations": 950,
                       "number of self-citations": 12, "total number of refereed citations": 800},
    "citation stats refereed": {"total number of citations": 940},
    "indicators": {"h": 1, "g": 1, "i10": 1, "i100": 1, "m": 0.14, "read10": 3.2, "tori": 1.1, "riq": 100},
    "indicators refereed": {"h": 1},
}


@pytest.fixture(autouse=True)
def _isolated(monkeypatch):
    """No real sleeps, a closed breaker, no tool deadline, default breaker policy."""
    slept = []
    monkeypatch.setattr(ads_mod.time, "sleep", lambda s: slept.append(s))
    for var in ("HOST_BREAKER_DISABLED", "HOST_BREAKER_TRIP_FAILURES", "QUASAR_TOOL_TIMEOUT_SECONDS",
                "QUASAR_TOOL_TIMEOUT_OVERRIDES"):
        monkeypatch.delenv(var, raising=False)
    HostBreaker.reset()
    tb.end_tool_deadline()
    yield slept
    HostBreaker.reset()
    tb.end_tool_deadline()


# ── the regression itself ────────────────────────────────────────────────


def test_every_registered_ads_tool_calls_an_existing_method():
    src = (REPO / "core" / "tool_registrations.py").read_text(encoding="utf-8", errors="replace")
    methods = sorted(set(re.findall(r"agent\.ads_client\.([A-Za-z_]\w*)\(", src)))
    assert {"get_author_papers", "get_paper_metrics", "get_author_metrics", "export_bibtex", "list_libraries",
            "get_library_papers", "create_library", "add_to_library", "get_paper_details"} <= set(methods)
    missing = [m for m in methods if not callable(getattr(ADSService, m, None))]
    assert missing == [], f"tool registrations call ADSService methods that do not exist: {missing}"


def test_budget_declarations_cover_every_ads_tool():
    one_call = ADSService.worst_case_seconds()
    assert one_call == ATTEMPTS * ADSService.DEFAULT_TIMEOUT_S + sum(2.0 ** i for i in range(ATTEMPTS - 1))
    assert one_call == sum(tb._ads().values()), "client arithmetic and the budget table agree"
    assert ADSService.worst_case_seconds(2) == 2 * one_call
    for tool in ADS_TOOLS_ONE_CALL | ADS_TOOLS_TWO_CALLS:
        assert tool in tb.INNER_BUDGETS, f"{tool} has no inner-budget declaration"
        assert f"{HOST}/v1" in tb.hosts_for(tool)  # the /v1 service circuit (guard CX-10)
        total = sum(tb.INNER_BUDGETS[tool]().values())
        assert total == (one_call if tool in ADS_TOOLS_ONE_CALL else 2 * one_call), (tool, total)
    assert tb.check_hierarchy(sorted(ADS_TOOLS_ONE_CALL | ADS_TOOLS_TWO_CALLS)) == []


# ── get_author_papers ────────────────────────────────────────────────────


def test_get_author_papers_queries_the_author_field():
    svc = _svc(FakeResponse(200, _search_payload([_doc(BIB), _doc(BIB2, "Second", "2020")], num_found=295)))
    out = svc.get_author_papers("Accomazzi, Alberto", max_results=2)
    call = svc.session.calls[0]
    assert call["method"] == "GET" and call["url"] == f"{BASE}/search/query"
    assert call["params"]["q"] == 'author:"Accomazzi, Alberto"'
    assert call["params"]["rows"] == 2 and call["params"]["sort"] == "date desc"
    assert call["headers"]["Authorization"] == "Bearer test-key"
    assert call["timeout"] == ADSService.DEFAULT_TIMEOUT_S
    assert out["success"] is True and out["num_found"] == 295 and out["returned"] == 2
    assert out["truncated"] is True
    assert out["papers"][0]["bibcode"] == BIB and out["papers"][0]["title"] == "Paper"
    assert out["papers"][0]["link"].endswith(BIB)


def test_get_author_papers_escapes_quotes_and_applies_refereed_filter():
    svc = _svc(FakeResponse(200, _search_payload([])))
    out = svc.get_author_papers('O"Neil, Kim', max_results="5", refereed_only=True)
    params = svc.session.calls[0]["params"]
    assert params["q"] == 'author:"O\\"Neil, Kim"'
    assert params["rows"] == 5 and params["fq"] == ["property:refereed"]
    assert out["success"] is True and out["papers"] == [] and out["num_found"] == 0


def test_get_author_papers_requires_a_name():
    svc = _svc()
    out = svc.get_author_papers("   ")
    assert out["success"] is False and "author name" in out["error"].lower()
    assert svc.session.calls == []


# ── get_paper_metrics ────────────────────────────────────────────────────


def test_get_paper_metrics_posts_to_the_metrics_service():
    svc = _svc(FakeResponse(200, METRICS))
    out = svc.get_paper_metrics(BIB)
    call = svc.session.calls[0]
    assert call["method"] == "POST" and call["url"] == f"{BASE}/metrics"
    assert call["json"] == {"bibcodes": [BIB], "types": ["basic", "citations", "indicators"]}
    assert call["headers"]["Content-Type"] == "application/json"
    assert out["success"] is True and out["bibcode"] == BIB
    assert out["total_citations"] == 950 and out["refereed_citations"] == 800
    assert out["total_reads"] == 1234 and out["h_index"] == 1 and out["tori"] == 1.1
    assert out["citation_stats"]["number of citing papers"] == 900


def test_get_paper_metrics_unknown_bibcode_is_a_typed_error():
    # Probed live 2026-09-21: HTTP 200 with an "Error" key, not a 404.
    svc = _svc(FakeResponse(200, {"Error": "Unable to get results!",
                                  "Error Info": "No data available to generate metrics"}))
    out = svc.get_paper_metrics("2099ZZZZ..999Z...9Z")
    assert out["success"] is False
    assert "no metrics" in out["error"].lower() and "No data available" in out["error"]


# ── get_author_metrics ───────────────────────────────────────────────────


def test_get_author_metrics_aggregates_the_authors_bibcodes():
    svc = _svc(
        FakeResponse(200, _search_payload([{"bibcode": BIB}, {"bibcode": BIB2}], num_found=295)),
        FakeResponse(200, {**METRICS, "indicators": {**METRICS["indicators"], "h": 42, "i10": 120}}),
    )
    out = svc.get_author_metrics("Accomazzi, Alberto")
    search, metrics = svc.session.calls
    assert search["method"] == "GET" and search["params"]["q"] == 'author:"Accomazzi, Alberto"'
    assert search["params"]["fl"] == "bibcode" and search["params"]["rows"] == ADSService.METRICS_MAX_BIBCODES
    assert metrics["method"] == "POST" and metrics["url"] == f"{BASE}/metrics"
    assert metrics["json"]["bibcodes"] == [BIB, BIB2]
    assert out["success"] is True and out["h_index"] == 42 and out["i10_index"] == 120
    assert out["papers_considered"] == 2 and out["num_found"] == 295 and out["truncated"] is True
    assert "295" in out["note"]


def test_get_author_metrics_without_papers_makes_no_metrics_call():
    svc = _svc(FakeResponse(200, _search_payload([])))
    out = svc.get_author_metrics("Nobody, Nemo")
    assert out["success"] is False and "No ADS papers found" in out["error"]
    assert len(svc.session.calls) == 1


# ── export_bibtex ────────────────────────────────────────────────────────


def test_export_bibtex_posts_bibcodes_and_returns_entries():
    export = f"@ARTICLE{{{BIB},\n author = {{Andrews}},\n}}\n\n@ARTICLE{{{BIB2},\n author = {{L}},\n}}\n"
    svc = _svc(FakeResponse(200, {"msg": "Retrieved 2 abstracts, starting with number 1.", "export": export}))
    out = svc.export_bibtex(f"{BIB}, {BIB2} {BIB}")  # string input: split, de-duplicated, order kept
    call = svc.session.calls[0]
    assert call["method"] == "POST" and call["url"] == f"{BASE}/export/bibtex"
    assert call["json"] == {"bibcode": [BIB, BIB2]}
    assert out["success"] is True and out["count"] == 2 and out["entries"] == 2
    assert out["bibtex"] == export and "missing_bibcodes" not in out


def test_export_bibtex_flags_bibcodes_the_service_dropped():
    svc = _svc(FakeResponse(200, {"msg": "Retrieved 1 abstracts", "export": f"@ARTICLE{{{BIB},\n}}\n"}))
    out = svc.export_bibtex([BIB, BIB2])
    assert out["success"] is True and out["missing_bibcodes"] == [BIB2]


@pytest.mark.parametrize("bibcodes,fmt", [([], "bibtex"), ("", "bibtex"), ([BIB], "docx")])
def test_export_bibtex_rejects_bad_input_without_network(bibcodes, fmt):
    svc = _svc()
    out = svc.export_bibtex(bibcodes, format=fmt)
    assert out["success"] is False and svc.session.calls == []


# ── libraries ────────────────────────────────────────────────────────────


def test_list_libraries_formats_the_account_libraries():
    payload = {"count": 1, "libraries": [{
        "id": "AbCdEf123", "name": "Disks", "description": "ALMA disk papers", "num_documents": 12,
        "public": False, "permission": "owner", "owner": "adam", "date_created": "2026-01-01T00:00:00",
        "date_last_modified": "2026-02-01T00:00:00", "num_users": 1,
    }]}
    svc = _svc(FakeResponse(200, payload))
    out = svc.list_libraries()
    call = svc.session.calls[0]
    assert call["method"] == "GET" and call["url"] == f"{BASE}/biblib/libraries"
    assert out["success"] is True and out["count"] == 1
    lib = out["libraries"][0]
    assert lib["id"] == "AbCdEf123" and lib["num_documents"] == 12 and lib["public"] is False
    assert lib["link"] == "https://ui.adsabs.harvard.edu/user/libraries/AbCdEf123"


def test_get_library_papers_uses_the_solr_block():
    payload = {
        "documents": [BIB, BIB2],
        "solr": {"response": {"numFound": 2, "start": 0, "docs": [_doc(BIB), _doc(BIB2, "Second", "2020")]}},
        "metadata": {"id": "LIB1", "name": "Disks", "num_documents": 2, "public": False, "permission": "owner"},
        "updates": {},
    }
    svc = _svc(FakeResponse(200, payload))
    out = svc.get_library_papers("LIB1", max_results=10)
    call = svc.session.calls[0]
    assert call["url"] == f"{BASE}/biblib/libraries/LIB1"
    assert call["params"]["rows"] == 10 and call["params"]["start"] == 0 and "title" in call["params"]["fl"]
    assert out["success"] is True and out["num_documents"] == 2 and out["returned"] == 2
    assert [p["bibcode"] for p in out["papers"]] == [BIB, BIB2]
    assert out["library"]["name"] == "Disks"
    assert len(svc.session.calls) == 1


def test_get_library_papers_falls_back_to_a_bibcode_search():
    svc = _svc(
        FakeResponse(200, {"documents": [BIB], "solr": {}, "metadata": {"num_documents": 1}}),
        FakeResponse(200, _search_payload([_doc(BIB)])),
    )
    out = svc.get_library_papers("LIB1")
    assert len(svc.session.calls) == 2
    assert svc.session.calls[1]["url"] == f"{BASE}/search/query"
    assert svc.session.calls[1]["params"]["q"] == f'bibcode:("{BIB}")'
    assert out["success"] is True and out["papers"][0]["title"] == "Paper" and out["bibcodes"] == [BIB]


@pytest.mark.parametrize("bad_id", ["../libraries", "", "a b", "x/y"])
def test_library_id_is_validated_before_any_request(bad_id):
    svc = _svc()
    assert svc.get_library_papers(bad_id)["success"] is False
    assert svc.add_to_library(bad_id, [BIB])["success"] is False
    assert svc.session.calls == []


def test_create_library_posts_the_payload():
    svc = _svc(FakeResponse(200, {"name": "Disks", "id": "NewLib1", "description": "d", "permission": "owner",
                                  "public": False, "bibcode": []}))
    out = svc.create_library("Disks", description="d", public="false")
    call = svc.session.calls[0]
    assert call["method"] == "POST" and call["url"] == f"{BASE}/biblib/libraries"
    assert call["json"] == {"name": "Disks", "description": "d", "public": False, "bibcode": []}
    assert out["success"] is True and out["library_id"] == "NewLib1"
    assert out["link"] == "https://ui.adsabs.harvard.edu/user/libraries/NewLib1"


def test_create_library_requires_a_name():
    svc = _svc()
    assert svc.create_library("")["success"] is False and svc.session.calls == []


def test_add_to_library_posts_the_add_action():
    svc = _svc(FakeResponse(200, {"number_added": 1}))
    out = svc.add_to_library("LIB1", BIB)  # single bibcode as a plain string
    call = svc.session.calls[0]
    assert call["method"] == "POST" and call["url"] == f"{BASE}/biblib/documents/LIB1"
    assert call["json"] == {"bibcode": [BIB], "action": "add"}
    assert out["success"] is True and out["number_added"] == 1 and out["requested"] == 1


def test_add_to_library_rejects_empty_bibcodes_and_unknown_action():
    svc = _svc()
    assert svc.add_to_library("LIB1", [])["success"] is False
    assert svc.add_to_library("LIB1", [BIB], action="purge")["success"] is False
    assert svc.session.calls == []


# ── shared failure contract ──────────────────────────────────────────────


@pytest.mark.parametrize("method,args", [
    ("get_author_papers", ("Accomazzi, Alberto",)),
    ("get_paper_metrics", (BIB,)),
    ("get_author_metrics", ("Accomazzi, Alberto",)),
    ("export_bibtex", ([BIB],)),
    ("list_libraries", ()),
    ("get_library_papers", ("LIB1",)),
    ("create_library", ("Disks",)),
    ("add_to_library", ("LIB1", [BIB])),
])
def test_missing_api_key_is_a_typed_error_without_network(monkeypatch, method, args):
    monkeypatch.delenv("NASA_ADS_API_KEY", raising=False)
    monkeypatch.delenv("SCIX_API_KEY", raising=False)
    svc = _svc(api_key=None)
    out = getattr(svc, method)(*args)
    assert out["success"] is False and "API key" in out["error"]
    assert svc.session.calls == []


def test_http_4xx_is_reported_without_retry():
    svc = _svc(FakeResponse(401, {"error": "Unauthorized"}))
    out = svc.list_libraries()
    assert out["success"] is False and "401" in out["error"]
    assert len(svc.session.calls) == 1
    assert not HostBreaker.is_open(HOST), "a 4xx is about the request, not the host"


def test_html_500_from_biblib_is_summarised_with_a_hint():
    # Probed live 2026-09-21: an unknown library id answers an HTML 500 page.
    html = ("<!doctype html>\n<html lang=en>\n<title>500 Internal Server Error</title>\n"
            "<h1>Internal Server Error</h1>")
    svc = _svc(FakeResponse(500, None, text=html))
    out = svc.get_library_papers("doesnotexist123")
    assert out["success"] is False
    assert "<" not in out["error"] and "500" in out["error"] and "does not exist" in out["error"]


def test_transport_failures_retry_then_report(monkeypatch, _isolated):
    monkeypatch.setenv("HOST_BREAKER_DISABLED", "1")  # isolate the retry ladder itself
    svc = _svc(*[requests.Timeout("read timed out") for _ in range(ATTEMPTS)])
    out = svc.get_paper_metrics(BIB)
    assert out["success"] is False and "timed out" in out["error"]
    assert len(svc.session.calls) == ATTEMPTS
    assert _isolated == [2.0 ** i for i in range(ATTEMPTS - 1)], "1 s, 2 s, ... backoff between attempts"


def test_gateway_status_is_a_soft_failure_and_is_not_retried():
    # A 503 is SOFT (UI benchmark 2026-09-22, L07): one gateway error from one
    # tool call is recorded but does not open the host; three from DISTINCT
    # calls spread over >= 20 s do (tests/unit/test_host_breaker.py).
    svc = _svc(FakeResponse(503, None, text="Service Unavailable"))
    out = svc.get_author_papers("Accomazzi, Alberto")
    assert out["success"] is False and "503" in out["error"]
    assert len(svc.session.calls) == 1
    assert not HostBreaker.is_open(HOST)


def test_a_transport_failure_opens_the_ads_breaker_so_the_next_call_fails_fast():
    svc = _svc(requests.ConnectionError("connection refused"), FakeResponse(200, METRICS))
    # Attempt 1 fails and opens the host (HOST_BREAKER_TRIP_FAILURES default 1);
    # attempt 2 is refused by the breaker instead of paying another timeout.
    # HostCircuitOpen propagates: the tool guard turns it into the structured
    # INFRASTRUCTURE FAILURE result (core/agent.py).
    with pytest.raises(HostCircuitOpen) as info:
        svc.get_paper_metrics(BIB)
    assert info.value.host == HOST
    assert len(svc.session.calls) == 1
    # Every later ADS call this turn — any tool — is refused before it is sent.
    other = _svc(FakeResponse(200, {"count": 0, "libraries": []}))
    with pytest.raises(HostCircuitOpen):
        other.list_libraries()
    assert other.session.calls == []


def test_timeout_is_clamped_to_the_tool_budget():
    tb.adopt_deadline(tb.Deadline(4.0, label="get_author_papers"))
    svc = _svc(FakeResponse(200, _search_payload([])))
    out = svc.get_author_papers("Accomazzi, Alberto")
    assert out["success"] is True
    assert 3.0 < svc.session.calls[0]["timeout"] <= 4.0, "10 s default clamped to the 4 s remaining"


def test_exhausted_budget_refuses_before_sending():
    tb.adopt_deadline(tb.Deadline(0.0, label="export_bibtex"))
    svc = _svc(FakeResponse(200, {"export": "@ARTICLE{x,}"}))
    out = svc.export_bibtex([BIB])
    assert out["success"] is False and "budget exhausted" in out["error"]
    assert svc.session.calls == []


def test_backoff_never_outlasts_the_budget(_isolated):
    tb.adopt_deadline(tb.Deadline(1.5, label="get_paper_metrics"))
    svc = _svc(requests.Timeout("read timed out"), FakeResponse(200, METRICS))
    out = svc.get_paper_metrics(BIB)
    assert out["success"] is False and "retry abandoned" in out["error"]
    assert len(svc.session.calls) == 1 and _isolated == [], "no 1 s sleep with 1.5 s left"


def test_429_with_a_short_retry_after_is_retried(_isolated):
    svc = _svc(FakeResponse(429, {"error": "rate"}, headers={"Retry-After": "2"}), FakeResponse(200, METRICS))
    out = svc.get_paper_metrics(BIB)
    assert out["success"] is True and len(svc.session.calls) == 2 and _isolated == [2.0]


def test_429_with_a_long_retry_after_fails_fast(_isolated):
    svc = _svc(FakeResponse(429, {"error": "rate"}, headers={"Retry-After": "3600"}), FakeResponse(200, METRICS))
    out = svc.get_paper_metrics(BIB)
    assert out["success"] is False and "rate limit" in out["error"].lower() and "3600" in out["error"]
    assert len(svc.session.calls) == 1 and _isolated == []


def test_get_paper_details_still_works_through_the_shared_core():
    svc = _svc(FakeResponse(200, _search_payload([_doc(BIB)])))
    details = svc.get_paper_details(BIB)
    assert details["bibcode"] == BIB and details["title"] == "Paper"
    assert svc.session.calls[0]["params"]["q"] == f"bibcode:{BIB}"


# ── requests hook: wired explicitly, not double counted ──────────────────


class _RefusingSession(requests.Session):
    """A real requests.Session whose transport always refuses — so the call
    passes through the (patched) Session.request like production traffic."""

    def send(self, request, **kwargs):
        raise requests.exceptions.ConnectionError("connection refused")


def test_ads_calls_run_suppressed_under_the_requests_hook(monkeypatch):
    monkeypatch.delenv("QUASAR_HTTP_BUDGET_HOOK", raising=False)
    monkeypatch.setenv("HOST_BREAKER_TRIP_FAILURES", "2")
    hook.uninstall()
    assert hook.install() is True
    try:
        tb.begin_tool_deadline("get_author_papers", 60.0)
        svc = ADSService(api_key="test-key", base_url=BASE, session=_RefusingSession(), retry_attempts=1)
        out = svc.get_author_papers("Accomazzi, Alberto")
        assert out["success"] is False and "connection refused" in out["error"]
        # One transport failure was recorded — by the client, not also by the hook.
        assert not HostBreaker.is_open(HOST)
        assert HostBreaker._failures.get(HOST) == 1
    finally:
        hook.uninstall()
