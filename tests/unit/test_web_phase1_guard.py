"""Regression tests for the Phase 1 guard review (task-25bee13-20119), one per finding."""
from __future__ import annotations

import json
import threading
import time

import pytest

from services.web_evidence import (
    EvidenceRegistry,
    PrepassEvidence,
    canonicalize_url,
    clean_citations,
    deep_read,
    find_citations,
    split_sentences,
    unsupported_citation_claims,
)


def _reg(*pages, origin="prepass"):
    reg = EvidenceRegistry()
    reg.add_from_payload(
        {"success": True, "results": [{"title": t, "url": f"https://s{i}.org/{i}", "snippet": s} for i, (t, s) in enumerate(pages)]},
        origin=origin,
    )
    return reg


def _render(reg):
    return reg.render_prompt_block(retrieved="2026-09-24")


# CX-04: a late deep read never changes what the model was shown, and the check uses the shown text


def test_cx04_late_deep_read_does_not_change_the_injected_evidence():
    reg = _reg(("Guide", "The proprietary period is 12 months."))
    pre = PrepassEvidence(reg)
    pre.record_search({"success": True, "results": []})
    block = _render(reg)                                     # injected: the model saw "12 months"
    assert "12 months" in block
    assert reg.update_excerpt("https://s0.org/0", "Completely different text about 24 months.") is False
    assert reg.get("W1").shown_text.endswith("12 months.")
    assert unsupported_citation_claims("The period is 12 months [W1].", reg) == []


def test_cx04_deep_read_skips_pages_already_shown():
    reg = _reg(("A", "snippet a"), ("B", "snippet b"))
    _render(reg)
    called = []
    assert deep_read(reg, "q", budget_s=1, extractor=lambda u, q: called.append(u)) == 0 and called == []


# CX-08: only ids the model was shown are citable


def test_cx08_registered_but_unshown_ids_are_not_citable():
    reg = EvidenceRegistry()
    reg.add_from_payload({"success": True, "results": [{"url": f"https://a{i}.org", "title": f"T{i}", "snippet": "x"} for i in range(12)]})
    reg.render_prompt_block(max_items=10)
    assert reg.citable_ids() == [f"W{i}" for i in range(1, 11)]
    text, cited, removed = clean_citations("Fact [W3]. Guess [W11].", reg.citable_ids())
    assert cited == ["W3"] and removed == ["W11"] and "W11" not in text


def test_cx08_tool_result_tags_make_ids_citable():
    import json as _json

    reg = EvidenceRegistry()
    tagged = reg.annotate_tool_result({"success": True, "results": [{"url": "https://x.org/p", "title": "P", "snippet": "tool text 42"}]},
                                      origin="tool:web_search")
    reg.confirm_tool_shown(_json.dumps(tagged))   # what the model is sent
    assert reg.citable_ids() == ["W1"] and reg.get("W1").shown_text == "tool text 42"


# CX-09: the deep read targets the pre-pass search's own top results


def test_cx09_contact_pages_registered_first_are_not_deep_read():
    reg = EvidenceRegistry()
    reg.add_from_payload({"success": True, "results": [{"url": f"https://contact{i}.org", "snippet": "c"} for i in range(3)]},
                         origin="researcher_email")
    reg.add_from_payload({"success": True, "results": [{"url": f"https://main{i}.org", "snippet": "m"} for i in range(4)]},
                         origin="prepass")
    seen = {}
    deep_read(reg, "q", budget_s=1, extractor=lambda urls, q: seen.setdefault("urls", urls) and None)
    assert seen["urls"] == ["https://main0.org", "https://main1.org", "https://main2.org"]


# CX-10: tool results well past 40 pages still get tags


def test_cx10_many_tool_results_keep_their_tags():
    reg = EvidenceRegistry()
    for call in range(8):
        tagged = reg.annotate_tool_result(
            {"success": True, "results": [{"url": f"https://c{call}-{i}.org", "snippet": "s"} for i in range(10)]},
            origin="tool:web_search",
        )
        assert all("cite_as" in r for r in tagged["results"])
    assert len(reg) == 80


# CX-11: the block never exceeds its budget, even with long titles


def test_cx11_block_respects_the_budget_with_long_titles():
    reg = EvidenceRegistry()
    reg.add_from_payload({"success": True, "results": [
        {"url": f"https://a{i}.org", "title": "T" * 400, "snippet": "x " * 800} for i in range(10)
    ]})
    for limit in (2000, 4000, 8000):
        block = EvidenceRegistry.render_prompt_block(reg, max_total_chars=limit)
        assert 0 < len(block) <= limit


# CX-12: only the scheme's own default port is dropped


def test_cx12_ports():
    assert canonicalize_url("https://example.org:443/a") == "https://example.org/a"
    assert canonicalize_url("http://example.org:80/a") == "https://example.org/a"
    assert canonicalize_url("https://example.org:80/a") == "https://example.org:80/a"
    assert canonicalize_url("http://example.org:443/a") == "https://example.org:443/a"


# CX-13 / CX-22 (backend): indented code, existing links and escaped tags are not citations


def test_cx13_indented_code_is_protected():
    text = "Intro.\n\n    x = arr[W9]\n    y = 2\n\nReal [W9]."
    cleaned, cited, removed = clean_citations(text, [])
    assert "    x = arr[W9]" in cleaned and removed == ["W9"] and cleaned.endswith("Real.")
    assert find_citations("\n    x = arr[W9]") == []


def test_cx13_nested_list_continuation_is_still_prose():
    text = "- item\n    continued claim [W1]"
    assert find_citations(text) == ["W1"]


def test_cx22_links_and_escaped_tags_are_not_citations():
    assert find_citations("See [W1](https://example.com) and \\[W2\\] and [W3].") == ["W3"]


# CX-14 .. CX-19: the numeric / date checker


def test_cx14_percentages_are_checked_as_percentages():
    reg = _reg(("A", "The grant covers 35 hours and 12% of the time."))
    assert unsupported_citation_claims("About 12% of the time [W1].", reg) == []
    assert unsupported_citation_claims("About 3% of the time [W1].", reg)
    assert unsupported_citation_claims("About 35% of the time [W1].", reg)


def test_cx15_ids_must_match_as_ids():
    reg = _reg(("A", "Cycle 13 has 14 months of observing."))
    assert unsupported_citation_claims("This applies to Cycle 13 [W1].", reg) == []
    assert unsupported_citation_claims("This applies to Cycle 14 [W1].", reg)


def test_cx16_dates_are_checked_whole():
    reg = _reg(("A", "Proposals were due 15 October 2025; the TAC met on 7 June 2025."))
    assert unsupported_citation_claims("The deadline was October 15, 2025 [W1].", reg) == []
    assert unsupported_citation_claims("The deadline was 15 June 2025 [W1].", reg)
    reg2 = _reg(("NRAO", "2026 Oct 29 - 2027 Feb 15: C configuration."))
    assert unsupported_citation_claims("C configuration starts on October 29, 2026 [W1].", reg2) == []


def test_cx18_sentence_splitting_handles_abbreviations_and_lowercase_starts():
    assert split_sentences("Deadline is 15 Oct. 2025 [W1]. next one.") == ["Deadline is 15 Oct. 2025 [W1].", "next one."]
    reg = _reg(("A", "The deadline is 15 October 2025."))
    assert unsupported_citation_claims("Deadline is 15 Oct. 2025 [W1].", reg) == []
    assert unsupported_citation_claims("Deadline is 15 Nov. 2025 [W1].", reg)
    # a lowercase next sentence is its own claim: 42 is not attributed to W1
    reg2 = _reg(("A", "Twelve months."))
    assert unsupported_citation_claims("It is twelve months [W1]. it has 42 parts.", reg2) == []


def test_cx18_each_citation_checks_its_own_claim():
    reg = _reg(("A", "The period is 12 months."), ("B", "The DDT period is 24 months."))
    assert unsupported_citation_claims("It is 12 months [W1], and 24 months for DDT [W2].", reg) == []
    assert unsupported_citation_claims("It is 24 months [W1], and 12 months for DDT [W2].", reg)


def test_cx19_times_and_scientific_notation():
    reg = _reg(("A", "Submit by 8:00 pm EDT. The mass is 2 x 10^6 solar masses."))
    assert unsupported_citation_claims("The deadline is 8:00 pm EDT [W1].", reg) == []
    assert unsupported_citation_claims("The deadline is 9:00 pm EDT [W1].", reg)
    assert unsupported_citation_claims("It weighs 2e6 solar masses [W1].", reg) == []
    assert unsupported_citation_claims("It weighs 1e6 solar masses [W1].", reg)
    assert unsupported_citation_claims("It weighs 3 × 10⁶ solar masses [W1].", reg)


def test_prepass_threads_can_record_and_render_concurrently():
    reg = EvidenceRegistry()
    pre = PrepassEvidence(reg)

    def worker():
        pre.record_search({"success": True, "results": [{"url": f"https://p{i}.org", "snippet": "s"} for i in range(10)]})
        time.sleep(0.05)
        pre.finish()

    t = threading.Thread(target=worker)
    t.start()
    assert pre.wait(2) is True
    t.join()
    assert sum(1 for line in _render(reg).splitlines() if line.startswith("[W")) == 10


# ── verify round 1 reopen items ──────────────────────────────────────────


def test_r2_cx08_only_tags_that_survive_serialization_become_citable():
    import json as _json

    reg = EvidenceRegistry()
    tagged = reg.annotate_tool_result(
        {"success": True, "results": [{"url": f"https://t{i}.org", "snippet": f"text {i}"} for i in range(10)]},
        origin="tool:web_search",
    )
    assert reg.citable_ids() == []                   # nothing shown before serialization
    capped = _json.dumps({"results": tagged["results"][:2]})   # a budget cut kept only two items
    assert reg.confirm_tool_shown(capped) == ["W1", "W2"]
    assert reg.citable_ids() == ["W1", "W2"]


def test_r2_cx09_deep_read_uses_the_search_order_even_for_pages_registered_first_by_contacts():
    reg = EvidenceRegistry()
    reg.add_from_payload({"success": True, "results": [{"url": "https://shared.org/top", "snippet": "c"}]},
                         origin="researcher_email")
    pre = PrepassEvidence(reg)
    pre.record_search({"success": True, "results": [
        {"url": "https://shared.org/top", "snippet": "m"}, {"url": "https://b.org", "snippet": "m"},
    ]})
    seen = {}
    deep_read(reg, "q", budget_s=1, candidates=pre.items, extractor=lambda urls, q: seen.setdefault("urls", urls) and None)
    assert seen["urls"] == ["https://shared.org/top", "https://b.org"]


def test_r2_cx10_no_cap_every_result_is_tagged():
    reg = EvidenceRegistry()
    for call in range(21):
        tagged = reg.annotate_tool_result(
            {"success": True, "results": [{"url": f"https://c{call}-{i}.org", "snippet": "s"} for i in range(10)]},
            origin="tool:web_search",
        )
        assert all("cite_as" in r for r in tagged["results"])
    assert len(reg) == 210


def test_r2_cx25_a_later_tool_result_for_the_same_page_counts_as_shown_text():
    import json as _json

    reg = _reg(("Guide", "Short overview"))
    _render(reg)                                      # W1 shown as "Short overview"
    tagged = reg.annotate_tool_result(
        {"success": True, "results": [{"url": "https://s0.org/0", "raw_content": "The period is 24 months."}]},
        origin="tool:web_extract_url",
    )
    reg.confirm_tool_shown(_json.dumps(tagged))
    assert unsupported_citation_claims("The period is 24 months [W1].", reg) == []


def test_r3_cx08_citation_note_example_alone_does_not_confirm():
    from core.turn_recovery import serialize_tool_result

    reg = EvidenceRegistry()
    tagged = reg.annotate_tool_result(
        {"success": True, "results": [{"url": "https://a.org", "snippet": "The fee is 40 EUR."}]},
        origin="tool:web_search",
    )
    assert "[W1]" in tagged["citation_note"]
    survived = {k: v for k, v in tagged.items() if k != "results"}   # every item cut, note kept
    survived["results"] = []
    assert reg.confirm_tool_shown(json.dumps(survived)) == []
    assert reg.citable_ids() == []
    # page text that merely mentions a tag does not count either
    reg2 = EvidenceRegistry()
    tagged2 = reg2.annotate_tool_result(
        {"success": True, "results": [{"url": "https://b.org", "snippet": "s"}]}, origin="tool:web_search")
    fake = json.dumps({"results": [], "note": 'see "cite_as": "[W1]" and [W1]'})
    assert reg2.confirm_tool_shown(fake) == []
    # a preview cut (context overflow) still confirms tags whose item survived
    reg3 = EvidenceRegistry()
    tagged3 = reg3.annotate_tool_result(
        {"success": True, "results": [{"url": f"https://p{i}.org", "snippet": "x" * 400} for i in range(6)]},
        origin="tool:web_search",
    )
    preview = serialize_tool_result(tagged3, max_chars=900)
    marked = reg3.confirm_tool_shown(preview)
    assert marked and set(marked) <= {"W1", "W2", "W3", "W4", "W5", "W6"}
    for cid in marked:
        assert f'"cite_as": "[{cid}]"' in (json.loads(preview).get("preview_text") or preview)


def test_r3_cx25_full_tool_item_text_counts_as_shown():
    reg = EvidenceRegistry()
    long_text = ("Filler sentence about the observatory. " * 40) + "The proprietary period is 24 months."
    assert long_text.index("24 months") > 1550
    tagged = reg.annotate_tool_result(
        {"success": True, "results": [{"url": "https://a.org/p", "title": "Policy", "raw_content": long_text}]},
        origin="tool:web_extract_url",
    )
    reg.confirm_tool_shown(json.dumps(tagged))
    assert unsupported_citation_claims("The proprietary period is 24 months [W1].", reg) == []
    # every text field the model read counts, not just the first one
    reg2 = EvidenceRegistry()
    tagged2 = reg2.annotate_tool_result(
        {"success": True, "results": [{"url": "https://b.org", "snippet": "Overview.",
                                       "raw_content": "The deadline is 15 October 2026."}]},
        origin="tool:web_search",
    )
    reg2.confirm_tool_shown(json.dumps(tagged2))
    assert unsupported_citation_claims("The deadline is 15 October 2026 [W1].", reg2) == []


def test_r4_cx08_nested_metadata_tags_and_wrong_urls_never_confirm():
    reg = EvidenceRegistry()
    tagged = reg.annotate_tool_result(
        {"success": True, "results": [{"url": "https://a.org", "snippet": "A"}, {"url": "https://b.org", "snippet": "B"}]},
        origin="tool:web_search",
    )
    kept = dict(tagged["results"][0])
    kept["meta"] = {"cite_as": "[W2]", "url": "https://b.org"}    # W2 smuggled inside the retained W1 item
    sent = json.dumps({"success": True, "results": [kept], "meta": {"results": [tagged["results"][1]]}})
    assert reg.confirm_tool_shown(sent) == ["W1"]
    assert reg.citable_ids() == ["W1"]
    # a top-level item with W2's tag but another page's URL does not confirm W2
    reg2 = EvidenceRegistry()
    tagged2 = reg2.annotate_tool_result(
        {"success": True, "results": [{"url": "https://a.org", "snippet": "A"}, {"url": "https://b.org", "snippet": "B"}]},
        origin="tool:web_search",
    )
    forged = {"url": "https://evil.org", "snippet": "x", "cite_as": "[W2]"}
    assert reg2.confirm_tool_shown(json.dumps({"results": [tagged2["results"][0], forged]})) == ["W1"]


def test_r4_cx08_cut_preview_walks_only_top_level_items():
    from services.web_evidence import _items_in_cut_json

    full = json.dumps({"success": True, "meta": {"results": [{"cite_as": "[W2]", "url": "https://b.org"}]},
                       "results": [{"url": "https://a.org", "snippet": "x", "cite_as": "[W1]"},
                                   {"url": "https://b.org", "snippet": "yyyy", "cite_as": "[W2]"}],
                       "response": {"sources": [{"url": "https://c.org", "cite_as": "[W3]"}]}})
    tags = lambda text: [i.get("cite_as") for i in _items_in_cut_json(text)]
    assert tags(full) == ["[W1]", "[W2]", "[W3]"]
    assert tags(full[: full.index("yyyy")]) == ["[W1]"]                # the cut item does not count
    assert tags(full[: len(full) - 5]) == ["[W1]", "[W2]"]             # cut inside response.sources
    assert tags(full[: full.index('"results": [{"url": "https://a')]) == []   # nested meta ignored


def test_r4_cx25_two_items_for_the_same_page_both_count():
    reg = EvidenceRegistry()
    tagged = reg.annotate_tool_result(
        {"success": True, "results": [
            {"url": "https://a.org/p", "snippet": "The period is 24 months."},
            {"url": "https://a.org/p?utm_source=x", "snippet": "Renewals take 12 months."},
        ]},
        origin="tool:web_search",
    )
    assert [r["cite_as"] for r in tagged["results"]] == ["[W1]", "[W1]"]
    assert reg.confirm_tool_shown(json.dumps(tagged)) == ["W1"]
    assert unsupported_citation_claims("The period is 24 months [W1].", reg) == []
    assert unsupported_citation_claims("Renewals take 12 months [W1].", reg) == []


def test_r3_cite_as_is_the_last_key_even_when_the_item_had_one():
    reg = EvidenceRegistry()
    tagged = reg.annotate_tool_result(
        {"success": True, "results": [{"cite_as": "[W9]", "url": "https://a.org", "snippet": "s"}]},
        origin="tool:web_search",
    )
    assert list(tagged["results"][0])[-1] == "cite_as"
    assert tagged["results"][0]["cite_as"] == "[W1]"


def test_r2_cx26_tool_citation_note_does_not_invite_url_dates():
    reg = EvidenceRegistry()
    tagged = reg.annotate_tool_result({"success": True, "results": [{"url": "https://a.org", "snippet": "x"}]},
                                      origin="tool:web_search")
    assert "(not its URL)" in tagged["citation_note"]


def test_r2_cx21_sse_merge_keeps_distinct_case_and_query_pages():
    import importlib
    import sys

    sys.path.insert(0, "ui-pro")
    sse = importlib.import_module("api.sse")
    merged = sse._merge_web_sources([], [
        {"id": "W1", "url": "https://x.org/Case", "title": "A"},
        {"id": "W2", "url": "https://x.org/case", "title": "B"},
        {"id": "W3", "url": "https://x.org/a?id=1", "title": "C"},
        {"id": "W4", "url": "https://x.org/a?id=2", "title": "D"},
        {"id": "W4", "url": "https://www.x.org/a?id=2&utm_source=z", "title": "D"},
    ])
    assert [s["id"] for s in merged] == ["W1", "W2", "W3", "W4"]


def test_r2_cx06_bounded_post_stops_a_trickling_response(monkeypatch):
    from services import web_search_service as wss

    class Trickle:
        status_code = 200

        def iter_content(self, chunk_size=16384):
            while True:
                time.sleep(0.2)
                yield b" "

        def close(self):
            self.closed = True

    resp = Trickle()
    monkeypatch.setattr(wss.requests.Session, "post", lambda self, *a, **k: resp)
    svc = wss.WebSearchService()
    svc.tavily_key = "t"
    t0 = time.monotonic()
    out = svc._tavily_post_bounded("extract", {"urls": ["https://a.org"]}, wall_s=0.6)
    assert out["success"] is False and "wall clock" in out["error"]
    assert time.monotonic() - t0 < 1.5
    deadline = time.monotonic() + 2.0
    while not getattr(resp, "closed", False) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert resp.closed  # the late worker stops at its next read


def test_r3_cx06_bounded_post_returns_when_headers_trickle(monkeypatch):
    """The POST itself blocks (a server trickling HEADERS, so no per-read
    timeout fires): the caller must still return at the wall clock."""
    from services import web_search_service as wss

    release = threading.Event()

    def blocking_post(self, *a, **k):
        release.wait(10)
        raise wss.requests.ConnectionError("closed")

    monkeypatch.setattr(wss.requests.Session, "post", blocking_post)
    svc = wss.WebSearchService()
    svc.tavily_key = "t"
    t0 = time.monotonic()
    try:
        out = svc._tavily_post_bounded("extract", {"urls": ["https://a.org"]}, wall_s=0.6)
    finally:
        release.set()
    assert out["success"] is False and "wall clock" in out["error"]
    assert time.monotonic() - t0 < 1.2


def test_r4_cx06_deadline_kills_a_real_socket_trickling_headers():
    """A real local server that trickles response HEADERS one byte every
    0.1 s (no per-read timeout ever fires): the caller returns at the wall
    clock AND the worker's socket is shut down, so the worker thread ends
    and the server sees the connection closed."""
    import socket as _socket

    from services import web_search_service as wss

    srv = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    port = srv.getsockname()[1]
    state = {}

    def serve():
        conn, _ = srv.accept()
        conn.settimeout(0.5)
        try:
            conn.recv(65536)                       # the request
            conn.sendall(b"HTTP/1.1 200 OK\r\nX-Slow: ")
            t_end = time.monotonic() + 8
            while time.monotonic() < t_end:
                try:
                    conn.sendall(b"a")
                except OSError:
                    state["closed_at"] = time.monotonic()
                    return
                try:                               # a closed peer reads as b""
                    conn.settimeout(0.1)
                    if conn.recv(1) == b"":
                        state["closed_at"] = time.monotonic()
                        return
                except _socket.timeout:
                    pass
        finally:
            conn.close()

    th = threading.Thread(target=serve, daemon=True)
    th.start()
    svc = wss.WebSearchService()
    svc.tavily_key = "t"
    svc._TAVILY_BASE = f"http://127.0.0.1:{port}"
    t0 = time.monotonic()
    out = svc._tavily_post_bounded("extract", {"urls": ["https://a.org"]}, wall_s=0.8)
    elapsed = time.monotonic() - t0
    assert out["success"] is False and "wall clock" in out["error"]
    assert elapsed < 1.4
    th.join(3)
    srv.close()
    assert "closed_at" in state and state["closed_at"] - t0 < 2.5   # the socket really went away
    deadline = time.monotonic() + 2
    while any(t.name == "tavily-extract" and t.is_alive() for t in threading.enumerate()) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not any(t.name == "tavily-extract" and t.is_alive() for t in threading.enumerate())


def _self_signed_cert(tmp_path, host):
    import datetime as _dt
    import ipaddress

    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.x509.oid import NameOID

    key = ec.generate_private_key(ec.SECP256R1())
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, host)])
    try:
        san = x509.IPAddress(ipaddress.ip_address(host))
    except ValueError:
        san = x509.DNSName(host)
    now = _dt.datetime.now(_dt.timezone.utc)
    cert = (x509.CertificateBuilder().subject_name(name).issuer_name(name).public_key(key.public_key())
            .serial_number(x509.random_serial_number()).not_valid_before(now - _dt.timedelta(minutes=5))
            .not_valid_after(now + _dt.timedelta(days=1))
            .add_extension(x509.SubjectAlternativeName([san]), critical=False)
            .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
            .sign(key, hashes.SHA256()))
    cert_file, key_file = tmp_path / f"{host}.pem", tmp_path / f"{host}.key"
    cert_file.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    key_file.write_bytes(key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                           serialization.NoEncryption()))
    return str(cert_file), str(key_file)


def _tls_server(cert_file, key_file, on_conn):
    import socket as _socket
    import ssl as _ssl

    ctx = _ssl.SSLContext(_ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(cert_file, key_file)
    srv = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)

    def serve():
        raw, _ = srv.accept()
        try:
            conn = ctx.wrap_socket(raw, server_side=True)
        except (OSError, _ssl.SSLError):
            raw.close()
            return
        try:
            on_conn(conn)
        finally:
            conn.close()

    th = threading.Thread(target=serve, daemon=True)
    th.start()
    return srv, srv.getsockname()[1], th


def _no_tavily_worker_left():
    deadline = time.monotonic() + 2
    while any(t.name == "tavily-extract" and t.is_alive() for t in threading.enumerate()) and time.monotonic() < deadline:
        time.sleep(0.05)
    return not any(t.name == "tavily-extract" and t.is_alive() for t in threading.enumerate())


def test_r5_cx06_https_trickled_headers_the_tls_socket_is_killed(tmp_path):
    """Round 4 gap: over HTTPS the raw socket is detached by the TLS wrap.
    A real TLS server trickles header bytes; at the deadline the live TLS
    socket must be shut down (the server sees the close) and the worker end."""
    import socket as _socket

    from services import web_search_service as wss

    cert_file, key_file = _self_signed_cert(tmp_path, "127.0.0.1")
    state = {}

    def on_conn(conn):
        conn.settimeout(1.0)
        conn.recv(65536)                               # the request
        conn.sendall(b"HTTP/1.1 200 OK\r\nX-Slow: ")
        t_end = time.monotonic() + 8
        while time.monotonic() < t_end:
            try:
                conn.sendall(b"a")
                conn.settimeout(0.1)
                if conn.recv(1) == b"":
                    state["closed_at"] = time.monotonic()
                    return
            except _socket.timeout:
                continue
            except OSError:
                state["closed_at"] = time.monotonic()
                return

    srv, port, th = _tls_server(cert_file, key_file, on_conn)
    svc = wss.WebSearchService()
    svc.tavily_key = "t"
    svc._TAVILY_BASE = f"https://127.0.0.1:{port}"
    svc._TAVILY_VERIFY = cert_file
    t0 = time.monotonic()
    out = svc._tavily_post_bounded("extract", {"urls": ["https://a.org"]}, wall_s=1.0)
    assert out["success"] is False and "wall clock" in out["error"]
    assert time.monotonic() - t0 < 1.6
    th.join(4)
    srv.close()
    assert "closed_at" in state and state["closed_at"] - t0 < 3.0
    assert _no_tavily_worker_left()


def test_r5_cx06_https_stalled_handshake_is_killed(tmp_path):
    """A server that accepts TCP but never answers the TLS ClientHello: the
    caller returns at the wall clock and the connection is gone within it.
    (A fully silent peer is also ended by the read timeout, which equals the
    wall clock, so this checks the bound; the trickle test above is the one
    that fails without TLS-socket tracking.)"""
    import socket as _socket

    from services import web_search_service as wss

    srv = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    port = srv.getsockname()[1]
    state = {}

    def serve():
        conn, _ = srv.accept()
        conn.settimeout(0.2)
        t_end = time.monotonic() + 8
        try:
            while time.monotonic() < t_end:
                try:
                    if conn.recv(65536) == b"":
                        state["closed_at"] = time.monotonic()
                        return
                except _socket.timeout:
                    continue
                except OSError:
                    state["closed_at"] = time.monotonic()
                    return
        finally:
            conn.close()

    th = threading.Thread(target=serve, daemon=True)
    th.start()
    svc = wss.WebSearchService()
    svc.tavily_key = "t"
    svc._TAVILY_BASE = f"https://127.0.0.1:{port}"
    t0 = time.monotonic()
    out = svc._tavily_post_bounded("extract", {"urls": ["https://a.org"]}, wall_s=1.0)
    assert out["success"] is False and "wall clock" in out["error"]
    th.join(4)
    srv.close()
    assert "closed_at" in state and state["closed_at"] - t0 < 3.0
    assert _no_tavily_worker_left()


def test_r5_cx06_https_success_and_hostname_verification_still_apply(tmp_path):
    from services import web_search_service as wss

    good_cert, good_key = _self_signed_cert(tmp_path, "127.0.0.1")
    body = b'{"results": [{"url": "https://a.org", "raw_content": "ok"}]}'

    def reply(conn):
        data = b""
        while b"\r\n\r\n" not in data:
            data += conn.recv(65536)
        head, _, rest = data.partition(b"\r\n\r\n")
        length = next((int(l.split(b":")[1]) for l in head.split(b"\r\n") if l.lower().startswith(b"content-length:")), 0)
        while len(rest) < length:
            rest += conn.recv(65536)
        conn.sendall(b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: "
                     + str(len(body)).encode() + b"\r\nConnection: close\r\n\r\n" + body)

    srv, port, th = _tls_server(good_cert, good_key, reply)
    svc = wss.WebSearchService()
    svc.tavily_key = "t"
    svc._TAVILY_BASE = f"https://127.0.0.1:{port}"
    svc._TAVILY_VERIFY = good_cert
    assert svc._tavily_post_bounded("extract", {}, wall_s=4.0) == {"results": [{"url": "https://a.org", "raw_content": "ok"}]}
    th.join(2)
    srv.close()

    # a certificate for another host must still be rejected
    bad_cert, bad_key = _self_signed_cert(tmp_path, "other.example")
    srv2, port2, th2 = _tls_server(bad_cert, bad_key, reply)
    svc._TAVILY_BASE = f"https://127.0.0.1:{port2}"
    svc._TAVILY_VERIFY = bad_cert
    with pytest.raises(wss.requests.exceptions.SSLError):
        svc._tavily_post_bounded("extract", {}, wall_s=4.0)
    th2.join(2)
    srv2.close()


def test_r4_cx06_slow_first_byte_within_the_wall_clock_succeeds():
    """Tavily Extract often needs over 2 s before its first byte: a reply that
    starts after 2.5 s must succeed under a 4 s wall clock (no short per-read
    timeout; the deadline kill bounds the total)."""
    import socket as _socket

    from services import web_search_service as wss

    srv = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    port = srv.getsockname()[1]

    def serve():
        conn, _ = srv.accept()
        try:
            conn.recv(65536)
            time.sleep(2.5)
            body = b'{"results": [{"url": "https://a.org", "raw_content": "ok"}]}'
            conn.sendall(b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: "
                         + str(len(body)).encode() + b"\r\nConnection: close\r\n\r\n" + body)
        finally:
            conn.close()

    th = threading.Thread(target=serve, daemon=True)
    th.start()
    svc = wss.WebSearchService()
    svc.tavily_key = "t"
    svc._TAVILY_BASE = f"http://127.0.0.1:{port}"
    out = svc._tavily_post_bounded("extract", {"urls": ["https://a.org"]}, wall_s=4.0)
    th.join(2)
    srv.close()
    assert out == {"results": [{"url": "https://a.org", "raw_content": "ok"}]}


def test_r3_cx06_bounded_post_returns_body_and_raises_worker_errors(monkeypatch):
    from services import web_search_service as wss

    class Ok:
        status_code = 200

        def iter_content(self, chunk_size=16384):
            yield b'{"results": []}'

        def close(self):
            pass

    monkeypatch.setattr(wss.requests.Session, "post", lambda self, *a, **k: Ok())
    svc = wss.WebSearchService()
    svc.tavily_key = "t"
    assert svc._tavily_post_bounded("extract", {}, wall_s=2.0) == {"results": []}

    def boom(self, *a, **k):
        raise wss.requests.ConnectionError("refused")

    monkeypatch.setattr(wss.requests.Session, "post", boom)
    with pytest.raises(wss.requests.ConnectionError):
        svc._tavily_post_bounded("extract", {}, wall_s=2.0)


def test_r2_cx06_deep_read_worker_runs_under_a_child_of_the_turn_deadline():
    from services import tool_budgets as tb

    reg = _reg(("A", "snippet"))
    seen = {}

    def extractor(urls, q):
        d = tb.current_deadline()
        seen["remaining"] = d.remaining() if d else None
        return None

    turn = tb.TurnCancellation(label="t")
    tb.adopt_deadline(tb.Deadline(60.0, label="prepass", turn=turn))
    try:
        deep_read(reg, "q", budget_s=1.0, extractor=extractor)
    finally:
        tb.adopt_deadline(None)
    assert seen["remaining"] is not None and seen["remaining"] <= 3.0 + 0.1
