"""Requested-link verification (guard CX-01..03, WP1 1.5).

When the user asks for a URL the link guard keeps model-emitted links and
probes them instead of stripping them. The probe must never reach loopback,
private, link-local (cloud metadata) or reserved addresses -- including via a
redirect -- must not fall back to GET, must check EVERY requested link (or
label the overflow), and must run concurrently under one wall-clock cap that
honours turn cancellation. Network is faked throughout.
"""
from __future__ import annotations

import socket
import threading
import time
import types

import pytest

from core.agent import QuasarAgent
from services.tool_budgets import TurnCancellation


def _agent():
    agent = QuasarAgent.__new__(QuasarAgent)
    agent._tls = threading.local()
    return agent


def _fake_dns(mapping):
    def getaddrinfo(host, port, *a, **k):
        if host not in mapping:
            raise socket.gaierror("nxdomain")
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (mapping[host], port))]
    return getaddrinfo


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1/admin",
        "http://localhost/",
        "http://169.254.169.254/latest/meta-data/",
        "http://10.0.0.5/",
        "http://[::1]/",
        "http://192.168.1.1/",
        "ftp://example.org/file",
        "https://example.org:8443/x",
        "https://user:pw@example.org/",
        "http://internal.example/",  # resolves to a private address
    ],
)
def test_non_public_targets_are_refused(monkeypatch, url):
    monkeypatch.setattr(socket, "getaddrinfo", _fake_dns({
        "127.0.0.1": "127.0.0.1", "169.254.169.254": "169.254.169.254", "10.0.0.5": "10.0.0.5", "::1": "::1",
        "192.168.1.1": "192.168.1.1", "example.org": "93.184.216.34", "internal.example": "10.1.2.3", "localhost": "127.0.0.1",
    }))
    ok, why = QuasarAgent._public_http_target(url)
    assert not ok and why


def test_public_target_is_allowed(monkeypatch):
    monkeypatch.setattr(socket, "getaddrinfo", _fake_dns({"almascience.org": "193.146.57.10"}))
    assert QuasarAgent._public_http_target("https://almascience.org/aq/?source=Orion") == (True, "")


class _Resp:
    def __init__(self, status, location=None):
        self.status_code = status
        self.headers = {"Location": location} if location else {}

    def close(self):
        pass


def test_redirect_into_a_private_address_is_not_followed_and_no_get_fallback(monkeypatch):
    monkeypatch.setattr(socket, "getaddrinfo", _fake_dns({"public.example": "93.184.216.34", "evil.example": "169.254.169.254"}))
    calls = []

    def head_once(url, ip, timeout):
        calls.append(("HEAD", url, ip))
        return (302, "http://evil.example/latest/meta-data/") if "public" in url else (200, None)

    monkeypatch.setattr(QuasarAgent, "_head_once", staticmethod(head_once))
    ok, detail = QuasarAgent._head_verify_url("https://public.example/page")
    assert not ok and "non-public" in detail
    assert calls == [("HEAD", "https://public.example/page", "93.184.216.34")]  # the metadata host was never contacted

    monkeypatch.setattr(QuasarAgent, "_head_once", staticmethod(lambda url, ip, timeout: (405, None)))
    ok, detail = QuasarAgent._head_verify_url("https://public.example/page")
    assert not ok and "refuses HEAD" in detail


def test_every_requested_link_is_probed_concurrently_with_a_labelled_overflow(monkeypatch):
    agent = _agent()
    probed = []

    def fake_probe(url, timeout=5.0, stop=None):
        probed.append(url)
        time.sleep(0.3)
        return True, "HTTP 200"

    monkeypatch.setattr(QuasarAgent, "_head_verify_url", staticmethod(fake_probe))
    urls = [f"https://example.org/p{i}" for i in range(QuasarAgent._LINK_PROBE_MAX + 3)]
    t0 = time.monotonic()
    out = agent._verify_requested_urls(urls)
    elapsed = time.monotonic() - t0
    assert elapsed < 0.3 * QuasarAgent._LINK_PROBE_MAX / 2  # concurrent, not sequential
    assert [u for u, ok, _ in out if ok] == urls[: QuasarAgent._LINK_PROBE_MAX]
    assert all("not probed" in d for _, ok, d in out[QuasarAgent._LINK_PROBE_MAX:])
    assert len(out) == len(urls)


def test_probes_stop_at_the_wall_clock_cap_and_on_turn_cancellation(monkeypatch):
    agent = _agent()
    monkeypatch.setattr(QuasarAgent, "_LINK_PROBE_TOTAL_S", 0.5)
    monkeypatch.setattr(QuasarAgent, "_head_verify_url", staticmethod(lambda url, timeout=5.0, stop=None: (time.sleep(3), (True, "HTTP 200"))[1]))
    t0 = time.monotonic()
    out = agent._verify_requested_urls(["https://example.org/slow"])
    assert time.monotonic() - t0 < 1.5
    assert out == [("https://example.org/slow", False, "probe did not finish in time")]

    tc = TurnCancellation("t")
    tc.cancel("client went away")
    agent._tls.turn_cancellation = tc
    out = agent._verify_requested_urls(["https://example.org/a"])
    assert out == [("https://example.org/a", False, "turn cancelled")]


def test_link_guard_lists_all_requested_links(monkeypatch):
    agent = _agent()
    monkeypatch.setattr(QuasarAgent, "_head_verify_url", staticmethod(lambda url, timeout=5.0, stop=None: (True, "HTTP 200")))
    urls = [f"https://example.org/q{i}" for i in range(7)]
    text = "Here you go:\n" + "\n".join(urls)
    out = agent._strip_unverified_urls(text, sources=[], user_query="give me the links")
    assert all(u in out for u in urls)
    assert out.count("verified (HTTP 200)") == 7
