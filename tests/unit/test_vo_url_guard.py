"""Offline tests for the VO SSRF guard (services/vo_url_guard.py) and its
enforcement on every HTTP hop of the VO session (services/vo_registry.py)."""

from __future__ import annotations

import pytest
import requests

import services.vo_url_guard as guard
from services.vo_registry import VoRegistryService, _TimeoutHTTPSession


@pytest.fixture(autouse=True)
def _no_allowlist(monkeypatch):
    monkeypatch.delenv("VO_ALLOWED_HOSTS", raising=False)


def _dns(monkeypatch, table):
    def fake(host):
        if host not in table:
            raise OSError("NXDOMAIN")
        return table[host]

    monkeypatch.setattr(guard, "_resolve", fake)


@pytest.mark.parametrize("url", [
    "http://127.0.0.1/tap",
    "http://10.0.0.5:8080/tap",
    "http://192.168.1.1/tap",
    "http://169.254.169.254/latest/meta-data",
    "http://[::1]/tap",
    "http://[::ffff:127.0.0.1]/tap",
    "http://0.0.0.0/tap",
    "http://localhost:8000/api",
    "http://backend.internal/tap",
    "http://printer.local/tap",
])
def test_private_targets_rejected_without_dns(url):
    ok, reason = guard.check_public_url(url, resolve=False)
    assert not ok
    assert "public network address" in reason
    # Never echo the address back (scanner oracle).
    assert "127.0.0.1" not in reason and "169.254" not in reason


@pytest.mark.parametrize("url,fragment", [
    ("ftp://archive.example/tap", "http(s)"),
    ("file:///etc/passwd", "http(s)"),
    ("https://user:pw@archive.example/tap", "credentials"),
    ("https:///tap", "no host"),
    ("https://archive.example:99999/tap", "invalid port"),
])
def test_malformed_urls_rejected(url, fragment):
    ok, reason = guard.check_public_url(url, resolve=False)
    assert not ok and fragment in reason


def test_public_host_allowed_any_port(monkeypatch):
    _dns(monkeypatch, {"tap.example.org": ["93.184.216.34"]})
    assert guard.check_public_url("https://tap.example.org:8443/tap") == (True, "")


def test_hostname_resolving_to_private_rejected(monkeypatch):
    _dns(monkeypatch, {"evil.example": ["10.1.2.3"]})
    ok, reason = guard.check_public_url("https://evil.example/tap")
    assert not ok and "public network address" in reason and "10.1.2.3" not in reason


def test_split_horizon_any_private_record_rejects(monkeypatch):
    _dns(monkeypatch, {"mixed.example": ["93.184.216.34", "192.168.0.9"]})
    assert guard.check_public_url("https://mixed.example/tap")[0] is False


def test_dns_failure_fails_closed_with_the_same_message(monkeypatch):
    """Same text as a private target: a distinct message would reveal which
    internal hostnames exist."""
    _dns(monkeypatch, {"evil.example": ["10.1.2.3"]})
    _, private_reason = guard.check_public_url("https://evil.example/tap")
    ok, reason = guard.check_public_url("https://nowhere.example/tap")
    assert not ok and reason == private_reason


@pytest.mark.parametrize("addr", [
    "64:ff9b::a9fe:a9fe",      # NAT64 form of 169.254.169.254
    "64:ff9b::a00:1",          # NAT64 form of 10.0.0.1
    "::127.0.0.1",             # IPv4-compatible (deprecated)
    "fec0::1",                 # site-local (deprecated)
    "2002:a00:1::1",           # 6to4 wrapping 10.0.0.1
    "::ffff:169.254.169.254",  # IPv4-mapped
])
def test_ipv6_embeddings_of_private_addresses_rejected(monkeypatch, addr):
    _dns(monkeypatch, {"sneaky.example": [addr]})
    assert guard.check_public_url("https://sneaky.example/tap")[0] is False


def test_teredo_with_private_server_rejected(monkeypatch):
    """CX-03: server 10.0.0.1, client 8.8.8.8 (obfuscated)."""
    _dns(monkeypatch, {"teredo.example": ["2001:0:a00:1:0:ffff:f7f7:f7f7"]})
    assert guard.check_public_url("https://teredo.example/tap")[0] is False


def test_blank_allowlist_fails_closed(monkeypatch):
    """CX-04: a configured-but-empty VO_ALLOWED_HOSTS allows nothing."""
    monkeypatch.setenv("VO_ALLOWED_HOSTS", " , ")
    _dns(monkeypatch, {"tap.example.org": ["93.184.216.34"]})
    ok, reason = guard.check_public_url("https://tap.example.org/tap")
    assert not ok and "VO_ALLOWED_HOSTS" in reason


def test_nat64_of_public_address_allowed(monkeypatch):
    _dns(monkeypatch, {"v6.example": ["64:ff9b::5db8:d822"]})  # 93.184.216.34
    assert guard.check_public_url("https://v6.example/tap")[0] is True


def test_no_dns_mode_does_not_resolve(monkeypatch):
    def boom(host):
        raise AssertionError("resolve=False must not hit DNS")

    monkeypatch.setattr(guard, "_resolve", boom)
    assert guard.check_public_url("https://tap.example.org/tap", resolve=False) == (True, "")


def test_allowlist_exact_and_subdomain_only(monkeypatch):
    monkeypatch.setenv("VO_ALLOWED_HOSTS", "eso.org, cadc-ccda.hia-iha.nrc-cnrc.gc.ca")
    _dns(monkeypatch, {"archive.eso.org": ["134.171.1.1"], "eso.org": ["134.171.1.2"],
                       "notaneso.org": ["93.184.216.34"]})
    assert guard.check_public_url("https://archive.eso.org/tap_obs")[0]
    assert guard.check_public_url("https://eso.org/x")[0]
    ok, reason = guard.check_public_url("https://notaneso.org/tap")
    assert not ok and "VO_ALLOWED_HOSTS" in reason


def test_ensure_raises_value_error_subclass():
    with pytest.raises(ValueError):
        guard.ensure_public_url("http://127.0.0.1/tap")


def test_service_rejects_private_access_url_before_building_service(monkeypatch):
    built = []
    svc = VoRegistryService(tap_factory=lambda url: built.append(url))
    out = svc.run_adql("http://169.254.169.254/tap", "SELECT 1")
    assert out["success"] is False and "public network address" in out["error"]
    assert built == []


def test_session_send_blocks_redirect_hop_to_private(monkeypatch):
    """A public service that 302s to the metadata address must not be
    followed: requests routes every redirect hop through send()."""
    _dns(monkeypatch, {"tap.example.org": ["93.184.216.34"]})
    sent = []

    # Patch the TRANSPORT, not Session.send: the real Session.send runs
    # resolve_redirects, which re-enters our send() override per hop.
    def fake_send(self, request, **kwargs):
        sent.append(request.url)
        resp = requests.Response()
        resp.request = request
        resp.url = request.url
        if request.url.startswith("https://tap.example.org"):
            resp.status_code = 302
            resp.headers["Location"] = "http://169.254.169.254/latest/meta-data"
        else:
            resp.status_code = 200
        resp._content = b""
        return resp

    monkeypatch.setattr(requests.adapters.HTTPAdapter, "send", fake_send)
    session = _TimeoutHTTPSession(timeout=5)
    with pytest.raises(guard.UnsafeUrlError):
        session.get("https://tap.example.org/tap/sync")
    assert sent == ["https://tap.example.org/tap/sync"]


def test_session_send_blocks_server_supplied_private_job_url(monkeypatch):
    _dns(monkeypatch, {})
    monkeypatch.setattr(requests.adapters.HTTPAdapter, "send",
                        lambda self, request, **kw: pytest.fail("must not send"))
    session = _TimeoutHTTPSession(timeout=5)
    with pytest.raises(guard.UnsafeUrlError):
        session.get("http://10.0.0.7:8080/tap/async/job123")
