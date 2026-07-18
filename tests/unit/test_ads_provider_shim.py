"""R8 — ADS/SciX provider abstraction in integrations.ads_client."""

import pytest

from integrations.ads_client import (
    ADSService,
    resolve_ads_api_key,
    resolve_ads_base_url,
)


CLASSIC = "https://api.adsabs.harvard.edu/v1"
SCIX = "https://api.scixplorer.org/v1"


def _clear_env(monkeypatch):
    for var in ("NASA_ADS_BASE_URL", "ADS_API_PROVIDER", "NASA_ADS_API_KEY", "SCIX_API_KEY"):
        monkeypatch.delenv(var, raising=False)


def test_default_provider_is_classic_ads(monkeypatch):
    _clear_env(monkeypatch)
    assert resolve_ads_base_url() == CLASSIC


def test_scix_provider_switches_host(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("ADS_API_PROVIDER", "scix")
    assert resolve_ads_base_url() == SCIX


def test_provider_name_is_case_insensitive(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("ADS_API_PROVIDER", "SciX")
    assert resolve_ads_base_url() == SCIX


def test_explicit_base_url_beats_provider(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("ADS_API_PROVIDER", "scix")
    monkeypatch.setenv("NASA_ADS_BASE_URL", "https://example.org/v1")
    assert resolve_ads_base_url() == "https://example.org/v1"


def test_unknown_provider_falls_back_to_classic(monkeypatch, caplog):
    _clear_env(monkeypatch)
    monkeypatch.setenv("ADS_API_PROVIDER", "bogus")
    with caplog.at_level("WARNING"):
        assert resolve_ads_base_url() == CLASSIC
    assert any("ADS_API_PROVIDER" in rec.message for rec in caplog.records)


def test_scix_key_fallback(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("SCIX_API_KEY", "scix-token")
    assert resolve_ads_api_key() == "scix-token"


def test_ads_key_wins_over_scix_key(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("NASA_ADS_API_KEY", "ads-token")
    monkeypatch.setenv("SCIX_API_KEY", "scix-token")
    assert resolve_ads_api_key() == "ads-token"


def test_service_picks_up_provider_and_key(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("ADS_API_PROVIDER", "scix")
    monkeypatch.setenv("SCIX_API_KEY", "scix-token")
    svc = ADSService()
    assert svc.base_url == SCIX
    assert svc.api_key == "scix-token"
    assert svc._build_headers()["Authorization"] == "Bearer scix-token"


def test_service_constructor_args_still_win(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("ADS_API_PROVIDER", "scix")
    svc = ADSService(api_key="explicit", base_url="https://example.org/v1")
    assert svc.base_url == "https://example.org/v1"
    assert svc.api_key == "explicit"
