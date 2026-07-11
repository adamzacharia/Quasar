# tests/conftest.py
"""
Shared pytest configuration and fixtures for Quasar test suite.

Provides:
  - Environment setup (dummy API keys for offline tests)
  - Shared fixtures for agent, services, and integrations
  - Markers for categorizing tests (unit, integration, smoke)
"""

import os
import sys
import pytest

# ── Ensure project root is on sys.path ───────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ── Never let tests reach a remote database ──────────────────────
# config/__init__.py and config/settings.py call load_dotenv() when imported,
# which copies the developer's real TURSO_* credentials into os.environ. Once
# they are there, services.db sends EVERY get_connection() to the cloud database
# — including the calls that pass an explicit local sqlite path, because that
# path is only the local fallback, never an override. So in a whole-directory
# run, the first test module that imported `config` silently moved the rest of
# the session onto one shared remote DB, and tests began seeing rows committed
# by earlier tests (and by earlier runs).
#
# This has to happen before anything imports services.db.
os.environ["QUASAR_FORCE_LOCAL_DB"] = "1"

# ── Set dummy env vars for offline testing ───────────────────────
# These prevent ImportError / KeyError during module import.
# They are NOT valid keys — no real API calls should be made in unit tests.
_DUMMY_KEYS = {
    "OPENAI_API_KEY": "sk-test-dummy-key-for-unit-tests",
    "NASA_ADS_API_KEY": "test-dummy-ads-key",
    "DEFAULT_LLM_MODEL": "gpt-5.4-mini",
    "QUASAR_ENV": "testing",
    # Keep the on-disk Splatalogue query cache off by default so unit tests are
    # deterministic and never serve stale cross-run results. Cache-specific
    # tests opt back in explicitly.
    "SPLATALOGUE_QUERY_CACHE_ENABLED": "0",
}
for key, val in _DUMMY_KEYS.items():
    os.environ.setdefault(key, val)


# ── Custom Markers ───────────────────────────────────────────────
def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "unit: Fast tests with no network or API calls")
    config.addinivalue_line("markers", "integration: Tests that call real APIs (need valid keys)")
    config.addinivalue_line("markers", "live: Opt-in tests that call public external archive services")
    config.addinivalue_line("markers", "smoke: End-to-end smoke tests against a live server")
    config.addinivalue_line("markers", "slow: Tests that take > 30 seconds")


# ── Fixtures ─────────────────────────────────────────────────────
@pytest.fixture
def project_root():
    """Return the absolute path to the project root."""
    return PROJECT_ROOT


@pytest.fixture
def dummy_config():
    """Return a minimal config dict for testing."""
    return {
        "model": "gpt-5.4-mini",
        "temperature": 0.7,
        "max_tokens": 1000,
    }
