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

# ── Set dummy env vars for offline testing ───────────────────────
# These prevent ImportError / KeyError during module import.
# They are NOT valid keys — no real API calls should be made in unit tests.
_DUMMY_KEYS = {
    "OPENAI_API_KEY": "sk-test-dummy-key-for-unit-tests",
    "NASA_ADS_API_KEY": "test-dummy-ads-key",
    "DEFAULT_LLM_MODEL": "gpt-4.1",
    "QUASAR_ENV": "testing",
}
for key, val in _DUMMY_KEYS.items():
    os.environ.setdefault(key, val)


# ── Custom Markers ───────────────────────────────────────────────
def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "unit: Fast tests with no network or API calls")
    config.addinivalue_line("markers", "integration: Tests that call real APIs (need valid keys)")
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
        "model": "gpt-4.1",
        "temperature": 0.7,
        "max_tokens": 1000,
    }
