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
# Never fire the background live TAP-schema refresh from unit tests: fake
# Data Lab clients record last_sql, and a daemon-thread tap_schema query would
# race those assertions (and hit the network with a real client). Deliberately
# a HARD set, not setdefault — a developer environment exporting =1 must not
# leak refreshes into tests (guard CX-04). Refresh-specific tests opt back in
# via monkeypatch.
os.environ["DATALAB_TAP_SCHEMA_AUTOREFRESH"] = "0"
# Isolate the TAP-schema disk cache per test session: without this, a developer
# machine with a real live cache under cache/datalab_tap_schema would flip the
# SQL governor's column check from soft (curated-only) to authoritative and make
# unit tests behave differently locally than in CI. Also a hard set.
import tempfile  # noqa: E402
os.environ["DATALAB_TAP_SCHEMA_CACHE_DIR"] = os.path.join(
    tempfile.gettempdir(), f"quasar-test-tap-schema-{os.getpid()}"
)
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


# ── Reset the process-wide login rate-limiter singletons between tests ──
# Two module-level LoginRateLimiter singletons accumulate failure/lockout state
# across test modules: services.login_rate_limit._default_limiter (the
# per-username limiter the shared AuthService uses) and
# ui-pro/api/routers/auth.py:_ip_limiter (per-IP). In a whole-directory run a
# lockout tripped by one module made a later module's login return 429 instead
# of 401 (the test_s3_s6_security.py flake). Clear both before every test.
# Reads are defensive: a unit-only run that never loaded ui-pro just skips it.
@pytest.fixture(autouse=True)
def _reset_login_rate_limiters():
    def _clear(limiter):
        if limiter is None:
            return
        try:
            with limiter._lock:
                limiter._failures.clear()
                limiter._locked_until.clear()
                limiter._last_sweep = float("-inf")
        except Exception:
            pass

    try:
        import services.login_rate_limit as _lrl
        _clear(getattr(_lrl, "_default_limiter", None))
    except Exception:
        pass

    _auth_mod = sys.modules.get("api.routers.auth")
    if _auth_mod is not None:
        _clear(getattr(_auth_mod, "_ip_limiter", None))

    yield
