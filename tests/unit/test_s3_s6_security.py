"""Regression tests for S3 (admin DB role) and S6 (compare_digest, login rate
limiting, and the wired httpOnly-cookie auth flow — set/read/clear, per-IP
throttle, XFF policy, CSRF Origin check).

These run fully offline against a local SQLite users.db in a tmp dir.
"""

import importlib
import sys
import types

import pytest


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def security_env(monkeypatch, tmp_path):
    """Fresh, isolated local-SQLite auth stack with caching disabled."""
    monkeypatch.delenv("TURSO_DATABASE_URL", raising=False)
    monkeypatch.delenv("TURSO_AUTH_TOKEN", raising=False)
    monkeypatch.setenv("QUASAR_FORCE_LOCAL_DB", "1")
    monkeypatch.setenv("QUASAR_ENV", "testing")
    monkeypatch.setenv("QUASAR_USERS_DB_PATH", str(tmp_path / "users.db"))
    # Disable the admin-role TTL cache so role changes are visible immediately.
    monkeypatch.setenv("QUASAR_ADMIN_ROLE_CACHE_TTL", "0")
    monkeypatch.delenv("ADMIN_EMAILS", raising=False)
    monkeypatch.delenv("QUASAR_TOKEN_LIMIT_EXEMPT_EMAILS", raising=False)
    monkeypatch.delenv("QUASAR_ENABLE_LOCAL_TEST_LOGIN", raising=False)

    for mod in (
        "services.db",
        "services.login_rate_limit",
        "services.auth",
        "services.admin_access",
    ):
        sys.modules.pop(mod, None)

    auth = importlib.import_module("services.auth")
    admin_access = importlib.import_module("services.admin_access")
    return types.SimpleNamespace(auth=auth, admin_access=admin_access)


class FakeClock:
    def __init__(self, start=1000.0):
        self.t = float(start)

    def __call__(self):
        return self.t

    def advance(self, dt):
        self.t += float(dt)


def _make_service(auth_module, rate_limiter=None):
    from services.login_rate_limit import LoginRateLimiter

    return auth_module.AuthService(
        rate_limiter=rate_limiter or LoginRateLimiter(max_attempts=5)
    )


# ── S3: no hard-coded admin; DB role column ──────────────────────────────────


def test_no_hardcoded_admin_default(security_env):
    admin_access = security_env.admin_access
    assert admin_access.DEFAULT_ADMIN_EMAILS == set()
    assert admin_access.configured_admin_emails() == set()

    # Empty users table → '1@1' is no longer implicitly an admin.
    _make_service(security_env.auth)
    assert admin_access.is_admin_email("1@1") is False
    assert admin_access.is_admin_user({"email": "1@1"}) is False


def test_env_allowlist_still_grants_admin(security_env, monkeypatch):
    admin_access = security_env.admin_access
    _make_service(security_env.auth)
    monkeypatch.setenv("ADMIN_EMAILS", "boss@obs.org, other@obs.org")
    assert admin_access.is_admin_email("boss@obs.org") is True
    assert admin_access.is_admin_email("BOSS@obs.org") is True  # case-insensitive
    assert admin_access.is_admin_email("nobody@obs.org") is False


def test_db_role_grants_admin(security_env):
    auth = security_env.auth
    admin_access = security_env.admin_access
    svc = _make_service(auth)

    ok, _msg, _tok = svc.register_user("scientist@obs.org", "pw123456")
    assert ok is True

    # Freshly registered users are not admin.
    assert svc.get_user_role("scientist@obs.org") == "user"
    assert admin_access.is_admin_email("scientist@obs.org") is False
    assert admin_access.is_quota_exempt_email("scientist@obs.org") is False

    # Grant admin via the DB role column — no redeploy, no hard-coded email.
    assert svc.set_user_role("scientist@obs.org", "admin") is True
    admin_access.clear_admin_role_cache()

    assert svc.get_user_role("scientist@obs.org") == "admin"
    assert admin_access.is_admin_email("scientist@obs.org") is True
    assert admin_access.is_admin_user({"email": "scientist@obs.org"}) is True
    # DB-role admins inherit quota exemption (prior admin behaviour).
    assert admin_access.is_quota_exempt_email("scientist@obs.org") is True


def test_seed_test_user_is_admin_via_role(security_env, monkeypatch):
    auth = security_env.auth
    admin_access = security_env.admin_access
    monkeypatch.setenv("QUASAR_ENABLE_LOCAL_TEST_LOGIN", "1")

    svc = _make_service(auth)

    # Login still works with 1@1 / 1.
    success, user_id, email, _display, message = svc.login_user("1@1", "1")
    assert success is True and user_id and email == "1@1"
    assert message == "Login successful"

    # Admin now comes from the seeded role, not a hard-coded email.
    assert svc.get_user_role("1@1") == "admin"
    assert admin_access.is_admin_email("1@1") is True


# ── S6: constant-time password comparison ────────────────────────────────────


def test_coerce_bytes_normalises_db_types(security_env):
    coerce = security_env.auth._coerce_bytes
    assert coerce(b"abc") == b"abc"
    assert coerce(bytearray(b"abc")) == b"abc"
    assert coerce(memoryview(b"abc")) == b"abc"
    assert coerce("abc") == b"abc"


def test_login_password_verification(security_env):
    svc = _make_service(security_env.auth)
    ok, _msg, _tok = svc.register_user("astro@obs.org", "correct horse")
    assert ok is True

    good = svc.login_user("astro@obs.org", "correct horse")
    assert good[0] is True and good[4] == "Login successful"

    bad = svc.login_user("astro@obs.org", "wrong password")
    assert bad[0] is False and bad[4] == "Invalid username or password"

    missing = svc.login_user("ghost@obs.org", "whatever")
    assert missing[0] is False and missing[4] == "Invalid username or password"


def test_register_rejects_duplicate_email(security_env):
    """CX-01: admin is granted by email (env allowlist AND DB role lookup), so a
    new account must not be allowed to claim an existing user's email — otherwise
    it inherits that identity's privileges via its JWT."""
    svc = _make_service(security_env.auth)
    ok, msg, _tok = svc.register_user("victim@obs.org", "pw-123456")
    assert ok is True

    # Same email, different username, different case → rejected.
    dup, dmsg, _ = svc.register_user("attacker", "pw-654321", email="Victim@OBS.org")
    assert dup is False and "already registered" in dmsg.lower()

    # And you cannot claim it as your username either (case-insensitive).
    dup2, _dmsg2, _ = svc.register_user("VICTIM@obs.org", "pw-654321")
    assert dup2 is False


def test_rate_key_is_exact_not_normalized(security_env):
    """CX-03: the throttle key must equal the exact identity the DB authenticates
    (WHERE username = ?), so case/whitespace variants get distinct buckets and one
    cannot reset another's throttle."""
    AuthService = security_env.auth.AuthService
    assert AuthService._rate_key("Admin@X.org") != AuthService._rate_key("admin@x.org")
    assert AuthService._rate_key("admin@x.org") == "admin@x.org"


def test_int_env_malformed_falls_back(monkeypatch):
    """CX-13: malformed limiter env must fall back to the default, not crash import."""
    from services.login_rate_limit import _int_env

    monkeypatch.setenv("QUASAR_TEST_MALFORMED_INT", "not-a-number")
    assert _int_env("QUASAR_TEST_MALFORMED_INT", 7) == 7
    monkeypatch.setenv("QUASAR_TEST_MALFORMED_INT", "-5")  # non-positive → default
    assert _int_env("QUASAR_TEST_MALFORMED_INT", 7) == 7
    monkeypatch.setenv("QUASAR_TEST_MALFORMED_INT", "")
    assert _int_env("QUASAR_TEST_MALFORMED_INT", 7) == 7
    monkeypatch.setenv("QUASAR_TEST_MALFORMED_INT", "9")
    assert _int_env("QUASAR_TEST_MALFORMED_INT", 7) == 9


def test_login_uses_constant_time_compare(security_env, monkeypatch):
    """The comparison must go through hmac.compare_digest, not ``==``."""
    import services.auth as auth_mod

    calls = {"n": 0}
    real = auth_mod.hmac.compare_digest

    def spy(a, b):
        calls["n"] += 1
        return real(a, b)

    monkeypatch.setattr(auth_mod.hmac, "compare_digest", spy)

    svc = _make_service(security_env.auth)
    svc.register_user("timing@obs.org", "s3cret-passphrase")
    svc.login_user("timing@obs.org", "s3cret-passphrase")
    assert calls["n"] >= 1


# ── S6: basic login rate limiting ────────────────────────────────────────────


def test_rate_limit_locks_after_threshold(security_env):
    from services.login_rate_limit import LoginRateLimiter

    clock = FakeClock()
    limiter = LoginRateLimiter(
        max_attempts=3, window_seconds=1000, lockout_seconds=100, clock=clock
    )
    svc = _make_service(security_env.auth, rate_limiter=limiter)
    svc.register_user("target@obs.org", "rightpass")

    for _ in range(3):
        res = svc.login_user("target@obs.org", "wrongpass")
        assert res[0] is False

    # Now locked out — even the correct password is rejected with a throttle msg.
    locked = svc.login_user("target@obs.org", "rightpass")
    assert locked[0] is False
    assert "Too many failed login attempts" in locked[4]

    # Lockout still active just before it expires.
    clock.advance(99)
    assert svc.login_user("target@obs.org", "rightpass")[0] is False

    # After the lockout window, the correct password works again.
    clock.advance(2)
    ok = svc.login_user("target@obs.org", "rightpass")
    assert ok[0] is True and ok[4] == "Login successful"


def test_rate_limit_resets_on_success(security_env):
    from services.login_rate_limit import LoginRateLimiter

    clock = FakeClock()
    limiter = LoginRateLimiter(
        max_attempts=3, window_seconds=1000, lockout_seconds=100, clock=clock
    )
    svc = _make_service(security_env.auth, rate_limiter=limiter)
    svc.register_user("resetme@obs.org", "goodpass")

    # Two failures (below threshold), then a success resets the counter.
    assert svc.login_user("resetme@obs.org", "nope")[0] is False
    assert svc.login_user("resetme@obs.org", "nope")[0] is False
    assert svc.login_user("resetme@obs.org", "goodpass")[0] is True

    # Two more failures must NOT lock (counter was reset by the success).
    assert svc.login_user("resetme@obs.org", "nope")[0] is False
    assert svc.login_user("resetme@obs.org", "nope")[0] is False
    still_ok = svc.login_user("resetme@obs.org", "goodpass")
    assert still_ok[0] is True


def test_first_login_never_rate_limited(security_env):
    """A legitimate user's first correct login is never throttled (1@1/1 case)."""
    svc = _make_service(security_env.auth)
    svc.register_user("firsttry@obs.org", "letmein!!")
    res = svc.login_user("firsttry@obs.org", "letmein!!")
    assert res[0] is True and res[4] == "Login successful"


# ── S6: httpOnly cookie helper ───────────────────────────────────────────────


def test_cookie_params_are_httponly():
    sys.modules.pop("services.auth_cookie", None)
    ac = importlib.import_module("services.auth_cookie")
    params = ac.cookie_params()
    assert params["httponly"] is True
    assert params["path"] == "/"
    assert params["max_age"] > 0


def test_cookie_security_by_environment(monkeypatch):
    sys.modules.pop("services.auth_cookie", None)
    ac = importlib.import_module("services.auth_cookie")

    monkeypatch.setenv("QUASAR_ENV", "production")
    monkeypatch.delenv("QUASAR_AUTH_COOKIE_SECURE", raising=False)
    monkeypatch.delenv("QUASAR_AUTH_COOKIE_SAMESITE", raising=False)
    assert ac.cookie_samesite() == "none"
    assert ac.cookie_secure() is True

    monkeypatch.setenv("QUASAR_ENV", "development")
    assert ac.cookie_samesite() == "lax"
    assert ac.cookie_secure() is False


def test_set_read_clear_cookie_roundtrip():
    from starlette.responses import Response

    sys.modules.pop("services.auth_cookie", None)
    ac = importlib.import_module("services.auth_cookie")

    resp = Response()
    ac.set_auth_cookie(resp, "jwt-token-value")
    header = resp.headers.get("set-cookie") or ""
    assert f"{ac.AUTH_COOKIE_NAME}=jwt-token-value" in header
    assert "HttpOnly" in header

    fake_request = types.SimpleNamespace(
        cookies={ac.AUTH_COOKIE_NAME: "jwt-token-value"}
    )
    assert ac.read_auth_cookie(fake_request) == "jwt-token-value"
    assert ac.read_auth_cookie(types.SimpleNamespace(cookies={})) is None

    cleared = Response()
    ac.clear_auth_cookie(cleared)
    clear_header = cleared.headers.get("set-cookie") or ""
    assert ac.AUTH_COOKIE_NAME in clear_header
    assert 'Max-Age=0' in clear_header or "expires=" in clear_header.lower()


# ── Adversarial-review fixes ─────────────────────────────────────────────────


def test_rate_limiter_memory_is_bounded(security_env):
    """An unauthenticated flood of distinct identities cannot leak memory."""
    from services.login_rate_limit import LoginRateLimiter

    clock = FakeClock()
    limiter = LoginRateLimiter(
        max_attempts=5, window_seconds=1000, lockout_seconds=1000,
        clock=clock, max_tracked=64,
    )
    for i in range(5000):
        limiter.record_failure(f"attacker-{i}@spam.test")
        clock.advance(0.001)

    assert len(limiter._failures) <= 64
    assert len(limiter._locked_until) <= 64


def test_rate_limiter_truncates_giant_keys():
    from services.login_rate_limit import LoginRateLimiter

    limiter = LoginRateLimiter()
    limiter.record_failure("x" * 100_000)
    assert limiter._failures
    assert all(len(k) <= 256 for k in limiter._failures)


def test_cookie_secure_forced_when_samesite_none(monkeypatch):
    sys.modules.pop("services.auth_cookie", None)
    ac = importlib.import_module("services.auth_cookie")

    # SameSite=None must remain Secure even with an explicit falsey override.
    monkeypatch.setenv("QUASAR_ENV", "development")
    monkeypatch.setenv("QUASAR_AUTH_COOKIE_SAMESITE", "none")
    monkeypatch.setenv("QUASAR_AUTH_COOKIE_SECURE", "0")
    assert ac.cookie_samesite() == "none"
    assert ac.cookie_secure() is True

    # Production must remain Secure even with override=0.
    monkeypatch.delenv("QUASAR_AUTH_COOKIE_SAMESITE", raising=False)
    monkeypatch.setenv("QUASAR_ENV", "production")
    assert ac.cookie_secure() is True

    # Where Secure is NOT forced, the override still applies.
    monkeypatch.setenv("QUASAR_ENV", "development")
    monkeypatch.setenv("QUASAR_AUTH_COOKIE_SAMESITE", "lax")
    monkeypatch.setenv("QUASAR_AUTH_COOKIE_SECURE", "1")
    assert ac.cookie_secure() is True
    monkeypatch.setenv("QUASAR_AUTH_COOKIE_SECURE", "0")
    assert ac.cookie_secure() is False


def test_admin_role_cache_ttl_and_expiry(security_env, monkeypatch):
    admin_access = security_env.admin_access
    monkeypatch.setenv("QUASAR_ADMIN_ROLE_CACHE_TTL", "60")
    admin_access.clear_admin_role_cache()

    clock = FakeClock()
    monkeypatch.setattr(admin_access, "time", types.SimpleNamespace(monotonic=clock))

    calls = {"n": 0}
    state = {"admin": True}

    def fake_query(email):
        calls["n"] += 1
        return state["admin"]

    monkeypatch.setattr(admin_access, "_query_db_role_is_admin", fake_query)

    # First call hits the DB; the second within the TTL is cached.
    assert admin_access.is_admin_email("cached@obs.org") is True
    assert admin_access.is_admin_email("cached@obs.org") is True
    assert calls["n"] == 1

    # A DB demotion is NOT visible within the TTL window (documented staleness).
    state["admin"] = False
    clock.advance(30)
    assert admin_access.is_admin_email("cached@obs.org") is True
    assert calls["n"] == 1

    # After the TTL expires it re-queries and reflects the demotion.
    clock.advance(31)
    assert admin_access.is_admin_email("cached@obs.org") is False
    assert calls["n"] == 2


def test_set_user_role_invalidates_admin_cache(security_env, monkeypatch):
    auth = security_env.auth
    admin_access = security_env.admin_access
    monkeypatch.setenv("QUASAR_ADMIN_ROLE_CACHE_TTL", "60")
    admin_access.clear_admin_role_cache()

    svc = _make_service(auth)
    svc.register_user("promote@obs.org", "pw12345678")
    assert svc.set_user_role("promote@obs.org", "admin") is True
    assert admin_access.is_admin_email("promote@obs.org") is True  # caches True

    # Revocation is visible immediately despite the 60s TTL, because
    # set_user_role clears the role cache.
    assert svc.set_user_role("promote@obs.org", "user") is True
    assert admin_access.is_admin_email("promote@obs.org") is False


# ── S6 API layer: 429 throttle semantics, per-IP limiter, cookie flow ────────
#
# Route-only FastAPI app mounting the REAL auth router (api.deps singletons are
# import-time, so a full-app TestClient is not per-test isolatable; instead we
# import once under a stable env and patch fresh limiters per test).

import os as _os
from pathlib import Path as _Path


@pytest.fixture(scope="module")
def auth_api(tmp_path_factory):
    repo = _Path(__file__).resolve().parents[2]
    saved = {}
    env = {
        "QUASAR_ENV": "testing",
        "QUASAR_FORCE_LOCAL_DB": "1",
        "QUASAR_USERS_DB_PATH": str(tmp_path_factory.mktemp("authapi") / "users.db"),
        "QUASAR_ADMIN_ROLE_CACHE_TTL": "0",
        "QUASAR_ENABLE_LOCAL_TEST_LOGIN": "1",
    }
    for key in list(env) + ["TURSO_DATABASE_URL", "TURSO_AUTH_TOKEN", "ADMIN_EMAILS", "QUASAR_TRUST_XFF"]:
        saved[key] = _os.environ.pop(key, None)
    _os.environ.update(env)

    # Invalidate the auth-relevant service modules AND every already-imported
    # ``api.*`` module, so ``api.deps`` (which builds the ``auth_service``
    # singleton at import) and ``api.routers.auth`` rebuild against a fresh
    # ``services.auth`` under this fixture's env. Without popping ``api.*`` a
    # prior test file can leave ``api.deps.auth_service`` bound to a different
    # env or — via test_auth_config's fake ``jwt`` — a ``verify_token`` that
    # "validates" any token, which would break the fail-closed / XFF assertions.
    #
    # CRUCIAL: snapshot every module we invalidate and RESTORE it on teardown,
    # so this fixture leaves zero global module-state pollution for the test
    # files that run after it (e.g. test_sky_monitor, which imports services.db).
    _svc_mods = ("services.db", "services.login_rate_limit", "services.auth",
                 "services.admin_access", "services.auth_cookie")

    def _affected():
        return [m for m in sys.modules if m == "api" or m.startswith("api.") or m in _svc_mods]

    mod_snapshot = {m: sys.modules[m] for m in _affected()}
    for m in list(mod_snapshot):
        sys.modules.pop(m, None)
    for m in _svc_mods:
        sys.modules.pop(m, None)
    for path in (str(repo), str(repo / "ui-pro")):
        if path not in sys.path:
            sys.path.insert(0, path)

    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    import api.routers.auth as auth_router_mod

    app = FastAPI()
    app.include_router(auth_router_mod.router)

    yield types.SimpleNamespace(
        app=app,
        mod=auth_router_mod,
        service=auth_router_mod.auth_service,
        client=lambda: TestClient(app),
    )

    # Restore module state exactly (drop what we imported under the test env,
    # put the originals back) so later test files are unaffected.
    for m in _affected():
        sys.modules.pop(m, None)
    sys.modules.update(mod_snapshot)

    for key, value in saved.items():
        if value is None:
            _os.environ.pop(key, None)
        else:
            _os.environ[key] = value


@pytest.fixture
def fresh_limiters(auth_api, monkeypatch):
    """Fresh fake-clock user + IP limiters so tests are isolated and time-controllable."""
    from services.login_rate_limit import LoginRateLimiter

    user_clock, ip_clock = FakeClock(), FakeClock()
    user_limiter = LoginRateLimiter(max_attempts=2, window_seconds=1000,
                                    lockout_seconds=100, clock=user_clock)
    ip_limiter = LoginRateLimiter(max_attempts=3, window_seconds=1000,
                                  lockout_seconds=100, clock=ip_clock)
    monkeypatch.setattr(auth_api.service, "_rate_limiter", user_limiter)
    monkeypatch.setattr(auth_api.mod, "_ip_limiter", ip_limiter)
    return types.SimpleNamespace(user=user_limiter, ip=ip_limiter,
                                 user_clock=user_clock, ip_clock=ip_clock)


_UNIQ = iter(range(10_000))


def _new_user(auth_api, password="pw-123456"):
    name = f"api-user-{next(_UNIQ)}@obs.org"
    ok, msg, _tok = auth_api.service.register_user(name, password)
    assert ok, msg
    return name


def test_login_trip_is_401_locked_is_429(auth_api, fresh_limiters):
    client = auth_api.client()
    user = _new_user(auth_api)

    # max_attempts=2: both wrong attempts (incl. the one that TRIPS) are 401.
    assert client.post("/api/auth/login", json={"username": user, "password": "x"}).status_code == 401
    assert client.post("/api/auth/login", json={"username": user, "password": "x"}).status_code == 401

    # Already locked → 429 + positive integer Retry-After, even with correct creds.
    locked = client.post("/api/auth/login", json={"username": user, "password": "pw-123456"})
    assert locked.status_code == 429
    assert int(locked.headers["Retry-After"]) >= 1


def test_login_429_per_ip_across_usernames(auth_api, fresh_limiters):
    client = auth_api.client()
    users = [_new_user(auth_api) for _ in range(4)]

    # ip max_attempts=3: three invalid-credential failures for DIFFERENT users…
    for user in users[:3]:
        assert client.post("/api/auth/login", json={"username": user, "password": "x"}).status_code == 401

    # …locks the source: a FOURTH username gets 429 despite zero prior failures.
    blocked = client.post("/api/auth/login", json={"username": users[3], "password": "pw-123456"})
    assert blocked.status_code == 429
    assert int(blocked.headers["Retry-After"]) >= 1


def test_ip_bucket_not_reset_on_success(auth_api, fresh_limiters):
    client = auth_api.client()
    u1, u2, u3 = _new_user(auth_api), _new_user(auth_api), _new_user(auth_api)

    assert client.post("/api/auth/login", json={"username": u1, "password": "x"}).status_code == 401
    # A successful login must NOT erase the source's failure history.
    assert client.post("/api/auth/login", json={"username": u1, "password": "pw-123456"}).status_code == 200
    assert client.post("/api/auth/login", json={"username": u2, "password": "x"}).status_code == 401
    assert client.post("/api/auth/login", json={"username": u3, "password": "x"}).status_code == 401
    # 3 failures accumulated across the success → IP locked.
    assert client.post("/api/auth/login", json={"username": u2, "password": "pw-123456"}).status_code == 429


def test_ip_lockout_expires(auth_api, fresh_limiters):
    client = auth_api.client()
    users = [_new_user(auth_api) for _ in range(3)]
    for user in users:
        client.post("/api/auth/login", json={"username": user, "password": "x"})
    assert client.post("/api/auth/login", json={"username": users[0], "password": "x"}).status_code == 429

    fresh_limiters.ip_clock.advance(101)  # past lockout_seconds=100
    # Back to normal failure semantics (401, not 429).
    assert client.post("/api/auth/login", json={"username": users[0], "password": "x"}).status_code == 401


def test_login_retry_after_helper(auth_api, fresh_limiters):
    user = _new_user(auth_api)
    assert auth_api.service.login_retry_after(user) is None
    auth_api.service.login_user(user, "wrong")
    auth_api.service.login_user(user, "wrong")  # trips (max_attempts=2)
    retry = auth_api.service.login_retry_after(user)
    assert retry is not None and retry > 0


def test_client_ip_xff_policy(auth_api, monkeypatch):
    mod = auth_api.mod

    def req(xff=None, host="203.0.113.7"):
        headers = {"x-forwarded-for": xff} if xff else {}
        return types.SimpleNamespace(
            headers=headers,
            client=types.SimpleNamespace(host=host) if host else None,
        )

    # Default: X-Forwarded-For is untrusted — the socket peer wins.
    monkeypatch.delenv("QUASAR_TRUST_XFF", raising=False)
    assert mod._client_ip(req(xff="6.6.6.6")) == "203.0.113.7"

    # Opt-in trusted ingress: the entry our OWN proxy appended is the RIGHTMOST
    # one. Everything to its left is client-supplied and forgeable — trusting it
    # would let an attacker mint a fresh limiter key per request (bypass) or
    # spend a victim IP's bucket (targeted lockout).
    monkeypatch.setenv("QUASAR_TRUST_XFF", "1")
    monkeypatch.delenv("QUASAR_TRUSTED_PROXY_HOPS", raising=False)
    assert mod._client_ip(req(xff="6.6.6.6, 10.0.0.1")) == "10.0.0.1"
    # A forged leftmost entry cannot change the key.
    forged = mod._client_ip(req(xff="1.1.1.1, 10.0.0.1"))
    honest = mod._client_ip(req(xff="9.9.9.9, 10.0.0.1"))
    assert forged == honest == "10.0.0.1"
    # Single-entry header (proxy appended, client sent none).
    assert mod._client_ip(req(xff="10.0.0.1")) == "10.0.0.1"

    # Two trusted proxies → the second-from-right is ours.
    monkeypatch.setenv("QUASAR_TRUSTED_PROXY_HOPS", "2")
    assert mod._client_ip(req(xff="6.6.6.6, 10.0.0.1, 10.0.0.2")) == "10.0.0.1"
    # Header shorter than the declared hop count → tampered/misconfigured →
    # fall back to the socket peer rather than trust a client-supplied entry.
    assert mod._client_ip(req(xff="6.6.6.6")) == "203.0.113.7"
    monkeypatch.delenv("QUASAR_TRUSTED_PROXY_HOPS", raising=False)

    # Malformed/empty XFF falls back to the socket peer.
    assert mod._client_ip(req(xff="   ")) == "203.0.113.7"
    # IPv6 normalization (brackets + zone stripped, canonical form).
    assert mod._client_ip(req(xff="[2001:DB8::1%eth0]")) == "2001:db8::1"
    # No socket info at all → stable non-empty key.
    monkeypatch.delenv("QUASAR_TRUST_XFF", raising=False)
    assert mod._client_ip(req(host=None)) == "unknown"


def test_no_cookie_on_failed_login(auth_api, fresh_limiters):
    client = auth_api.client()
    user = _new_user(auth_api)
    res = client.post("/api/auth/login", json={"username": user, "password": "x"})
    assert res.status_code == 401
    assert "quasar_auth" not in (res.headers.get("set-cookie") or "")


def test_cookie_only_me_and_logout_roundtrip(auth_api, fresh_limiters):
    client = auth_api.client()
    user = _new_user(auth_api)
    login = client.post("/api/auth/login", json={"username": user, "password": "pw-123456"})
    assert login.status_code == 200
    assert "HttpOnly" in (login.headers.get("set-cookie") or "")

    # Cookie-only (no Authorization header) reaches /me.
    me = client.get("/api/auth/me")
    assert me.status_code == 200 and me.json()["user"]["username"] == user

    # Logout clears the cookie → /me is 401 again.
    assert client.post("/api/auth/logout").status_code == 200
    assert client.get("/api/auth/me").status_code == 401


def test_cors_exposes_retry_after_header(auth_api):
    """Retry-After must be in the CORS expose list, or cross-origin browsers
    can't read it off the 429 (TestClient can't see this — see main.py)."""
    import api.main as main_mod

    exposed = None
    for mw in main_mod.app.user_middleware:
        # Starlette stores kwargs on the Middleware wrapper (kwargs on newer,
        # options on older releases).
        opts = getattr(mw, "kwargs", None) or getattr(mw, "options", {})
        if "expose_headers" in opts:
            exposed = opts["expose_headers"]
            break
    assert exposed is not None, "CORS middleware not found / no expose_headers"
    assert any(h.lower() == "retry-after" for h in exposed)


def test_bearer_auth_still_works(auth_api, fresh_limiters):
    client = auth_api.client()
    user = _new_user(auth_api)
    token = client.post(
        "/api/auth/login", json={"username": user, "password": "pw-123456"}
    ).json()["token"]

    fresh = auth_api.client()  # no cookie jar
    me = fresh.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert me.status_code == 200 and me.json()["user"]["username"] == user


def test_ip_budget_only_invalid_credentials(auth_api, fresh_limiters):
    """CX-13: only wrong-password failures consume the per-IP budget. A
    provider-mismatch rejection (Google account, local login attempt) must not."""
    email = f"google-{next(_UNIQ)}@obs.org"
    auth_api.service.register_or_login_google_user(email, "G User", None)

    client = auth_api.client()
    # ip max_attempts=3: five provider-mismatch 401s must NOT trip the IP lock.
    for _ in range(5):
        r = client.post("/api/auth/login", json={"username": email, "password": "x"})
        assert r.status_code == 401 and "Google" in r.json()["detail"]

    # A brand-new user from the same client still gets normal (non-429) handling.
    fresh = _new_user(auth_api)
    ok = client.post("/api/auth/login", json={"username": fresh, "password": "pw-123456"})
    assert ok.status_code == 200


def test_invalid_bearer_fails_closed_not_cookie_fallback(auth_api, fresh_limiters):
    """An EXPLICIT but invalid Bearer must NOT silently fall back to the ambient
    cookie — otherwise a caller with a low-privilege/expired Bearer executes as
    whatever identity the browser cookie carries (confused-deputy escalation)."""
    client = auth_api.client()
    user = _new_user(auth_api)
    assert client.post(
        "/api/auth/login", json={"username": user, "password": "pw-123456"}
    ).status_code == 200  # admin/user cookie now in the jar

    # Garbage Bearer present → 401 (fail closed), NOT 200-as-cookie-identity.
    forced = client.get("/api/auth/me", headers={"Authorization": "Bearer garbage.stale.jwt"})
    assert forced.status_code == 401

    # No Authorization header at all → the cookie authenticates normally.
    cookie_only = client.get("/api/auth/me")
    assert cookie_only.status_code == 200 and cookie_only.json()["user"]["username"] == user


def test_forged_xff_cannot_bypass_or_weaponize_ip_limiter(auth_api, fresh_limiters, monkeypatch):
    """Under a trusted ingress, a rotating forged leftmost XFF must neither mint a
    fresh limiter key per request (bypass) nor let an attacker spend a victim's
    bucket (targeted lockout). Only the proxy-appended rightmost entry counts."""
    monkeypatch.setenv("QUASAR_TRUST_XFF", "1")
    client = auth_api.client()
    users = [_new_user(auth_api) for _ in range(4)]

    # ip limiter max_attempts=3. Attacker rotates the forged leftmost entry; the
    # real peer (rightmost, appended by our proxy) stays constant → still locks.
    for i, user in enumerate(users[:3]):
        res = client.post(
            "/api/auth/login",
            json={"username": user, "password": "x"},
            headers={"x-forwarded-for": f"9.9.9.{i}, 198.51.100.5"},
        )
        assert res.status_code == 401

    blocked = client.post(
        "/api/auth/login",
        json={"username": users[3], "password": "pw-123456"},
        headers={"x-forwarded-for": "9.9.9.250, 198.51.100.5"},
    )
    assert blocked.status_code == 429  # bypass prevented

    # A DIFFERENT real peer is unaffected — the attacker could not poison it.
    other = client.post(
        "/api/auth/login",
        json={"username": users[3], "password": "pw-123456"},
        headers={"x-forwarded-for": "198.51.100.5, 203.0.113.99"},
    )
    assert other.status_code == 200  # victim IP never locked out


def test_csrf_blocks_cookie_auth_cross_site_write(auth_api):
    """Cookie auth (SameSite=None in prod) reintroduces CSRF that Bearer never had:
    /api/chat/upload, /api/personalization/upload and /api/proposals/review are
    multipart (preflight-free), and /api/feedback parses body via req.json() so it
    accepts a preflight-free text/plain. The Origin check must reject those."""
    from fastapi.testclient import TestClient
    import api.main as main_mod

    client = TestClient(main_mod.app)
    client.cookies.set("quasar_auth", "any-value")  # presence is what arms the check

    evil = {"Origin": "https://evil.example"}
    # Preflight-free multipart write from a hostile origin → blocked.
    blocked = client.post("/api/chat/upload", data={"message": "hi"}, headers=evil)
    assert blocked.status_code == 403, blocked.status_code
    assert "Cross-site" in blocked.json()["detail"]

    # Preflight-free text/plain JSON write → blocked.
    blocked2 = client.post(
        "/api/feedback",
        content=b'{"message_id":"m","feedback":"like"}',
        headers={**evil, "Content-Type": "text/plain"},
    )
    assert blocked2.status_code == 403


def test_csrf_allows_legitimate_and_non_browser_traffic(auth_api):
    from fastapi.testclient import TestClient
    import api.main as main_mod

    client = TestClient(main_mod.app)
    client.cookies.set("quasar_auth", "any-value")

    # Allow-listed dev origin is not blocked (401/422/etc are fine — just not 403).
    ok_origin = client.post(
        "/api/feedback",
        json={"message_id": "m", "feedback": "like"},
        headers={"Origin": "http://localhost:3001"},
    )
    assert ok_origin.status_code != 403

    # No Origin header (curl / server-to-server) is not a CSRF victim → not blocked.
    no_origin = client.post("/api/feedback", json={"message_id": "m", "feedback": "like"})
    assert no_origin.status_code != 403

    # Safe methods are never blocked, even cross-site.
    safe = client.get("/api/auth/me", headers={"Origin": "https://evil.example"})
    assert safe.status_code != 403

    # Bearer-only clients (no auth cookie) are untouched even from a weird origin.
    bare = TestClient(main_mod.app)
    bearer_cross = bare.post(
        "/api/feedback",
        json={"message_id": "m", "feedback": "like"},
        headers={"Origin": "https://evil.example", "Authorization": "Bearer x"},
    )
    assert bearer_cross.status_code != 403


def test_cors_exposes_retry_after(auth_api):
    """Retry-After is not CORS-safelisted: without expose_headers the frontend
    JS can never read the throttle countdown. Verified against the REAL app's
    CORS middleware (browser-found defect — TestClient bypasses CORS, so this
    asserts the Access-Control-Expose-Headers contract instead)."""
    from fastapi.testclient import TestClient
    import api.main as main_mod

    client = TestClient(main_mod.app)
    res = client.post(
        "/api/auth/login",
        json={"username": "cors-probe@obs.org", "password": "wrong"},
        headers={"Origin": "http://localhost:3001"},  # matches the dev regex
    )
    exposed = res.headers.get("access-control-expose-headers", "")
    assert "retry-after" in exposed.lower()
