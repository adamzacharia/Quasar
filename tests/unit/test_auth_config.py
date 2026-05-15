import importlib
import sys
import types
from typing import Optional

import pytest


class _FakePyJWTError(Exception):
    pass


def _install_fake_jwt(monkeypatch):
    fake_jwt = types.SimpleNamespace(
        PyJWTError=_FakePyJWTError,
        encode=lambda payload, secret, algorithm=None: f"token:{secret}",
        decode=lambda token, secret, algorithms=None: {"sub": "test-user"},
    )
    monkeypatch.setitem(sys.modules, "jwt", fake_jwt)


def _fresh_auth_import(monkeypatch, *, env: str, jwt_secret: Optional[str]):
    _install_fake_jwt(monkeypatch)
    monkeypatch.setenv("QUASAR_ENV", env)
    monkeypatch.delenv("APP_ENV", raising=False)
    monkeypatch.delenv("ENVIRONMENT", raising=False)
    if jwt_secret is None:
        monkeypatch.delenv("JWT_SECRET", raising=False)
    else:
        monkeypatch.setenv("JWT_SECRET", jwt_secret)

    sys.modules.pop("services.auth", None)
    return importlib.import_module("services.auth")


def test_production_requires_jwt_secret(monkeypatch):
    with pytest.raises(RuntimeError, match="JWT_SECRET is required"):
        _fresh_auth_import(monkeypatch, env="production", jwt_secret=None)


def test_production_uses_configured_jwt_secret(monkeypatch):
    auth = _fresh_auth_import(
        monkeypatch,
        env="production",
        jwt_secret="real-production-secret-value",
    )

    assert auth.JWT_SECRET == "real-production-secret-value"


def test_production_rejects_known_default_jwt_secret(monkeypatch):
    with pytest.raises(RuntimeError, match="private deployment-specific"):
        _fresh_auth_import(
            monkeypatch,
            env="production",
            jwt_secret="quasar-research-assistant-super-secret-key-2026",
        )


def test_non_production_uses_local_only_fallback(monkeypatch):
    auth = _fresh_auth_import(monkeypatch, env="testing", jwt_secret=None)

    assert auth.JWT_SECRET == "quasar-local-development-jwt-secret"
