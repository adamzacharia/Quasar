"""The frontend origins this API trusts (CORS, CSRF, OAuth result pages).

Production origins are fixed; localhost dev is matched by a tight regex
(localhost/127.0.0.1:300x is not attacker-registerable). Operators can add
specific deploy origins (e.g. a Vercel preview or a raw onrender URL) via the
comma-separated ``QUASAR_CORS_ORIGINS`` env var, never a wildcard (S4).
"""

import os
import re

_DEFAULT_CORS_ORIGINS = [
    "https://quasarassistant.com",
    "https://www.quasarassistant.com",
]
_cors_env = os.getenv("QUASAR_CORS_ORIGINS", "").strip()
_extra_cors_origins = [o.strip() for o in _cors_env.split(",") if o.strip()]
CORS_ALLOWED_ORIGINS = list(dict.fromkeys(_DEFAULT_CORS_ORIGINS + _extra_cors_origins))
CORS_ALLOWED_ORIGIN_REGEX = r"http://localhost:300[0-9]|http://127\.0\.0\.1:300[0-9]"


def is_allowed_origin(origin: str) -> bool:
    if origin in CORS_ALLOWED_ORIGINS:
        return True
    return re.fullmatch(CORS_ALLOWED_ORIGIN_REGEX, origin or "") is not None


def default_frontend_origin() -> str:
    """Where to send a user back when a request did not say where it came
    from: ``QUASAR_FRONTEND_URL`` if it is an allowed origin, else the
    production site."""
    env = os.getenv("QUASAR_FRONTEND_URL", "").strip().rstrip("/")
    if env and is_allowed_origin(env):
        return env
    return "https://www.quasarassistant.com"
