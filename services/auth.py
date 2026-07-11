# services/auth.py
"""
Authentication Service
Handles user registration and login using PBKDF2 hashing.
Uses Turso (cloud) when TURSO_DATABASE_URL is set, else local SQLite.
"""

import hashlib
import hmac
import os
import uuid
from typing import Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path
from services.db import get_connection, is_using_turso
from services.login_rate_limit import LoginRateLimiter, get_default_login_rate_limiter
import jwt

_LOCAL_JWT_SECRET = "quasar-local-development-jwt-secret"
_PRODUCTION_ENVIRONMENTS = {"production", "prod"}
_UNSAFE_PRODUCTION_SECRETS = {
    _LOCAL_JWT_SECRET,
    "quasar-research-assistant-super-secret-key-2026",
    "change-me",
    "change-me-in-production",
}


def _current_environment() -> str:
    """Return the configured app environment."""
    return (
        os.environ.get("QUASAR_ENV")
        or os.environ.get("APP_ENV")
        or os.environ.get("ENVIRONMENT")
        or "development"
    ).strip().lower()


def resolve_jwt_secret(secret: Optional[str] = None, environment: Optional[str] = None) -> str:
    """Resolve the JWT signing secret and fail fast when production is unsafe."""
    raw_secret = os.environ.get("JWT_SECRET") if secret is None else secret
    env = (_current_environment() if environment is None else environment.strip().lower())

    if raw_secret and raw_secret.strip():
        resolved_secret = raw_secret.strip()
        if env in _PRODUCTION_ENVIRONMENTS and resolved_secret in _UNSAFE_PRODUCTION_SECRETS:
            raise RuntimeError(
                "JWT_SECRET must be a private deployment-specific value in production. "
                "Replace the default/example value with a long random secret."
            )
        return resolved_secret

    if env in _PRODUCTION_ENVIRONMENTS:
        raise RuntimeError(
            "JWT_SECRET is required when QUASAR_ENV=production. "
            "Set it to a long random value in the deployment environment."
        )

    return _LOCAL_JWT_SECRET


JWT_SECRET = resolve_jwt_secret()
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_HOURS = 24 * 7  # 1 week


def _default_db_path() -> str:
    """Return a stable DB path relative to the project root (local fallback only)."""
    override = os.environ.get("QUASAR_USERS_DB_PATH")
    if override and override.strip():
        return override.strip()
    root = Path(__file__).resolve().parent.parent
    data_dir = root / "data"
    data_dir.mkdir(exist_ok=True)
    return str(data_dir / "users.db")


def _truthy(value: str) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _coerce_bytes(value) -> bytes:
    """Normalise a DB-returned value to ``bytes`` for constant-time comparison.

    SQLite returns BLOBs as ``bytes``; the Turso wrapper may return
    ``memoryview``/``bytearray``. ``hmac.compare_digest`` requires matching
    bytes-like operands, so coerce defensively.
    """
    if isinstance(value, (bytes, bytearray, memoryview)):
        return bytes(value)
    if isinstance(value, str):
        return value.encode("utf-8")
    return bytes(value)


class AuthService:
    """
    Authentication Service
    Handles user registration and login using SQLite and PBKDF2 hashing.
    """

    def __init__(self, db_path: str = None, rate_limiter: LoginRateLimiter = None):
        self._local_db_path = db_path or _default_db_path()
        self._rate_limiter = rate_limiter or get_default_login_rate_limiter()
        self._init_db()
        self._seed_local_test_user_if_enabled()

    def _get_conn(self):
        """Get a database connection (Turso cloud or local SQLite)."""
        return get_connection(self._local_db_path)

    def _init_db(self):
        """Initialize the users database and handle migrations"""
        with self._get_conn() as conn:
            cursor = conn.cursor()
            
            # Create base table if it doesn't exist
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    id TEXT PRIMARY KEY,
                    username TEXT UNIQUE NOT NULL,
                    password_hash BLOB,
                    salt BLOB,
                    created_at TEXT NOT NULL
                )
            ''')
            
            # Simple migration: add columns if they don't exist
            try:
                cursor.execute("ALTER TABLE users ADD COLUMN email TEXT")
            except Exception:
                pass # Column exists
                
            try:
                cursor.execute("ALTER TABLE users ADD COLUMN display_name TEXT")
            except Exception:
                pass
                
            try:
                cursor.execute("ALTER TABLE users ADD COLUMN auth_provider TEXT DEFAULT 'local'")
            except Exception:
                pass

            try:
                cursor.execute("ALTER TABLE users ADD COLUMN picture_url TEXT")
            except Exception:
                pass

            # S3: per-user admin role (replaces the hard-coded admin email).
            try:
                cursor.execute("ALTER TABLE users ADD COLUMN role TEXT DEFAULT 'user'")
            except Exception:
                pass

            conn.commit()

    def _seed_local_test_user_if_enabled(self) -> None:
        """Create a disposable local-only login for manual development testing."""
        if not _truthy(os.environ.get("QUASAR_ENABLE_LOCAL_TEST_LOGIN")):
            return
        if _current_environment() in _PRODUCTION_ENVIRONMENTS:
            raise RuntimeError(
                "QUASAR_ENABLE_LOCAL_TEST_LOGIN is not allowed in production mode."
            )
        if is_using_turso():
            raise RuntimeError(
                "QUASAR_ENABLE_LOCAL_TEST_LOGIN is local-only. Set QUASAR_FORCE_LOCAL_DB=1 "
                "or unset TURSO_DATABASE_URL/TURSO_AUTH_TOKEN before enabling it."
            )

        username = os.environ.get("QUASAR_LOCAL_TEST_USERNAME", "1@1").strip()
        email = os.environ.get("QUASAR_LOCAL_TEST_EMAIL", username).strip()
        display_name = os.environ.get("QUASAR_LOCAL_TEST_DISPLAY_NAME", "1").strip()
        password = os.environ.get("QUASAR_LOCAL_TEST_PASSWORD", "1")
        # The disposable local login carries admin via the DB role column (S3),
        # replacing the removed hard-coded '1@1' admin email.
        role = os.environ.get("QUASAR_LOCAL_TEST_ROLE", "admin").strip() or "admin"

        if not username or not password:
            raise RuntimeError("Local test username and password cannot be empty.")

        pwd_hash, salt = self._hash_password(password)
        created_at = datetime.now().isoformat()

        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id FROM users WHERE username = ?", (username,))
            existing = cursor.fetchone()
            if existing:
                cursor.execute(
                    """UPDATE users
                       SET password_hash = ?, salt = ?, email = ?, display_name = ?,
                           auth_provider = 'local', role = ?
                       WHERE username = ?""",
                    (pwd_hash, salt, email, display_name, role, username),
                )
            else:
                cursor.execute(
                    """INSERT INTO users
                       (id, username, password_hash, salt, created_at, email, display_name, auth_provider, role)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        str(uuid.uuid4()),
                        username,
                        pwd_hash,
                        salt,
                        created_at,
                        email,
                        display_name,
                        "local",
                        role,
                    ),
                )
            conn.commit()

    def generate_token(self, user_id: str, email: str = None, display_name: str = None, picture_url: str = None) -> str:
        """Generate a JWT token for a user"""
        payload = {
            "sub": user_id,
            "email": email,
            "name": display_name,
            "picture": picture_url,
            "exp": datetime.utcnow() + timedelta(hours=JWT_EXPIRATION_HOURS),
            "iat": datetime.utcnow()
        }
        return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)
        
    def verify_token(self, token: str) -> Optional[dict]:
        """Verify a JWT token and return its payload"""
        try:
            payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
            return payload
        except jwt.PyJWTError:
            return None

    def _hash_password(self, password: str, salt: bytes = None) -> Tuple[bytes, bytes]:
        """Hash a password using PBKDF2"""
        if salt is None:
            salt = os.urandom(32)

        pwd_hash = hashlib.pbkdf2_hmac(
            'sha256',
            password.encode('utf-8'),
            salt,
            100000
        )
        return pwd_hash, salt

    def register_user(self, username: str, password: str, email: str = None, display_name: str = None) -> Tuple[bool, str, str]:
        """Register a new user. Returns (success, message, token)"""
        if not username or not password:
            return False, "Username and password required", ""

        if not email:
            email = username

        if not display_name:
            display_name = username.split('@')[0]

        try:
            with self._get_conn() as conn:
                cursor = conn.cursor()

                # Check if user exists
                cursor.execute("SELECT id FROM users WHERE username = ?", (username,))
                if cursor.fetchone():
                    return False, "Username already exists", ""

                # SECURITY (CX-01): email must be unique. Admin is granted by
                # email — via the ADMIN_EMAILS allowlist AND the DB role lookup,
                # both of which match on the email column. Without this check a
                # new account could register under an existing admin's email and
                # inherit that admin's privileges (its JWT carries that email).
                # Matched case-insensitively against BOTH columns because for
                # local/Google users the username IS the email.
                normalized_email = email.strip().lower()
                cursor.execute(
                    "SELECT id FROM users "
                    "WHERE lower(email) = ? OR lower(username) = ? LIMIT 1",
                    (normalized_email, normalized_email),
                )
                if cursor.fetchone():
                    return False, "Email already registered", ""

                # Create user
                user_id = str(uuid.uuid4())
                pwd_hash, salt = self._hash_password(password)
                created_at = datetime.now().isoformat()

                cursor.execute(
                    "INSERT INTO users (id, username, password_hash, salt, created_at, email, display_name, auth_provider) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (user_id, username, pwd_hash, salt, created_at, email, display_name, 'local')
                )
                conn.commit()
                token = self.generate_token(user_id, email, display_name)
                return True, "Registration successful", token

        except Exception as e:
            return False, f"Registration failed: {str(e)}", ""

    @staticmethod
    def _rate_key(username: str) -> str:
        """Limiter key = the EXACT identity the DB authenticates.

        SECURITY (CX-03): login resolves the account with ``WHERE username = ?``
        (exact match), so the throttle bucket must key on that same exact string.
        Normalizing (``strip().lower()``) would merge case/whitespace-variant
        accounts into one bucket, letting the owner of an alias account
        (``Admin@x.org``) reset a victim's (``admin@x.org``) throttle by logging
        into their own alias.
        """
        return username or ""

    def login_user(self, username: str, password: str) -> Tuple[bool, Optional[str], Optional[str], Optional[str], str]:
        """Login a user. Returns (success, user_id, email, display_name, message)"""
        rate_key = self._rate_key(username)

        # S6: basic rate limiting — block online password guessing after too
        # many failed attempts for this identity.
        retry_after = self._rate_limiter.seconds_until_unblocked(rate_key)
        if retry_after is not None:
            return (
                False,
                None,
                None,
                None,
                f"Too many failed login attempts. Please try again in {int(retry_after) + 1} seconds.",
            )

        try:
            with self._get_conn() as conn:
                cursor = conn.cursor()

                cursor.execute("SELECT id, password_hash, salt, auth_provider, email, display_name FROM users WHERE username = ?", (username,))
                result = cursor.fetchone()

            if not result:
                self._rate_limiter.record_failure(rate_key)
                return False, None, None, None, "Invalid username or password"

            user_id, stored_hash, salt, auth_provider, email, display_name = result

            if auth_provider == 'google':
                return False, None, None, None, "Please login with Google"

            # Verify password with a constant-time comparison (S6).
            pwd_hash, _ = self._hash_password(password, _coerce_bytes(salt) if salt is not None else None)

            if stored_hash is not None and hmac.compare_digest(_coerce_bytes(pwd_hash), _coerce_bytes(stored_hash)):
                self._rate_limiter.reset(rate_key)
                return True, user_id, email, display_name, "Login successful"
            else:
                self._rate_limiter.record_failure(rate_key)
                return False, None, None, None, "Invalid username or password"

        except Exception as e:
            return False, None, None, None, f"Login failed: {str(e)}"

    def login_retry_after(self, username: str) -> Optional[float]:
        """Seconds until this identity may attempt login again, or None.

        Lets the API layer answer an already-locked identity with HTTP 429 +
        Retry-After BEFORE hashing anything (S6). The attempt that trips the
        limiter still flows through login_user and returns a normal failure.
        """
        key = self._rate_key(username)
        if not key:
            return None
        return self._rate_limiter.seconds_until_unblocked(key)

    def get_user_role(self, identity: str) -> Optional[str]:
        """Return the ``role`` for a user matched by username or email."""
        key = (identity or "").strip().lower()
        if not key:
            return None
        try:
            with self._get_conn() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT role FROM users WHERE lower(username) = ? OR lower(email) = ? LIMIT 1",
                    (key, key),
                )
                row = cursor.fetchone()
            if not row:
                return None
            return row[0]
        except Exception:
            return None

    def set_user_role(self, identity: str, role: str) -> bool:
        """Set a user's ``role`` (e.g. 'admin'/'user'). Returns True on match.

        Grants admin without a hard-coded email or a redeploy (S3). Callers
        that gate on admin should invalidate the admin-role cache afterwards
        via ``services.admin_access.clear_admin_role_cache()``.
        """
        key = (identity or "").strip().lower()
        if not key:
            return False
        try:
            with self._get_conn() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "UPDATE users SET role = ? WHERE lower(username) = ? OR lower(email) = ?",
                    (role, key, key),
                )
                conn.commit()
                changed = getattr(cursor, "rowcount", 0)
            # Invalidate the admin-role cache so a grant/revoke takes effect
            # immediately (the cache is keyed by username OR email; clear all).
            try:
                from services.admin_access import clear_admin_role_cache

                clear_admin_role_cache()
            except Exception:
                pass
            return bool(changed) if changed and changed > 0 else self.get_user_role(key) == role
        except Exception:
            return False

    def register_or_login_google_user(self, email: str, display_name: str, picture_url: str = None) -> Tuple[bool, str, str, str]:
        """Register or login a user via Google. Returns (success, user_id, message, token)"""
        username = email # Use email as username for Google auth
        
        try:
            with self._get_conn() as conn:
                cursor = conn.cursor()
                
                # Check if user exists
                cursor.execute("SELECT id FROM users WHERE username = ?", (username,))
                result = cursor.fetchone()
                
                if result:
                    user_id = result[0]
                    # Update display name and picture just in case they changed
                    cursor.execute(
                        "UPDATE users SET display_name = ?, picture_url = ?, auth_provider = 'google' WHERE id = ?",
                        (display_name, picture_url, user_id),
                    )
                    conn.commit()
                    token = self.generate_token(user_id, email, display_name, picture_url)
                    return True, user_id, "Login successful", token
                
                # Create new user
                user_id = str(uuid.uuid4())
                created_at = datetime.now().isoformat()
                
                cursor.execute(
                    """INSERT INTO users 
                       (id, username, email, display_name, auth_provider, picture_url, created_at) 
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (user_id, username, email, display_name, 'google', picture_url, created_at)
                )
                conn.commit()
                token = self.generate_token(user_id, email, display_name, picture_url)
                return True, user_id, "Registration successful", token
                
        except Exception as e:
            return False, "", f"Google Auth failed: {str(e)}", ""
