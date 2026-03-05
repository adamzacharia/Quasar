# services/auth.py
"""
Authentication Service
Handles user registration and login using SQLite and PBKDF2 hashing.
"""

import sqlite3
import hashlib
import os
import uuid
from typing import Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import jwt

# Secure default or loaded from env
JWT_SECRET = os.environ.get("JWT_SECRET", "quasar-research-assistant-super-secret-key-2026")
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_HOURS = 24 * 7  # 1 week


def _default_db_path() -> str:
    """Return a stable DB path relative to the project root."""
    root = Path(__file__).resolve().parent.parent
    return str(root / "users.db")


class AuthService:
    """
    Authentication Service
    Handles user registration and login using SQLite and PBKDF2 hashing.
    """

    def __init__(self, db_path: str = None):
        self.db_path = db_path or _default_db_path()
        self._init_db()

    def _init_db(self):
        """Initialize the users database and handle migrations"""
        with sqlite3.connect(self.db_path) as conn:
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
            except sqlite3.OperationalError:
                pass # Column exists
                
            try:
                cursor.execute("ALTER TABLE users ADD COLUMN display_name TEXT")
            except sqlite3.OperationalError:
                pass
                
            try:
                cursor.execute("ALTER TABLE users ADD COLUMN auth_provider TEXT DEFAULT 'local'")
            except sqlite3.OperationalError:
                pass
                
            conn.commit()

    def generate_token(self, user_id: str, email: str = None, display_name: str = None) -> str:
        """Generate a JWT token for a user"""
        payload = {
            "sub": user_id,
            "email": email,
            "name": display_name,
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
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # Check if user exists
                cursor.execute("SELECT id FROM users WHERE username = ?", (username,))
                if cursor.fetchone():
                    return False, "Username already exists", ""

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

    def login_user(self, username: str, password: str) -> Tuple[bool, Optional[str], Optional[str], Optional[str], str]:
        """Login a user. Returns (success, user_id, email, display_name, message)"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                cursor.execute("SELECT id, password_hash, salt, auth_provider, email, display_name FROM users WHERE username = ?", (username,))
                result = cursor.fetchone()

            if not result:
                return False, None, None, None, "Invalid username or password"

            user_id, stored_hash, salt, auth_provider, email, display_name = result
            
            if auth_provider == 'google':
                return False, None, None, None, "Please login with Google"

            # Verify password
            pwd_hash, _ = self._hash_password(password, salt)

            if pwd_hash == stored_hash:
                return True, user_id, email, display_name, "Login successful"
            else:
                return False, None, None, None, "Invalid username or password"

        except Exception as e:
            return False, None, None, None, f"Login failed: {str(e)}"
            
    def register_or_login_google_user(self, email: str, display_name: str) -> Tuple[bool, str, str, str]:
        """Register or login a user via Google. Returns (success, user_id, message, token)"""
        username = email # Use email as username for Google auth
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Check if user exists
                cursor.execute("SELECT id FROM users WHERE username = ?", (username,))
                result = cursor.fetchone()
                
                if result:
                    user_id = result[0]
                    # Update display name just in case it changed
                    cursor.execute("UPDATE users SET display_name = ?, auth_provider = 'google' WHERE id = ?", 
                                  (display_name, user_id))
                    conn.commit()
                    token = self.generate_token(user_id, email, display_name)
                    return True, user_id, "Login successful", token
                
                # Create new user
                user_id = str(uuid.uuid4())
                created_at = datetime.now().isoformat()
                
                cursor.execute(
                    """INSERT INTO users 
                       (id, username, email, display_name, auth_provider, created_at) 
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (user_id, username, email, display_name, 'google', created_at)
                )
                conn.commit()
                token = self.generate_token(user_id, email, display_name)
                return True, user_id, "Registration successful", token
                
        except Exception as e:
            return False, "", f"Google Auth failed: {str(e)}", ""
