import sqlite3
import hashlib
import os
import uuid
from typing import Optional, Tuple
from datetime import datetime

class AuthService:
    """
    Authentication Service
    Handles user registration and login using SQLite and PBKDF2 hashing.
    """
    
    def __init__(self, db_path: str = "users.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Initialize the users database"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                username TEXT UNIQUE NOT NULL,
                password_hash BLOB NOT NULL,
                salt BLOB NOT NULL,
                created_at TEXT NOT NULL
            )
        ''')
        conn.commit()
        conn.close()

    def _hash_password(self, password: str, salt: bytes = None) -> Tuple[bytes, bytes]:
        """Hash a password using PBKDF2"""
        if salt is None:
            salt = os.urandom(32)
        
        # PBKDF2-HMAC-SHA256
        pwd_hash = hashlib.pbkdf2_hmac(
            'sha256',
            password.encode('utf-8'),
            salt,
            100000
        )
        return pwd_hash, salt

    def register_user(self, username: str, password: str) -> Tuple[bool, str]:
        """Register a new user"""
        if not username or not password:
            return False, "Username and password required"
            
        try:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            
            # Check if user exists
            c.execute("SELECT id FROM users WHERE username = ?", (username,))
            if c.fetchone():
                conn.close()
                return False, "Username already exists"
            
            # Create user
            user_id = str(uuid.uuid4())
            pwd_hash, salt = self._hash_password(password)
            created_at = datetime.now().isoformat()
            
            c.execute(
                "INSERT INTO users (id, username, password_hash, salt, created_at) VALUES (?, ?, ?, ?, ?)",
                (user_id, username, pwd_hash, salt, created_at)
            )
            conn.commit()
            conn.close()
            return True, "Registration successful"
            
        except Exception as e:
            return False, f"Registration failed: {str(e)}"

    def login_user(self, username: str, password: str) -> Tuple[bool, Optional[str], str]:
        """Login a user. Returns (success, user_id, message)"""
        try:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            
            c.execute("SELECT id, password_hash, salt FROM users WHERE username = ?", (username,))
            result = c.fetchone()
            conn.close()
            
            if not result:
                return False, None, "Invalid username or password"
            
            user_id, stored_hash, salt = result
            
            # Verify password
            pwd_hash, _ = self._hash_password(password, salt)
            
            if pwd_hash == stored_hash:
                return True, user_id, "Login successful"
            else:
                return False, None, "Invalid username or password"
                
        except Exception as e:
            return False, None, f"Login failed: {str(e)}"
