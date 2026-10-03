# services/saved_papers_service.py
"""
Saved (bookmarked) papers, per user.

Bookmarks used to live only in the browser's in-memory store, so every page
reload, and therefore every redeploy, emptied the Saved Papers panel. They are
now server state in the same database as conversations (Turso in production,
local SQLite in development), so they survive reloads, redeploys and devices.

The paper itself is stored as an opaque JSON blob: the frontend owns its shape,
and the server only needs the key to dedupe and delete.
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Optional

from services.db import get_connection

# One bookmark is a card's metadata plus an abstract; anything far beyond that
# is not a paper. The count cap keeps one account from growing the table
# without bound.
MAX_PAPER_JSON_BYTES = 64 * 1024
MAX_SAVED_PAPERS_PER_USER = 2000
MAX_PAPER_KEY_LENGTH = 512


class SavedPaperError(ValueError):
    """Rejected bookmark (bad key, oversized payload, or over the cap)."""


class SavedPapersService:
    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            data_dir = os.path.join(root_dir, "data")
            os.makedirs(data_dir, exist_ok=True)
            db_path = os.path.join(data_dir, "conversations.db")
        self._local_db_path = db_path
        self._init_db()

    def _get_conn(self):
        return get_connection(self._local_db_path)

    def _init_db(self):
        conn = self._get_conn()
        try:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS saved_papers (
                    user_id TEXT NOT NULL,
                    paper_key TEXT NOT NULL,
                    paper_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (user_id, paper_key)
                )
            ''')
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_saved_papers_user
                ON saved_papers(user_id, created_at)
            ''')
            conn.commit()
        finally:
            conn.close()

    @staticmethod
    def _clean_key(paper_key: str) -> str:
        key = str(paper_key or "").strip()
        if not key or len(key) > MAX_PAPER_KEY_LENGTH:
            raise SavedPaperError("Invalid paper key")
        return key

    def list_papers(self, user_id: str) -> List[Dict]:
        """A user's bookmarks, oldest first (the order they were saved in)."""
        conn = self._get_conn()
        try:
            cursor = conn.cursor()
            cursor.execute(
                '''
                SELECT paper_key, paper_json FROM saved_papers
                WHERE user_id = ?
                ORDER BY created_at ASC
                ''',
                (user_id,),
            )
            rows = cursor.fetchall()
        finally:
            conn.close()

        papers: List[Dict] = []
        for paper_key, paper_json in rows:
            try:
                paper = json.loads(paper_json)
            except (TypeError, ValueError):
                continue  # one corrupt row must not empty the whole panel
            if not isinstance(paper, dict):
                continue
            paper["id"] = paper_key
            papers.append(paper)
        return papers

    def save_paper(self, user_id: str, paper_key: str, paper: Dict) -> None:
        """Insert or refresh a bookmark. Re-saving keeps the original position."""
        key = self._clean_key(paper_key)
        if not isinstance(paper, dict):
            raise SavedPaperError("Paper must be an object")
        payload = json.dumps({**paper, "id": key}, ensure_ascii=False)
        if len(payload.encode("utf-8")) > MAX_PAPER_JSON_BYTES:
            raise SavedPaperError("Paper payload too large")

        conn = self._get_conn()
        try:
            cursor = conn.cursor()
            cursor.execute(
                'SELECT 1 FROM saved_papers WHERE user_id = ? AND paper_key = ?',
                (user_id, key),
            )
            exists = cursor.fetchone() is not None
            if exists:
                cursor.execute(
                    'UPDATE saved_papers SET paper_json = ? WHERE user_id = ? AND paper_key = ?',
                    (payload, user_id, key),
                )
            else:
                cursor.execute('SELECT COUNT(*) FROM saved_papers WHERE user_id = ?', (user_id,))
                row = cursor.fetchone()
                if row and int(row[0]) >= MAX_SAVED_PAPERS_PER_USER:
                    raise SavedPaperError("Saved paper limit reached")
                cursor.execute(
                    '''
                    INSERT INTO saved_papers (user_id, paper_key, paper_json, created_at)
                    VALUES (?, ?, ?, ?)
                    ''',
                    (user_id, key, payload, datetime.now().isoformat()),
                )
            conn.commit()
        finally:
            conn.close()

    def remove_paper(self, user_id: str, paper_key: str) -> None:
        """Delete a bookmark. Removing one that is not there is not an error."""
        key = self._clean_key(paper_key)
        conn = self._get_conn()
        try:
            conn.cursor().execute(
                'DELETE FROM saved_papers WHERE user_id = ? AND paper_key = ?',
                (user_id, key),
            )
            conn.commit()
        finally:
            conn.close()
