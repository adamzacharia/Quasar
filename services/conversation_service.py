# services/conversation_service.py
"""
Conversation Storage Service
Manages multiple chat conversations per user.
Uses Turso (cloud) when TURSO_DATABASE_URL is set, else local SQLite.
"""

import json
import os
from datetime import datetime
from typing import List, Dict, Optional
import uuid
from services.db import get_connection

class ConversationService:
    """
    Manages conversation storage and retrieval per user.
    Uses Turso cloud DB or local SQLite fallback.
    """
    
    def __init__(self, db_path: str = None):
        if db_path is None:
            root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            data_dir = os.path.join(root_dir, "data")
            os.makedirs(data_dir, exist_ok=True)
            db_path = os.path.join(data_dir, "conversations.db")
        
        self._local_db_path = db_path
        self._init_db()

    def _get_conn(self):
        """Get a database connection (Turso cloud or local SQLite)."""
        return get_connection(self._local_db_path)
    
    def _init_db(self):
        """Initialize database tables"""
        conn = self._get_conn()
        try:
            cursor = conn.cursor()
        
            # Conversations table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    title TEXT,
                    model TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            ''')
        
            # Safe migration for existing installations
            try:
                cursor.execute('ALTER TABLE conversations ADD COLUMN model TEXT')
            except Exception:
                pass  # Column already exists
        
            # Messages table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    message_type TEXT DEFAULT 'general',
                    metadata TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (conversation_id) REFERENCES conversations(id)
                )
            ''')
        
            # Index for faster queries
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_conv_user 
                ON conversations(user_id, updated_at DESC)
            ''')
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_msg_conv 
                ON messages(conversation_id, created_at)
            ''')

            # Conversation-scoped provider file references for document reuse.
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS conversation_file_refs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    conversation_id TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    model_family TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    mime_type TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    provider_file_id TEXT,
                    provider_file_name TEXT,
                    provider_file_uri TEXT,
                    key_scope TEXT NOT NULL DEFAULT 'platform',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            ''')
            try:
                cursor.execute("ALTER TABLE conversation_file_refs ADD COLUMN key_scope TEXT NOT NULL DEFAULT 'platform'")
            except Exception:
                pass
            cursor.execute('DROP INDEX IF EXISTS idx_conv_file_unique')
            cursor.execute('''
                CREATE UNIQUE INDEX IF NOT EXISTS idx_conv_file_unique
                ON conversation_file_refs(user_id, conversation_id, provider, key_scope, content_hash)
            ''')
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_conv_file_lookup
                ON conversation_file_refs(user_id, conversation_id, provider, updated_at DESC)
            ''')
        
            conn.commit()
        finally:
            conn.close()
    
    def create_conversation(self, user_id: str, title: str = None, model: str = None) -> str:
        """Create a new conversation, returns conversation ID"""
        conv_id = str(uuid.uuid4())
        now = datetime.now().isoformat()
        
        conn = self._get_conn()
        try:
            cursor = conn.cursor()
        
            cursor.execute('''
                INSERT INTO conversations (id, user_id, title, model, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (conv_id, user_id, title or "New Chat", model, now, now))
        
            conn.commit()
        finally:
            conn.close()
        
        return conv_id
    
    def save_message(self, conversation_id: str, role: str, content: str, 
                     message_type: str = "general", metadata: Dict = None):
        """Save a message to a conversation"""
        now = datetime.now().isoformat()
        
        conn = self._get_conn()
        try:
            cursor = conn.cursor()
        
            cursor.execute('''
                INSERT INTO messages (conversation_id, role, content, message_type, metadata, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (conversation_id, role, content, message_type, 
                  json.dumps(metadata) if metadata else None, now))
        
            # Update conversation timestamp
            cursor.execute('''
                UPDATE conversations SET updated_at = ? WHERE id = ?
            ''', (now, conversation_id))
        
            conn.commit()
        finally:
            conn.close()
    
    def save_full_conversation(self, conversation_id: str, messages: List[Dict]):
        """Save all messages for a conversation (replaces existing)
        
        Properly serializes DataFrames and all message fields.
        """
        conn = self._get_conn()
        try:
            cursor = conn.cursor()
        
            # Delete existing messages
            cursor.execute('DELETE FROM messages WHERE conversation_id = ?', (conversation_id,))
        
            # Insert all messages
            now = datetime.now().isoformat()
            for msg in messages:
                # Serialize content
                content = msg.get("content", "")
            
                # Store extra fields in metadata
                metadata = msg.get("metadata", {}) or {}
            
                # Handle DataFrame data
                if "data" in msg:
                    data = msg["data"]
                    if hasattr(data, 'to_dict'):  # It's a DataFrame
                        try:
                            metadata["data_json"] = data.to_dict(orient='records')
                            metadata["data_columns"] = list(data.columns)
                        except Exception:
                            metadata["data_json"] = str(data)
                    else:
                        metadata["data_json"] = data
            
                # Store source name
                if "source" in msg:
                    metadata["source"] = msg["source"]
            
                # Store image (if bytes, skip - too large; if path, store path)
                if "image" in msg:
                    if isinstance(msg["image"], str):  # Path
                        metadata["image_path"] = msg["image"]
                    # Skip bytes - images won't persist across sessions
            
                # Store caption
                if "caption" in msg:
                    metadata["caption"] = msg["caption"]
            
                cursor.execute('''
                    INSERT INTO messages (conversation_id, role, content, message_type, metadata, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    conversation_id, 
                    msg.get("role", "user"),
                    content,
                    msg.get("type", "general"),
                    json.dumps(metadata) if metadata else None,
                    now
                ))
        
            # Update conversation timestamp
            cursor.execute('UPDATE conversations SET updated_at = ? WHERE id = ?', (now, conversation_id))
        
            conn.commit()
        finally:
            conn.close()
    
    def get_conversation_messages(self, conversation_id: str) -> List[Dict]:
        """Get all messages for a conversation
        
        Properly deserializes DataFrames and restores all message fields.
        """
        conn = self._get_conn()
        try:
            cursor = conn.cursor()
        
            cursor.execute('''
                SELECT role, content, message_type, metadata, created_at
                FROM messages 
                WHERE conversation_id = ?
                ORDER BY created_at
            ''', (conversation_id,))
        
            messages = []
            for row in cursor.fetchall():
                msg = {
                    "role": row[0],
                    "content": row[1] or "",
                    "type": row[2] or "general"
                }
            
                # Parse metadata and restore fields
                if row[3]:
                    try:
                        metadata = json.loads(row[3])
                    
                        # Restore DataFrame data
                        if "data_json" in metadata:
                            try:
                                if isinstance(metadata["data_json"], list):
                                    import pandas as pd
                                    msg["data"] = pd.DataFrame(metadata["data_json"])
                                else:
                                    msg["data"] = metadata["data_json"]
                            except (ValueError, TypeError):
                                pass
                    
                        # Restore source
                        if "source" in metadata:
                            msg["source"] = metadata["source"]
                    
                        # Restore image path
                        if "image_path" in metadata:
                            msg["image"] = metadata["image_path"]
                    
                        # Restore caption
                        if "caption" in metadata:
                            msg["caption"] = metadata["caption"]
                    
                        # Keep any other metadata
                        remaining_meta = {k: v for k, v in metadata.items() 
                                         if k not in ["data_json", "data_columns", "source", "image_path", "caption"]}
                        if remaining_meta:
                            msg["metadata"] = remaining_meta
                        
                    except Exception as e:
                        print(f"Failed to parse message metadata: {e}")
            
                messages.append(msg)
        
        finally:
            conn.close()
        return messages

    def conversation_belongs_to_user(self, conversation_id: str, user_id: str) -> bool:
        """Return True when the conversation exists and belongs to user_id."""
        conn = self._get_conn()
        try:
            cursor = conn.cursor()
            cursor.execute(
                'SELECT 1 FROM conversations WHERE id = ? AND user_id = ?',
                (conversation_id, user_id),
            )
            exists = cursor.fetchone() is not None
        finally:
            conn.close()
        return exists

    def get_conversation_messages_for_user(
        self,
        conversation_id: str,
        user_id: str,
    ) -> Optional[List[Dict]]:
        """Get messages only when the conversation belongs to user_id."""
        if not self.conversation_belongs_to_user(conversation_id, user_id):
            return None
        return self.get_conversation_messages(conversation_id)
    
    def get_user_conversations(self, user_id: str, limit: int = 20) -> List[Dict]:
        """Get list of conversations for a user, most recent first"""
        conn = self._get_conn()
        try:
            cursor = conn.cursor()
        
            cursor.execute('''
                SELECT id, title, created_at, updated_at, model
                FROM conversations 
                WHERE user_id = ?
                ORDER BY updated_at DESC
                LIMIT ?
            ''', (user_id, limit))
        
            conversations = []
            for row in cursor.fetchall():
                conversations.append({
                    "id": row[0],
                    "title": row[1],
                    "created_at": row[2],
                    "updated_at": row[3],
                    "model": row[4] if len(row) > 4 else None
                })
        
        finally:
            conn.close()
        return conversations
    
    def update_conversation_title(self, conversation_id: str, title: str):
        """Update conversation title"""
        conn = self._get_conn()
        try:
            cursor = conn.cursor()
        
            cursor.execute('''
                UPDATE conversations SET title = ? WHERE id = ?
            ''', (title, conversation_id))
        
            conn.commit()
        finally:
            conn.close()

    def update_conversation_model(self, conversation_id: str, model: str):
        """Update conversation model"""
        conn = self._get_conn()
        try:
            cursor = conn.cursor()
        
            cursor.execute('''
                UPDATE conversations SET model = ? WHERE id = ?
            ''', (model, conversation_id))
        
            conn.commit()
        finally:
            conn.close()

    def update_conversation_title_for_user(
        self,
        conversation_id: str,
        user_id: str,
        title: str,
    ) -> bool:
        """Update a conversation title only when it belongs to user_id."""
        now = datetime.now().isoformat()
        conn = self._get_conn()
        try:
            cursor = conn.cursor()

            cursor.execute(
                '''
                UPDATE conversations
                SET title = ?, updated_at = ?
                WHERE id = ? AND user_id = ?
                ''',
                (title, now, conversation_id, user_id),
            )
            conn.commit()
            updated = cursor.rowcount > 0
        finally:
            conn.close()
        return updated
    
    def generate_title_from_message(self, first_message: str) -> str:
        """Generate a short title from the first message"""
        # Take first 40 chars, clean it up
        title = first_message[:40].strip()
        
        # Add ellipsis if truncated
        if len(first_message) > 40:
            title += "..."
        
        return title or "New Chat"
    
    def delete_conversation(self, conversation_id: str):
        """Delete a conversation and all its messages"""
        conn = self._get_conn()
        try:
            cursor = conn.cursor()
        
            # Verify the conversation exists first
            cursor.execute('SELECT id FROM conversations WHERE id = ?', (conversation_id,))
            row = cursor.fetchone()
            if not row:
                print(f"[WARN] delete_conversation: conversation {conversation_id} not found in DB")
                return
        
            cursor.execute('DELETE FROM messages WHERE conversation_id = ?', (conversation_id,))
            cursor.execute('DELETE FROM conversations WHERE id = ?', (conversation_id,))
            cursor.execute('DELETE FROM conversation_file_refs WHERE conversation_id = ?', (conversation_id,))
        
            conn.commit()
        finally:
            conn.close()
        print(f"[INFO] delete_conversation: successfully deleted {conversation_id}")

    def delete_conversation_for_user(self, conversation_id: str, user_id: str) -> bool:
        """Delete a conversation only when it belongs to user_id."""
        conn = self._get_conn()
        try:
            cursor = conn.cursor()

            cursor.execute(
                'SELECT id FROM conversations WHERE id = ? AND user_id = ?',
                (conversation_id, user_id),
            )
            row = cursor.fetchone()
            if not row:
                return False

            cursor.execute('DELETE FROM messages WHERE conversation_id = ?', (conversation_id,))
            cursor.execute(
                'DELETE FROM conversation_file_refs WHERE user_id = ? AND conversation_id = ?',
                (user_id, conversation_id),
            )
            cursor.execute(
                'DELETE FROM conversations WHERE id = ? AND user_id = ?',
                (conversation_id, user_id),
            )

            conn.commit()
        finally:
            conn.close()
        return True
    
    def get_or_create_current(self, user_id: str) -> str:
        """Get most recent conversation or create new one"""
        conversations = self.get_user_conversations(user_id, limit=1)
        
        if conversations:
            return conversations[0]["id"]
        else:
            return self.create_conversation(user_id)

    def save_conversation_file_ref(
        self,
        user_id: str,
        conversation_id: str,
        provider: str,
        model_family: str,
        filename: str,
        mime_type: str,
        content_hash: str,
        provider_file_id: Optional[str] = None,
        provider_file_name: Optional[str] = None,
        provider_file_uri: Optional[str] = None,
        key_scope: str = "platform",
    ) -> Dict:
        """Insert or update a provider file reference for a conversation."""
        now = datetime.now().isoformat()
        conn = self._get_conn()
        try:
            cursor = conn.cursor()

            cursor.execute(
                '''
                SELECT id FROM conversation_file_refs
                WHERE user_id = ? AND conversation_id = ? AND provider = ? AND key_scope = ? AND content_hash = ?
                ''',
                (user_id, conversation_id, provider, key_scope, content_hash),
            )
            row = cursor.fetchone()

            if row:
                ref_id = row[0]
                cursor.execute(
                    '''
                    UPDATE conversation_file_refs
                    SET model_family = ?, filename = ?, mime_type = ?, provider_file_id = ?,
                        provider_file_name = ?, provider_file_uri = ?, key_scope = ?, updated_at = ?
                    WHERE id = ?
                    ''',
                    (
                        model_family,
                        filename,
                        mime_type,
                        provider_file_id,
                        provider_file_name,
                        provider_file_uri,
                        key_scope,
                        now,
                        ref_id,
                    ),
                )
            else:
                cursor.execute(
                    '''
                    INSERT INTO conversation_file_refs (
                        user_id, conversation_id, provider, model_family, filename, mime_type,
                        content_hash, provider_file_id, provider_file_name, provider_file_uri,
                        key_scope, created_at, updated_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''',
                    (
                        user_id,
                        conversation_id,
                        provider,
                        model_family,
                        filename,
                        mime_type,
                        content_hash,
                        provider_file_id,
                        provider_file_name,
                        provider_file_uri,
                        key_scope,
                        now,
                        now,
                    ),
                )
                ref_id = None

            conn.commit()
            if ref_id is None:
                ref_id = cursor.lastrowid
        finally:
            conn.close()

        return {
            "id": ref_id,
            "user_id": user_id,
            "conversation_id": conversation_id,
            "provider": provider,
            "model_family": model_family,
            "filename": filename,
            "mime_type": mime_type,
            "content_hash": content_hash,
            "provider_file_id": provider_file_id,
            "provider_file_name": provider_file_name,
            "provider_file_uri": provider_file_uri,
            "key_scope": key_scope,
            "updated_at": now,
        }

    def list_conversation_file_refs(
        self,
        user_id: str,
        conversation_id: str,
        provider: Optional[str] = None,
        key_scope: Optional[str] = None,
    ) -> List[Dict]:
        """List stored provider file references for a user conversation."""
        conn = self._get_conn()
        try:
            cursor = conn.cursor()

            if provider and key_scope:
                cursor.execute(
                    '''
                    SELECT provider, model_family, filename, mime_type, content_hash,
                           provider_file_id, provider_file_name, provider_file_uri,
                           created_at, updated_at, key_scope
                    FROM conversation_file_refs
                    WHERE user_id = ? AND conversation_id = ? AND provider = ? AND key_scope = ?
                    ORDER BY updated_at ASC, id ASC
                    ''',
                    (user_id, conversation_id, provider, key_scope),
                )
            elif provider:
                cursor.execute(
                    '''
                    SELECT provider, model_family, filename, mime_type, content_hash,
                           provider_file_id, provider_file_name, provider_file_uri,
                           created_at, updated_at, key_scope
                    FROM conversation_file_refs
                    WHERE user_id = ? AND conversation_id = ? AND provider = ?
                    ORDER BY updated_at ASC, id ASC
                    ''',
                    (user_id, conversation_id, provider),
                )
            else:
                cursor.execute(
                    '''
                    SELECT provider, model_family, filename, mime_type, content_hash,
                           provider_file_id, provider_file_name, provider_file_uri,
                           created_at, updated_at, key_scope
                    FROM conversation_file_refs
                    WHERE user_id = ? AND conversation_id = ?
                    ORDER BY updated_at ASC, id ASC
                    ''',
                    (user_id, conversation_id),
                )

            rows = cursor.fetchall()
        finally:
            conn.close()

        refs = []
        for row in rows:
            refs.append({
                "provider": row[0],
                "model_family": row[1],
                "filename": row[2],
                "mime_type": row[3],
                "content_hash": row[4],
                "provider_file_id": row[5],
                "provider_file_name": row[6],
                "provider_file_uri": row[7],
                "created_at": row[8],
                "updated_at": row[9],
                "key_scope": row[10],
            })
        return refs

    def clear_conversation_file_refs(
        self,
        user_id: str,
        conversation_id: str,
        provider: Optional[str] = None,
    ):
        """Delete stored provider file references for a user conversation."""
        conn = self._get_conn()
        try:
            cursor = conn.cursor()

            if provider:
                cursor.execute(
                    '''
                    DELETE FROM conversation_file_refs
                    WHERE user_id = ? AND conversation_id = ? AND provider = ?
                    ''',
                    (user_id, conversation_id, provider),
                )
            else:
                cursor.execute(
                    '''
                    DELETE FROM conversation_file_refs
                    WHERE user_id = ? AND conversation_id = ?
                    ''',
                    (user_id, conversation_id),
                )

            conn.commit()
        finally:
            conn.close()
