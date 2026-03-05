# core/memory.py
"""
Enhanced Conversation memory management for Quasar
- Topic detection to prevent context bleeding
- Smart history filtering
- Sync support with UI
"""

from typing import List, Dict, Optional, Set
from collections import deque
from datetime import datetime
import re

class ConversationMemory:
    """
    Manages conversation history with smart context management.
    
    Features:
    - Sliding window for recent messages
    - Topic detection to identify topic changes
    - Smart history filtering to prevent context bleeding
    """

    def __init__(self, max_turns: int = 10):
        self.max_turns = max_turns
        self.messages = deque(maxlen=max_turns * 2)  # user + assistant messages
        self.metadata = {}
        self.current_topic_keywords: Set[str] = set()

    def add_message(self, role: str, content: str, metadata: Optional[Dict] = None):
        """Add a message to memory"""
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        self.messages.append(message)
        
        # Update topic keywords for user messages
        if role == "user":
            self.current_topic_keywords = self._extract_keywords(content)

    def _extract_keywords(self, text: str) -> Set[str]:
        """Extract meaningful keywords from text for topic comparison"""
        # Clean and tokenize
        text = text.lower()
        # Remove common words and commands
        stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'what', 'how', 
                    'why', 'when', 'where', 'who', 'which', 'this', 'that', 'it',
                    'for', 'to', 'of', 'in', 'on', 'at', 'with', 'by', 'from',
                    'can', 'you', 'i', 'me', 'my', 'please', 'tell', 'show', 'find',
                    '@archive', '@search', '@paper', 'alma', 'data', 'search'}
        
        # Extract words (alphanumeric)
        words = set(re.findall(r'\b[a-z]{3,}\b', text))
        keywords = words - stopwords
        
        # Also extract astronomical targets (capitalized words, coordinates)
        targets = set(re.findall(r'\b[A-Z][a-zA-Z0-9\-\+]+\b', text.upper()))
        
        return keywords | targets

    def is_new_topic(self, query: str) -> bool:
        """
        Detect if the query is about a completely new topic.
        Returns True if we should minimize history inclusion.
        """
        if not self.messages:
            return True  # First message, no history anyway
        
        # Extract keywords from new query
        new_keywords = self._extract_keywords(query)
        
        if not new_keywords or not self.current_topic_keywords:
            return False  # Can't determine, be conservative
        
        # Calculate keyword overlap
        overlap = new_keywords & self.current_topic_keywords
        overlap_ratio = len(overlap) / max(len(new_keywords), 1)
        
        # If less than 20% overlap, consider it a new topic
        return overlap_ratio < 0.2

    def get_history(self) -> List[Dict]:
        """Get full conversation history"""
        return list(self.messages)
    
    def get_relevant_history(self, query: str, max_messages: int = 4) -> List[Dict]:
        """
        Get only history relevant to the current query.
        Prevents context bleeding from unrelated previous conversations.
        """
        if self.is_new_topic(query):
            # New topic - only include last 1-2 messages for minimal context
            return list(self.messages)[-2:] if self.messages else []
        
        # Continuing topic - include more history but cap it
        return list(self.messages)[-max_messages:]
    
    def get_last_user_query(self) -> Optional[str]:
        """Get the last user message (for follow-up detection)"""
        for msg in reversed(self.messages):
            if msg["role"] == "user":
                return msg["content"]
        return None

    def get_last_n_turns(self, n: int) -> List[Dict]:
        """Get last n conversation turns"""
        return list(self.messages)[-n*2:]

    def clear(self):
        """Clear conversation memory"""
        self.messages.clear()
        self.metadata.clear()
        self.current_topic_keywords.clear()

    def sync_from_ui(self, ui_messages: List[Dict]):
        """
        Sync memory from UI session state.
        Call this to ensure agent memory matches UI display.
        """
        self.messages.clear()
        for msg in ui_messages[-self.max_turns * 2:]:
            self.messages.append({
                "role": msg.get("role", "user"),
                "content": msg.get("content", str(msg.get("data", ""))),
                "timestamp": datetime.now().isoformat(),
                "metadata": {}
            })

    def get_context_summary(self) -> str:
        """Generate a summary of the conversation context"""
        if not self.messages:
            return "No previous conversation"

        topics = []
        for msg in self.messages:
            if msg["role"] == "user":
                content = msg["content"]
                # Extract key terms
                if "VLA" in content or "VLBA" in content:
                    topics.append("VLA/VLBA observations")
                if "pulsar" in content.lower():
                    topics.append("pulsar observations")
                if "download" in content.lower():
                    topics.append("data downloads")
                if "alma" in content.lower():
                    topics.append("ALMA data")
                if "paper" in content.lower():
                    topics.append("literature search")

        if topics:
            return f"Previous topics: {', '.join(set(topics))}"
        return "General radio astronomy queries"
