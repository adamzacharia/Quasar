# core/memory.py
"""
Conversation memory management for Quasar
"""

from typing import List, Dict, Optional
from collections import deque
from datetime import datetime

class ConversationMemory:
    """Manages conversation history with sliding window"""

    def __init__(self, max_turns: int = 10):
        self.max_turns = max_turns
        self.messages = deque(maxlen=max_turns * 2)  # user + assistant messages
        self.metadata = {}

    def add_message(self, role: str, content: str, metadata: Optional[Dict] = None):
        """Add a message to memory"""
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        self.messages.append(message)

    def get_history(self) -> List[Dict]:
        """Get conversation history"""
        return list(self.messages)

    def get_last_n_turns(self, n: int) -> List[Dict]:
        """Get last n conversation turns"""
        return list(self.messages)[-n*2:]

    def clear(self):
        """Clear conversation memory"""
        self.messages.clear()
        self.metadata.clear()

    def get_context_summary(self) -> str:
        """Generate a summary of the conversation context"""
        if not self.messages:
            return "No previous conversation"

        topics = []
        for msg in self.messages:
            if msg["role"] == "user":
                # Extract key terms (simple approach)
                if "VLA" in msg["content"] or "VLBA" in msg["content"]:
                    topics.append("VLA/VLBA observations")
                if "pulsar" in msg["content"].lower():
                    topics.append("pulsar observations")
                if "download" in msg["content"].lower():
                    topics.append("data downloads")

        if topics:
            return f"Previous topics: {', '.join(set(topics))}"
        return "General radio astronomy queries"
