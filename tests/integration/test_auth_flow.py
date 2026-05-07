import sys
import os
import time
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Mock OpenAIEmbeddings to avoid API calls
from typing import List
class MockEmbeddings:
    def __init__(self, **kwargs):
        pass
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # Return dummy 1536-dim vectors (standard OpenAI size)
        return [[0.1] * 1536 for _ in texts]
        
    def embed_query(self, text: str) -> List[float]:
        return [0.1] * 1536

# Patch the import in memory_service
import services.memory_service
services.memory_service.OpenAIEmbeddings = MockEmbeddings

from services.auth import AuthService
from services.memory_service import MemoryService
from core.agent import QuasarAgent, AgentConfig

def test_auth_and_memory():
    print("Testing Auth and Memory Isolation...")
    
    # 1. Test Auth Service
    print("\n1. Testing AuthService...")
    auth = AuthService("test_users.db")
    
    # Register Alice
    success, msg = auth.register_user("alice", "password123")
    print(f"Register Alice: {success} - {msg}")
    
    # Login Alice
    success, alice_id, msg = auth.login_user("alice", "password123")
    print(f"Login Alice: {success} - ID: {alice_id}")
    
    if not success:
        print("[FAIL] Alice login failed")
        return

    # Register Bob
    success, msg = auth.register_user("bob", "password123")
    print(f"Register Bob: {success} - {msg}")
    
    # Login Bob
    success, bob_id, msg = auth.login_user("bob", "password123")
    print(f"Login Bob: {success} - ID: {bob_id}")

    # 2. Test Memory Isolation via Agent
    print("\n2. Testing Memory Isolation...")
    
    # Always use MemoryService directly for reliable testing of isolation logic
    # (Testing via Agent is harder without mocking LLM)
    mem_service = MemoryService()
    
    # Alice's Memory
    print(f"Adding memory for Alice ({alice_id})...")
    mem_service.add_memory("Alice likes pulsars", user_id=alice_id)
    
    # Bob's Memory
    print(f"Adding memory for Bob ({bob_id})...")
    mem_service.add_memory("Bob likes quasars", user_id=bob_id)
    
    # Verify Alice
    print("Searching Alice's memory...")
    alice_mems = mem_service.search_memories("likes", user_id=alice_id)
    print(f"Alice's memories: {alice_mems}")
    
    if any("pulsars" in m for m in alice_mems) and not any("quasars" in m for m in alice_mems):
        print("[PASS] Alice's memory is correct and isolated")
    else:
        print("[FAIL] Alice's memory check failed")

    # Verify Bob
    print("Searching Bob's memory...")
    bob_mems = mem_service.search_memories("likes", user_id=bob_id)
    print(f"Bob's memories: {bob_mems}")
    
    if any("quasars" in m for m in bob_mems) and not any("pulsars" in m for m in bob_mems):
        print("[PASS] Bob's memory is correct and isolated")
    else:
        print("[FAIL] Bob's memory check failed")
            
    # Clean up
    if os.path.exists("test_users.db"):
        os.remove("test_users.db")

if __name__ == "__main__":
    test_auth_and_memory()
