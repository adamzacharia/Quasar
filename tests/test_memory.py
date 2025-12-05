import sys
import os
import time
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from services.memory_service import MemoryService

def test_memory_persistence():
    print("Testing Memory Persistence...")
    
    # 1. Initialize Service
    mem_service = MemoryService()
    
    # 2. Add a unique memory
    unique_fact = f"User loves radio astronomy {time.time()}"
    print(f"Adding memory: {unique_fact}")
    mem_service.add_memory(unique_fact)
    
    # 3. Search for it immediately
    print("Searching immediately...")
    results = mem_service.search_memories("radio astronomy")
    if any(unique_fact in r for r in results):
        print("✅ Immediate retrieval successful")
    else:
        print("❌ Immediate retrieval failed")
        print(f"Results: {results}")

    # 4. Re-initialize (Simulate restart)
    print("Re-initializing service (Simulating restart)...")
    mem_service_2 = MemoryService()
    results_2 = mem_service_2.search_memories("radio astronomy")
    
    if any(unique_fact in r for r in results_2):
        print("✅ Persistence successful")
    else:
        print("❌ Persistence failed")
        print(f"Results after restart: {results_2}")

if __name__ == "__main__":
    try:
        test_memory_persistence()
    except Exception as e:
        print(f"Test failed with error: {e}")
