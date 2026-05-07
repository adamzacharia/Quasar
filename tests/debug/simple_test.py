import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

print("Testing imports...")
try:
    from core.agent import QuasarAgent
    print("[OK] core.agent imported")
except ImportError as e:
    print(f"[FAIL] core.agent failed: {e}")

try:
    from services.search import SearchService
    print("[OK] services.search imported")
except ImportError as e:
    print(f"[FAIL] services.search failed: {e}")

try:
    from services.rag_service import RAGService
    print("[OK] services.rag_service imported")
    
    # Test RAG path resolution
    rag = RAGService()
    print(f"[OK] RAGService initialized with path: {rag.persist_directory}")
    if os.path.exists(rag.persist_directory):
        print("[OK] ChromaDB directory found")
    else:
        print(f"[!] ChromaDB directory not found at {rag.persist_directory}")
        
except ImportError as e:
    print(f"[FAIL] services.rag_service failed: {e}")
except Exception as e:
    print(f"[FAIL] RAGService init failed: {e}")

print("\nTests complete.")
