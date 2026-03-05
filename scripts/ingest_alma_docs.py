"""
Script to ingest all ALMA documentation into the RAG vector database
"""
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Load environment variables from .env
from dotenv import load_dotenv
load_dotenv()

from services.rag_service import RAGService

# Documents to ingest
ALMA_DOCS = [
    "ALMA Pipeline Known Issues.txt",
    "alma-large-program-data-products.pdf",
    "alma-ot-quickstart.pdf",
    "alma-ot-refmanual.pdf",
    "alma-ot-usermanual.pdf",
    "alma-proposers-guide.pdf",
    "alma-technical-handbook.pdf",
    "alma-user-policies.pdf",
    "alma_pipeline_users_guide_2025.pdf",
    "alma_tap_columns.txt",
]

def main():
    print("=" * 60)
    print("ALMA Documentation RAG Ingestion")
    print("=" * 60)
    
    # Initialize RAG service
    rag = RAGService()
    
    # Get current stats
    stats = rag.get_collection_stats()
    print(f"\nCurrent vector store: {stats}")
    
    # Get project root
    project_root = os.path.dirname(os.path.abspath(__file__))
    
    # Ingest each document
    results = {}
    for doc in ALMA_DOCS:
        file_path = os.path.join(project_root, doc)
        if os.path.exists(file_path):
            results[doc] = rag.ingest_document(file_path)
        else:
            print(f"✗ File not found: {doc}")
            results[doc] = False
    
    # Summary
    print("\n" + "=" * 60)
    print("INGESTION SUMMARY")
    print("=" * 60)
    success = sum(1 for v in results.values() if v)
    print(f"Successful: {success}/{len(results)}")
    
    for doc, status in results.items():
        icon = "✓" if status else "✗"
        print(f"  {icon} {doc}")
    
    # Final stats
    stats = rag.get_collection_stats()
    print(f"\nFinal vector store: {stats}")
    
    # Test search
    print("\n" + "=" * 60)
    print("TEST SEARCH: 'What is ALMA Band 6?'")
    print("=" * 60)
    results = rag.search("What is ALMA Band 6?", k=2)
    for i, doc in enumerate(results):
        print(f"\n[Result {i+1}] Source: {doc.metadata.get('source_file', 'unknown')}")
        print(f"Content: {doc.page_content[:300]}...")

if __name__ == "__main__":
    main()
