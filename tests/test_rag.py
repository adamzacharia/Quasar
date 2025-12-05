import sys
from pathlib import Path

# Add project root to path
root_path = Path(__file__).resolve().parent
sys.path.append(str(root_path))

from services.rag_service import RAGService
from dotenv import load_dotenv

load_dotenv()

def main():
    print("Initializing RAG Service...")
    rag = RAGService()
    
    query = "How do I use the sensitivity calculator?"
    print(f"\nQuery: {query}")
    
    docs = rag.search(query)
    
    if docs:
        print(f"\nFound {len(docs)} relevant documents:")
        for i, doc in enumerate(docs):
            print(f"\n--- Document {i+1} ---")
            print(doc.page_content[:200] + "...")
    else:
        print("\nNo documents found.")

if __name__ == "__main__":
    main()
